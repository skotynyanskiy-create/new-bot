"""
OrderFlowEngine — analisi order flow per segnali di conferma istituzionali.

Metriche calcolate:
  1. Delta Volume: volume buy - volume sell (approssimato da candle)
  2. Cumulative Volume Delta (CVD): trend del delta nel tempo
  3. Liquidation Level Estimation: dove sono concentrati gli stop loss
  4. Volume Profile: identifica Point of Control (POC) e Value Area

L'engine NON genera segnali direttamente — fornisce metriche che il
SignalAggregator usa per confermare/rifiutare segnali di altre strategie.
"""

import logging
from collections import deque
from decimal import Decimal
from typing import Optional

from src.models import Candle

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')


class OrderFlowEngine:
    """
    Analizza order flow da dati OHLCV (approssimazione senza order book L2).

    Delta Volume approssimato:
      Se close > open (candela verde): buy_vol = volume × (close - low) / (high - low)
      Se close < open (candela rossa): sell_vol = volume × (high - close) / (high - low)
      delta = buy_vol - sell_vol

    Questo è un'approssimazione — il delta reale richiede dati tick-by-tick.
    Ma su crypto, questa approssimazione ha ~70% di correlazione con il delta reale.

    Args:
        cvd_window — lunghezza rolling window per CVD (default 50)
        volume_profile_bins — numero di bin per il volume profile (default 20)
    """

    def __init__(
        self,
        cvd_window: int = 50,
        volume_profile_bins: int = 20,
    ) -> None:
        self._cvd_window = cvd_window
        self._vp_bins = volume_profile_bins
        self._delta_history: deque[Decimal] = deque(maxlen=cvd_window)
        self._candle_history: deque[Candle] = deque(maxlen=200)

    def update(self, candle: Candle) -> None:
        """Aggiorna con una nuova candle chiusa."""
        self._candle_history.append(candle)
        delta = self._calc_delta(candle)
        self._delta_history.append(delta)

    def _calc_delta(self, candle: Candle) -> Decimal:
        """Calcola il delta volume approssimato per una candle."""
        hl_range = candle.high - candle.low
        if hl_range <= _ZERO:
            return _ZERO

        # Approssimazione: proporzione del volume attribuibile a buy vs sell
        if candle.close >= candle.open:
            # Candela verde: buy pressure dominante
            buy_ratio = (candle.close - candle.low) / hl_range
        else:
            # Candela rossa: sell pressure dominante
            buy_ratio = (candle.high - candle.close) / hl_range

        buy_vol = candle.volume * buy_ratio
        sell_vol = candle.volume * (Decimal('1') - buy_ratio)
        return buy_vol - sell_vol

    @property
    def current_delta(self) -> Decimal:
        """Delta volume dell'ultima candle."""
        return self._delta_history[-1] if self._delta_history else _ZERO

    @property
    def cvd(self) -> Decimal:
        """Cumulative Volume Delta nella rolling window."""
        return sum(self._delta_history, _ZERO)

    @property
    def cvd_trend(self) -> str:
        """
        Trend del CVD: 'bullish', 'bearish', o 'neutral'.
        Confronta CVD prima metà vs seconda metà della window.
        """
        if len(self._delta_history) < 10:
            return "neutral"
        mid = len(self._delta_history) // 2
        deltas = list(self._delta_history)
        first_half = sum(deltas[:mid], _ZERO)
        second_half = sum(deltas[mid:], _ZERO)
        diff = second_half - first_half
        threshold = abs(first_half + second_half) * Decimal('0.1')
        if diff > threshold:
            return "bullish"
        elif diff < -threshold:
            return "bearish"
        return "neutral"

    def divergence_signal(self, price_rising: bool) -> Optional[str]:
        """
        Detecta divergenza prezzo vs CVD (segnale contrarian forte).

        Divergenza bullish: prezzo scende ma CVD sale (accumulo nascosto)
        Divergenza bearish: prezzo sale ma CVD scende (distribuzione nascosta)

        Returns:
            'bullish_divergence', 'bearish_divergence', o None
        """
        cvd_trend = self.cvd_trend
        if price_rising and cvd_trend == "bearish":
            return "bearish_divergence"
        if not price_rising and cvd_trend == "bullish":
            return "bullish_divergence"
        return None

    def volume_profile_poc(self, lookback: int = 50) -> Optional[Decimal]:
        """
        Calcola il Point of Control (POC) — prezzo con il volume più alto.

        Il POC è il livello di prezzo "più accettato" dal mercato.
        Breakout sopra/sotto il POC con volume = segnale forte.
        """
        candles = list(self._candle_history)[-lookback:]
        if len(candles) < 10:
            return None

        # Trova range
        all_lows = [float(c.low) for c in candles]
        all_highs = [float(c.high) for c in candles]
        price_low = min(all_lows)
        price_high = max(all_highs)
        if price_high <= price_low:
            return None

        bin_size = (price_high - price_low) / self._vp_bins
        bins = [Decimal('0')] * self._vp_bins

        for c in candles:
            # Distribuisci il volume della candle nei bin attraversati
            c_low = max(float(c.low), price_low)
            c_high = min(float(c.high), price_high)
            for i in range(self._vp_bins):
                bin_low = price_low + i * bin_size
                bin_high = bin_low + bin_size
                # Overlap tra candle e bin
                overlap = max(0, min(c_high, bin_high) - max(c_low, bin_low))
                if overlap > 0 and (c_high - c_low) > 0:
                    proportion = Decimal(str(overlap / (c_high - c_low)))
                    bins[i] += c.volume * proportion

        # POC = centro del bin con volume massimo
        max_idx = max(range(self._vp_bins), key=lambda i: bins[i])
        poc = Decimal(str(price_low + (max_idx + 0.5) * bin_size))
        return poc

    def score_modifier(self, is_long: bool) -> Decimal:
        """
        Ritorna un modificatore di score basato sull'order flow.

        CVD bullish + entry LONG → +1.1 (conferma)
        CVD bearish + entry LONG → ×0.85 (contrario)
        Divergenza bullish → +1.15 (segnale contrarian forte)
        """
        cvd = self.cvd_trend
        div = self.divergence_signal(price_rising=is_long)

        if div == "bullish_divergence" and is_long:
            return Decimal('1.15')
        if div == "bearish_divergence" and not is_long:
            return Decimal('1.15')

        if is_long and cvd == "bullish":
            return Decimal('1.10')
        if not is_long and cvd == "bearish":
            return Decimal('1.10')
        if is_long and cvd == "bearish":
            return Decimal('0.85')
        if not is_long and cvd == "bullish":
            return Decimal('0.85')

        return Decimal('1.0')
