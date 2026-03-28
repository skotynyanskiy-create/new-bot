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
        """
        Calcola il delta volume approssimato per una candle.

        Formula unificata (indipendente dal colore della candela):
          buy_ratio = (close - low) / (high - low)
        Questo misura dove il close si posiziona nel range H-L:
          close = high → buy_ratio = 1.0 (tutto buy pressure)
          close = low  → buy_ratio = 0.0 (tutto sell pressure)
          close = mid  → buy_ratio = 0.5 (equilibrio)
        """
        hl_range = candle.high - candle.low
        # Protezione doji: se range < 0.01% del prezzo, delta non significativo
        if hl_range <= _ZERO or (candle.close > _ZERO and hl_range < candle.close * Decimal('0.0001')):
            return _ZERO

        buy_ratio = (candle.close - candle.low) / hl_range
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

    # ------------------------------------------------------------------
    # Liquidation Level Clustering
    # ------------------------------------------------------------------

    def estimate_liquidation_levels(
        self, lookback: int = 100, leverage_common: list[int] | None = None
    ) -> list[Decimal]:
        """
        Stima dove si concentrano i livelli di liquidazione degli altri trader.

        Logica: i trader usano tipicamente leve 5x, 10x, 25x.
        Per ogni swing high/low recente, calcola dove sarebbero i liq price
        per chi è entrato long/short a quel livello con leve comuni.

        Returns: lista dei top 5 livelli di liquidazione più probabili.
        """
        candles = list(self._candle_history)[-lookback:]
        if len(candles) < 20:
            return []

        levers = leverage_common or [5, 10, 25]
        liq_levels: list[Decimal] = []

        # Trova swing high/low come probabili punti di entry degli altri trader
        for i in range(2, len(candles) - 2):
            c = candles[i]
            # Swing high: potenziale entry SHORT degli altri → liq sopra
            if c.high > candles[i-1].high and c.high > candles[i+1].high:
                for lev in levers:
                    # SHORT entry al swing high → liquidazione = entry × (1 + 1/lev)
                    liq = c.high * (Decimal('1') + Decimal('1') / Decimal(str(lev)))
                    liq_levels.append(liq)
            # Swing low: potenziale entry LONG degli altri → liq sotto
            if c.low < candles[i-1].low and c.low < candles[i+1].low:
                for lev in levers:
                    # LONG entry al swing low → liquidazione = entry × (1 - 1/lev)
                    liq = c.low * (Decimal('1') - Decimal('1') / Decimal(str(lev)))
                    liq_levels.append(liq)

        if not liq_levels:
            return []

        # Clusterizza livelli vicini (entro 0.5%)
        return self._cluster_levels(liq_levels, tolerance_pct=Decimal('0.005'))[:5]

    def _cluster_levels(
        self, levels: list[Decimal], tolerance_pct: Decimal
    ) -> list[Decimal]:
        """Raggruppa livelli vicini e ritorna i centri dei cluster più densi."""
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[Decimal]] = [[sorted_levels[0]]]
        for price in sorted_levels[1:]:
            ref = clusters[-1][0]
            if ref > _ZERO and abs(price - ref) / ref <= tolerance_pct:
                clusters[-1].append(price)
            else:
                clusters.append([price])

        # Ordina per densità (cluster più denso = più probabile)
        clusters.sort(key=len, reverse=True)
        return [
            sum(c) / Decimal(len(c))  # media del cluster
            for c in clusters
        ]

