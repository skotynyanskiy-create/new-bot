"""
MeanReversionEngine — segnali di mean-reversion su mercati laterali.

Condizioni LONG:
  1. Prezzo <= BB_lower × 1.002  (tocca o buca la banda inferiore)
  2. RSI < rsi_oversold (default 32) — ipervenduto
  3. Volume < vol_ma × 1.3 — basso volume = no trend
  4. ADX < adx_max (default 22) — mercato non trending
  5. EMA_fast > EMA_slow × 0.995 — non in downtrend forte

Condizioni SHORT: speculari.

TP = BB_middle (ritorno alla media)
SL = entry ∓ sl_atr_mult × ATR
Score base: 65 (alta affidabilità in ranging)
"""

import logging
import time
from collections import deque
from decimal import Decimal

from src.config import BotSettings
from src.models import Candle, Signal, SignalType
from src.strategy.indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')
_MIN_NOTIONAL = Decimal('6')


class MeanReversionEngine:
    """
    Produce segnali di mean-reversion quando ADX < 22 (mercato ranging).

    Args:
        config     — BotSettings con parametri strategia
        indicators — IndicatorEngine già aggiornato per questo symbol
        capital    — capitale USDT disponibile
    """

    def __init__(
        self,
        config: BotSettings,
        indicators: IndicatorEngine,
        capital: Decimal,
    ) -> None:
        self._top_cfg    = config
        self._cfg        = config.strategy
        self._indicators = indicators
        self._capital    = capital

    # ------------------------------------------------------------------
    # Metodo principale
    # ------------------------------------------------------------------

    def detect(self, candles: deque[Candle]) -> Signal:
        """Analizza l'ultima candela e ritorna un Signal LONG/SHORT/NONE."""
        if not self._indicators.ready():
            return self._no_signal(candles[-1] if candles else None)

        candle = candles[-1]

        try:
            ema_fast   = self._indicators.ema_fast()
            ema_slow   = self._indicators.ema_slow()
            rsi_val    = self._indicators.rsi()
            atr_val    = self._indicators.atr()
            vol_ma     = self._indicators.volume_ma()
            adx_val    = self._indicators.adx()
            bb_upper   = self._indicators.bb_upper()
            bb_lower   = self._indicators.bb_lower()
            bb_middle  = self._indicators.bb_middle()
        except ValueError as e:
            logger.debug("Indicatori MR non pronti: %s", e)
            return self._no_signal(candle)

        # Filtro ATR minimo
        min_atr = candle.close * Decimal('0.0005')
        if atr_val < min_atr:
            return self._no_signal(candle)

        # Condizioni comuni
        adx_ok = adx_val < self._cfg.mr_adx_max

        # Volume: sia basso volume (ranging puro) che spike (capitolazione) sono validi
        # ma lo spike ha score più alto (segnale più affidabile)
        vol_ratio = candle.volume / vol_ma if vol_ma > _ZERO else Decimal('1')
        volume_ok = True  # Mean reversion accetta qualsiasi volume in ranging
        vol_score_bonus = _ZERO
        if vol_ratio > Decimal('3.0'):
            vol_score_bonus = Decimal('-5')   # Volume eccessivo = possibile trend forte, cautela
        elif vol_ratio > Decimal('2.0'):
            vol_score_bonus = Decimal('8')    # Spike volume = capitolazione → segnale forte
        elif vol_ratio > Decimal('1.5'):
            vol_score_bonus = Decimal('4')

        # BB width check: evita entry quando le bande sono troppo strette (squeeze imminente)
        bb_width = _ZERO
        if bb_middle > _ZERO:
            bb_width = (bb_upper - bb_lower) / bb_middle
        if bb_width < Decimal('0.005'):  # BB width < 0.5% del prezzo → squeeze, evita
            return self._no_signal(candle)

        base_score = Decimal('65')

        # ---- LONG (rimbalzo da BB lower) --------------------------------
        if (
            adx_ok
            and volume_ok
            and candle.close <= bb_lower * Decimal('1.002')
            and rsi_val < self._cfg.mr_rsi_oversold
            and ema_fast > ema_slow * Decimal('0.995')
        ):
            entry = candle.close
            tp    = bb_middle        # obiettivo = ritorno alla media
            sl    = entry - self._cfg.sl_atr_mult * atr_val
            # Verifica R:R minimo 1:1
            if (tp - entry) > (entry - sl) * Decimal('0.8'):
                size = self._calc_size(atr_val, entry)
                if size > _ZERO:
                    score = min(base_score + vol_score_bonus, Decimal('85'))
                    logger.info(
                        "LONG mean-rev %s | entry=%.4f tp=%.4f sl=%.4f adx=%.1f rsi=%.1f vol_ratio=%.1f",
                        candle.symbol, entry, tp, sl, adx_val, rsi_val, float(vol_ratio),
                    )
                    return Signal(
                        symbol=candle.symbol,
                        signal_type=SignalType.LONG,
                        entry_price=entry,
                        take_profit=tp,
                        stop_loss=sl,
                        size=size,
                        atr=atr_val,
                        timestamp=time.time(),
                        score=score,
                        strategy_name="mean_reversion",
                    )

        # ---- SHORT (rimbalzo da BB upper) --------------------------------
        if (
            adx_ok
            and volume_ok
            and candle.close >= bb_upper * Decimal('0.998')
            and rsi_val > self._cfg.mr_rsi_overbought
            and ema_fast < ema_slow * Decimal('1.005')
        ):
            entry = candle.close
            tp    = bb_middle        # obiettivo = ritorno alla media
            sl    = entry + self._cfg.sl_atr_mult * atr_val
            if (entry - tp) > (sl - entry) * Decimal('0.8'):
                size = self._calc_size(atr_val, entry)
                if size > _ZERO:
                    score = min(base_score + vol_score_bonus, Decimal('85'))
                    logger.info(
                        "SHORT mean-rev %s | entry=%.4f tp=%.4f sl=%.4f adx=%.1f rsi=%.1f vol_ratio=%.1f",
                        candle.symbol, entry, tp, sl, adx_val, rsi_val, float(vol_ratio),
                    )
                    return Signal(
                        symbol=candle.symbol,
                        signal_type=SignalType.SHORT,
                        entry_price=entry,
                        take_profit=tp,
                        stop_loss=sl,
                        size=size,
                        atr=atr_val,
                        timestamp=time.time(),
                        score=score,
                        strategy_name="mean_reversion",
                    )

        return self._no_signal(candle)

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _calc_size(self, atr: Decimal, price: Decimal) -> Decimal:
        risk_usdt   = self._capital * self._top_cfg.risk_per_trade_pct
        sl_distance = self._cfg.sl_atr_mult * atr
        if sl_distance == _ZERO or price == _ZERO:
            return _ZERO

        raw_size     = risk_usdt / sl_distance
        max_notional = self._capital * Decimal(str(self._top_cfg.leverage))
        max_size     = max_notional / price
        size         = min(raw_size, max_size).quantize(Decimal('0.001'))

        if size * price < _MIN_NOTIONAL:
            return _ZERO
        return size

    @staticmethod
    def _no_signal(candle: Candle | None) -> Signal:
        entry  = candle.close  if candle else _ZERO
        symbol = candle.symbol if candle else ""
        return Signal(
            symbol=symbol,
            signal_type=SignalType.NONE,
            entry_price=entry,
            take_profit=_ZERO,
            stop_loss=_ZERO,
            size=_ZERO,
            atr=_ZERO,
            timestamp=time.time(),
            score=_ZERO,
            strategy_name="mean_reversion",
        )
