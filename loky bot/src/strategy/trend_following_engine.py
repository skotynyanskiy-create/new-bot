"""
TrendFollowingEngine — cattura mega-trend con EMA Ribbon + ADX.

Condizioni LONG:
  1. EMA ribbon aligned bullish: 8 > 13 > 21 > 34 > 55
  2. ADX > adx_min (default 28) — trend forte
  3. DI+ > DI- — pressione rialzista confermata
  4. Pullback: close tra EMA21 e EMA13 (entry sul pullback, non all'apice)
  5. RSI > 45 — non ipervenduto nel trend

Condizioni SHORT: speculari.

TP = entry ± tf_tp_atr_mult × ATR  (default 3.0×, R:R 3:1)
SL = entry ∓ tf_sl_atr_mult × ATR  (default 1.2×)
Score base: 72 (trend forti = alta affidabilità)
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


class TrendFollowingEngine:
    """
    Produce segnali trend-following basati su EMA ribbon e ADX.

    Args:
        config     — BotSettings
        indicators — IndicatorEngine già aggiornato
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
            rsi_val  = self._indicators.rsi()
            atr_val  = self._indicators.atr()
            adx_val  = self._indicators.adx()
            di_plus  = self._indicators.di_plus()
            di_minus = self._indicators.di_minus()
            ribbon   = self._indicators.ema_ribbon()   # (e8, e13, e21, e34, e55)
        except ValueError as e:
            logger.debug("Indicatori TF non pronti: %s", e)
            return self._no_signal(candle)

        e8, e13, e21, e34, e55 = ribbon

        # Filtro ATR minimo
        min_atr = candle.close * Decimal('0.0005')
        if atr_val < min_atr:
            return self._no_signal(candle)

        adx_strong = adx_val > self._cfg.tf_adx_min

        # ---- LONG (pullback in uptrend) ----------------------------------
        # Pullback ampio: prezzo tra EMA21 e EMA8 (finestra più larga per catturare più setup)
        # Confirmation candle: close > open (candela rialzista = momentum ripreso)
        is_bullish_candle = candle.close > candle.open
        if (
            adx_strong
            and di_plus > di_minus
            and self._indicators.ribbon_aligned_bullish()
            and e21 <= candle.close <= e8    # pullback ampio: tra EMA21 e EMA8
            and is_bullish_candle            # conferma: candela verde (momentum ripreso)
            and rsi_val > Decimal('45')
        ):
            entry = candle.close
            tp    = entry + self._cfg.tf_tp_atr_mult * atr_val
            sl    = entry - self._cfg.tf_sl_atr_mult * atr_val
            # Score dinamico: pullback profondo (vicino a EMA21) → score più alto (entry migliore)
            # Skip se EMAs collassate (< 0.1% spread → trend debole, non entrare)
            if abs(e8 - e21) < candle.close * Decimal('0.001'):
                return self._no_signal(candle)
            depth_ratio = (e8 - candle.close) / (e8 - e21)
            score = Decimal('72') + min(depth_ratio * Decimal('8'), Decimal('8'))
            size  = self._calc_size(atr_val, entry, self._cfg.tf_sl_atr_mult)
            if size > _ZERO:
                logger.info(
                    "LONG trend-follow %s | entry=%.4f tp=%.4f sl=%.4f adx=%.1f DI+/DI-=%.1f/%.1f depth=%.0f%%",
                    candle.symbol, entry, tp, sl, adx_val, di_plus, di_minus, float(depth_ratio * 100),
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
                    strategy_name="trend_following",
                )

        # ---- SHORT (pullback in downtrend) ------------------------------
        is_bearish_candle = candle.close < candle.open
        if (
            adx_strong
            and di_minus > di_plus
            and self._indicators.ribbon_aligned_bearish()
            and e8 <= candle.close <= e21    # pullback ampio: tra EMA8 e EMA21
            and is_bearish_candle            # conferma: candela rossa (momentum ripreso)
            and rsi_val < Decimal('55')
        ):
            entry = candle.close
            tp    = entry - self._cfg.tf_tp_atr_mult * atr_val
            sl    = entry + self._cfg.tf_sl_atr_mult * atr_val
            if abs(e8 - e21) < candle.close * Decimal('0.001'):
                return self._no_signal(candle)
            depth_ratio = (candle.close - e8) / (e21 - e8)
            score = Decimal('72') + min(depth_ratio * Decimal('8'), Decimal('8'))
            size  = self._calc_size(atr_val, entry, self._cfg.tf_sl_atr_mult)
            if size > _ZERO:
                logger.info(
                    "SHORT trend-follow %s | entry=%.4f tp=%.4f sl=%.4f adx=%.1f DI-/DI+=%.1f/%.1f depth=%.0f%%",
                    candle.symbol, entry, tp, sl, adx_val, di_minus, di_plus, float(depth_ratio * 100),
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
                    strategy_name="trend_following",
                )

        return self._no_signal(candle)

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _calc_size(self, atr: Decimal, price: Decimal, sl_mult: Decimal) -> Decimal:
        risk_usdt   = self._capital * self._top_cfg.risk_per_trade_pct
        sl_distance = sl_mult * atr
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
            strategy_name="trend_following",
        )
