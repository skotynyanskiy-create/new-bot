"""
BreakoutEngine — rileva segnali di breakout con conferma momentum.

Condizioni LONG:
  1. Chiusura candela > highest_high delle ultime N candele
  2. Volume > volume_multiplier × media_volume
  3. RSI tra rsi_min e rsi_max (momentum, non ipercomprato)
  4. EMA_fast > EMA_slow (trend rialzista)
  5. ATR > soglia minima (mercato con sufficiente volatilità)

Condizioni SHORT: speculari (RSI fuori dalla banda long, non dentro).

TP = entry ± tp_atr_mult × ATR
SL = entry ∓ sl_atr_mult × ATR
Size = (capitale × risk_pct) / (sl_atr_mult × ATR)

Minimo notional Binance Futures: $5 per ordine (validato prima di emettere segnale).
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

# Notional minimo accettato da Binance Futures (USDT)
_MIN_NOTIONAL = Decimal('6')


class BreakoutEngine:
    """
    Valuta ogni candela chiusa e produce un Signal (LONG / SHORT / NONE).

    Args:
        config     — BotSettings con strategia annidata
        indicators — IndicatorEngine già aggiornato per questo symbol
        capital    — capitale USDT disponibile per questo bot
    """

    def __init__(
        self,
        config: BotSettings,
        indicators: IndicatorEngine,
        capital: Decimal,
    ) -> None:
        self._top_cfg    = config           # risk_per_trade_pct, leverage
        self._cfg        = config.strategy  # breakout params
        self._indicators = indicators
        self._capital    = capital

    # ------------------------------------------------------------------
    # Metodo principale
    # ------------------------------------------------------------------

    def detect(self, candles: deque[Candle]) -> Signal:
        """
        Analizza l'ultima candela chiusa e ritorna un Signal.
        Ritorna SignalType.NONE se nessuna condizione è soddisfatta.
        """
        if not self._indicators.ready():
            return self._no_signal(candles[-1] if candles else None)

        candle = candles[-1]

        try:
            ema_fast = self._indicators.ema_fast()
            ema_slow = self._indicators.ema_slow()
            rsi_val  = self._indicators.rsi()
            atr_val  = self._indicators.atr()
            vol_ma   = self._indicators.volume_ma()
            hh       = self._indicators.highest_high(self._cfg.breakout_lookback)
            ll       = self._indicators.lowest_low(self._cfg.breakout_lookback)
        except ValueError as e:
            logger.debug("Indicatori non pronti: %s", e)
            return self._no_signal(candle)

        # Filtro mercato fermo: ATR deve essere almeno 0.05% del prezzo
        min_atr = candle.close * Decimal('0.0005')
        if atr_val < min_atr:
            return self._no_signal(candle)

        volume_ok = self._volume_ok(candle, vol_ma)

        # Multi-candle level validation: il HH/LL deve essere stato testato
        # da almeno 2 candle per essere un livello reale (evita spike singoli)
        lookback = self._cfg.breakout_lookback
        if len(candles) >= lookback:
            recent = list(candles)[-lookback:]
            hh_touches = sum(1 for c in recent if c.high >= hh * Decimal('0.998'))
            ll_touches = sum(1 for c in recent if c.low <= ll * Decimal('1.002'))
        else:
            hh_touches = 1
            ll_touches = 1

        # VWAP filter: LONG solo se prezzo > VWAP (bullish volume bias)
        vwap_long_ok  = self._indicators.price_above_vwap(candle.close)
        vwap_short_ok = self._indicators.price_below_vwap(candle.close)

        # ---- LONG --------------------------------------------------------
        # Confirmation candle: close > open (candela verde) conferma la rottura
        is_bullish = candle.close > candle.open
        if (
            candle.close > hh
            and is_bullish
            and volume_ok
            and hh_touches >= 2   # livello testato da almeno 2 candle
            and self._cfg.rsi_min <= rsi_val <= self._cfg.rsi_max
            and ema_fast > ema_slow
            and vwap_long_ok
        ):
            entry = candle.close
            tp    = entry + self._cfg.tp_atr_mult * atr_val
            sl    = entry - self._cfg.sl_atr_mult * atr_val
            size  = self._calc_size(atr_val, entry)
            if size > _ZERO:
                # Score dinamico: volume ratio più alto → breakout più convincente
                score = Decimal('60')
                vol_ratio = candle.volume / vol_ma if vol_ma > _ZERO else Decimal('1')
                if vol_ratio >= Decimal('2.5'):
                    score = Decimal('75')
                elif vol_ratio >= Decimal('2.0'):
                    score = Decimal('70')
                elif vol_ratio >= Decimal('1.5'):
                    score = Decimal('65')
                logger.info(
                    "LONG breakout %s | entry=%.4f tp=%.4f sl=%.4f size=%.6f atr=%.4f score=%.0f",
                    candle.symbol, entry, tp, sl, size, atr_val, score,
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
                )

        # ---- SHORT -------------------------------------------------------
        # RSI SHORT: breakout ribassista richiede momentum ribassista (RSI basso).
        # Specularmente al LONG (rsi_min <= rsi <= rsi_max), lo SHORT richiede
        # RSI nella banda bassa: (100 - rsi_max) <= RSI <= (100 - rsi_min).
        # Default: 28 <= RSI <= 55 per SHORT (momentum ribassista, non ipervenduto estremo).
        rsi_short_ok = (Decimal('100') - self._cfg.rsi_max) <= rsi_val <= (Decimal('100') - self._cfg.rsi_min)
        is_bearish = candle.close < candle.open
        if (
            candle.close < ll
            and is_bearish
            and volume_ok
            and ll_touches >= 2   # livello testato da almeno 2 candle
            and rsi_short_ok
            and ema_fast < ema_slow
            and vwap_short_ok
        ):
            entry = candle.close
            tp    = entry - self._cfg.tp_atr_mult * atr_val
            sl    = entry + self._cfg.sl_atr_mult * atr_val
            size  = self._calc_size(atr_val, entry)
            if size > _ZERO:
                score = Decimal('60')
                vol_ratio = candle.volume / vol_ma if vol_ma > _ZERO else Decimal('1')
                if vol_ratio >= Decimal('2.5'):
                    score = Decimal('75')
                elif vol_ratio >= Decimal('2.0'):
                    score = Decimal('70')
                elif vol_ratio >= Decimal('1.5'):
                    score = Decimal('65')
                logger.info(
                    "SHORT breakout %s | entry=%.4f tp=%.4f sl=%.4f size=%.6f atr=%.4f score=%.0f",
                    candle.symbol, entry, tp, sl, size, atr_val, score,
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
                )

        return self._no_signal(candle)

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _volume_ok(self, candle: Candle, vol_ma: Decimal) -> bool:
        if vol_ma == _ZERO:
            return False
        return candle.volume >= self._cfg.volume_multiplier * vol_ma

    def _calc_size(self, atr: Decimal, price: Decimal) -> Decimal:
        """
        size = (capitale × risk_pct) / (sl_atr_mult × ATR)
        Limita a: notional <= capitale × leva
        Valida: notional >= _MIN_NOTIONAL (minimo Binance Futures $6)
        """
        risk_usdt   = self._capital * self._top_cfg.risk_per_trade_pct
        sl_distance = self._cfg.sl_atr_mult * atr
        if sl_distance == _ZERO or price == _ZERO:
            return _ZERO

        raw_size = risk_usdt / sl_distance

        # Limita a notional massimo = capitale × leva
        max_notional = self._capital * Decimal(str(self._top_cfg.leverage))
        max_size     = max_notional / price
        size         = min(raw_size, max_size)

        # Arrotonda a 3 decimali (step size comune per BTC/ETH Futures)
        size = size.quantize(Decimal('0.001'))

        # Verifica notional minimo Binance Futures
        if size * price < _MIN_NOTIONAL:
            logger.debug(
                "Size %.6f troppo piccola (notional=%.2f USDT < %.0f USDT min). Segnale scartato.",
                size, float(size * price), float(_MIN_NOTIONAL),
            )
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
        )
