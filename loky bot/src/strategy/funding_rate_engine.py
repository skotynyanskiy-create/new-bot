"""
FundingRateEngine — harvesting del funding rate Bybit Perpetual.

Logica:
  - Polling REST /v5/market/funding/history ogni ciclo
  - funding_rate > +threshold (default 0.08% per 8h) → SHORT
    (i long pagano i short → ricevi funding stando short)
  - funding_rate < -threshold → LONG
    (gli short pagano i long → ricevi funding stando long)
  - Exit dopo max_hold_hours (default 8h = 1 ciclo funding)
  - SL stretto: 0.5×ATR (protezione da spike, il rendimento è il funding)
  - Size ridotta: 50% del normale (basso rischio, rendimento costante)

Score base: 80 (funding > threshold = alta certezza)

NOTA: Questo engine è asincrono (deve fare una chiamata REST per ottenere
il funding rate). Viene chiamato dalla versione asincrona on_candle().
"""

import asyncio
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Optional

import aiohttp

from src.config import BotSettings
from src.models import Candle, Signal, SignalType
from src.strategy.indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')
_MIN_NOTIONAL = Decimal('6')

# Endpoint Bybit V5 funding rate
_BYBIT_FUNDING_URL      = "https://api.bybit.com/v5/market/funding/history"
_BYBIT_TESTNET_FUND_URL = "https://api-testnet.bybit.com/v5/market/funding/history"


class FundingRateEngine:
    """
    Harvesta il funding rate su Bybit Linear Perpetual.

    Args:
        config     — BotSettings
        indicators — IndicatorEngine già aggiornato
        capital    — capitale USDT disponibile
        testnet    — True = usa endpoint testnet
    """

    def __init__(
        self,
        config: BotSettings,
        indicators: IndicatorEngine,
        capital: Decimal,
        testnet: bool = False,
    ) -> None:
        self._top_cfg    = config
        self._cfg        = config.strategy
        self._indicators = indicators
        self._capital    = capital
        self._base_url   = _BYBIT_TESTNET_FUND_URL if testnet else _BYBIT_FUNDING_URL
        self._last_rate: Optional[Decimal] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Metodo principale (async)
    # ------------------------------------------------------------------

    async def detect(self, candles: deque[Candle]) -> Signal:
        """
        Recupera il funding rate corrente e genera segnale se sopra/sotto threshold.
        In paper/backtest mode, skip HTTP call (no funding rate data disponibile).
        """
        if not self._indicators.ready():
            return self._no_signal(candles[-1] if candles else None)

        # Skip HTTP in paper/backtest: funding rate non disponibile senza API live
        if not self._top_cfg.live_trading_enabled:
            return self._no_signal(candles[-1] if candles else None)

        candle = candles[-1]

        try:
            atr_val = self._indicators.atr()
        except ValueError:
            return self._no_signal(candle)

        funding_rate = await self._fetch_funding_rate(candle.symbol)
        if funding_rate is None:
            return self._no_signal(candle)

        self._last_rate = funding_rate
        threshold = self._cfg.funding_threshold

        entry = candle.close
        sl_distance = Decimal('0.5') * atr_val

        # TP a 1.0×ATR, SL a 0.8×ATR → R:R 1.25:1 (ragionevole per funding harvest).
        # Il profitto principale viene dal funding stesso, il TP è un bonus.
        # Score proporzionale al funding rate: rate più alto = segnale più forte.
        tp_distance = Decimal('1.0') * atr_val
        sl_distance = Decimal('0.8') * atr_val

        # Score dinamico: funding rate più alto → più affidabile
        abs_rate = abs(funding_rate)
        if abs_rate > threshold * 3:
            score = Decimal('85')   # funding estremo
        elif abs_rate > threshold * 2:
            score = Decimal('80')
        else:
            score = Decimal('72')   # appena sopra threshold

        # funding positivo → SHORT (long pagano short)
        if funding_rate > threshold:
            tp   = entry - tp_distance
            sl   = entry + sl_distance
            size = self._calc_size(sl_distance, entry, fraction=Decimal('0.5'))
            if size > _ZERO:
                logger.info(
                    "SHORT funding-harvest %s | rate=%.4f%% tp=%.4f sl=%.4f score=%.0f",
                    candle.symbol, float(funding_rate * 100), tp, sl, score,
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
                    strategy_name="funding_rate",
                )

        # funding negativo → LONG (short pagano long)
        if funding_rate < -threshold:
            tp   = entry + tp_distance
            sl   = entry - sl_distance
            size = self._calc_size(sl_distance, entry, fraction=Decimal('0.5'))
            if size > _ZERO:
                logger.info(
                    "LONG funding-harvest %s | rate=%.4f%% tp=%.4f sl=%.4f score=%.0f",
                    candle.symbol, float(funding_rate * 100), tp, sl, score,
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
                    strategy_name="funding_rate",
                )

        return self._no_signal(candle)

    # ------------------------------------------------------------------
    # Fetch funding rate
    # ------------------------------------------------------------------

    async def _fetch_funding_rate(self, symbol: str) -> Optional[Decimal]:
        """Chiama REST Bybit e ritorna l'ultimo funding rate (come Decimal)."""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": "1",
            }
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=8)
                    )
                session = self._session
            async with session.get(self._base_url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    items = data.get("result", {}).get("list", [])
                    if items:
                        rate_str = items[0].get("fundingRate", "0")
                        return Decimal(str(rate_str))
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Errore fetch funding rate %s: %s", symbol, e)
            # Chiudi sessione corrotta per forzare ricreazione al prossimo ciclo
            async with self._session_lock:
                if self._session and not self._session.closed:
                    await self._session.close()
                self._session = None
        except Exception as e:
            logger.warning("Errore inatteso fetch funding rate %s: %s", symbol, e)
        return None

    async def close(self) -> None:
        """Chiude la sessione HTTP e nullifica il riferimento."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _calc_size(
        self, sl_distance: Decimal, price: Decimal, fraction: Decimal = Decimal('1')
    ) -> Decimal:
        risk_usdt = self._capital * self._top_cfg.risk_per_trade_pct * fraction
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
            strategy_name="funding_rate",
        )
