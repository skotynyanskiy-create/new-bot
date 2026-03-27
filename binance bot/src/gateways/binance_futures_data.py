"""
BinanceFuturesDataGateway — dati di mercato in tempo reale da Binance USDT-M Futures.

Funzionalità:
  • WebSocket kline (candele) su /fapi/v1/stream (combinedStream)
  • Bootstrap delle candele storiche via REST GET /fapi/v1/klines
  • Reconnect automatico con backoff esponenziale
  • Callback on_candle_close(symbol, candle) su candela chiusa

URL base: wss://fstream.binance.com/stream
REST base: https://fapi.binance.com
"""

import asyncio
import json
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Callable, Awaitable

import aiohttp

from src.models import Candle

logger = logging.getLogger(__name__)

_WS_MAINNET   = "wss://fstream.binance.com/stream"
_WS_TESTNET   = "wss://stream.binancefuture.com/stream"
_REST_MAINNET = "https://fapi.binance.com"
_REST_TESTNET = "https://testnet.binancefuture.com"

# Secondi per timeframe → per bootstrap storico
_TF_SECONDS = {
    "1m":  60, "3m":  180, "5m":  300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


class BinanceFuturesDataGateway:
    """
    Gateway dati Futures. Emette eventi Candle al callback on_candle_close.

    Uso:
        gw = BinanceFuturesDataGateway(on_candle_close=my_callback)
        await gw.start(["BTCUSDT", "SOLUSDT"], ["15m", "1h"])
    """

    def __init__(
        self,
        on_candle_close: Callable[[str, Candle], Awaitable[None]],
        bootstrap_bars: int = 100,
        testnet: bool = False,
    ) -> None:
        self._on_candle_close = on_candle_close
        self._bootstrap_bars  = bootstrap_bars
        self._running         = False
        self._ws_task: asyncio.Task | None = None
        self._ws_base   = _WS_TESTNET   if testnet else _WS_MAINNET
        self._rest_base = _REST_TESTNET  if testnet else _REST_MAINNET
        # candle buffer per symbol+timeframe — per indicatori
        self._buffers: dict[str, deque[Candle]] = {}

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        """Avvia il gateway: bootstrap storico + WebSocket streaming."""
        self._running = True
        self._symbols    = [s.upper() for s in symbols]
        self._timeframes = timeframes

        # Inizializza buffer
        for sym in self._symbols:
            for tf in self._timeframes:
                self._buffers[f"{sym}_{tf}"] = deque(maxlen=200)

        # Bootstrap storico (REST)
        await self._bootstrap_history()

        # Stream WebSocket
        self._ws_task = asyncio.create_task(self._run_websocket())

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()

    def get_buffer(self, symbol: str, timeframe: str) -> deque[Candle]:
        return self._buffers.get(f"{symbol.upper()}_{timeframe}", deque())

    # ------------------------------------------------------------------
    # Bootstrap storico via REST
    # ------------------------------------------------------------------

    async def _bootstrap_history(self) -> None:
        async with aiohttp.ClientSession() as session:
            for sym in self._symbols:
                for tf in self._timeframes:
                    await self._fetch_klines(session, sym, tf)
                    await asyncio.sleep(0.05)  # rate limit

    async def _fetch_klines(
        self, session: aiohttp.ClientSession, symbol: str, timeframe: str
    ) -> None:
        url = f"{self._rest_base}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": timeframe, "limit": self._bootstrap_bars}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
            key = f"{symbol}_{timeframe}"
            for row in data[:-1]:  # esclude l'ultima (non chiusa)
                candle = _parse_rest_kline(row, symbol, timeframe)
                self._buffers[key].append(candle)
            logger.info("Bootstrap %s %s: %d candele caricate", symbol, timeframe, len(data) - 1)
        except Exception as e:
            logger.warning("Bootstrap fallito %s %s: %s", symbol, timeframe, e)

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def _run_websocket(self) -> None:
        # Costruisce combined stream: btcusdt@kline_15m/btcusdt@kline_1h/...
        streams = "/".join(
            f"{sym.lower()}@kline_{tf}"
            for sym in self._symbols
            for tf in self._timeframes
        )
        url = f"{self._ws_base}?streams={streams}"
        backoff = 1

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info("WebSocket Futures connesso: %d stream", len(self._symbols) * len(self._timeframes))
                        backoff = 1
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning("WebSocket chiuso/errore, reconnect...")
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WebSocket errore: %s — retry in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            # Combined stream wrappa in {"stream": "...", "data": {...}}
            data = msg.get("data", msg)
            if data.get("e") != "kline":
                return
            k = data["k"]
            if not k["x"]:   # x = is_closed
                return
            symbol   = k["s"]
            timeframe = k["i"]
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                open=Decimal(k["o"]),
                high=Decimal(k["h"]),
                low=Decimal(k["l"]),
                close=Decimal(k["c"]),
                volume=Decimal(k["v"]),
                timestamp=k["t"] / 1000.0,
                is_closed=True,
            )
            key = f"{symbol}_{timeframe}"
            if key in self._buffers:
                self._buffers[key].append(candle)
            await self._on_candle_close(symbol, candle)
        except Exception as e:
            logger.error("Errore parsing messaggio WebSocket: %s", e)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _parse_rest_kline(row: list, symbol: str, timeframe: str) -> Candle:
    """Converte una riga REST kline in Candle."""
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=Decimal(str(row[5])),
        timestamp=row[0] / 1000.0,
        is_closed=True,
    )
