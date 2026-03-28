"""
BybitFuturesDataGateway — dati di mercato in tempo reale da Bybit V5 Linear Futures.

Funzionalità:
  • WebSocket kline su wss://stream.bybit.com/v5/public/linear
  • Bootstrap storico via REST GET /v5/market/kline
  • Reconnect automatico con backoff esponenziale
  • Callback on_candle_close(symbol, candle) su candela chiusa (confirm=true)

Conversione timeframe: "15m" → "15", "1h" → "60", "4h" → "240"
"""

import asyncio
import json
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Awaitable, Callable

import aiohttp

from src.models import Candle

logger = logging.getLogger(__name__)

_WS_MAINNET  = "wss://stream.bybit.com/v5/public/linear"
_WS_TESTNET  = "wss://stream-testnet.bybit.com/v5/public/linear"
_REST_MAINNET = "https://api.bybit.com"
_REST_TESTNET = "https://api-testnet.bybit.com"

# Binance → Bybit timeframe conversion
_TF_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D",
}


def _to_bybit_tf(tf: str) -> str:
    """Converte timeframe Binance-style in Bybit interval."""
    return _TF_MAP.get(tf, tf)


class BybitFuturesDataGateway:
    """
    Gateway dati Bybit Linear Futures V5.
    Emette eventi Candle al callback on_candle_close.

    Uso:
        gw = BybitFuturesDataGateway(on_candle_close=my_callback)
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
        self._buffers: dict[str, deque[Candle]] = {}
        self._last_candle_ts: dict[str, float] = {}   # Dedup: ultimo timestamp per key

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        """Avvia il gateway: bootstrap storico + WebSocket streaming."""
        self._running    = True
        self._symbols    = [s.upper() for s in symbols]
        self._timeframes = timeframes

        for sym in self._symbols:
            for tf in self._timeframes:
                self._buffers[f"{sym}_{tf}"] = deque(maxlen=200)
                self._last_candle_ts[f"{sym}_{tf}"] = 0.0

        await self._bootstrap_history()
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
                    await asyncio.sleep(0.05)

    async def _fetch_klines(
        self, session: aiohttp.ClientSession, symbol: str, timeframe: str
    ) -> None:
        url = f"{self._rest_base}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": _to_bybit_tf(timeframe),
            "limit":    self._bootstrap_bars,
        }
        try:
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            rows = data.get("result", {}).get("list", [])
            # Bybit restituisce le candele in ordine inverso (più recente prima)
            rows = list(reversed(rows))

            key = f"{symbol}_{timeframe}"
            if not rows:
                logger.warning("Bootstrap %s %s: nessuna candela ricevuta", symbol, timeframe)
                return
            for row in rows[:-1]:  # esclude l'ultima (non ancora chiusa)
                candle = _parse_bybit_kline(row, symbol, timeframe)
                self._buffers[key].append(candle)
                self._last_candle_ts[key] = candle.timestamp

            logger.info(
                "Bootstrap %s %s: %d candele caricate", symbol, timeframe, len(rows) - 1
            )
        except Exception as e:
            logger.warning("Bootstrap fallito %s %s: %s", symbol, timeframe, e)

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def _run_websocket(self) -> None:
        backoff = 1
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self._ws_base, heartbeat=20,
                        timeout=aiohttp.ClientTimeout(total=None, sock_read=60),
                    ) as ws:
                        # Sottoscrivi tutti i topic kline
                        args = [
                            f"kline.{_to_bybit_tf(tf)}.{sym}"
                            for sym in self._symbols
                            for tf in self._timeframes
                        ]
                        await ws.send_json({"op": "subscribe", "args": args})
                        logger.info(
                            "Bybit WebSocket connesso: %d stream", len(args)
                        )
                        backoff = 1

                        # Gap fill: recupera candle perse durante la disconnessione
                        await self._fill_reconnect_gaps()

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(msg.data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                logger.warning("WS chiuso/errore, reconnect...")
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WS errore: %s — retry in %ds", e, backoff)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)

            # Ignora messaggi di controllo (subscribe response, pong, ecc.)
            if "topic" not in msg:
                return

            topic: str = msg["topic"]  # es. "kline.15.BTCUSDT"
            if not topic.startswith("kline."):
                return

            parts = topic.split(".")  # ["kline", "15", "BTCUSDT"]
            if len(parts) < 3:
                return

            bybit_interval = parts[1]
            symbol         = parts[2]

            # Converti Bybit interval → timeframe originale ("15" → "15m")
            timeframe = _from_bybit_tf(bybit_interval)

            # Valida timeframe: ignora stream non sottoscritti
            if timeframe not in self._timeframes:
                return

            for item in msg.get("data", []):
                if not item.get("confirm", False):
                    continue  # candela non ancora chiusa

                candle_ts = item["start"] / 1000.0
                key = f"{symbol}_{timeframe}"

                # Dedup: ignora se timestamp già processato (Bybit può reinviare)
                if key in self._last_candle_ts and candle_ts <= self._last_candle_ts[key]:
                    continue

                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=Decimal(str(item["open"])),
                    high=Decimal(str(item["high"])),
                    low=Decimal(str(item["low"])),
                    close=Decimal(str(item["close"])),
                    volume=Decimal(str(item["volume"])),
                    timestamp=candle_ts,
                    is_closed=True,
                )

                if key in self._buffers:
                    self._buffers[key].append(candle)
                    self._last_candle_ts[key] = candle_ts

                await self._on_candle_close(symbol, candle)

        except Exception as e:
            logger.error("Errore parsing WS: %s", e)

    async def _fill_reconnect_gaps(self) -> None:
        """
        Dopo reconnessione WS, detecta gap nelle candle e recupera
        le mancanti via REST API per mantenere gli indicatori accurati.
        """
        now = time.time()
        tf_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400,
        }
        for sym in self._symbols:
            for tf in self._timeframes:
                key = f"{sym}_{tf}"
                last_ts = self._last_candle_ts.get(key, 0)
                if last_ts <= 0:
                    continue
                gap_s = now - last_ts
                tf_s = tf_seconds.get(tf, 900)
                # Se gap > 2× la durata del timeframe, recupera candle mancanti
                if gap_s > tf_s * 2:
                    n_missed = int(gap_s / tf_s)
                    logger.warning(
                        "Gap detected %s %s: %.0fs gap (~%d candle). Fetching...",
                        sym, tf, gap_s, n_missed,
                    )
                    try:
                        candles = await self._fetch_gap_candles(
                            sym, tf, int(last_ts * 1000), int(now * 1000), min(n_missed + 2, 200)
                        )
                        for c in candles:
                            if c.timestamp > last_ts:
                                await self._on_candle_close(sym, c)
                                self._last_candle_ts[key] = c.timestamp
                        logger.info("Gap filled: %s %s → %d candle recuperate", sym, tf, len(candles))
                    except Exception as e:
                        logger.error("Gap fill failed %s %s: %s", sym, tf, e)

    async def _fetch_gap_candles(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int, limit: int
    ) -> list:
        """Fetch candle mancanti via REST per riempire il gap."""
        interval = _TF_MAP.get(timeframe, "15")
        params = {
            "category": "linear", "symbol": symbol,
            "interval": interval, "start": start_ms, "end": end_ms, "limit": limit,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self._rest_base}/v5/market/kline", params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    return []
                rows = data.get("result", {}).get("list", [])
                candles = []
                for row in reversed(rows):
                    ts = int(row[0]) / 1000.0
                    candles.append(Candle(
                        symbol=symbol, timeframe=timeframe,
                        open=Decimal(str(row[1])), high=Decimal(str(row[2])),
                        low=Decimal(str(row[3])), close=Decimal(str(row[4])),
                        volume=Decimal(str(row[5])), timestamp=ts, is_closed=True,
                    ))
                return candles


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

# Mappa inversa: Bybit interval → timeframe Binance-style
_TF_MAP_INVERSE = {v: k for k, v in _TF_MAP.items()}


def _from_bybit_tf(interval: str) -> str:
    """Converte Bybit interval in timeframe Binance-style."""
    return _TF_MAP_INVERSE.get(interval, interval + "m")


def _parse_bybit_kline(row: list, symbol: str, timeframe: str) -> Candle:
    """
    Converte una riga REST Bybit kline in Candle.
    Formato Bybit: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    """
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open=Decimal(str(row[1])),
        high=Decimal(str(row[2])),
        low=Decimal(str(row[3])),
        close=Decimal(str(row[4])),
        volume=Decimal(str(row[5])),
        timestamp=int(row[0]) / 1000.0,
        is_closed=True,
    )
