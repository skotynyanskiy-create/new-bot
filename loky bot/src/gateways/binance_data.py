import asyncio
import heapq
import orjson
import aiohttp
import logging
import websockets
from decimal import Decimal
from typing import Callable, Coroutine, Any
import time

from src.gateways.interfaces import MarketDataGateway

logger = logging.getLogger("BinanceWS")

# Binance Diff Depth: validare sequenza per evitare book corrotto su reconnect
# Protocollo: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
#   1. Fetch snapshot REST
#   2. Buffer diffs durante il fetch
#   3. Trovare primo diff con U <= lastUpdateId+1 <= u
#   4. Applicare da quel punto

class BinanceDataGateway(MarketDataGateway):
    def __init__(self, asset_ids: list[str]):
        self.asset_ids_upper = [a.upper() for a in asset_ids]
        self.asset_ids_lower = [a.lower() for a in asset_ids]

        streams = "/".join([f"{a}@depth@100ms" for a in self.asset_ids_lower])
        self.ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"

        self.on_market_data: Callable[[str, Decimal, float], Coroutine[Any, Any, None]] = None

        # Orderbook locale (RAM): {symbol_lower: {"bids": {price: qty}, "asks": ..., "lastUpdateId": int}}
        # FIX A: cache best bid/ask aggiornata incrementalmente → evita sort O(n log n) ad ogni diff
        self.orderbooks = {
            a: {"bids": {}, "asks": {}, "lastUpdateId": 0,
                "_best_bid": None, "_best_ask": None}
            for a in self.asset_ids_lower
        }

        # Buffer diffs ricevuti durante il fetch snapshot (per sequence validation)
        self._diff_buffer: list[dict] = []
        self._snapshot_done = False

    def set_on_market_data_callback(self, callback):
        self.on_market_data = callback

    # ------------------------------------------------------------------ #
    #  Snapshot REST con sequence sync                                     #
    # ------------------------------------------------------------------ #
    async def _fetch_snapshots(self):
        async with aiohttp.ClientSession() as session:
            for symbol in self.asset_ids_lower:
                logger.info(f"📸 Snapshot REST L2 per {symbol.upper()}...")
                url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit=1000"
                async with session.get(url) as resp:
                    snap = await resp.json(content_type=None)

                book = self.orderbooks[symbol]
                book["bids"].clear()
                book["asks"].clear()
                book["lastUpdateId"] = snap.get("lastUpdateId", 0)

                for b in snap.get("bids", []):
                    book["bids"][Decimal(b[0])] = Decimal(b[1])
                for a in snap.get("asks", []):
                    book["asks"][Decimal(a[0])] = Decimal(a[1])

                logger.info(
                    f"✅ Snapshot {symbol.upper()}: "
                    f"{len(book['bids'])} bids, {len(book['asks'])} asks "
                    f"(lastUpdateId={book['lastUpdateId']})"
                )

        # Applica i diff bufferizzati durante il fetch, rispettando la sequenza
        for payload in self._diff_buffer:
            self._apply_diff_validated(payload)
        self._diff_buffer.clear()
        self._snapshot_done = True

    # ------------------------------------------------------------------ #
    #  Main WebSocket loop con exponential backoff                         #
    # ------------------------------------------------------------------ #
    async def run_forever(self):
        self._snapshot_done = False
        await self._fetch_snapshots()

        logger.info(f"🔗 HFT Diff Stream per {self.asset_ids_upper}...")

        attempt = 0
        while True:
            try:
                # Ping attivi: dead connection rilevata in ~30s
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=15,
                    close_timeout=5,
                ) as ws:
                    logger.info("✅ WebSocket Binance connesso.")
                    attempt = 0  # reset backoff su connessione riuscita

                    async for message in ws:
                        payload = orjson.loads(message)
                        if self._snapshot_done:
                            self._apply_diff_validated(payload)
                        else:
                            self._diff_buffer.append(payload)

            except websockets.ConnectionClosed as e:
                logger.warning(f"⚠️ WS chiuso ({e.code}). Reconnect...")
            except Exception as e:
                logger.error(f"❌ Errore WS Binance: {e}")

            # Exponential backoff: 1s, 2s, 4s, 8s … max 60s
            delay = min(2 ** attempt, 60)
            attempt += 1
            logger.info(f"⏳ Reconnect tra {delay}s (tentativo {attempt})...")
            await asyncio.sleep(delay)

            # Re-fetch snapshot per evitare book stale su reconnect lungo
            self._snapshot_done = False
            self._diff_buffer.clear()
            await self._fetch_snapshots()

    # ------------------------------------------------------------------ #
    #  Applicazione diff con validazione sequence                          #
    # ------------------------------------------------------------------ #
    def _apply_diff_validated(self, payload: dict):
        stream_name = payload.get("stream", "")
        data        = payload.get("data", {})
        symbol_lower = stream_name.split("@")[0]

        if symbol_lower not in self.asset_ids_lower:
            return

        book = self.orderbooks[symbol_lower]
        last_id = book["lastUpdateId"]

        # u = final update ID di questo evento, U = primo update ID
        U = data.get("U", 0)
        u = data.get("u", 0)

        # Scarta eventi già applicati o fuori sequenza
        if u <= last_id:
            return
        if last_id > 0 and U > last_id + 1:
            logger.warning(
                f"[{symbol_lower.upper()}] Gap sequenza! "
                f"lastUpdateId={last_id} U={U} → refetch snapshot"
            )
            # Schedula un refetch asincrono senza bloccare il loop
            asyncio.create_task(self._refetch_single(symbol_lower))
            return

        # Applica bids patch e aggiorna cache best_bid
        for b in data.get("b", []):
            price, qty = Decimal(b[0]), Decimal(b[1])
            if qty == Decimal('0'):
                book["bids"].pop(price, None)
                if book["_best_bid"] == price:
                    book["_best_bid"] = max(book["bids"]) if book["bids"] else None
            else:
                book["bids"][price] = qty
                if book["_best_bid"] is None or price > book["_best_bid"]:
                    book["_best_bid"] = price

        # Applica asks patch e aggiorna cache best_ask
        for a in data.get("a", []):
            price, qty = Decimal(a[0]), Decimal(a[1])
            if qty == Decimal('0'):
                book["asks"].pop(price, None)
                if book["_best_ask"] == price:
                    book["_best_ask"] = min(book["asks"]) if book["asks"] else None
            else:
                book["asks"][price] = qty
                if book["_best_ask"] is None or price < book["_best_ask"]:
                    book["_best_ask"] = price

        book["lastUpdateId"] = u

        # Calcola e pubblica microprice + imbalance
        self._publish_microprice(symbol_lower)

    async def _refetch_single(self, symbol_lower: str):
        """Re-fetch snapshot per un singolo symbol dopo gap di sequenza."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.binance.com/api/v3/depth?symbol={symbol_lower.upper()}&limit=1000"
                async with session.get(url) as resp:
                    snap = await resp.json(content_type=None)

            book = self.orderbooks[symbol_lower]
            book["bids"].clear()
            book["asks"].clear()
            book["lastUpdateId"] = snap.get("lastUpdateId", 0)
            for b in snap.get("bids", []):
                book["bids"][Decimal(b[0])] = Decimal(b[1])
            for a in snap.get("asks", []):
                book["asks"][Decimal(a[0])] = Decimal(a[1])

            logger.info(f"🔄 Snapshot ricaricato per {symbol_lower.upper()}")
        except Exception as e:
            logger.error(f"Errore refetch snapshot {symbol_lower}: {e}")

    # ------------------------------------------------------------------ #
    #  Calcolo microprice e pubblicazione callback                         #
    # ------------------------------------------------------------------ #
    def _publish_microprice(self, symbol_lower: str):
        book = self.orderbooks[symbol_lower]

        # FIX A: usa cache best bid/ask invece di sorted() O(n log n) ad ogni tick
        best_bid = book["_best_bid"]
        best_ask = book["_best_ask"]

        if best_bid is None or best_ask is None:
            return

        # Depth dinamica: spread tossico (>0.05%) → usa 50 livelli per VWAP
        spread_pct = (best_ask - best_bid) / best_bid
        depth = 50 if spread_pct > Decimal('0.0005') else 5

        # FIX A: heapq.nlargest/nsmallest è O(n log depth) invece di O(n log n)
        top_bids = heapq.nlargest(depth, book["bids"].items(), key=lambda x: x[0])
        top_asks = heapq.nsmallest(depth, book["asks"].items(), key=lambda x: x[0])

        bid_sum = bid_vol = Decimal('0')
        for p, s in top_bids:
            bid_sum += p * s
            bid_vol += s

        ask_sum = ask_vol = Decimal('0')
        for p, s in top_asks:
            ask_sum += p * s
            ask_vol += s

        if bid_vol <= Decimal('0') or ask_vol <= Decimal('0'):
            return

        avg_bid   = bid_sum / bid_vol
        avg_ask   = ask_sum / ask_vol
        imbalance = bid_vol / (bid_vol + ask_vol)

        # Microprice: media pesata per imbalance
        micro_price = (avg_ask * imbalance) + (avg_bid * (Decimal('1') - imbalance))
        decimals    = 2 if micro_price > Decimal('10') else 4
        micro_price = Decimal(str(round(micro_price, decimals)))

        asset_id  = symbol_lower.upper()
        timestamp = time.monotonic()

        if self.on_market_data:
            # FIX F: aggiungi callback per loggare eccezioni silenziate da create_task
            task = asyncio.create_task(self.on_market_data(asset_id, micro_price, timestamp))
            task.add_done_callback(
                lambda t: logger.error(f"Errore market data callback: {t.exception()}")
                if not t.cancelled() and t.exception() else None
            )
