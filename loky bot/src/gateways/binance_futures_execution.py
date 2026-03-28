"""
BinanceFuturesExecutionGateway — esecuzione ordini su Binance USDT-M Futures.

Differenze rispetto al gateway Spot:
  • Endpoint REST: https://fapi.binance.com/fapi/v1/
  • UserStream: /fapi/v1/listenKey  (WS: wss://fstream.binance.com/ws/{key})
  • Supporto LONG/SHORT (positionSide=BOTH con hedge mode disattivato)
  • Set leverage prima di aprire posizione
  • Ordini TP/SL: TAKE_PROFIT_MARKET e STOP_MARKET con reduceOnly=True
  • Fill event: "ORDER_TRADE_UPDATE" invece di "executionReport"
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
import aiohttp
import websockets
import orjson
from decimal import Decimal
from typing import Any, Callable, Coroutine, Optional

from dotenv import load_dotenv

from src.models import Order, OrderStatus, Side
from src.gateways.interfaces import ExecutionGateway

load_dotenv()
logger = logging.getLogger(__name__)

_REST_MAINNET       = "https://fapi.binance.com"
_REST_TESTNET       = "https://testnet.binancefuture.com"
_WS_MAINNET         = "wss://fstream.binance.com/ws/{listen_key}"
_WS_TESTNET         = "wss://stream.binancefuture.com/ws/{listen_key}"
_LISTEN_KEY_REFRESH = 30 * 60   # secondi


class BinanceFuturesExecutionGateway(ExecutionGateway):
    """
    Gateway di esecuzione per Binance USDT-M Futures.

    Flusso tipico:
        await gw.set_leverage("BTCUSDT", 3)
        await gw.submit_order(entry_order)
        await gw.submit_tp_sl(symbol, tp_price, sl_price, size, side)
        # fill notificati via on_order_update callback
        await gw.start_userstream()  # task continuo, da lanciare in background
    """

    def __init__(self, testnet: bool = False) -> None:
        self._testnet      = testnet
        self._rest_base    = _REST_TESTNET if testnet else _REST_MAINNET
        self._ws_base      = _WS_TESTNET   if testnet else _WS_MAINNET
        self._listen_key_url = f"{self._rest_base}/fapi/v1/listenKey"

        env_key    = "BINANCE_TESTNET_API_KEY"    if testnet else "BINANCE_API_KEY"
        env_secret = "BINANCE_TESTNET_SECRET_KEY" if testnet else "BINANCE_SECRET_KEY"
        api_key    = os.getenv(env_key)
        secret_key = os.getenv(env_secret)

        if not api_key or not secret_key:
            logger.error("%s / %s mancanti in .env", env_key, env_secret)
            self._api_key  = None
            self._secret   = None
        else:
            self._api_key  = api_key
            self._secret   = secret_key
            net = "TESTNET" if testnet else "mainnet"
            logger.info("Client Binance Futures inizializzato (%s)", net)

        self.on_order_update: Optional[Callable[[Order], Coroutine[Any, Any, None]]] = None
        self._pending_orders: dict[str, Order] = {}  # binance orderId → Order
        self._listen_key: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    def set_on_order_update_callback(self, callback) -> None:
        self.on_order_update = callback

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """POST /fapi/v1/leverage — imposta la leva per un symbol."""
        if not self._api_key:
            return
        try:
            params = {"symbol": symbol, "leverage": leverage}
            resp = await self._signed_post("/fapi/v1/leverage", params)
            logger.info("Leva impostata: %s × %dx", symbol, resp.get("leverage", leverage))
        except Exception as e:
            logger.error("Errore set_leverage %s: %s", symbol, e)

    # ------------------------------------------------------------------
    # Ordini
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> None:
        """Invia ordine LIMIT (entry) al Futures exchange."""
        if not self._api_key:
            order.status = OrderStatus.REJECTED
            return
        side = "BUY" if order.side == Side.BUY else "SELL"
        params = {
            "symbol":      order.symbol,
            "side":        side,
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    str(order.size),
            "price":       str(order.price),
        }
        logger.warning("FUTURES LIVE: %s %s %s @ %s", side, order.size, order.symbol, order.price)
        try:
            resp = await self._signed_post("/fapi/v1/order", params)
            binance_id = str(resp["orderId"])
            order.id     = binance_id
            order.status = OrderStatus.OPEN
            self._pending_orders[binance_id] = order
            if self.on_order_update:
                await self.on_order_update(order)
        except Exception as e:
            logger.error("Errore submit_order: %s", e)
            order.status = OrderStatus.REJECTED

    async def submit_market_order(self, symbol: str, side: Side, size: Decimal) -> Optional[Order]:
        """Invia ordine MARKET (entry immediato o chiusura)."""
        if not self._api_key:
            return None
        side_str = "BUY" if side == Side.BUY else "SELL"
        params = {
            "symbol":   symbol,
            "side":     side_str,
            "type":     "MARKET",
            "quantity": str(size),
        }
        try:
            resp = await self._signed_post("/fapi/v1/order", params)
            order = Order(
                id=str(resp["orderId"]),
                symbol=symbol,
                side=side,
                price=Decimal(str(resp.get("avgPrice", "0"))),
                size=size,
                status=OrderStatus.OPEN,
                filled_size=Decimal("0"),
            )
            self._pending_orders[order.id] = order
            logger.info("Market order inviato: %s %s %s", side_str, size, symbol)
            return order
        except Exception as e:
            logger.error("Errore market order %s: %s", symbol, e)
            return None

    async def submit_tp_sl(
        self,
        symbol: str,
        side: Side,             # lato della posizione aperta (BUY = long)
        size: Decimal,
        tp_price: Decimal,
        sl_price: Decimal,
    ) -> None:
        """
        Piazza ordini TAKE_PROFIT_MARKET e STOP_MARKET con reduceOnly=True.
        Il lato degli ordini di uscita è opposto alla posizione aperta.
        """
        if not self._api_key:
            return
        close_side = "SELL" if side == Side.BUY else "BUY"

        # Take profit
        tp_params = {
            "symbol":       symbol,
            "side":         close_side,
            "type":         "TAKE_PROFIT_MARKET",
            "stopPrice":    str(tp_price),
            "closePosition":"true",
            "timeInForce":  "GTE_GTC",
        }
        # Stop loss
        sl_params = {
            "symbol":       symbol,
            "side":         close_side,
            "type":         "STOP_MARKET",
            "stopPrice":    str(sl_price),
            "closePosition":"true",
            "timeInForce":  "GTE_GTC",
        }
        try:
            tp_resp = await self._signed_post("/fapi/v1/order", tp_params)
            logger.info("TP piazzato: %s @ %s (id=%s)", symbol, tp_price, tp_resp.get("orderId"))
        except Exception as e:
            logger.error("Errore submit TP %s: %s", symbol, e)
        try:
            sl_resp = await self._signed_post("/fapi/v1/order", sl_params)
            logger.info("SL piazzato: %s @ %s (id=%s)", symbol, sl_price, sl_resp.get("orderId"))
        except Exception as e:
            logger.error("Errore submit SL %s: %s", symbol, e)

    async def cancel_order(self, order: Order) -> None:
        if not self._api_key:
            return
        try:
            params = {"symbol": order.symbol, "orderId": order.id}
            await self._signed_delete("/fapi/v1/order", params)
            order.status = OrderStatus.CANCELED
            self._pending_orders.pop(str(order.id), None)
            if self.on_order_update:
                await self.on_order_update(order)
        except Exception as e:
            logger.error("Errore cancel_order %s: %s", order.id, e)

    async def cancel_all_orders(self) -> None:
        """DELETE /fapi/v1/allOpenOrders per ogni symbol con posizioni aperte."""
        if not self._api_key:
            return
        # Recupera tutti gli ordini aperti e cancella per symbol
        symbols_seen = set(o.symbol for o in self._pending_orders.values())
        for sym in symbols_seen:
            try:
                params = {"symbol": sym}
                await self._signed_delete("/fapi/v1/allOpenOrders", params)
                logger.info("Tutti gli ordini Futures cancellati: %s", sym)
            except Exception as e:
                logger.error("Errore cancel_all %s: %s", sym, e)
        self._pending_orders.clear()

    async def fetch_open_orders_count(self) -> int:
        if not self._api_key:
            return 0
        try:
            resp = await self._signed_get("/fapi/v1/openOrders", {})
            return len(resp)
        except Exception as e:
            logger.error("Errore fetch open orders: %s", e)
            return 0

    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        """Ritorna il balance USDT disponibile (margine libero)."""
        if not self._api_key:
            return Decimal("0")
        try:
            resp = await self._signed_get("/fapi/v2/balance", {})
            for item in resp:
                if item.get("asset") == "USDT":
                    return Decimal(str(item.get("availableBalance", "0")))
            return Decimal("0")
        except Exception as e:
            logger.error("Errore fetch inventory: %s", e)
            return Decimal("0")

    async def get_position(self, symbol: str) -> Optional[dict]:
        """Ritorna info posizione aperta per symbol (o None)."""
        if not self._api_key:
            return None
        try:
            resp = await self._signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
            for pos in resp:
                if pos.get("symbol") == symbol and Decimal(str(pos.get("positionAmt", "0"))) != Decimal("0"):
                    return pos
            return None
        except Exception as e:
            logger.error("Errore get_position %s: %s", symbol, e)
            return None

    async def match_engine_tick(self):
        pass  # fill arrivano via UserStream

    # ------------------------------------------------------------------
    # UserStream (fill real-time)
    # ------------------------------------------------------------------

    async def start_userstream(self) -> None:
        """Avvia UserDataStream Futures per fill in tempo reale."""
        if not self._api_key:
            logger.warning("UserStream disabilitato: nessuna API key")
            return

        attempt = 0
        while True:
            try:
                self._listen_key = await self._create_listen_key()
                if not self._listen_key:
                    raise RuntimeError("listenKey non ottenuto")

                ws_url = self._ws_base.format(listen_key=self._listen_key)
                refresh_task = asyncio.create_task(self._refresh_listen_key_loop())

                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=15) as ws:
                    logger.info("UserStream Futures connesso.")
                    attempt = 0
                    async for message in ws:
                        event = orjson.loads(message)
                        await self._handle_userstream_event(event)

            except websockets.ConnectionClosed as e:
                logger.warning("UserStream WS chiuso (%s). Reconnect...", e.code)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Errore UserStream: %s", e)
            finally:
                try:
                    refresh_task.cancel()
                except Exception:
                    pass

            delay = min(2 ** attempt, 60)
            attempt += 1
            logger.info("UserStream reconnect tra %ds...", delay)
            await asyncio.sleep(delay)

    async def _create_listen_key(self) -> Optional[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._listen_key_url,
                    headers={"X-MBX-APIKEY": self._api_key},
                ) as resp:
                    data = await resp.json(content_type=None)
                    return data.get("listenKey")
        except Exception as e:
            logger.error("Errore creazione listenKey Futures: %s", e)
            return None

    async def _refresh_listen_key_loop(self) -> None:
        while True:
            await asyncio.sleep(_LISTEN_KEY_REFRESH)
            if not self._listen_key or not self._api_key:
                continue
            try:
                async with aiohttp.ClientSession() as session:
                    await session.put(
                        self._listen_key_url,
                        headers={"X-MBX-APIKEY": self._api_key},
                        params={"listenKey": self._listen_key},
                    )
                logger.debug("listenKey Futures refreshato")
            except Exception as e:
                logger.warning("Errore refresh listenKey: %s", e)

    async def _handle_userstream_event(self, event: dict) -> None:
        """
        Futures UserStream emette "ORDER_TRADE_UPDATE" invece di "executionReport".
        """
        if event.get("e") != "ORDER_TRADE_UPDATE":
            return

        o = event.get("o", {})
        exec_type    = o.get("x")   # "TRADE", "CANCELED", "EXPIRED", ecc.
        order_status = o.get("X")   # "NEW", "PARTIALLY_FILLED", "FILLED", ecc.
        binance_id   = str(o.get("i"))
        symbol       = o.get("s")
        side_str     = o.get("S")
        last_qty     = Decimal(str(o.get("l", "0")))
        last_price   = Decimal(str(o.get("L", "0")))
        commission   = Decimal(str(o.get("n", "0")))
        comm_asset   = o.get("N", "USDT")

        if exec_type not in ("TRADE",) and order_status not in ("FILLED", "PARTIALLY_FILLED"):
            if order_status == "CANCELED":
                local = self._pending_orders.pop(binance_id, None)
                if local:
                    local.status = OrderStatus.CANCELED
                    if self.on_order_update:
                        await self.on_order_update(local)
            return

        if last_qty <= Decimal("0"):
            return

        local_order = self._pending_orders.get(binance_id)
        if not local_order:
            side = Side.BUY if side_str == "BUY" else Side.SELL
            local_order = Order(
                id=binance_id,
                symbol=symbol,
                side=side,
                price=last_price,
                size=last_qty,
                status=OrderStatus.OPEN,
                filled_size=Decimal("0"),
            )

        local_order.filled_size += last_qty
        local_order.price        = last_price
        local_order.size         = last_qty

        if order_status == "FILLED":
            local_order.status = OrderStatus.FILLED
            self._pending_orders.pop(binance_id, None)
        else:
            local_order.status = OrderStatus.OPEN

        logger.info(
            "Futures FILL: %s %s %s @ %s | fee=%s %s | status=%s",
            side_str, last_qty, symbol, last_price, commission, comm_asset, order_status,
        )

        if self.on_order_update:
            await self.on_order_update(local_order)

    # ------------------------------------------------------------------
    # Helper HTTP firmati (HMAC-SHA256)
    # ------------------------------------------------------------------

    async def _signed_post(self, path: str, params: dict) -> dict:
        return await self._signed_request("POST", path, params)

    async def _signed_get(self, path: str, params: dict) -> Any:
        return await self._signed_request("GET", path, params)

    async def _signed_delete(self, path: str, params: dict) -> dict:
        return await self._signed_request("DELETE", path, params)

    async def _signed_request(self, method: str, path: str, params: dict) -> Any:
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        query += f"&signature={signature}"

        url = f"{self._rest_base}{path}?{query}"
        headers = {"X-MBX-APIKEY": self._api_key}

        async with aiohttp.ClientSession() as session:
            req = getattr(session, method.lower())
            async with req(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json(content_type=None)
                if isinstance(data, dict) and "code" in data and data["code"] != 200:
                    raise RuntimeError(f"Binance Futures API error: {data}")
                return data

