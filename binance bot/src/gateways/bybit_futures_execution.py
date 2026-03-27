"""
BybitFuturesExecutionGateway — esecuzione ordini su Bybit V5 Linear Futures.

Differenze rispetto al gateway Binance:
  • Autenticazione: HMAC-SHA256 su (timestamp + api_key + recv_window + body)
  • Headers: X-BAPI-API-KEY, X-BAPI-SIGN, X-BAPI-TIMESTAMP, X-BAPI-RECV-WINDOW
  • TP/SL: endpoint separato POST /v5/position/trading-stop
  • Private WebSocket: auth con expires + firma "GET/realtime{expires}"
  • Fill events: topic "order" con orderStatus="Filled"
  • Side: "Buy" / "Sell" (non "BUY"/"SELL")
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from decimal import Decimal
from typing import Any, Callable, Coroutine, Optional

import aiohttp
import websockets
from dotenv import load_dotenv

from src.models import Order, OrderStatus, Side
from src.gateways.interfaces import ExecutionGateway

load_dotenv()
logger = logging.getLogger(__name__)

_REST_MAINNET = "https://api.bybit.com"
_REST_TESTNET = "https://api-testnet.bybit.com"
_WS_MAINNET   = "wss://stream.bybit.com/v5/private"
_WS_TESTNET   = "wss://stream-testnet.bybit.com/v5/private"
_RECV_WINDOW  = "5000"


class BybitFuturesExecutionGateway(ExecutionGateway):
    """
    Gateway di esecuzione per Bybit Linear Futures V5.

    Flusso tipico:
        await gw.set_leverage("BTCUSDT", 3)
        order = await gw.submit_market_order("BTCUSDT", Side.BUY, size)
        await gw.submit_tp_sl("BTCUSDT", Side.BUY, size, tp_price, sl_price)
        await gw.start_userstream()  # task continuo
    """

    def __init__(self, testnet: bool = False) -> None:
        self._testnet      = testnet
        self._rest_base    = _REST_TESTNET if testnet else _REST_MAINNET
        self._ws_url       = _WS_TESTNET   if testnet else _WS_MAINNET

        env_key    = "BYBIT_TESTNET_API_KEY"    if testnet else "BYBIT_API_KEY"
        env_secret = "BYBIT_TESTNET_SECRET_KEY" if testnet else "BYBIT_SECRET_KEY"
        api_key    = os.getenv(env_key)
        secret_key = os.getenv(env_secret)

        if not api_key or not secret_key:
            raise RuntimeError(
                f"Credenziali Bybit mancanti in .env: {env_key} / {env_secret}"
            )

        self._api_key = api_key
        self._secret  = secret_key
        net = "TESTNET" if testnet else "mainnet"
        logger.info("Bybit gateway inizializzato (%s)", net)

        self._pending_orders: dict[str, Order] = {}
        self.on_order_update: Optional[Callable] = None

    # ------------------------------------------------------------------
    # ExecutionGateway interface
    # ------------------------------------------------------------------

    def set_on_order_update_callback(self, cb) -> None:
        self.on_order_update = cb

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Imposta la leva per buy e sell side."""
        try:
            body = {
                "category":     "linear",
                "symbol":       symbol,
                "buyLeverage":  str(leverage),
                "sellLeverage": str(leverage),
            }
            resp = await self._signed_post("/v5/position/set-leverage", body)
            ret_code = resp.get("retCode", -1)
            if ret_code == 0 or ret_code == 110043:  # 110043 = leva già impostata
                logger.info("Leva impostata: %s ×%d", symbol, leverage)
            else:
                logger.error("Errore set_leverage %s: %s", symbol, resp)
        except Exception as e:
            logger.error("Errore set_leverage %s: %s", symbol, e)

    async def submit_market_order(
        self, symbol: str, side: Side, size: Decimal
    ) -> Optional[Order]:
        """Invia ordine MARKET su Bybit Linear Futures."""
        side_str = "Buy" if side == Side.BUY else "Sell"
        body = {
            "category":  "linear",
            "symbol":    symbol,
            "side":      side_str,
            "orderType": "Market",
            "qty":       str(size),
        }
        try:
            resp = await self._signed_post("/v5/order/create", body)
            if resp.get("retCode") != 0:
                logger.error("Market order rifiutato %s: %s", symbol, resp)
                return None

            order_id = resp["result"]["orderId"]
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                price=Decimal("0"),   # market, prezzo noto al fill
                size=size,
                status=OrderStatus.OPEN,
                filled_size=Decimal("0"),
            )
            self._pending_orders[order_id] = order
            logger.info("Market order inviato: %s %s %s", side_str, size, symbol)
            return order
        except Exception as e:
            logger.error("Errore market order %s: %s", symbol, e)
            return None

    async def submit_tp_sl(
        self,
        symbol: str,
        side: Side,
        size: Decimal,
        tp_price: Decimal,
        sl_price: Decimal,
        max_retries: int = 3,
    ) -> bool:
        """
        Imposta TP e SL sulla posizione aperta tramite /v5/position/trading-stop.
        Bybit gestisce TP/SL a livello di posizione, non come ordini separati.
        Ritenta fino a max_retries volte con backoff esponenziale.
        Returns True se TP/SL piazzati con successo.
        """
        body = {
            "category":    "linear",
            "symbol":      symbol,
            "takeProfit":  str(tp_price),
            "stopLoss":    str(sl_price),
            "tpTriggerBy": "MarkPrice",
            "slTriggerBy": "MarkPrice",
            "tpslMode":    "Full",
            "positionIdx": 0,  # 0 = one-way mode
        }
        for attempt in range(max_retries):
            try:
                resp = await self._signed_post("/v5/position/trading-stop", body)
                if resp.get("retCode") == 0:
                    logger.info(
                        "TP/SL impostati: %s | TP=%.4f SL=%.4f", symbol, tp_price, sl_price
                    )
                    return True
                else:
                    logger.error(
                        "Errore submit_tp_sl %s (tentativo %d/%d): %s",
                        symbol, attempt + 1, max_retries, resp,
                    )
            except Exception as e:
                logger.error(
                    "Errore submit_tp_sl %s (tentativo %d/%d): %s",
                    symbol, attempt + 1, max_retries, e,
                )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        logger.critical(
            "ATTENZIONE: TP/SL NON piazzati per %s dopo %d tentativi! "
            "Posizione SENZA protezione.", symbol, max_retries,
        )
        return False

    async def submit_order(self, order: Order) -> None:
        """Invia ordine LIMIT."""
        side_str = "Buy" if order.side == Side.BUY else "Sell"
        body = {
            "category":    "linear",
            "symbol":      order.symbol,
            "side":        side_str,
            "orderType":   "Limit",
            "qty":         str(order.size),
            "price":       str(order.price),
            "timeInForce": "GTC",
        }
        try:
            resp = await self._signed_post("/v5/order/create", body)
            if resp.get("retCode") != 0:
                order.status = OrderStatus.REJECTED
                logger.error("Limit order rifiutato: %s", resp)
                return
            order_id = resp["result"]["orderId"]
            order.id     = order_id
            order.status = OrderStatus.OPEN
            self._pending_orders[order_id] = order
        except Exception as e:
            logger.error("Errore submit_order: %s", e)
            order.status = OrderStatus.REJECTED

    async def cancel_order(self, order: Order) -> None:
        body = {
            "category": "linear",
            "symbol":   order.symbol,
            "orderId":  str(order.id),
        }
        try:
            resp = await self._signed_post("/v5/order/cancel", body)
            if resp.get("retCode") == 0:
                order.status = OrderStatus.CANCELED
                self._pending_orders.pop(str(order.id), None)
                if self.on_order_update:
                    await self.on_order_update(order)
        except Exception as e:
            logger.error("Errore cancel_order %s: %s", order.id, e)

    async def cancel_all_orders(self) -> None:
        """Cancella tutti gli ordini aperti per ogni symbol con ordini pendenti."""
        symbols_seen = set(o.symbol for o in self._pending_orders.values())
        for sym in symbols_seen:
            body = {"category": "linear", "symbol": sym}
            try:
                resp = await self._signed_post("/v5/order/cancel-all", body)
                if resp.get("retCode") == 0:
                    logger.info("Ordini cancellati: %s", sym)
                else:
                    logger.error("Errore cancel_all %s: %s", sym, resp)
            except Exception as e:
                logger.error("Errore cancel_all %s: %s", sym, e)
        self._pending_orders.clear()

    async def fetch_open_orders_count(self) -> int:
        try:
            resp = await self._signed_get(
                "/v5/order/realtime", {"category": "linear"}
            )
            return len(resp.get("result", {}).get("list", []))
        except Exception as e:
            logger.error("Errore fetch open orders: %s", e)
            return 0

    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        """Ritorna il balance USDT disponibile (Unified Account)."""
        try:
            resp = await self._signed_get(
                "/v5/account/wallet-balance",
                {"accountType": "UNIFIED", "coin": "USDT"},
            )
            coins = (
                resp.get("result", {})
                    .get("list", [{}])[0]
                    .get("coin", [])
            )
            for coin in coins:
                if coin.get("coin") == "USDT":
                    return Decimal(str(coin.get("availableToWithdraw", "0")))
            return Decimal("0")
        except Exception as e:
            logger.error("Errore fetch inventory: %s", e)
            return Decimal("0")

    async def get_position(self, symbol: str) -> Optional[dict]:
        """Ritorna info posizione aperta per symbol (o None)."""
        try:
            resp = await self._signed_get(
                "/v5/position/list",
                {"category": "linear", "symbol": symbol},
            )
            positions = resp.get("result", {}).get("list", [])
            for pos in positions:
                if Decimal(str(pos.get("size", "0"))) != Decimal("0"):
                    return pos
            return None
        except Exception as e:
            logger.error("Errore get_position %s: %s", symbol, e)
            return None

    async def match_engine_tick(self) -> None:
        pass  # fill arrivano via Private WebSocket

    # ------------------------------------------------------------------
    # Private WebSocket (fill real-time)
    # ------------------------------------------------------------------

    async def start_userstream(self) -> None:
        """Avvia il Private WebSocket Bybit per fill in tempo reale."""
        attempt = 0
        while True:
            try:
                async with websockets.connect(
                    self._ws_url, ping_interval=20, ping_timeout=15
                ) as ws:
                    # Autenticazione
                    expires = int(time.time() * 1000) + 5000
                    sign_str = f"GET/realtime{expires}"
                    signature = hmac.new(
                        self._secret.encode(), sign_str.encode(), hashlib.sha256
                    ).hexdigest()
                    await ws.send(json.dumps({
                        "op":   "auth",
                        "args": [self._api_key, expires, signature],
                    }))

                    # Sottoscrivi ordini
                    await ws.send(json.dumps({
                        "op":   "subscribe",
                        "args": ["order"],
                    }))

                    logger.info("Bybit Private WS connesso.")
                    # Sync pending orders dopo riconnessione per recuperare fill persi
                    if attempt > 0:
                        await self._sync_pending_orders()
                    attempt = 0

                    async for message in ws:
                        event = json.loads(message)
                        await self._handle_ws_event(event)

            except websockets.ConnectionClosed as e:
                logger.warning("Bybit WS chiuso (%s). Reconnect...", e.code)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Errore Bybit WS: %s", e)

            delay = min(2 ** attempt, 60)
            attempt += 1
            logger.info("Bybit WS reconnect tra %ds...", delay)
            await asyncio.sleep(delay)

    async def _handle_ws_event(self, event: dict) -> None:
        """
        Gestisce messaggi dal Private WebSocket Bybit.
        Gli aggiornamenti ordini arrivano come topic "order".
        """
        topic = event.get("topic", "")
        op    = event.get("op", "")

        # Log messaggi di controllo
        if op in ("auth", "subscribe", "pong"):
            success = event.get("success", False)
            logger.debug("Bybit WS %s: success=%s", op, success)
            return

        if topic != "order":
            return

        for order_data in event.get("data", []):
            await self._process_order_update(order_data)

    async def _process_order_update(self, o: dict) -> None:
        order_id     = str(o.get("orderId", ""))
        order_status = o.get("orderStatus", "")
        symbol       = o.get("symbol", "")
        side_str     = o.get("side", "Buy")
        avg_price    = Decimal(str(o.get("avgPrice", "0")))
        qty          = Decimal(str(o.get("qty", "0")))
        cum_exec_qty = Decimal(str(o.get("cumExecQty", "0")))
        fee          = Decimal(str(o.get("cumExecFee", "0")))

        if order_status not in ("Filled", "PartiallyFilled", "Cancelled"):
            return

        side = Side.BUY if side_str == "Buy" else Side.SELL

        local_order = self._pending_orders.get(order_id)
        if not local_order:
            local_order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                price=avg_price,
                size=qty,
                status=OrderStatus.OPEN,
                filled_size=Decimal("0"),
            )

        local_order.filled_size = cum_exec_qty
        local_order.price       = avg_price if avg_price > Decimal("0") else local_order.price

        if order_status == "Filled":
            local_order.status = OrderStatus.FILLED
            self._pending_orders.pop(order_id, None)
            logger.info(
                "Bybit FILL: %s %s %s @ %.4f | fee=%.4f USDT",
                side_str, cum_exec_qty, symbol, avg_price, fee,
            )
        elif order_status == "Cancelled":
            local_order.status = OrderStatus.CANCELED
            self._pending_orders.pop(order_id, None)
        else:
            local_order.status = OrderStatus.OPEN

        if self.on_order_update:
            await self.on_order_update(local_order)

    # ------------------------------------------------------------------
    # Sync dopo riconnessione WS
    # ------------------------------------------------------------------

    async def _sync_pending_orders(self) -> None:
        """Dopo riconnessione WS, controlla ordini pending che potrebbero aver fillato."""
        if not self._pending_orders:
            return
        logger.info("Sync post-reconnect: verifico %d ordini pending...", len(self._pending_orders))
        try:
            resp = await self._signed_get(
                "/v5/order/realtime", {"category": "linear", "limit": "50"}
            )
            remote_orders = {
                o.get("orderId"): o
                for o in resp.get("result", {}).get("list", [])
            }
            for order_id, local_order in list(self._pending_orders.items()):
                remote = remote_orders.get(order_id)
                if remote and remote.get("orderStatus") == "Filled":
                    logger.warning("Ordine %s fillato durante disconnessione — sync!", order_id)
                    await self._process_order_update(remote)
                elif not remote:
                    # Ordine non trovato, potrebbe essere già chiuso
                    logger.warning("Ordine %s non trovato su exchange — rimosso da pending.", order_id)
                    self._pending_orders.pop(order_id, None)
        except Exception as e:
            logger.error("Errore sync pending orders: %s", e)

    # ------------------------------------------------------------------
    # Helper HTTP firmati (Bybit V5 HMAC-SHA256)
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, payload: str) -> str:
        sign_str = timestamp + self._api_key + _RECV_WINDOW + payload
        return hmac.new(
            self._secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

    async def _signed_post(self, path: str, body: dict) -> dict:
        timestamp = str(int(time.time() * 1000))
        body_str  = json.dumps(body)
        signature = self._sign(timestamp, body_str)
        headers = {
            "X-BAPI-API-KEY":     self._api_key,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
            "Content-Type":       "application/json",
        }
        url = f"{self._rest_base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, data=body_str,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json(content_type=None)

    async def _signed_get(self, path: str, params: dict) -> dict:
        timestamp  = str(int(time.time() * 1000))
        import urllib.parse
        query_str  = urllib.parse.urlencode(params)
        signature  = self._sign(timestamp, query_str)
        headers = {
            "X-BAPI-API-KEY":     self._api_key,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-TIMESTAMP":   timestamp,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
        }
        url = f"{self._rest_base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json(content_type=None)
