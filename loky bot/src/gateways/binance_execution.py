import asyncio
import logging
import os
import time
import websockets
import orjson
import aiohttp
from decimal import Decimal
from typing import Callable, Coroutine, Any, Optional

from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

from src.models import Order, OrderStatus, Side
from src.gateways.interfaces import ExecutionGateway

load_dotenv()
logger = logging.getLogger(__name__)

# Binance UserStream URL
_USER_STREAM_WS = "wss://stream.binance.com:9443/ws/{listen_key}"
_USER_STREAM_REST = "https://api.binance.com/api/v3/userDataStream"
_LISTEN_KEY_REFRESH_INTERVAL = 30 * 60  # 30 minuti (scade a 60m)


class BinanceExecutionGateway(ExecutionGateway):
    def __init__(self):
        api_key    = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not secret_key:
            logger.error("❌ BINANCE_API_KEY / BINANCE_SECRET_KEY mancanti in .env")
            self.client     = None
            self._api_key   = None
            self._secret    = None
        else:
            self.client   = BinanceClient(api_key, secret_key, testnet=False)
            self._api_key = api_key
            self._secret  = secret_key
            logger.info("✅ Client Binance inizializzato (mainnet)")

        self.on_order_update: Optional[Callable[[Order], Coroutine[Any, Any, None]]] = None

        # Mappa orderId Binance → Order locale
        self._pending_orders: dict[str, Order] = {}

        # UserStream state
        self._listen_key: Optional[str] = None
        self._userstream_task: Optional[asyncio.Task] = None

    def set_on_order_update_callback(self, callback):
        self.on_order_update = callback

    # ------------------------------------------------------------------ #
    #  Order submission                                                     #
    # ------------------------------------------------------------------ #
    async def submit_order(self, order: Order) -> None:
        if not self.client:
            order.status = OrderStatus.REJECTED
            return

        logger.warning(
            f"💸 BINANCE LIVE: {order.side.name} {order.size} {order.symbol} @ {order.price}"
        )
        try:
            side     = 'BUY' if order.side == Side.BUY else 'SELL'
            response = await asyncio.to_thread(
                self.client.create_order,
                symbol=order.symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=float(order.size),
                price=float(order.price),
            )
            binance_order_id = str(response['orderId'])
            order.id     = binance_order_id
            order.status = OrderStatus.OPEN
            self._pending_orders[binance_order_id] = order

            if self.on_order_update:
                await self.on_order_update(order)

        except BinanceAPIException as e:
            logger.error(f"API Binance rifiuto submit: {e}")
            order.status = OrderStatus.REJECTED
        except Exception as e:
            logger.error(f"Errore generico submit: {e}")
            order.status = OrderStatus.REJECTED

    # ------------------------------------------------------------------ #
    #  Order cancellation                                                   #
    # ------------------------------------------------------------------ #
    async def cancel_order(self, order: Order) -> None:
        if not self.client:
            order.status = OrderStatus.OPEN
            return

        logger.info(f"🚫 Cancellazione Binance: {order.id}")
        try:
            await asyncio.to_thread(
                self.client.cancel_order,
                symbol=order.symbol,
                orderId=order.id,
            )
            order.status = OrderStatus.CANCELED
            self._pending_orders.pop(str(order.id), None)
            if self.on_order_update:
                await self.on_order_update(order)
        except BinanceAPIException as e:
            logger.error(f"API Binance rifiuto cancel: {e}")
            order.status = OrderStatus.OPEN
        except Exception as e:
            logger.error(f"Errore generico cancel: {e}")
            order.status = OrderStatus.OPEN

    # ------------------------------------------------------------------ #
    #  Open orders count                                                    #
    # ------------------------------------------------------------------ #
    async def fetch_open_orders_count(self) -> int:
        if not self.client:
            return 0
        try:
            orders = await asyncio.to_thread(self.client.get_open_orders)
            return len(orders)
        except Exception as e:
            logger.error(f"Errore fetch open orders: {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  Cancel all                                                           #
    # ------------------------------------------------------------------ #
    async def cancel_all_orders(self) -> None:
        if not self.client:
            return
        logger.warning("🧹 CLEAN-SLATE: cancellazione tutti gli ordini aperti")
        try:
            open_orders = await asyncio.to_thread(self.client.get_open_orders)
            for o in open_orders:
                try:
                    await asyncio.to_thread(
                        self.client.cancel_order,
                        symbol=o['symbol'],
                        orderId=o['orderId'],
                    )
                except Exception:
                    pass
            logger.info("✅ Clean-slate completato.")
        except Exception as e:
            logger.error(f"Errore cancel-all: {e}")

    # ------------------------------------------------------------------ #
    #  Inventory sync                                                       #
    # ------------------------------------------------------------------ #
    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        if not self.client:
            return Decimal('0')
        try:
            account = await asyncio.to_thread(self.client.get_account)
            # FIX E: rimossa riga dead code (base_asset veniva subito sovrascritta dal for)
            base_asset = asset_id
            for suffix in ['USDT', 'BTC', 'ETH', 'BNB']:
                if asset_id.endswith(suffix):
                    base_asset = asset_id[: -len(suffix)]
                    break
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    return Decimal(balance['free'])
            return Decimal('0')
        except Exception as e:
            logger.error(f"Errore fetch inventory {asset_id}: {e}")
            return Decimal('0')

    # ------------------------------------------------------------------ #
    #  match_engine_tick: no-op per Binance (fill via UserStream)          #
    # ------------------------------------------------------------------ #
    async def match_engine_tick(self):
        pass  # I fill arrivano via UserStream WebSocket

    # ================================================================== #
    #  USER DATA STREAM — fill real-time                                   #
    # ================================================================== #
    async def start_userstream(self):
        """Avvia il UserDataStream Binance per ricevere fill in tempo reale."""
        if not self._api_key:
            logger.warning("UserStream disabilitato: nessuna API key")
            return

        logger.info("🔌 Avvio UserDataStream Binance...")
        attempt = 0
        while True:
            try:
                self._listen_key = await self._create_listen_key()
                if not self._listen_key:
                    raise RuntimeError("listenKey non ottenuto")

                ws_url = _USER_STREAM_WS.format(listen_key=self._listen_key)

                # Task parallelo per refresh listenKey ogni 30 min
                refresh_task = asyncio.create_task(self._refresh_listen_key_loop())

                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=15,
                ) as ws:
                    logger.info("✅ UserDataStream connesso.")
                    attempt = 0

                    async for message in ws:
                        event = orjson.loads(message)
                        await self._handle_userstream_event(event)

            except websockets.ConnectionClosed as e:
                logger.warning(f"⚠️ UserStream WS chiuso ({e.code}). Reconnect...")
            except Exception as e:
                logger.error(f"❌ Errore UserStream: {e}")
            finally:
                try:
                    refresh_task.cancel()
                except Exception:
                    pass

            delay = min(2 ** attempt, 60)
            attempt += 1
            logger.info(f"⏳ UserStream reconnect tra {delay}s...")
            await asyncio.sleep(delay)

    async def _create_listen_key(self) -> Optional[str]:
        """POST /api/v3/userDataStream → listenKey"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _USER_STREAM_REST,
                    headers={"X-MBX-APIKEY": self._api_key},
                ) as resp:
                    data = await resp.json(content_type=None)
                    return data.get("listenKey")
        except Exception as e:
            logger.error(f"Errore creazione listenKey: {e}")
            return None

    async def _refresh_listen_key_loop(self):
        """PUT /api/v3/userDataStream ogni 30 minuti per tenere vivo il listenKey."""
        while True:
            await asyncio.sleep(_LISTEN_KEY_REFRESH_INTERVAL)
            if not self._listen_key or not self._api_key:
                continue
            try:
                async with aiohttp.ClientSession() as session:
                    await session.put(
                        _USER_STREAM_REST,
                        headers={"X-MBX-APIKEY": self._api_key},
                        params={"listenKey": self._listen_key},
                    )
                logger.debug("🔑 listenKey refreshato")
            except Exception as e:
                logger.warning(f"Errore refresh listenKey: {e}")

    async def _handle_userstream_event(self, event: dict):
        """Gestisce eventi executionReport dal UserDataStream."""
        event_type = event.get("e")

        if event_type != "executionReport":
            return

        execution_type = event.get("x")  # TRADE = partial fill, FILLED = full fill
        order_status   = event.get("X")  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED

        if execution_type not in ("TRADE", "FILLED") and order_status not in ("FILLED", "PARTIALLY_FILLED"):
            return

        binance_order_id = str(event.get("i"))  # orderId
        symbol           = event.get("s")        # e.g. BTCUSDT
        side_str         = event.get("S")        # BUY / SELL
        last_qty         = Decimal(str(event.get("l", "0")))   # lastFilledQty
        last_price       = Decimal(str(event.get("L", "0")))   # lastFilledPrice
        commission       = Decimal(str(event.get("n", "0")))   # commissione
        commission_asset = event.get("N", "USDT")

        if last_qty <= Decimal('0'):
            return

        # Recupera o crea l'ordine locale
        local_order = self._pending_orders.get(binance_order_id)
        if not local_order:
            # Ordine non nella mappa locale (es. piazzato manualmente): crea un fantasma
            side = Side.BUY if side_str == 'BUY' else Side.SELL
            local_order = Order(
                id=binance_order_id,
                symbol=symbol,
                side=side,
                price=last_price,
                size=last_qty,
                status=OrderStatus.OPEN,
                filled_size=Decimal('0'),
            )

        local_order.filled_size += last_qty
        local_order.price        = last_price  # usa prezzo di fill effettivo
        local_order.size         = last_qty

        if order_status in ("FILLED",):
            local_order.status = OrderStatus.FILLED
            self._pending_orders.pop(binance_order_id, None)
        else:
            local_order.status = OrderStatus.OPEN  # partial fill, rimane aperto

        logger.info(
            f"📨 UserStream FILL: {side_str} {last_qty} {symbol} @ {last_price} "
            f"| fee={commission} {commission_asset} | status={order_status}"
        )

        if self.on_order_update:
            await self.on_order_update(local_order)
