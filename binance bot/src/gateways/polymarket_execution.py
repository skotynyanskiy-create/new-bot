import logging
import os
import asyncio
from decimal import Decimal
from src.models import Order, OrderStatus
from src.gateways.interfaces import ExecutionGateway

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, Funder
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False

logger = logging.getLogger("PolymarketExecution")

class PolymarketExecutionGateway(ExecutionGateway):
    def __init__(self, host="https://clob.polymarket.com", chain_id=137):
        self.client = None
        self.on_order_update = None
        
        if not CLOB_AVAILABLE:
            logger.error("🛑 py-clob-client assente! Esecuzione reale interdetta.")
            return
            
        private_key = os.getenv("PK")
        funder_addr = os.getenv("FUNDER")
        
        if not private_key:
            logger.warning("⚠️ Nessuna Private Key (PK) trovata in ENV. Gateway in modalità Mocking Pura.")
            return
            
        self.funder = Funder(funder_addr) if funder_addr else None
        
        try:
            self.client = ClobClient(host, key=private_key, chain_id=chain_id, funder=self.funder)
            self.client.set_credentials()
            logger.info("✅ Credenziali L2 ClobClient registrate attivamente in memoria.")
        except Exception as e:
            logger.error(f"Errore caricamento Wallet: {e}")

    def set_on_order_update_callback(self, callback):
        self.on_order_update = callback

    async def submit_order(self, order: Order) -> None:
        if not self.client or not self.client.creds:
            # Fallback sicuro: se l'utente accende il live trading senza mettere chiavi, drop totale
            order.status = OrderStatus.REJECTED
            return
            
        logger.warning(f"💸 ESECUZIONE REALE AL MERCATO: {order.side.name} | Size: {order.size} | Prezzo: {order.price}")
        try:
            # Integrazione Crittografica: 
            args = OrderArgs(price=float(order.price), size=float(order.size), side=order.side.name, token_id=order.symbol)
            await asyncio.to_thread(self.client.create_and_post_order, args)
            
            # Simuliamo la latenza della rete REST Reale di Polymarket (circa 80ms)
            # await asyncio.sleep(0.08) 
            order.status = OrderStatus.OPEN
            if self.on_order_update:
                await self.on_order_update(order)
        except Exception as e:
            logger.error(f"Rigetto API Polymarket su Submit: {e}")
            order.status = OrderStatus.REJECTED

    async def cancel_order(self, order: Order) -> None:
        if not self.client or not self.client.creds:
            order.status = OrderStatus.OPEN
            return
            
        logger.info(f"🚫 CANCELLAZIONE REALE: {order.id}")
        try:
            await asyncio.to_thread(self.client.cancel, order.id)
            # await asyncio.sleep(0.05)
            order.status = OrderStatus.CANCELED
            if self.on_order_update:
                await self.on_order_update(order)
        except Exception as e:
            logger.error(f"Rigetto API Polymarket su Cancel: {e}")
            order.status = OrderStatus.OPEN
            
    async def fetch_open_orders_count(self) -> int:
        if not self.client or not self.client.creds: return 0
        try:
            orders = await asyncio.to_thread(self.client.get_orders)
            return len([o for o in orders if o['status'] == 'OPEN'])
        except Exception as e:
            logger.error(f"Errore fetch open orders: {e}")
            return 0

    async def cancel_all_orders(self) -> None:
        if not self.client or not self.client.creds: return
        logger.warning("🧹 CLEAN-SLATE BOOT: Esecuzione GLOBALE 'Cancel All Orders' su Polymarket...")
        try:
            await asyncio.to_thread(self.client.cancel_all)
            # await asyncio.sleep(0.5)
            logger.info("✅ Clean-Slate completato (Phantom Orders Azzerati).")
        except Exception as e:
            logger.error(f"Errore Polymarket Cancel-All: {e}")

    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        if not self.client or not self.client.creds: return Decimal('0')
        logger.info(f"🔍 Sincronizzazione Portafoglio Wallet Polygon per Asset {asset_id[:8]}...")
        try:
            result = await asyncio.to_thread(self.client.get_positions, self.funder.address)
            if result:
                total_size = Decimal('0')
                for position in result:
                    if position.get('token_id') == asset_id:  # Assumendo che asset_id sia token_id
                        total_size += Decimal(str(position.get('size', 0)))
                return total_size
            # await asyncio.sleep(0.5)
            return Decimal('0')
        except Exception as e:
            logger.error(f"Errore Polymarket Portfolio API: {e}")
            return Decimal('0')

    async def match_engine_tick(self):
        # A differenza del Simulator locale, l'execution reale NON inventa i fills.
        # Li riceverebbe passivamente dai WSS utente standard (da implementare), quindu qui e' no-op
        pass
