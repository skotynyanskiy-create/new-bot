import asyncio
from decimal import Decimal
from typing import Dict, Callable, Coroutine, Any, Optional
from src.models import Order, OrderStatus, Side
from src.gateways.interfaces import ExecutionGateway


class SimulatorGateway(ExecutionGateway):
    """
    Paper-trading gateway con fill realistici basati su price-crossing.

    Un ordine BUY viene eseguito quando il mercato scende sotto il bid price.
    Un ordine SELL viene eseguito quando il mercato sale sopra l'ask price.
    Questo emula il comportamento reale di un limit order su exchange.
    """

    def __init__(self):
        self.active_orders: Dict[str, Order] = {}
        self.on_order_update: Optional[Callable[[Order], Coroutine[Any, Any, None]]] = None

        # Prezzi correnti di mercato per symbol: {symbol: Decimal}
        self._market_prices: Dict[str, Decimal] = {}

        # Latenza di rete simulata
        self._network_latency = 0.005  # 5ms

    def set_on_order_update_callback(self, callback):
        self.on_order_update = callback

    def update_market_price(self, symbol: str, price: Decimal):
        """Chiamato da EventDrivenBot ad ogni tick per aggiornare il prezzo."""
        self._market_prices[symbol] = price

    # ------------------------------------------------------------------ #
    #  Submit order                                                         #
    # ------------------------------------------------------------------ #
    async def submit_order(self, order: Order) -> None:
        await asyncio.sleep(self._network_latency)
        order.status = OrderStatus.OPEN
        self.active_orders[order.id] = order
        if self.on_order_update:
            await self.on_order_update(order)

    # ------------------------------------------------------------------ #
    #  Cancel order                                                         #
    # ------------------------------------------------------------------ #
    async def cancel_order(self, order: Order) -> None:
        await asyncio.sleep(self._network_latency)
        if order.id in self.active_orders:
            order.status = OrderStatus.CANCELED
            del self.active_orders[order.id]
            if self.on_order_update:
                await self.on_order_update(order)

    async def fetch_open_orders_count(self) -> int:
        return len(self.active_orders)

    async def cancel_all_orders(self) -> None:
        keys = list(self.active_orders.keys())
        for k in keys:
            order = self.active_orders.pop(k, None)
            if order:
                order.status = OrderStatus.CANCELED
                if self.on_order_update:
                    await self.on_order_update(order)

    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        return Decimal('0')

    # ------------------------------------------------------------------ #
    #  Matching engine: fill basato su price-crossing                      #
    # ------------------------------------------------------------------ #
    async def match_engine_tick(self):
        """
        Controlla se il prezzo di mercato attuale ha attraversato i prezzi
        degli ordini aperti, simulando i fill di un limit order book reale.

        BUY  @ prezzo P → fill se market_price ≤ P  (il mercato scende fino al bid)
        SELL @ prezzo P → fill se market_price ≥ P  (il mercato sale fino all'ask)
        """
        if not self.active_orders:
            return

        to_fill = []
        for order_id, order in self.active_orders.items():
            market_price = self._market_prices.get(order.symbol)
            if market_price is None:
                continue

            if order.side == Side.BUY and market_price <= order.price:
                to_fill.append(order_id)
            elif order.side == Side.SELL and market_price >= order.price:
                to_fill.append(order_id)

        for order_id in to_fill:
            order = self.active_orders.pop(order_id, None)
            if not order:
                continue
            order.status      = OrderStatus.FILLED
            order.filled_size = order.size
            if self.on_order_update:
                await self.on_order_update(order)
