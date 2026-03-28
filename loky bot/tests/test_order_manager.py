import pytest
import asyncio
from decimal import Decimal
from src.models import Order, Side, OrderStatus
from src.state.order_manager import OrderManager

class MockGateway:
    def __init__(self):
        self.submitted = 0
        self.canceled = 0

    async def submit_order(self, order: Order):
        self.submitted += 1
        
    async def cancel_order(self, order: Order):
        self.canceled += 1

@pytest.mark.asyncio
async def test_order_manager_pending_cancel_protection():
    manager = OrderManager()
    gateway = MockGateway()
    
    # 1. Configurazione: Ordine In-Flight
    active_test_order = Order(
        id="ord_test_01",
        symbol="POL-USD",
        side=Side.BUY,
        price=Decimal('0.5'),
        size=Decimal('10'),
        status=OrderStatus.PENDING_CANCEL,
        filled_size=Decimal('0')
    )
    manager.active_orders[Side.BUY] = active_test_order
    
    # 2. Replay Market Data -> quote diversa ma protetta da differential lock
    await manager.sync_target_quote(
        symbol="POL-USD",
        gateway=gateway,
        side=Side.BUY,
        target_price=Decimal('0.4'), # Prezzo differente
        target_size=Decimal('10')
    )
    
    # 3. Validazioni In-Flight Guarantee: nessuna modifica al gateway
    assert gateway.submitted == 0
    assert gateway.canceled == 0
    
    # L'ordine rimane intatto in PENDING_CANCEL e non viene ricancellato
    assert manager.active_orders[Side.BUY].id == "ord_test_01"
    assert manager.active_orders[Side.BUY].status == OrderStatus.PENDING_CANCEL
