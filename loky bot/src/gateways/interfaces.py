from abc import ABC, abstractmethod
from typing import Callable, Coroutine, Any
from decimal import Decimal
from src.models import Order

class MarketDataGateway(ABC):
    @abstractmethod
    def set_on_market_data_callback(self, callback: Callable[[str, Decimal, float], Coroutine[Any, Any, None]]):
        """Registra la callback multi-asset (asset_id, micro_price, timestamp)"""
        pass

class ExecutionGateway(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> None:
        pass
        
    @abstractmethod
    async def cancel_order(self, order: Order) -> None:
        pass

    @abstractmethod
    async def fetch_open_orders_count(self) -> int:
        pass
        
    @abstractmethod
    async def cancel_all_orders(self) -> None:
        pass
        
    @abstractmethod
    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        pass
