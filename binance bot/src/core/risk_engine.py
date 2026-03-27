import time
import logging
from dataclasses import dataclass
from decimal import Decimal
from datetime import date

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class SystemState:
    net_inventory: Decimal
    pnl: Decimal
    last_market_data_ts: float
    last_orderbook_ts: float
    local_open_orders: int
    remote_open_orders: int

class KillSwitchException(Exception):
    pass

class RiskEngine:
    def __init__(
        self,
        rate_limit_rps: int = 8,
        max_daily_loss: Decimal = Decimal('-50'),
        max_position_per_asset: Decimal = Decimal('0.05'),
    ):
        # Token Bucket rate limiter
        self.tokens_per_second = rate_limit_rps
        self.max_tokens = float(rate_limit_rps)
        self.current_tokens = self.max_tokens
        self.last_refill_ts = time.monotonic()

        # Risk parameters
        self.max_daily_loss = max_daily_loss
        self.max_position_per_asset = max_position_per_asset

        # Daily PnL tracking con reset automatico a mezzanotte UTC
        self.daily_pnl_start = Decimal('0')
        self.last_reset_date: date = date.today()

    # ------------------------------------------------------------------ #
    #  Position guard                                                       #
    # ------------------------------------------------------------------ #
    def validate_open_position(self, symbol: str, size: Decimal, current_inventory: Decimal) -> bool:
        new_inventory = current_inventory + size
        if abs(new_inventory) > self.max_position_per_asset:
            logger.error(
                f"Rischio: Posizione troppo grande per {symbol} "
                f"({new_inventory} > {self.max_position_per_asset})"
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    #  System health (market data staleness + order sync)                  #
    # ------------------------------------------------------------------ #
    def validate_system_health(self, state: SystemState) -> bool:
        current_time = time.monotonic()

        data_age = current_time - state.last_market_data_ts
        ob_age   = current_time - state.last_orderbook_ts

        if data_age > 0.5:
            logger.error(f"Kill Switch: dati mercato scaduti ({data_age:.3f}s)")
            raise KillSwitchException("Market data staleness detected")

        if ob_age > 0.5:
            logger.error(f"Kill Switch: orderbook scaduto ({ob_age:.3f}s)")
            raise KillSwitchException("Orderbook staleness detected")

        if state.local_open_orders != state.remote_open_orders:
            logger.error(
                f"Kill Switch: desync ordini "
                f"(locale {state.local_open_orders} ≠ remoto {state.remote_open_orders})"
            )
            raise KillSwitchException("Order state desynchronization detected")

        return True

    # ------------------------------------------------------------------ #
    #  Daily PnL stop — con reset automatico a mezzanotte UTC             #
    # ------------------------------------------------------------------ #
    def check_daily_pnl_stop(self, current_pnl: Decimal) -> bool:
        today = date.today()
        if today > self.last_reset_date:
            logger.info(
                f"📅 Nuovo giorno trading: reset PnL giornaliero "
                f"(era {float(current_pnl - self.daily_pnl_start):.4f} USDT)"
            )
            self.daily_pnl_start = current_pnl
            self.last_reset_date = today

        daily_pnl = current_pnl - self.daily_pnl_start
        if daily_pnl < self.max_daily_loss:
            logger.critical(
                f"Stop Loss Giornaliero: perdita {float(daily_pnl):.4f} < "
                f"{float(self.max_daily_loss):.4f} USDT"
            )
            raise KillSwitchException("Daily loss limit exceeded")
        return True

    # ------------------------------------------------------------------ #
    #  Rate limiter (token bucket)                                         #
    # ------------------------------------------------------------------ #
    def _refill_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill_ts
        self.current_tokens = min(
            self.max_tokens,
            self.current_tokens + elapsed * self.tokens_per_second
        )
        self.last_refill_ts = now

    def check_rate_limit(self) -> bool:
        self._refill_tokens()
        if self.current_tokens >= 1.0:
            self.current_tokens -= 1.0
            return True
        logger.debug("Rate limit: token esauriti, tick saltato")
        return False
