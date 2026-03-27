from enum import Enum, auto
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderStatus(Enum):
    PENDING = auto()
    OPEN = auto()
    FILLED = auto()
    CANCELED = auto()
    PENDING_CANCEL = auto()
    REJECTED = auto()

@dataclass(slots=True)
class Order:
    id: str
    symbol: str
    side: Side
    price: Decimal
    size: Decimal
    status: OrderStatus
    filled_size: Decimal

@dataclass(slots=True)
class QuoteIntent:
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal

@dataclass(slots=True)
class Position:
    symbol: str
    size: Decimal
    average_entry_price: Decimal

@dataclass
class Trade:
    """Record immutabile di un'esecuzione avvenuta, per audit trail e PnL."""
    symbol: str
    side: Side
    size: Decimal
    price: Decimal
    commission: Decimal
    commission_asset: str
    order_id: str
    timestamp: float
    realized_pnl: Decimal = Decimal('0')


# ---------------------------------------------------------------------------
# Nuovi tipi per il bot direzionale Breakout/Momentum
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Candle:
    """OHLCV candela per un dato symbol e timeframe."""
    symbol: str
    timeframe: str      # es. "15m", "1h"
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: float    # Unix timestamp apertura candela (secondi)
    is_closed: bool     # True solo quando la candela è chiusa


class SignalType(Enum):
    LONG = auto()
    SHORT = auto()
    NONE = auto()


@dataclass
class Signal:
    """Segnale di ingresso generato da qualsiasi strategy engine."""
    symbol: str
    signal_type: SignalType
    entry_price: Decimal
    take_profit: Decimal
    stop_loss: Decimal
    size: Decimal           # contratti / qty calcolata dal risk sizing
    atr: Decimal            # ATR al momento del segnale
    timestamp: float
    score: Decimal = Decimal('50')          # 0-100 confidence score
    strategy_name: str = "breakout"         # nome della strategia origine


@dataclass
class TPLevel:
    """Livello parziale di Take Profit."""
    price: Decimal
    qty_fraction: Decimal   # frazione del position size totale (es. 0.50, 0.25)
    hit: bool = False
