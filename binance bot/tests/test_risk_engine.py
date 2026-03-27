import time
import pytest
from decimal import Decimal
from src.core.risk_engine import RiskEngine, SystemState, KillSwitchException

def test_risk_engine_staleness():
    engine = RiskEngine(rate_limit_rps=10)
    
    current_time = time.monotonic()
    
    # Health Pass -> OK
    state_ok = SystemState(
        net_inventory=Decimal('0'),
        pnl=Decimal('0'),
        last_market_data_ts=current_time,
        last_orderbook_ts=current_time,
        local_open_orders=0,
        remote_open_orders=0
    )
    assert engine.validate_system_health(state_ok) is True

    # Staleness Fail -> Exception
    state_stale = SystemState(
        net_inventory=Decimal('0'),
        pnl=Decimal('0'),
        last_market_data_ts=current_time - 0.6, # > 0.5s theshold
        last_orderbook_ts=current_time,
        local_open_orders=0,
        remote_open_orders=0
    )
    with pytest.raises(KillSwitchException, match="Market data staleness detected"):
        engine.validate_system_health(state_stale)
