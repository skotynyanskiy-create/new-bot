import pytest
from decimal import Decimal
from src.core.quote_engine import QuoteEngine

def test_exponential_skew_and_kelly_fraction():
    # Base spread ignorato in init se overridato dalla method call
    engine = QuoteEngine(skew_factor=Decimal('0.005'))
    
    # Caso 1: Inventory neutrale (Tutto aperto a 10)
    # FIX G: half_spread = base_spread * fair_value / 2 = 0.04 * 0.5 / 2 = 0.01
    # bid = 0.5 - 0.01 = 0.49 (non più 0.48, perché spread è ratio × prezzo)
    quotes_neutral = engine.calculate_quotes(
        symbol="TEST",
        fair_value=Decimal('0.5'),
        base_spread=Decimal('0.04'),
        net_inventory=Decimal('0'),
        quote_size=Decimal('10.0'),
        max_inventory=Decimal('100.0')
    )

    assert quotes_neutral.bid_size == Decimal('10.0')
    assert quotes_neutral.ask_size == Decimal('10.0')
    assert quotes_neutral.bid_price == Decimal('0.49')
    
    # Caso 2: Fractional Sizing Anti-Liquidation (Abbiamo 90 pezzi, 10 liberi)
    # Ratio = 0.9 -> buy_size_factor = 0.1 -> quote_size (10) * 0.1 = 1.00
    quotes_skewed = engine.calculate_quotes(
        symbol="TEST",
        fair_value=Decimal('0.50'),
        base_spread=Decimal('0.04'),
        net_inventory=Decimal('90'),
        quote_size=Decimal('10.0'),
        max_inventory=Decimal('100.0')
    )
    
    assert quotes_skewed.bid_size == Decimal('1.00')
    assert quotes_skewed.ask_size == Decimal('10.00')

    # Caso 3: Inventory Block
    quotes_blocked = engine.calculate_quotes(
        symbol="TEST",
        fair_value=Decimal('0.50'),
        base_spread=Decimal('0.04'),
        net_inventory=Decimal('100'),
        quote_size=Decimal('10.0'),
        max_inventory=Decimal('100.0')
    )
    
    assert quotes_blocked.bid_size == Decimal('0')
    assert quotes_blocked.ask_size == Decimal('10.00')
