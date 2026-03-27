from decimal import Decimal
from src.models import QuoteIntent

class QuoteEngine:
    def __init__(
        self,
        skew_factor: Decimal = Decimal('1.0'),
        fee_maker: Decimal = Decimal('0.001'),
        fee_taker: Decimal = Decimal('0.001'),
    ):
        self.skew_factor = skew_factor
        # Spread minimo per essere profittevoli: copre fee maker su entrambi i lati
        self.min_spread = (fee_maker + fee_taker) * Decimal('1.05')  # +5% margine

    def calculate_quotes(
        self,
        symbol: str,
        fair_value: Decimal,
        base_spread: Decimal,
        net_inventory: Decimal,
        quote_size: Decimal,
        max_inventory: Decimal = Decimal('0.01'),
        volatility_factor: Decimal = Decimal('1.0'),
        imbalance_factor: Decimal = Decimal('0'),
    ) -> QuoteIntent:
        # Spread effettivo: mai sotto il minimo profittevole
        effective_spread = max(base_spread * volatility_factor, self.min_spread)
        # FIX G: base_spread è un ratio (es. 0.0025 = 25 bps), va moltiplicato per fair_value
        # → per BTC a $40.000 con spread 0.25%: half_spread = 40000 * 0.0025 / 2 = $50
        half_spread = (effective_spread * fair_value) / Decimal('2')

        # --- SKEW (Avellaneda-Stoikov) ---
        # inv_ratio ∈ [-1, +1]: quando long → positivo, quando short → negativo
        if max_inventory <= Decimal('0'):
            max_inventory = Decimal('1')
        inv_ratio = net_inventory / max_inventory
        inv_ratio = max(Decimal('-1'), min(Decimal('1'), inv_ratio))

        # skew sposta il reservation price: long → skew < 0 (prezzi scendono per incentivare sell)
        skew = -self.skew_factor * inv_ratio * half_spread

        bid_price = fair_value - half_spread + skew
        ask_price = fair_value + half_spread + skew

        # --- BOUNDS (Crypto: tick size $0.01, nessun cap superiore) ---
        tick_size = Decimal('0.01')
        bid_price = max(tick_size, bid_price.quantize(Decimal('0.01')))
        ask_price = max(bid_price + tick_size, ask_price.quantize(Decimal('0.01')))

        # --- POSITION SIZING (Kelly fraction anti-liquidation) ---
        # A max inventory: smetti di comprare. A -max: smetti di vendere.
        buy_fraction  = max(Decimal('0'), Decimal('1') - max(Decimal('0'),  inv_ratio))
        sell_fraction = max(Decimal('0'), Decimal('1') - max(Decimal('0'), -inv_ratio))

        bid_size = (quote_size * buy_fraction).quantize(Decimal('0.00001'))
        ask_size = (quote_size * sell_fraction).quantize(Decimal('0.00001'))

        # Drop size sotto minimo Binance (0.00001 BTC per BTC/USDT)
        min_qty = Decimal('0.00001')
        if bid_size < min_qty:
            bid_size = Decimal('0')
        if ask_size < min_qty:
            ask_size = Decimal('0')

        return QuoteIntent(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
        )
