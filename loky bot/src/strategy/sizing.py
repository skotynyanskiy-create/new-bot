"""
PositionSizer — logica di sizing condivisa tra tutti gli strategy engine.

Elimina la duplicazione di _calc_size() presente in 4 engine separati.
Formula unica:
  size = (capital × risk_pct) / sl_distance
  size = min(size, capital × leverage / price)
  size = 0 se notional < min_notional (Bybit $6)
"""

from decimal import Decimal

_ZERO = Decimal('0')
_MIN_NOTIONAL = Decimal('6')


def calc_risk_size(
    capital: Decimal,
    risk_pct: Decimal,
    sl_distance: Decimal,
    leverage: int,
    price: Decimal,
    fraction: Decimal = Decimal('1'),
) -> Decimal:
    """
    Calcola la size basata sul rischio per trade.

    Args:
        capital — USDT disponibili
        risk_pct — % del capitale da rischiare (es. 0.015 = 1.5%)
        sl_distance — distanza SL in unità di prezzo
        leverage — leva massima
        price — prezzo di entry
        fraction — moltiplicatore opzionale (es. 0.5 per funding rate)

    Returns:
        size in contratti, o 0 se invalida.
    """
    if sl_distance <= _ZERO or price <= _ZERO:
        return _ZERO

    risk_usdt = capital * risk_pct * fraction
    raw_size = risk_usdt / sl_distance

    max_notional = capital * Decimal(str(leverage))
    max_size = max_notional / price
    size = min(raw_size, max_size).quantize(Decimal('0.001'))

    if size * price < _MIN_NOTIONAL:
        return _ZERO
    return size
