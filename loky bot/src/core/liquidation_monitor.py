"""
LiquidationMonitor — monitoraggio prezzo di liquidazione in real-time.

Calcola il prezzo di liquidazione approssimato per posizioni isolate
Bybit Linear Perpetual e genera alert quando il margine si avvicina
alla soglia critica.

Formula (approssimata, Bybit Isolated Margin):
  LONG:  liq_price = entry × (1 - 1/leverage + maintenance_margin_rate)
  SHORT: liq_price = entry × (1 + 1/leverage - maintenance_margin_rate)

Alert levels:
  WARNING: prezzo entro il 30% del margine rimasto
  CRITICAL: prezzo entro il 15% del margine rimasto → auto-close consigliato
"""

import logging
from decimal import Decimal
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')
# Bybit maintenance margin rate per tier più basso (0-2M USDT)
_MAINTENANCE_MARGIN_RATE = Decimal('0.005')  # 0.5%


class LiquidationAlert(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


class LiquidationMonitor:
    """
    Monitora la distanza dal prezzo di liquidazione per ogni posizione.

    Args:
        leverage — leva utilizzata (es. 3, 5, 10)
        maintenance_margin_rate — tasso di margine di mantenimento (default 0.5%)
        warning_pct — soglia warning: alert se prezzo entro X% del liq price (default 30%)
        critical_pct — soglia critica: auto-close consigliato (default 15%)
    """

    def __init__(
        self,
        leverage: int = 3,
        maintenance_margin_rate: Decimal = _MAINTENANCE_MARGIN_RATE,
        warning_pct: Decimal = Decimal('0.30'),
        critical_pct: Decimal = Decimal('0.15'),
    ) -> None:
        self._leverage = Decimal(str(leverage))
        self._mmr = maintenance_margin_rate
        self._warning_pct = warning_pct
        self._critical_pct = critical_pct

    def liquidation_price(
        self, entry_price: Decimal, is_long: bool
    ) -> Decimal:
        """Calcola il prezzo di liquidazione approssimato."""
        if entry_price <= _ZERO or self._leverage <= _ZERO:
            return _ZERO

        inv_lev = Decimal('1') / self._leverage
        if is_long:
            return entry_price * (Decimal('1') - inv_lev + self._mmr)
        else:
            return entry_price * (Decimal('1') + inv_lev - self._mmr)

    def margin_distance_pct(
        self, entry_price: Decimal, current_price: Decimal, is_long: bool
    ) -> Decimal:
        """
        Percentuale di margine rimasto prima della liquidazione.
        100% = posizione appena aperta, 0% = liquidazione.
        """
        liq = self.liquidation_price(entry_price, is_long)
        if liq <= _ZERO or entry_price <= _ZERO:
            return Decimal('1')

        if is_long:
            total_margin = entry_price - liq
            remaining = current_price - liq
        else:
            total_margin = liq - entry_price
            remaining = liq - current_price

        if total_margin <= _ZERO:
            return _ZERO

        return max(_ZERO, remaining / total_margin)

    def check(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        is_long: bool,
        symbol: str = "",
    ) -> LiquidationAlert:
        """
        Verifica la distanza dal prezzo di liquidazione.

        Returns:
            LiquidationAlert.SAFE — margine sufficiente
            LiquidationAlert.WARNING — prezzo entro il 30% del liq price
            LiquidationAlert.CRITICAL — prezzo entro il 15% del liq price
        """
        margin_pct = self.margin_distance_pct(entry_price, current_price, is_long)

        if margin_pct <= self._critical_pct:
            liq = self.liquidation_price(entry_price, is_long)
            logger.warning(
                "LIQUIDATION CRITICAL %s | margin=%.1f%% | liq=%.4f | current=%.4f",
                symbol, float(margin_pct * 100), liq, current_price,
            )
            return LiquidationAlert.CRITICAL

        if margin_pct <= self._warning_pct:
            liq = self.liquidation_price(entry_price, is_long)
            logger.warning(
                "LIQUIDATION WARNING %s | margin=%.1f%% | liq=%.4f | current=%.4f",
                symbol, float(margin_pct * 100), liq, current_price,
            )
            return LiquidationAlert.WARNING

        return LiquidationAlert.SAFE
