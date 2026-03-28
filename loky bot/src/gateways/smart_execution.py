"""
SmartExecutionManager — esecuzione avanzata con routing intelligente.

Funzionalità:
  1. Slippage Estimation: stima lo slippage atteso prima di piazzare l'ordine
  2. Aggressive Chase: limit order che insegue il prezzo se non fillato
  3. Smart Size Splitting: divide ordini grandi in mini-ordini
  4. Execution Analytics: traccia slippage reale vs atteso

Integra con qualsiasi ExecutionGateway (Bybit, Binance, Paper).
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Optional

from src.models import Order, OrderStatus, Side

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')


class SlippageEstimator:
    """
    Stima lo slippage atteso basandosi su volume medio e size dell'ordine.

    Formula:
      base_slippage = config.slippage_pct (default 0.05%)
      market_impact = (order_notional / avg_daily_volume) × impact_factor
      estimated_slippage = base_slippage + market_impact

    Se lo slippage stimato > max_acceptable: riduce la size o segnala.
    """

    def __init__(
        self,
        base_slippage_pct: Decimal = Decimal('0.0005'),
        impact_factor: Decimal = Decimal('10'),
        max_acceptable_pct: Decimal = Decimal('0.002'),
    ) -> None:
        self._base_slippage = base_slippage_pct
        self._impact_factor = impact_factor
        self._max_acceptable = max_acceptable_pct

    def estimate(
        self, price: Decimal, size: Decimal, avg_volume: Decimal
    ) -> Decimal:
        """
        Stima lo slippage in percentuale del prezzo.

        Returns:
            Decimal — slippage stimato (es. 0.0008 = 0.08%)
        """
        if avg_volume <= _ZERO or price <= _ZERO:
            return self._base_slippage

        notional = price * size
        volume_ratio = notional / (avg_volume * price)
        market_impact = volume_ratio * self._impact_factor * self._base_slippage

        total = self._base_slippage + market_impact
        return min(total, Decimal('0.01'))  # cap 1%

    def is_acceptable(
        self, price: Decimal, size: Decimal, avg_volume: Decimal
    ) -> tuple[bool, Decimal]:
        """
        Verifica se lo slippage è accettabile.

        Returns:
            (True, slippage_pct) — ok, procedi
            (False, slippage_pct) — troppo alto, riduci size o attendi
        """
        slip = self.estimate(price, size, avg_volume)
        return slip <= self._max_acceptable, slip

    def adjusted_size(
        self, price: Decimal, size: Decimal, avg_volume: Decimal
    ) -> Decimal:
        """
        Se lo slippage è troppo alto, riduce la size fino a renderlo accettabile.

        Returns:
            size ridotta (o originale se già ok)
        """
        ok, slip = self.is_acceptable(price, size, avg_volume)
        if ok:
            return size

        # Riduci iterativamente la size
        adjusted = size
        for _ in range(5):
            adjusted = (adjusted * Decimal('0.7')).quantize(Decimal('0.001'))
            if adjusted <= _ZERO:
                return _ZERO
            ok, _ = self.is_acceptable(price, adjusted, avg_volume)
            if ok:
                logger.info(
                    "Slippage alto (%.2f%%): size ridotta da %.3f a %.3f",
                    float(slip * 100), float(size), float(adjusted),
                )
                return adjusted

        return adjusted


class AggressiveChaser:
    """
    Limit order con chase aggressivo: se non fillato, cancella e riposta
    1 tick più aggressivo. Max N tentativi prima di market fallback.

    Risparmio medio: 30-50% rispetto a market order diretto
    (maker 0.02% vs taker 0.04% × tasso di fill limit ~60-70%)
    """

    def __init__(
        self,
        max_chase_attempts: int = 3,
        chase_interval_s: float = 2.0,
        tick_size: Decimal = Decimal('0.1'),
    ) -> None:
        self._max_attempts = max_chase_attempts
        self._interval = chase_interval_s
        self._tick_size = tick_size

    async def execute(
        self,
        gateway,
        symbol: str,
        side: Side,
        size: Decimal,
        initial_price: Decimal,
    ) -> Optional[Order]:
        """
        Tenta di fillare con limit order aggressivo.
        Se fallisce dopo max_attempts, cade su market order.

        Returns:
            Order fillato o None se fallito anche il market fallback
        """
        price = initial_price
        order_id = None

        for attempt in range(self._max_attempts):
            # Piazza limit
            side_str = "Buy" if side == Side.BUY else "Sell"
            logger.debug(
                "Chase attempt %d/%d: %s %s %s @ %.4f",
                attempt + 1, self._max_attempts, side_str, size, symbol, price,
            )

            order = await gateway.submit_limit_order_raw(symbol, side, size, price)
            if order is None:
                break

            order_id = order.id

            # Attendi fill
            await asyncio.sleep(self._interval)

            # Controlla se fillato
            current = gateway._pending_orders.get(order_id)
            if current and current.status == OrderStatus.FILLED:
                logger.info(
                    "Chase fill al tentativo %d: %s %s @ %.4f (risparmiato vs market)",
                    attempt + 1, side_str, size, price,
                )
                return current

            # Non fillato: cancella e aggiungi 1 tick più aggressivo
            try:
                await gateway.cancel_order(order)
            except Exception:
                pass
            gateway._pending_orders.pop(order_id, None)

            # Aggiusta prezzo: BUY → alza, SELL → abbassa
            if side == Side.BUY:
                price += self._tick_size
            else:
                price -= self._tick_size

        # Tutti i tentativi falliti: market fallback
        logger.info(
            "Chase esaurito dopo %d tentativi. Fallback a market order.",
            self._max_attempts,
        )
        return await gateway.submit_market_order(symbol, side, size)


class ExecutionAnalytics:
    """
    Traccia slippage reale vs atteso per migliorare le stime future.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(
        self,
        symbol: str,
        side: Side,
        expected_price: Decimal,
        actual_price: Decimal,
        size: Decimal,
        estimated_slippage: Decimal,
        execution_method: str,
    ) -> None:
        """Registra un'esecuzione per analytics."""
        if expected_price <= _ZERO:
            return
        actual_slippage = abs(actual_price - expected_price) / expected_price
        self._records.append({
            "symbol": symbol,
            "side": side.name,
            "expected_price": float(expected_price),
            "actual_price": float(actual_price),
            "size": float(size),
            "estimated_slippage": float(estimated_slippage),
            "actual_slippage": float(actual_slippage),
            "method": execution_method,
            "timestamp": time.time(),
        })

        if len(self._records) > 1000:
            self._records = self._records[-500:]

    @property
    def avg_slippage(self) -> Decimal:
        """Slippage medio reale delle ultime esecuzioni."""
        if not self._records:
            return _ZERO
        total = sum(r["actual_slippage"] for r in self._records)
        return Decimal(str(total / len(self._records)))

    @property
    def estimation_accuracy(self) -> Decimal:
        """
        Accuratezza media della stima: 1.0 = perfetta, >1 = sottostimato, <1 = sovrastimato.
        """
        if not self._records:
            return Decimal('1')
        ratios = []
        for r in self._records:
            if r["estimated_slippage"] > 0:
                ratios.append(r["actual_slippage"] / r["estimated_slippage"])
        if not ratios:
            return Decimal('1')
        return Decimal(str(sum(ratios) / len(ratios)))

    def stats(self) -> dict:
        """Statistiche di esecuzione."""
        if not self._records:
            return {"n_executions": 0}
        market_count = sum(1 for r in self._records if r["method"] == "market")
        limit_count = sum(1 for r in self._records if r["method"] == "limit")
        chase_count = sum(1 for r in self._records if r["method"] == "chase")
        return {
            "n_executions": len(self._records),
            "avg_slippage_pct": float(self.avg_slippage * 100),
            "estimation_accuracy": float(self.estimation_accuracy),
            "market_orders": market_count,
            "limit_orders": limit_count,
            "chase_orders": chase_count,
        }
