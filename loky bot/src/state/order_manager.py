import asyncio
import uuid
import logging
from decimal import Decimal
from typing import Dict, Optional
from src.models import Order, Side, OrderStatus
from src.config import config

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self):
        # Mappa del lato del book con l'ordine attivo
        self.active_orders: Dict[Side, Optional[Order]] = {
            Side.BUY: None,
            Side.SELL: None
        }
        # FIX #4: letto una volta sola in __init__, non ad ogni tick
        self.quote_tolerance: Decimal = config.quote_tolerance

    async def sync_target_quote(self, symbol: str, gateway, side: Side, target_price: Decimal, target_size: Decimal):
        """
        Logica Differenziale (Non Cancel-All).
        Compara la quote desiderata con quella attiva. Evita double-post (In-Flight checks).
        Se coincidono o c'è un'azione in corso, ignora. Se differiscono, invia una cancellazione.
        """
        active_order = self.active_orders[side]

        # 1. Nessun ordine attivo -> Nuovo ordine
        if active_order is None or active_order.status in [OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED]:
            await self._place_order(symbol, gateway, side, target_price, target_size)
            return

        # 2. Controllo In-Flight: evita incroci per ordini in fase di modifica/cancellazione
        if active_order.status in [OrderStatus.PENDING, OrderStatus.PENDING_CANCEL]:
            logger.debug(f"[{side.name}] Operazione in-flight. Nessuna modifica eseguita.")
            return

        # 3. Confronto differenziale con l'ordine aperto
        price_differs = active_order.price != target_price
        size_differs = abs(active_order.size - target_size) > self.quote_tolerance

        if not price_differs and not size_differs:
            # Nessun cambiamento richiesto
            return

        # 4. Differenza rilevata -> Invia cancellazione. Il nuovo ordine sarà piazzato al prossimo tick
        logger.info(f"[{side.name}] Discrepanza trovata -> Active: {active_order.price}@{active_order.size} | Target: {target_price}@{target_size}. Cancello ordine attivo.")
        await self._cancel_order(gateway, active_order)

    async def _place_order(self, symbol: str, gateway, side: Side, price: Decimal, size: Decimal):
        if size <= 0:
            return
        # FIX #7: uuid garantisce unicità anche a bassa risoluzione temporale
        new_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.PENDING,
            filled_size=Decimal('0')
        )
        self.active_orders[side] = new_order
        try:
            await gateway.submit_order(new_order)
            # Simuliamo un immediato passaggio in OPEN per test (normalmente gestito da eventi WS)
            new_order.status = OrderStatus.OPEN
        except Exception as e:
            logger.error(f"Errore piazzamento ordine: {e}")
            new_order.status = OrderStatus.REJECTED

    async def _cancel_order(self, gateway, order: Order):
        order.status = OrderStatus.PENDING_CANCEL
        try:
            await gateway.cancel_order(order)
            order.status = OrderStatus.CANCELED # simulato, solitamente WS handler lo fa
        except Exception as e:
            logger.error(f"Errore cancellazione ordine: {e}")
            order.status = OrderStatus.OPEN # rollback

    async def place_twap_order(
        self,
        symbol: str,
        gateway,
        side: Side,
        target_price: Decimal,
        total_size: Decimal,
        duration_seconds: int = 60,
        slices: int = 6,
        volume_weights: list[float] | None = None,
    ):
        """
        Esegue un ordine TWAP con ponderazione volume opzionale.

        Args:
            volume_weights — pesi per ogni slice (normalizzati internamente).
                             Se None, distribuzione uniforme.
                             Es. [0.5, 1.0, 1.5, 1.5, 1.0, 0.5] per concentrare
                             il volume nel mezzo dell'esecuzione.
        """
        if total_size <= 0 or slices <= 0 or duration_seconds <= 0:
            logger.warning("Parametri TWAP non validi")
            return

        interval = duration_seconds / slices

        # Calcola pesi normalizzati
        if volume_weights and len(volume_weights) == slices:
            total_weight = sum(volume_weights)
            weights = [Decimal(str(w / total_weight)) for w in volume_weights]
        else:
            weights = [Decimal('1') / Decimal(str(slices))] * slices

        logger.info(
            "TWAP: %s %s %.6f su %ds in %d slice (volume-weighted=%s)",
            symbol, side.name, total_size, duration_seconds, slices,
            volume_weights is not None,
        )

        for i in range(slices):
            slice_size = (total_size * weights[i]).quantize(Decimal('0.00000001'))
            if slice_size > 0:
                await self._place_order(symbol, gateway, side, target_price, slice_size)
            if i < slices - 1:
                await asyncio.sleep(interval)

        logger.info("TWAP completato: %s %s %.6f", symbol, side.name, total_size)
