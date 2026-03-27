"""
AccountRiskManager — controlli di rischio a livello di conto (cross-symbol).

Responsabilità:
  • Daily loss stop aggregato: ferma TUTTI i bot se le perdite totali superano la soglia
  • Cap posizioni aperte simultanee: non più di max_concurrent_positions in totale
  • Registrazione e deregistrazione posizioni aperte

Ogni DirectionalBot chiama can_open_position() prima di entrare.
"""

import logging
import time
from decimal import Decimal
from typing import Set

logger = logging.getLogger("AccountRisk")

_ZERO = Decimal("0")


class AccountRiskManager:
    """
    Singleton condiviso tra tutti i bot dello stesso orchestratore.

    Args:
        max_daily_loss_account  — perdita USDT giornaliera massima sull'intero conto
        max_concurrent_positions — numero massimo di posizioni aperte in contemporanea
    """

    def __init__(
        self,
        max_daily_loss_account: Decimal = Decimal("-50"),
        max_concurrent_positions: int = 2,
    ) -> None:
        self._max_daily_loss    = max_daily_loss_account
        self._max_concurrent    = max_concurrent_positions
        self._realized_pnl_day: Decimal = _ZERO
        self._open_symbols: Set[str]    = set()
        self._day_start: float          = self._today_start()
        self._stop_triggered: bool      = False

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def can_open_position(self, symbol: str) -> bool:
        """
        Ritorna True se il bot può aprire una nuova posizione.
        Nega se:
          • daily loss account-level è raggiunto
          • numero posizioni aperte >= max_concurrent
        """
        self._maybe_reset_daily()

        if self._stop_triggered:
            logger.warning("Account daily stop attivo. Nessuna nuova posizione consentita.")
            return False

        if len(self._open_symbols) >= self._max_concurrent:
            logger.debug(
                "Max posizioni concurrent raggiunte (%d/%d). %s bloccato.",
                len(self._open_symbols), self._max_concurrent, symbol,
            )
            return False

        return True

    def register_open(self, symbol: str) -> None:
        """Registra che il bot ha aperto una posizione."""
        self._open_symbols.add(symbol)
        logger.debug("Posizione aperta: %s | totale open: %d", symbol, len(self._open_symbols))

    def register_close(self, symbol: str, pnl: Decimal) -> None:
        """Registra che il bot ha chiuso una posizione con questo PnL."""
        self._open_symbols.discard(symbol)
        self._realized_pnl_day += pnl

        logger.debug(
            "Posizione chiusa: %s | pnl=%.4f | PnL giorno=%.4f | open rimasti=%d",
            symbol, pnl, self._realized_pnl_day, len(self._open_symbols),
        )

        if self._realized_pnl_day <= self._max_daily_loss and not self._stop_triggered:
            logger.warning(
                "ACCOUNT DAILY STOP: PnL giornaliero=%.4f USDT (limite=%.4f). "
                "Tutti i bot sospesi fino a mezzanotte UTC.",
                self._realized_pnl_day, self._max_daily_loss,
            )
            self._stop_triggered = True

    @property
    def daily_pnl(self) -> Decimal:
        return self._realized_pnl_day

    @property
    def open_count(self) -> int:
        return len(self._open_symbols)

    # ------------------------------------------------------------------
    # Reset giornaliero a mezzanotte UTC
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        now_day = self._today_start()
        if now_day > self._day_start:
            logger.info(
                "Reset daily PnL (era %.4f USDT). Nuovo giorno UTC.",
                self._realized_pnl_day,
            )
            self._realized_pnl_day = _ZERO
            self._stop_triggered   = False
            self._day_start        = now_day

    @staticmethod
    def _today_start() -> float:
        """Timestamp Unix dell'inizio del giorno UTC corrente."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return datetime.datetime(now.year, now.month, now.day,
                                 tzinfo=datetime.timezone.utc).timestamp()
