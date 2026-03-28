"""
AccountRiskManager — controlli di rischio a livello di conto (cross-symbol).

Responsabilità:
  • Daily loss stop aggregato: ferma TUTTI i bot se le perdite totali superano la soglia
  • Cap posizioni aperte simultanee: non più di max_concurrent_positions in totale
  • Registrazione e deregistrazione posizioni aperte

Ogni LokyBot chiama can_open_position() prima di entrare.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Callable, Optional, Set

logger = logging.getLogger("AccountRisk")

_ZERO = Decimal("0")


class AccountRiskManager:
    """
    Singleton condiviso tra tutti i bot dello stesso orchestratore.

    Args:
        max_daily_loss_pct       — perdita giornaliera massima come % del capitale (es. 0.05 = 5%)
        max_concurrent_positions — numero massimo di posizioni aperte in contemporanea
        max_peak_drawdown_pct    — drawdown massimo dal picco equity (es. 0.15 = 15%)
        initial_capital          — capitale iniziale per calcolo drawdown (USDT)
    """

    def __init__(
        self,
        max_daily_loss_pct: Decimal = Decimal("0.05"),
        max_concurrent_positions: int = 2,
        max_peak_drawdown_pct: Decimal = Decimal("0.15"),
        initial_capital: Decimal = Decimal("500"),
    ) -> None:
        self._max_daily_loss_pct = max_daily_loss_pct
        self._base_daily_loss   = -(initial_capital * max_daily_loss_pct)  # base dal capitale
        self._max_daily_loss    = self._base_daily_loss                    # attivo (adattivo)
        self._max_concurrent    = max_concurrent_positions
        self._max_dd_pct        = max_peak_drawdown_pct
        self._realized_pnl_day: Decimal = _ZERO
        self._realized_pnl_total: Decimal = _ZERO      # cumulativo dall'avvio
        self._unrealized_pnl: Decimal = _ZERO           # PnL non realizzato posizioni aperte
        self._equity_peak: Decimal = initial_capital   # picco equity per drawdown
        self._initial_capital: Decimal = initial_capital
        self._open_symbols: Set[str]    = set()
        self._day_start: float          = self._today_start()
        self._stop_triggered: bool      = False
        self._drawdown_stop: bool       = False
        self._on_drawdown_stop: Optional[Callable] = None  # callback async per notifica

    def set_on_drawdown_stop(self, callback) -> None:
        """Imposta callback chiamato quando il peak drawdown stop si attiva."""
        self._on_drawdown_stop = callback

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def update_unrealized_pnl(self, unrealized: Decimal) -> None:
        """
        Aggiorna il PnL non realizzato delle posizioni aperte.
        Chiamato periodicamente dall'orchestratore o dai bot.
        """
        self._unrealized_pnl = unrealized

    def adjust_daily_stop_for_volatility(self, vol_factor: Decimal) -> None:
        """
        Adatta il daily stop alla volatilità corrente del mercato.

        vol_factor = ATR_percentile (0-1):
          > 0.70 (alta vol)  → stringe il daily stop (×0.7)
          0.30-0.70 (media)  → daily stop invariato (×1.0)
          < 0.30 (bassa vol) → allarga il daily stop (×1.3)

        Questo evita di bruciare il daily stop in mercati molto volatili
        e permette più spazio in mercati tranquilli.
        """
        if vol_factor > Decimal('0.70'):
            multiplier = Decimal('0.7')
        elif vol_factor < Decimal('0.30'):
            multiplier = Decimal('1.3')
        else:
            multiplier = Decimal('1.0')

        self._max_daily_loss = self._base_daily_loss * multiplier
        logger.debug(
            "Daily stop adattivo: vol_factor=%.2f → multiplier=%.1f → limit=%.2f USDT",
            float(vol_factor), float(multiplier), float(self._max_daily_loss),
        )

    def can_open_position(self, symbol: str) -> bool:
        """
        Ritorna True se il bot può aprire una nuova posizione.
        Nega se:
          • daily loss account-level è raggiunto (realizzato + non realizzato)
          • drawdown dal picco equity supera max_peak_drawdown_pct
          • numero posizioni aperte >= max_concurrent
        """
        self._maybe_reset_daily()

        if self._stop_triggered:
            logger.warning("Account daily stop attivo. Nessuna nuova posizione consentita.")
            return False

        if self._drawdown_stop:
            logger.warning("Peak drawdown stop attivo. Nessuna nuova posizione consentita.")
            return False

        # Controlla PnL giornaliero includendo perdite non realizzate
        total_daily = self._realized_pnl_day + self._unrealized_pnl
        if total_daily <= self._max_daily_loss and not self._stop_triggered:
            logger.warning(
                "ACCOUNT DAILY STOP (unrealized): PnL giorno=%.4f USDT "
                "(realized=%.4f + unrealized=%.4f, limite=%.4f).",
                total_daily, self._realized_pnl_day, self._unrealized_pnl, self._max_daily_loss,
            )
            self._stop_triggered = True
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
        self._realized_pnl_day   += pnl
        self._realized_pnl_total += pnl

        logger.debug(
            "Posizione chiusa: %s | pnl=%.4f | PnL giorno=%.4f | open rimasti=%d",
            symbol, pnl, self._realized_pnl_day, len(self._open_symbols),
        )

        # Daily loss stop
        if self._realized_pnl_day <= self._max_daily_loss and not self._stop_triggered:
            logger.warning(
                "ACCOUNT DAILY STOP: PnL giornaliero=%.4f USDT (limite=%.4f). "
                "Tutti i bot sospesi fino a mezzanotte UTC.",
                self._realized_pnl_day, self._max_daily_loss,
            )
            self._stop_triggered = True

        # Peak drawdown stop: aggiorna picco equity e controlla drawdown
        current_equity = self._initial_capital + self._realized_pnl_total
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity

        if self._equity_peak > _ZERO:
            drawdown = (self._equity_peak - current_equity) / self._equity_peak
            if drawdown >= self._max_dd_pct and not self._drawdown_stop:
                logger.warning(
                    "PEAK DRAWDOWN STOP: equity=%.2f USDT, picco=%.2f USDT, "
                    "drawdown=%.1f%% (limite=%.1f%%). Bot sospesi.",
                    float(current_equity), float(self._equity_peak),
                    float(drawdown * 100), float(self._max_dd_pct * 100),
                )
                self._drawdown_stop = True
                if self._on_drawdown_stop:
                    asyncio.create_task(self._on_drawdown_stop(
                        float(current_equity), float(self._equity_peak), float(drawdown * 100)
                    ))

    @property
    def daily_pnl(self) -> Decimal:
        return self._realized_pnl_day

    @property
    def total_pnl(self) -> Decimal:
        return self._realized_pnl_total

    @property
    def open_count(self) -> int:
        return len(self._open_symbols)

    @property
    def peak_drawdown_active(self) -> bool:
        return self._drawdown_stop

    @property
    def current_drawdown_pct(self) -> Decimal:
        """Drawdown corrente dal picco equity (0-1)."""
        if self._equity_peak <= _ZERO:
            return _ZERO
        equity = self._initial_capital + self._realized_pnl_total
        dd = (self._equity_peak - equity) / self._equity_peak
        return max(_ZERO, dd)

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
            # Il peak drawdown stop NON viene resettato giornalmente:
            # protegge il capitale dall'inizio della sessione, non si azzera mai da solo.
            # (L'operatore deve riavviare il bot dopo aver valutato la situazione.)
            self._day_start        = now_day

    @staticmethod
    def _today_start() -> float:
        """Timestamp Unix dell'inizio del giorno UTC corrente."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        return datetime.datetime(now.year, now.month, now.day,
                                 tzinfo=datetime.timezone.utc).timestamp()
