"""
PortfolioRiskManager — gestione rischio a livello di portafoglio.

Controlli:
  1. Notional totale account < capital × max_leverage
  2. Notional singolo symbol < capital × max_single_position_pct
  3. No altro asset dello stesso correlation_group già aperto (es. BTC + ETH)

Leva dinamica basata su ATR percentile:
  ATR percentile > 0.80 → leva 3x  (alta vol → meno rischio)
  ATR percentile 0.40-0.80 → leva 5x  (normale)
  ATR percentile < 0.40 → leva 7x  (bassa vol → sicuro aumentare)

Correlation groups default:
  group 0 → BTCUSDT, ETHUSDT  (correlati ~85%)
  group 1 → SOLUSDT, AVAXUSDT (correlati ~70%)
"""

import logging
from collections import deque
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')


# Gruppi di correlazione di default (fallback se non configurati da config.yaml)
_DEFAULT_CORRELATION_GROUPS: dict[str, int] = {
    "BTCUSDT": 0,
    "ETHUSDT": 0,
    "SOLUSDT": 1,
    "AVAXUSDT": 1,
}
# NOTA: Da config.yaml, usa BotSettings.correlation_groups_as_dict() per
# override personalizzato senza toccare questo file.


class PortfolioRiskManager:
    """
    Gestisce il rischio a livello di portafoglio per il multi-strategy bot.

    Args:
        capital                — capitale totale USDT del conto
        max_leverage           — leva massima account (hard limit)
        max_single_position_pct — max fraction del capitale su un singolo symbol
        correlation_groups     — dict symbol → group_id; None = no filtro correlazione
        atr_history_len        — lunghezza rolling window per ATR percentile
    """

    def __init__(
        self,
        capital: Decimal,
        max_leverage: int = 5,
        max_single_position_pct: Decimal = Decimal('0.25'),
        correlation_groups: Optional[dict[str, int]] = None,
        atr_history_len: int = 100,
    ) -> None:
        self._capital                = capital
        self._max_leverage           = max_leverage
        self._max_single_pct         = max_single_position_pct
        self._corr_groups            = correlation_groups or _DEFAULT_CORRELATION_GROUPS
        self._atr_history: deque[Decimal] = deque(maxlen=atr_history_len)

        # Stato posizioni aperte: symbol → notional
        self._open_notional: dict[str, Decimal] = {}

    # ------------------------------------------------------------------
    # Verifica apertura nuova posizione
    # ------------------------------------------------------------------

    def can_open(
        self, symbol: str, notional: Decimal
    ) -> tuple[bool, str]:
        """
        Verifica se è possibile aprire una nuova posizione.

        Returns:
            (True, "")          se tutto ok
            (False, "motivo")   se bloccato
        """
        max_total    = self._capital * Decimal(str(self._max_leverage))
        max_symbol   = self._capital * self._max_single_pct
        total_now    = self.total_notional()

        # 1. Notional totale non supera capital × max_leverage
        if total_now + notional > max_total:
            return False, (
                f"Notional totale {total_now + notional:.0f} USDT > "
                f"max {max_total:.0f} USDT (leva {self._max_leverage}x)"
            )

        # 2. Notional singolo symbol
        current_symbol_notional = self._open_notional.get(symbol, _ZERO)
        if current_symbol_notional + notional > max_symbol:
            return False, (
                f"{symbol} notional {current_symbol_notional + notional:.0f} USDT > "
                f"max {max_symbol:.0f} USDT ({float(self._max_single_pct)*100:.0f}% capitale)"
            )

        # 3. Correlation group: nessun altro asset dello stesso gruppo già aperto
        my_group = self._corr_groups.get(symbol)
        if my_group is not None:
            for open_sym in self._open_notional:
                if open_sym == symbol:
                    continue
                other_group = self._corr_groups.get(open_sym)
                if other_group is not None and other_group == my_group:
                    return False, (
                        f"{symbol} e {open_sym} sono nello stesso gruppo di correlazione "
                        f"(group {my_group}) — posizione già aperta su {open_sym}"
                    )

        return True, ""

    # ------------------------------------------------------------------
    # Leva dinamica
    # ------------------------------------------------------------------

    def record_atr(self, atr: Decimal) -> None:
        """Registra un valore ATR per calcolare il percentile rolling."""
        self._atr_history.append(atr)

    def atr_percentile(self, current_atr: Decimal) -> Decimal:
        """
        Restituisce il percentile 0-1 dell'ATR corrente nella rolling window.
        Usato dal trailing stop adattivo per adeguare la distanza del trail.

        Usa mid-rank interpolation per evitare bias:
          - Conta i valori strettamente minori (count_less)
          - Conta i valori uguali (count_equal)
          - Rank = count_less + count_equal / 2  (centro del gruppo di uguali)
          - Percentile = rank / n

        Questo evita che un ATR costante (es. range basso) salti da 0.0 a 1.0
        in un solo tick quando arriva uno spike.

        Ritorna 0.5 come fallback se non ci sono abbastanza dati.
        """
        if len(self._atr_history) < 10:
            return Decimal('0.5')

        sorted_atrs = sorted(self._atr_history)
        n = len(sorted_atrs)

        count_less  = sum(1 for a in sorted_atrs if a < current_atr)
        count_equal = sum(1 for a in sorted_atrs if a == current_atr)

        # Mid-rank: centro del blocco di valori uguali
        if count_equal == 0:
            # Valore non presente nella storia: interpolazione lineare
            rank = Decimal(count_less) + Decimal('0.5')
        else:
            rank = Decimal(count_less) + Decimal(count_equal) / Decimal('2')

        percentile = rank / Decimal(n)
        # Clamp in [0, 1] per sicurezza numerica
        return max(_ZERO, min(Decimal('1'), percentile))

    def dynamic_leverage(
        self, current_atr: Decimal, drawdown_pct: Decimal = _ZERO
    ) -> int:
        """
        Calcola la leva dinamica basata su ATR percentile e drawdown corrente.

        ATR-based:
          atr_percentile > 0.80 → 3x  (alta volatilità)
          atr_percentile 0.40-0.80 → 5x  (normale)
          atr_percentile < 0.40 → 7x  (bassa volatilità)

        Drawdown override (più conservativo):
          drawdown > 10% → max 1x (sopravvivenza)
          drawdown > 5%  → dimezza la leva ATR-based
          drawdown ≤ 5%  → leva ATR-based piena

        Falls back a max_leverage se non ci sono abbastanza dati.
        """
        if len(self._atr_history) < 20:
            base_lev = min(self._max_leverage, 5)
        else:
            percentile = self.atr_percentile(current_atr)
            if percentile > Decimal('0.80'):
                base_lev = 3
            elif percentile > Decimal('0.40'):
                base_lev = 5
            else:
                base_lev = 7

        # Drawdown override: riduce leva progressivamente
        if drawdown_pct > Decimal('0.10'):
            lev = 1  # sopravvivenza: leva minima
        elif drawdown_pct > Decimal('0.05'):
            lev = max(1, base_lev // 2)  # dimezza
        else:
            lev = base_lev

        # Rispetta comunque il cap configurato
        lev = min(lev, self._max_leverage)
        logger.debug(
            "Leva dinamica: ATR percentile, drawdown=%.1f%% → %dx",
            float(drawdown_pct * 100), lev,
        )
        return lev

    # ------------------------------------------------------------------
    # Registro posizioni
    # ------------------------------------------------------------------

    def register_open(self, symbol: str, notional: Decimal) -> None:
        """Registra apertura posizione."""
        self._open_notional[symbol] = self._open_notional.get(symbol, _ZERO) + notional
        logger.debug("PortfolioRisk: aperta %s notional=%.0f USDT", symbol, notional)

    def register_close(self, symbol: str) -> None:
        """Registra chiusura posizione (rimuove il symbol dal registro)."""
        if symbol in self._open_notional:
            del self._open_notional[symbol]
            logger.debug("PortfolioRisk: chiusa %s", symbol)

    def total_notional(self) -> Decimal:
        """Notional totale di tutte le posizioni aperte."""
        return sum(self._open_notional.values(), _ZERO)

    def open_positions(self) -> list[str]:
        """Lista dei symbol con posizioni aperte."""
        return list(self._open_notional.keys())

    def update_capital(self, new_capital: Decimal) -> None:
        """Aggiorna il capitale (es. dopo un trade chiuso con PnL)."""
        self._capital = new_capital
