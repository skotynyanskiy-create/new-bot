"""
PortfolioRiskManager — gestione rischio a livello di portafoglio.

Controlli:
  1. Notional totale account < capital × max_leverage
  2. Notional singolo symbol < capital × max_single_position_pct
  3. Correlazione dinamica rolling: blocca se correlazione > threshold con posizione aperta

Leva dinamica basata su ATR percentile + drawdown:
  ATR percentile > 0.80 → leva 3x  (alta vol → meno rischio)
  ATR percentile 0.40-0.80 → leva 5x  (normale)
  ATR percentile < 0.40 → leva 7x  (bassa vol → sicuro aumentare)
  Drawdown > 5% → dimezza, > 10% → 1x

Correlazione dinamica:
  Calcola Spearman rolling su ultimi 50 rendimenti per coppia.
  Se correlazione > 0.7 → blocca entry simultanea.
  Sostituisce i gruppi statici con dati reali.
"""

import logging
import math
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
        correlation_threshold: Decimal = Decimal('0.70'),
        correlation_window: int = 50,
    ) -> None:
        self._capital                = capital
        self._max_leverage           = max_leverage
        self._max_single_pct         = max_single_position_pct
        self._corr_groups            = correlation_groups or _DEFAULT_CORRELATION_GROUPS
        self._atr_history: deque[Decimal] = deque(maxlen=atr_history_len)
        self._corr_threshold         = correlation_threshold
        self._corr_window            = correlation_window

        # Rolling returns per symbol per correlazione dinamica
        self._returns: dict[str, deque[float]] = {}

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

        # 3. Correlazione: dinamica rolling (priorità) o gruppi statici (fallback)
        blocked, reason = self.is_correlated_with_open(symbol)
        if blocked:
            return False, reason

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

    # ------------------------------------------------------------------
    # Correlazione dinamica rolling
    # ------------------------------------------------------------------

    def record_return(self, symbol: str, log_return: float) -> None:
        """Registra un rendimento log per il calcolo della correlazione rolling."""
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._corr_window)
        self._returns[symbol].append(log_return)

    def rolling_correlation(self, sym_a: str, sym_b: str) -> Optional[Decimal]:
        """
        Calcola la correlazione Spearman rolling tra due symbol.

        Spearman è più robusta di Pearson per dati non normali (crypto).
        Ritorna None se dati insufficienti (< 20 punti).
        """
        ra = self._returns.get(sym_a)
        rb = self._returns.get(sym_b)
        if not ra or not rb:
            return None
        n = min(len(ra), len(rb))
        if n < 20:
            return None

        # Allinea gli ultimi N rendimenti
        a = list(ra)[-n:]
        b = list(rb)[-n:]

        # Spearman: correlazione sui rank
        def _rank(data: list[float]) -> list[float]:
            indexed = sorted(range(len(data)), key=lambda i: data[i])
            ranks = [0.0] * len(data)
            for rank_pos, idx in enumerate(indexed):
                ranks[idx] = float(rank_pos)
            return ranks

        rank_a = _rank(a)
        rank_b = _rank(b)

        # Pearson sui rank
        mean_a = sum(rank_a) / n
        mean_b = sum(rank_b) / n
        cov = sum((rank_a[i] - mean_a) * (rank_b[i] - mean_b) for i in range(n))
        var_a = sum((x - mean_a) ** 2 for x in rank_a)
        var_b = sum((x - mean_b) ** 2 for x in rank_b)
        denom = math.sqrt(var_a * var_b)
        if denom < 1e-10:
            return _ZERO
        corr = cov / denom
        return Decimal(str(round(corr, 4)))

    def is_correlated_with_open(self, symbol: str) -> tuple[bool, str]:
        """
        Verifica se il symbol ha correlazione > threshold con qualsiasi
        posizione aperta. Usa prima la correlazione dinamica rolling,
        poi fallback ai gruppi statici.

        Returns:
            (True, "motivo") se bloccato
            (False, "") se ok
        """
        for open_sym in self._open_notional:
            if open_sym == symbol:
                continue

            # 1. Correlazione dinamica (se disponibile)
            corr = self.rolling_correlation(symbol, open_sym)
            if corr is not None:
                if corr > self._corr_threshold:
                    return True, (
                        f"{symbol} correlato con {open_sym} "
                        f"(Spearman={float(corr):.2f} > {float(self._corr_threshold):.2f})"
                    )
                # Correlazione bassa → PERMETTI anche se nello stesso gruppo statico
                continue

            # 2. Fallback: gruppi statici (se non ci sono dati rolling)
            my_group = self._corr_groups.get(symbol)
            other_group = self._corr_groups.get(open_sym)
            if my_group is not None and other_group is not None and my_group == other_group:
                return True, (
                    f"{symbol} e {open_sym} nello stesso gruppo statico "
                    f"(group {my_group}) — dati rolling insufficienti"
                )

        return False, ""
