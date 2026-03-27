"""
KellySizer — dimensionamento posizione con Kelly Criterion (Half-Kelly).

Formula Kelly:
  f* = (p × avg_win_ratio - q) / avg_win_ratio
  Dove:
    p           = win_rate (frazione di trade vincenti)
    q           = 1 - p
    avg_win_ratio = avg_win / avg_loss

Half-Kelly: usa f*/2 per ridurre volatilità del capitale (prassi standard).
Clamped tra min_fraction e max_fraction per sicurezza.

Funziona solo dopo aver accumulato almeno `history_trades` trade.
Prima del raggiungimento della soglia → fallback a risk_per_trade_pct.
"""

import logging
from collections import deque
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')
_ONE  = Decimal('1')
_TWO  = Decimal('2')


class KellySizer:
    """
    Calcola la frazione ottimale di capitale da rischiare per trade.

    Args:
        history_trades — numero di trade recenti da considerare (rolling window)
        half_kelly     — usa Half-Kelly (f*/2) per ridurre volatilità
        min_fraction   — frazione minima (floor di sicurezza)
        max_fraction   — frazione massima (cap di sicurezza)
    """

    def __init__(
        self,
        history_trades: int = 30,
        half_kelly: bool = True,
        min_fraction: Decimal = Decimal('0.005'),
        max_fraction: Decimal = Decimal('0.03'),
    ) -> None:
        self._history_trades = history_trades
        self._half_kelly     = half_kelly
        self._min_fraction   = min_fraction
        self._max_fraction   = max_fraction

        # Rolling history di (pnl, risk_amount) per ogni trade
        self._pnls:    deque[Decimal] = deque(maxlen=history_trades)
        self._risks:   deque[Decimal] = deque(maxlen=history_trades)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, pnl: Decimal, risk_amount: Decimal) -> None:
        """
        Registra il risultato di un trade chiuso.

        Args:
            pnl         — PnL netto del trade (positivo = win, negativo = loss)
            risk_amount — USD rischiati nel trade = |entry - SL| × size
        """
        self._pnls.append(pnl)
        self._risks.append(risk_amount if risk_amount > _ZERO else _ONE)

    # ------------------------------------------------------------------
    # Calcolo frazione ottimale
    # ------------------------------------------------------------------

    def optimal_fraction(self) -> Optional[Decimal]:
        """
        Calcola la frazione Kelly ottimale.
        Ritorna None se non ci sono abbastanza trade in history.
        """
        if len(self._pnls) < self._history_trades:
            return None

        wins  = [p for p in self._pnls if p > _ZERO]
        losses = [p for p in self._pnls if p <= _ZERO]

        if not wins or not losses:
            return None

        p = Decimal(len(wins)) / Decimal(len(self._pnls))   # win rate
        q = _ONE - p

        avg_win  = sum(wins)  / Decimal(len(wins))
        avg_loss = sum(abs(l) for l in losses) / Decimal(len(losses))

        if avg_loss == _ZERO:
            return self._max_fraction

        avg_win_ratio = avg_win / avg_loss  # R (reward/risk ratio medio)

        if avg_win_ratio == _ZERO:
            return self._min_fraction

        # Kelly formula
        kelly_f = (p * avg_win_ratio - q) / avg_win_ratio

        if kelly_f <= _ZERO:
            logger.debug("Kelly f* <= 0 (win_rate=%.2f, R=%.2f). Uso min_fraction.", p, avg_win_ratio)
            return self._min_fraction

        if self._half_kelly:
            kelly_f = kelly_f / _TWO

        # Clamp tra min e max
        result = max(self._min_fraction, min(kelly_f, self._max_fraction))
        logger.debug(
            "Kelly: p=%.2f R=%.2f f*=%.4f → clamped=%.4f",
            float(p), float(avg_win_ratio), float(kelly_f * _TWO), float(result),
        )
        return result

    # ------------------------------------------------------------------
    # Position size
    # ------------------------------------------------------------------

    def position_size(
        self,
        capital: Decimal,
        sl_distance: Decimal,
        fallback_fraction: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calcola la size ottimale basata su Kelly.

        size = (capital × kelly_fraction) / sl_distance

        Args:
            capital           — capitale USDT totale del bot
            sl_distance       — distanza SL in USDT (|entry - sl_price|)
            fallback_fraction — da usare se Kelly non disponibile
        Returns:
            size in contratti (Decimal). Ritorna _ZERO se sl_distance = 0.
        """
        if sl_distance == _ZERO:
            return _ZERO

        fraction = self.optimal_fraction()
        if fraction is None:
            if fallback_fraction is not None:
                fraction = fallback_fraction
            else:
                return _ZERO

        risk_usdt = capital * fraction
        return risk_usdt / sl_distance

    # ------------------------------------------------------------------
    # Diagnostica
    # ------------------------------------------------------------------

    @property
    def n_trades(self) -> int:
        """Numero di trade registrati."""
        return len(self._pnls)

    @property
    def is_ready(self) -> bool:
        """True se ha abbastanza dati per calcolare Kelly."""
        return len(self._pnls) >= self._history_trades

    def stats(self) -> dict:
        """Ritorna statistiche di diagnostica."""
        if not self._pnls:
            return {"n_trades": 0}
        wins   = [p for p in self._pnls if p > _ZERO]
        losses = [p for p in self._pnls if p <= _ZERO]
        win_rate = Decimal(len(wins)) / Decimal(len(self._pnls)) if self._pnls else _ZERO
        return {
            "n_trades":  len(self._pnls),
            "win_rate":  float(win_rate),
            "avg_win":   float(sum(wins) / len(wins)) if wins else 0,
            "avg_loss":  float(sum(abs(l) for l in losses) / len(losses)) if losses else 0,
            "kelly_f":   float(self.optimal_fraction() or _ZERO),
        }
