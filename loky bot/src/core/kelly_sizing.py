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

    Usa un prior bayesiano conservativo per i primi trade, poi converge
    progressivamente verso i dati reali. Questo evita sizing erratico
    nei primi giorni live (quando si hanno pochissimi trade storici).

    Prior bayesiano:
      - win_rate prior = 0.45 (leggermente pessimistico)
      - avg_win_ratio prior = 1.5 (R:R conservativo)
      - Peso del prior = max(0, 1 - n_trades / blend_threshold)
      - Blend threshold = history_trades / 2 (es. 10 trade su 20 totali)

    Args:
        history_trades    — numero di trade per attivare Kelly puro
        half_kelly        — usa Half-Kelly (f*/2)
        min_fraction      — frazione minima
        max_fraction      — frazione massima
        prior_win_rate    — win rate prior bayesiano (default 0.45)
        prior_win_ratio   — R:R prior bayesiano (default 1.5)
    """

    # Prior per strategia: win_rate e R:R tipici per ogni tipo di strategia
    STRATEGY_PRIORS = {
        "breakout":        (Decimal('0.40'), Decimal('2.5')),
        "mean_reversion":  (Decimal('0.55'), Decimal('1.2')),
        "trend_following": (Decimal('0.45'), Decimal('2.0')),
        "funding_rate":    (Decimal('0.70'), Decimal('0.8')),
    }

    def __init__(
        self,
        history_trades: int = 30,
        kelly_divisor: int = 2,
        min_fraction: Decimal = Decimal('0.005'),
        max_fraction: Decimal = Decimal('0.03'),
        prior_win_rate: Decimal = Decimal('0.45'),
        prior_win_ratio: Decimal = Decimal('1.5'),
        use_optimal_f: bool = False,
    ) -> None:
        self._history_trades  = history_trades
        self._kelly_divisor   = max(1, kelly_divisor)  # 1=full, 2=half, 3=third, 4=quarter
        self._min_fraction    = min_fraction
        self._max_fraction    = max_fraction
        self._prior_win_rate  = prior_win_rate
        self._prior_win_ratio = prior_win_ratio
        self._use_optimal_f   = use_optimal_f
        self._blend_threshold = max(history_trades // 2, 5)

        # Rolling history di (pnl, risk_amount) per ogni trade
        self._pnls:    deque[Decimal] = deque(maxlen=history_trades)
        self._risks:   deque[Decimal] = deque(maxlen=history_trades)
        # Returns normalizzati per Optimal-f
        self._returns: deque[float] = deque(maxlen=history_trades)

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
        # Return normalizzato per Optimal-f: pnl / risk_amount
        ra = risk_amount if risk_amount > _ZERO else _ONE
        self._returns.append(float(pnl / ra))

    # ------------------------------------------------------------------
    # Calcolo frazione ottimale
    # ------------------------------------------------------------------

    def optimal_fraction(self) -> Optional[Decimal]:
        """
        Calcola la frazione Kelly ottimale con blending bayesiano.

        Nei primi trade usa il prior bayesiano (conservativo).
        Man mano che accumula dati, converge verso il Kelly empirico.

        Formula blending:
          blend_weight = min(1.0, n_trades / blend_threshold)
          p_blended = prior × (1 - w) + actual × w
          r_blended = prior_r × (1 - w) + actual_r × w

        Ritorna None se non ci sono ancora dati (0 trade).
        """
        n = len(self._pnls)
        if n == 0:
            return None

        wins   = [p for p in self._pnls if p > _ZERO]
        losses = [p for p in self._pnls if p <= _ZERO]

        # Peso blending: 0 all'inizio, 1 a blend_threshold trade
        blend_weight = min(_ONE, Decimal(n) / Decimal(self._blend_threshold))

        # Win rate blended
        actual_p = Decimal(len(wins)) / Decimal(n)
        p = self._prior_win_rate * (_ONE - blend_weight) + actual_p * blend_weight

        # Avg win ratio blended
        # Gestione esplicita di tutti gli scenari: solo win, solo loss, misto
        if wins and losses:
            avg_win  = sum(wins) / Decimal(len(wins))
            avg_loss = sum(abs(l) for l in losses) / Decimal(len(losses))
            actual_r = avg_win / avg_loss if avg_loss > _ZERO else self._prior_win_ratio
        elif wins:
            # Solo vincite: usa il prior R:R (non possiamo calcolare ratio senza perdite)
            actual_r = self._prior_win_ratio
        else:
            # Solo perdite: R:R degenerato
            actual_r = _ZERO
        avg_win_ratio = self._prior_win_ratio * (_ONE - blend_weight) + actual_r * blend_weight

        q = _ONE - p

        if avg_win_ratio == _ZERO:
            return self._min_fraction

        # Kelly formula
        kelly_f = (p * avg_win_ratio - q) / avg_win_ratio

        if kelly_f <= _ZERO:
            logger.debug(
                "Kelly f* <= 0 (p=%.2f, R=%.2f, n=%d, blend=%.2f). Uso min_fraction.",
                float(p), float(avg_win_ratio), n, float(blend_weight),
            )
            return self._min_fraction

        # Optimal-f override: usa ricerca numerica se abilitato e dati sufficienti
        if self._use_optimal_f and n >= 15:
            opt_f = self._compute_optimal_f()
            if opt_f is not None and opt_f > _ZERO:
                # Optimal-f / 3 per safety (equivalente a third-Kelly per fat tails)
                kelly_f = opt_f / Decimal('3')
            else:
                kelly_f = kelly_f / Decimal(str(self._kelly_divisor))
        else:
            kelly_f = kelly_f / Decimal(str(self._kelly_divisor))

        # Clamp tra min e max
        result = max(self._min_fraction, min(kelly_f, self._max_fraction))
        logger.debug(
            "Kelly (n=%d blend=%.2f div=%d optf=%s): p=%.2f R=%.2f → f=%.4f",
            n, float(blend_weight), self._kelly_divisor,
            self._use_optimal_f, float(p), float(avg_win_ratio), float(result),
        )
        return result

    def _compute_optimal_f(self) -> Optional[Decimal]:
        """
        Optimal-f (Ralph Vince) con safeguard per stabilità.

        Cerca f in [0.01, 0.30] (cap 30% per safety) che massimizza TWR.
        Requisiti minimi: almeno 15 trade con almeno 5 vincenti.
        TWR minimo: deve superare 1.05 (5% gain) per essere valido.
        """
        returns = list(self._returns)
        if len(returns) < 15:
            return None
        # Richiedi almeno 5 trade positivi per evitare bias da poche vincite
        n_positive = sum(1 for r in returns if r > 0)
        if n_positive < 5:
            return None

        best_f = Decimal('0.01')
        best_twr = Decimal('1')

        # Cerca in step di 0.5% (60 punti) da 0.5% a 30% — 2x risoluzione vs 1%
        for f_int in range(1, 61):
            f = f_int / 200.0
            twr = 1.0
            valid = True
            for r in returns:
                hpr = 1.0 + f * r
                if hpr <= 0.01:  # quasi-ruin, non solo <= 0
                    valid = False
                    break
                twr *= hpr
            if valid and twr > float(best_twr):
                best_twr = Decimal(str(round(twr, 6)))
                best_f = Decimal(str(f))

        # TWR minimo: deve mostrare almeno 5% di gain complessivo
        if best_twr < Decimal('1.05'):
            return None
        return best_f

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
        """
        True se il Kelly Sizer può fornire una frazione (sempre True con blending bayesiano).
        La qualità del sizing migliora man mano che si accumulano trade.
        """
        return True  # Con il prior bayesiano, è sempre pronto (anche a 0 trade)

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
