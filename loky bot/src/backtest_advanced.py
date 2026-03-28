"""
Backtesting avanzato — Walk-Forward Analysis e Monte Carlo Simulation.

Walk-Forward:
  Divide i dati in finestre rolling (es. 60gg train + 15gg test).
  Ottimizza su train, valida su test. Se il test è profittevole,
  la strategia non è overfitted.

Monte Carlo:
  Permuta la sequenza dei trade 1000 volte.
  Calcola la distribuzione di drawdown e rendimento.
  Confidence interval al 95% per profittabilità.

Queste analisi NON richiedono dati live — lavorano su trade già eseguiti
dal CandleBacktestEngine.
"""

import logging
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Risultato di una simulazione Monte Carlo."""
    n_simulations: int
    median_pnl: float
    mean_pnl: float
    pct_5_pnl: float          # 5° percentile (worst case 95% CI)
    pct_95_pnl: float         # 95° percentile (best case)
    median_max_dd: float
    pct_95_max_dd: float      # 95° percentile drawdown (worst case)
    prob_profitable: float    # % di simulazioni con PnL > 0
    prob_ruin: float          # % con drawdown > 50%


@dataclass
class WalkForwardWindow:
    """Risultato di una singola finestra walk-forward."""
    window_id: int
    train_trades: int
    test_trades: int
    train_pnl: float
    test_pnl: float
    train_win_rate: float
    test_win_rate: float
    is_profitable: bool       # test_pnl > 0


@dataclass
class WalkForwardResult:
    """Risultato aggregato del walk-forward."""
    n_windows: int
    windows: List[WalkForwardWindow]
    pct_profitable_windows: float
    avg_test_pnl: float
    total_test_pnl: float
    robustness_score: float   # 0-100: quanto è robusto il sistema


class MonteCarloSimulator:
    """
    Monte Carlo sui trade di un backtest.

    Permuta la sequenza dei PnL N volte per stimare la distribuzione
    di rendimento e drawdown. Questo è più robusto del singolo backtest
    perché elimina il bias dell'ordine cronologico.

    Args:
        trade_pnls — lista di PnL per trade (Decimal o float)
        capital — capitale iniziale
        n_simulations — numero di permutazioni (default 1000)
    """

    def __init__(
        self,
        trade_pnls: List[float],
        capital: float = 500.0,
        n_simulations: int = 1000,
    ) -> None:
        self._pnls = trade_pnls
        self._capital = capital
        self._n_sims = n_simulations

    def run(self, seed: int = 42) -> MonteCarloResult:
        """Esegue la simulazione Monte Carlo."""
        if not self._pnls:
            return MonteCarloResult(
                n_simulations=0, median_pnl=0, mean_pnl=0,
                pct_5_pnl=0, pct_95_pnl=0, median_max_dd=0,
                pct_95_max_dd=0, prob_profitable=0, prob_ruin=0,
            )

        rng = random.Random(seed)
        final_pnls: List[float] = []
        max_drawdowns: List[float] = []

        for _ in range(self._n_sims):
            shuffled = self._pnls.copy()
            rng.shuffle(shuffled)

            # Calcola equity curve e max drawdown
            equity = self._capital
            peak = equity
            max_dd = 0.0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            final_pnls.append(equity - self._capital)
            max_drawdowns.append(max_dd * 100)  # in percentuale

        # Statistiche
        final_pnls.sort()
        max_drawdowns.sort()
        n = len(final_pnls)

        return MonteCarloResult(
            n_simulations=self._n_sims,
            median_pnl=final_pnls[n // 2],
            mean_pnl=sum(final_pnls) / n,
            pct_5_pnl=final_pnls[int(n * 0.05)],
            pct_95_pnl=final_pnls[int(n * 0.95)],
            median_max_dd=max_drawdowns[n // 2],
            pct_95_max_dd=max_drawdowns[int(n * 0.95)],
            prob_profitable=sum(1 for p in final_pnls if p > 0) / n * 100,
            prob_ruin=sum(1 for d in max_drawdowns if d > 50) / n * 100,
        )

    def print_report(self, result: Optional[MonteCarloResult] = None) -> None:
        """Stampa il report Monte Carlo."""
        r = result or self.run()
        w = 55
        print("\n" + "=" * w)
        print(f"  MONTE CARLO — {r.n_simulations} simulazioni")
        print("=" * w)
        print(f"  PnL medio         : {r.mean_pnl:+.2f} USDT")
        print(f"  PnL mediano       : {r.median_pnl:+.2f} USDT")
        print(f"  5° percentile     : {r.pct_5_pnl:+.2f} USDT (worst 95% CI)")
        print(f"  95° percentile    : {r.pct_95_pnl:+.2f} USDT (best 95% CI)")
        print(f"  Max DD mediano    : {r.median_max_dd:.1f}%")
        print(f"  Max DD 95° pctile : {r.pct_95_max_dd:.1f}%")
        print(f"  Prob. profittevole: {r.prob_profitable:.1f}%")
        print(f"  Prob. rovina (>50%): {r.prob_ruin:.1f}%")
        print("=" * w)

        if r.prob_profitable >= 80:
            print("  Verdetto: ROBUSTO — alta probabilità di profitto")
        elif r.prob_profitable >= 60:
            print("  Verdetto: MODERATO — profittevole ma con rischi")
        else:
            print("  Verdetto: FRAGILE — richiede ottimizzazione")
        print()


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis sui trade di un backtest.

    Divide i trade in finestre rolling:
      - Train: primi N trade → calcola metriche
      - Test: successivi M trade → valida profittabilità

    Se la maggioranza dei test sono profittevoli, il sistema non è overfitted.

    Args:
        trade_pnls — lista di PnL per trade
        train_ratio — frazione dei dati per training (default 0.8)
        n_windows — numero di finestre rolling (default 5)
    """

    def __init__(
        self,
        trade_pnls: List[float],
        train_ratio: float = 0.8,
        n_windows: int = 5,
    ) -> None:
        self._pnls = trade_pnls
        self._train_ratio = train_ratio
        self._n_windows = n_windows

    def run(self) -> WalkForwardResult:
        """Esegue l'analisi walk-forward."""
        n = len(self._pnls)
        if n < 10:
            return WalkForwardResult(
                n_windows=0, windows=[], pct_profitable_windows=0,
                avg_test_pnl=0, total_test_pnl=0, robustness_score=0,
            )

        window_size = n // self._n_windows
        if window_size < 4:
            self._n_windows = max(1, n // 4)
            window_size = n // self._n_windows

        train_size = int(window_size * self._train_ratio)
        test_size = window_size - train_size

        windows: List[WalkForwardWindow] = []

        for i in range(self._n_windows):
            start = i * window_size
            end = min(start + window_size, n)
            train_end = start + train_size

            train_pnls = self._pnls[start:train_end]
            test_pnls = self._pnls[train_end:end]

            if not train_pnls or not test_pnls:
                continue

            train_total = sum(train_pnls)
            test_total = sum(test_pnls)
            train_wins = sum(1 for p in train_pnls if p > 0)
            test_wins = sum(1 for p in test_pnls if p > 0)

            windows.append(WalkForwardWindow(
                window_id=i + 1,
                train_trades=len(train_pnls),
                test_trades=len(test_pnls),
                train_pnl=train_total,
                test_pnl=test_total,
                train_win_rate=train_wins / len(train_pnls) * 100,
                test_win_rate=test_wins / len(test_pnls) * 100 if test_pnls else 0,
                is_profitable=test_total > 0,
            ))

        if not windows:
            return WalkForwardResult(
                n_windows=0, windows=[], pct_profitable_windows=0,
                avg_test_pnl=0, total_test_pnl=0, robustness_score=0,
            )

        profitable = sum(1 for w in windows if w.is_profitable)
        pct_profitable = profitable / len(windows) * 100
        total_test_pnl = sum(w.test_pnl for w in windows)
        avg_test_pnl = total_test_pnl / len(windows)

        # Robustness score: combina % finestre profittevoli + consistenza
        consistency = 1 - (
            max(abs(w.test_pnl - avg_test_pnl) for w in windows) /
            (abs(avg_test_pnl) + 0.01)
        ) if avg_test_pnl != 0 else 0
        robustness = min(100, pct_profitable * 0.7 + max(0, consistency * 30))

        return WalkForwardResult(
            n_windows=len(windows),
            windows=windows,
            pct_profitable_windows=pct_profitable,
            avg_test_pnl=avg_test_pnl,
            total_test_pnl=total_test_pnl,
            robustness_score=robustness,
        )

    def print_report(self, result: Optional[WalkForwardResult] = None) -> None:
        """Stampa il report walk-forward."""
        r = result or self.run()
        w = 60
        print("\n" + "=" * w)
        print(f"  WALK-FORWARD ANALYSIS — {r.n_windows} finestre")
        print("=" * w)

        for win in r.windows:
            status = "+" if win.is_profitable else "X"
            print(
                f"  [{status}] Window {win.window_id}: "
                f"train={win.train_trades}t PnL={win.train_pnl:+.2f} WR={win.train_win_rate:.0f}% | "
                f"test={win.test_trades}t PnL={win.test_pnl:+.2f} WR={win.test_win_rate:.0f}%"
            )

        print("-" * w)
        print(f"  Finestre profittevoli: {r.pct_profitable_windows:.0f}%")
        print(f"  PnL test medio      : {r.avg_test_pnl:+.2f} USDT")
        print(f"  PnL test totale     : {r.total_test_pnl:+.2f} USDT")
        print(f"  Robustness score    : {r.robustness_score:.0f}/100")
        print("=" * w)

        if r.robustness_score >= 70:
            print("  Verdetto: NON OVERFITTED — sistema robusto")
        elif r.robustness_score >= 40:
            print("  Verdetto: PARZIALE — potrebbe essere overfitted")
        else:
            print("  Verdetto: OVERFITTED — richiede ricalibrazione")
        print()
