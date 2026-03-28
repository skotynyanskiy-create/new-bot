"""
Walk-Forward Optimization — Loky Bot.

Valida i parametri di configurazione su dati reali Bybit usando la metodologia
walk-forward per evitare overfitting.

Fasi:
  1. Scarica N mesi di dati storici Bybit per ogni symbol/timeframe
  2. Suddivide in finestre: train (4 mesi) → test (1 mese) con slide di 1 mese
  3. Per ogni finestra: grid search dei parametri su train → valuta OOS su test
  4. Reporta Sharpe OOS medio e i parametri più robusti

Uso:
    python -m src.optimization
    python -m src.optimization --symbol BTCUSDT --months 8 --timeframe 15m
    python -m src.optimization --symbol BTCUSDT ETHUSDT --months 10

Output:
    optimization_results.json  — report completo
    best_params.yaml           — parametri ottimali pronti da incollare in config.yaml
"""

import argparse
import asyncio
import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from src.backtest import (
    CandleBacktestEngine,
    _BacktestGateway,
    _InMemoryStateManager,
    _TF_MAP,
    _TF_MINUTES,
    BacktestTrade,
)
from src.bot import BotState, LokyBot
from src.config import BotSettings
from src.models import Candle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Griglia parametri
# ---------------------------------------------------------------------------

PARAM_GRID: Dict[str, List[Any]] = {
    # ATR multiplier per SL (più alto = SL più largo = meno whipsaw ma perdita maggiore)
    "sl_atr_mult":    [1.5, 2.0, 2.5],
    # ATR multiplier per TP1 (primo target parziale)
    "partial_tp1_atr": [1.5, 2.0],
    # ATR multiplier per TP2
    "partial_tp2_atr": [2.5, 3.0],
    # ADX threshold per definire mercato trending (default 25)
    "adx_trend_threshold": [20, 25, 30],
    # RSI overbought per MeanReversion
    "rsi_ob": [65, 70, 75],
    # RSI oversold per MeanReversion
    "rsi_os": [25, 30, 35],
}

# Combinazioni massime da valutare (limita il tempo di run)
_MAX_COMBINATIONS = 30


# ---------------------------------------------------------------------------
# Dati per una singola finestra walk-forward
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx: int
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    best_params: Dict[str, Any]
    train_sharpe: float
    oos_sharpe: float
    oos_trades: int
    oos_pnl: float
    oos_win_rate: float
    oos_max_dd_pct: float


@dataclass
class OptimizationReport:
    symbol: str
    timeframe: str
    total_months: int
    n_windows: int
    avg_oos_sharpe: float
    best_params: Dict[str, Any]
    windows: List[WindowResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fetcher candles storici
# ---------------------------------------------------------------------------

async def fetch_all_candles(
    symbol: str,
    timeframe: str,
    days: int,
    confirm_tf: str = "1h",
) -> Tuple[List[Candle], List[Candle]]:
    """
    Scarica tutti i candle storici per un symbol da Bybit V5.
    Restituisce (primary_candles, confirm_candles).
    """
    engine = CandleBacktestEngine(symbol=symbol, timeframe=timeframe, days=days)
    primary  = await engine._fetch_candles(timeframe)
    confirm  = await engine._fetch_candles(confirm_tf)
    return primary, confirm


# ---------------------------------------------------------------------------
# Backtest con parametri personalizzati (senza ri-scaricare dati)
# ---------------------------------------------------------------------------

async def run_backtest_on_candles(
    primary_candles: List[Candle],
    confirm_candles: List[Candle],
    symbol: str,
    timeframe: str,
    capital: Decimal,
    param_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Esegue un backtest su un sottoinsieme di candle già scaricati.
    param_overrides: dizionario di chiavi config.strategy.* da sovrascrivere.
    """
    from dotenv import load_dotenv
    load_dotenv()

    cfg = BotSettings.load()

    # Applica overrides alla strategia
    for key, val in param_overrides.items():
        if hasattr(cfg.strategy, key):
            object.__setattr__(cfg.strategy, key, type(getattr(cfg.strategy, key))(val))

    gw = _BacktestGateway()
    sm = _InMemoryStateManager()

    bot = LokyBot(
        symbol=symbol,
        config=cfg,
        execution_gw=gw,
        state_manager=sm,
        capital=capital,
    )

    # Merge candles in ordine cronologico
    all_candles = sorted(
        primary_candles + confirm_candles,
        key=lambda c: c.timestamp,
    )

    equity_curve: List[float] = []
    for candle in all_candles:
        await bot.on_candle(candle)
        if candle.timeframe == timeframe:
            equity_curve.append(float(capital + bot.realized_pnl))

    # Forza chiusura
    if bot._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
        last = primary_candles[-1] if primary_candles else None
        if last:
            await bot._close_remaining(last.close, "end_of_data")

    # Costruisci metriche
    trades_raw = sm.trades
    if not trades_raw:
        return _empty_metrics()

    pnls = [float(t.realized_pnl) for t in trades_raw]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n = len(pnls)
    win_rate = len(wins) / n * 100 if n else 0.0

    # Sharpe (annualizzato)
    tf_min = _TF_MINUTES.get(timeframe, 15)
    periods_per_year = 365 * 24 * 60 // tf_min
    mu = total_pnl / n
    sigma = math.sqrt(sum((p - mu) ** 2 for p in pnls) / n) if n > 1 else 0
    sharpe = (mu / sigma * math.sqrt(periods_per_year)) if sigma > 0 else 0.0

    # Max drawdown
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / float(capital) * 100) if float(capital) > 0 else 0.0

    # Profit factor
    gross_win  = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0

    return {
        "n_trades":      n,
        "total_pnl":     total_pnl,
        "win_rate":      win_rate,
        "sharpe":        sharpe,
        "max_dd_pct":    max_dd_pct,
        "profit_factor": pf,
        "equity_curve":  equity_curve,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
        "sharpe": -999.0, "max_dd_pct": 100.0, "profit_factor": 0.0,
        "equity_curve": [],
    }


# ---------------------------------------------------------------------------
# Grid search su finestra train
# ---------------------------------------------------------------------------

def _generate_param_combinations(grid: Dict[str, List], max_n: int) -> List[Dict[str, Any]]:
    """
    Genera le combinazioni dalla griglia.
    Se il totale supera max_n, campiona uniformemente max_n.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = [dict(zip(keys, combo)) for combo in product(*values)]

    if len(all_combos) <= max_n:
        return all_combos

    # Campionamento uniforme deterministico
    step = len(all_combos) // max_n
    return [all_combos[i * step] for i in range(max_n)]


async def grid_search_window(
    primary_candles: List[Candle],
    confirm_candles: List[Candle],
    symbol: str,
    timeframe: str,
    capital: Decimal,
    param_grid: Dict[str, List],
    max_combinations: int = _MAX_COMBINATIONS,
) -> Tuple[Dict[str, Any], float]:
    """
    Esegue grid search su una finestra di training.
    Restituisce (best_params, best_sharpe).
    """
    combos = _generate_param_combinations(param_grid, max_combinations)
    best_sharpe = -999.0
    best_params: Dict[str, Any] = {}

    logger.info(
        "Grid search: %d combinazioni su %d candle primari",
        len(combos), len(primary_candles),
    )

    for i, params in enumerate(combos):
        try:
            metrics = await run_backtest_on_candles(
                primary_candles, confirm_candles,
                symbol, timeframe, capital, params,
            )
            sh = metrics["sharpe"]
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = params
                logger.debug("Grid [%d/%d] Sharpe=%.2f params=%s", i + 1, len(combos), sh, params)
        except Exception as e:
            logger.warning("Grid [%d/%d] errore: %s", i + 1, len(combos), e)

    return best_params, best_sharpe


# ---------------------------------------------------------------------------
# Walk-Forward Optimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """
    Walk-forward optimizer per Loky Bot.

    Metodologia:
      • Scarica total_months di dati storici
      • Crea finestre scorrevoli: train=train_months, test=test_months, slide=1 mese
      • Per ogni finestra: grid search su train → valuta OOS su test
      • Report: Sharpe OOS medio, best params robusti (mediana su tutte le finestre)

    Args:
        symbol        — coppia Bybit USDT-M (es. "BTCUSDT")
        timeframe     — timeframe primario (es. "15m")
        total_months  — mesi totali di dati storici da scaricare (default 8)
        train_months  — mesi usati per il training in ogni finestra (default 5)
        test_months   — mesi usati per il test OOS in ogni finestra (default 1)
        capital       — USDT simulati per ogni backtest (default 500)
        param_grid    — griglia parametri da ottimizzare
    """

    SECS_PER_MONTH = 30 * 86_400

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        total_months: int = 8,
        train_months: int = 5,
        test_months: int = 1,
        capital: Decimal = Decimal("500"),
        param_grid: Optional[Dict[str, List]] = None,
    ) -> None:
        self.symbol        = symbol
        self.timeframe     = timeframe
        self.total_months  = total_months
        self.train_months  = train_months
        self.test_months   = test_months
        self.capital       = capital
        self.param_grid    = param_grid or PARAM_GRID

        cfg = BotSettings.load()
        self._confirm_tf   = cfg.confirmation_timeframe

    async def run(self) -> OptimizationReport:
        """
        Esegue l'ottimizzazione walk-forward completa.
        Ritorna un OptimizationReport con tutti i risultati.
        """
        logger.info(
            "Walk-Forward Optimizer | %s %s | %d mesi totali | "
            "train=%dm test=%dm",
            self.symbol, self.timeframe, self.total_months,
            self.train_months, self.test_months,
        )

        # Download unico di tutti i dati
        all_primary, all_confirm = await fetch_all_candles(
            symbol=self.symbol,
            timeframe=self.timeframe,
            days=self.total_months * 30 + 10,
            confirm_tf=self._confirm_tf,
        )

        if len(all_primary) < 100:
            logger.error("Dati insufficienti: %d candle primari.", len(all_primary))
            return OptimizationReport(
                symbol=self.symbol, timeframe=self.timeframe,
                total_months=self.total_months, n_windows=0,
                avg_oos_sharpe=0.0, best_params={}, windows=[],
            )

        # Definisci le finestre temporali
        windows = self._build_windows(all_primary)
        logger.info("%d finestre walk-forward definite", len(windows))

        results: List[WindowResult] = []
        all_best_params: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                "Finestra %d/%d | train: %.0fd | test: %.0fd",
                i + 1, len(windows),
                (train_end - train_start) / 86_400,
                (test_end - test_start) / 86_400,
            )

            # Slice candles per questa finestra
            train_prim = [c for c in all_primary if train_start <= c.timestamp < train_end]
            train_conf = [c for c in all_confirm if train_start <= c.timestamp < train_end]

            # OOS: includi warm-up buffer (60 candle prima del test) per evitare
            # che gli indicatori partano da zero nella finestra test.
            # Le prime 60 candle servono solo per scaldare gli indicatori,
            # i trade vengono contati solo dopo il warm-up.
            warmup_candles = 60
            warmup_prim = sorted(
                [c for c in all_primary if c.timestamp < test_start],
                key=lambda c: c.timestamp,
            )[-warmup_candles:]  # ultime 60 candle prima del test
            warmup_conf = sorted(
                [c for c in all_confirm if c.timestamp < test_start],
                key=lambda c: c.timestamp,
            )[-warmup_candles:]

            test_prim_raw = [c for c in all_primary if test_start <= c.timestamp < test_end]
            test_conf_raw = [c for c in all_confirm if test_start <= c.timestamp < test_end]

            # Combina warm-up + test: il backtest parte con indicatori già pronti
            test_prim = warmup_prim + test_prim_raw
            test_conf = warmup_conf + test_conf_raw

            if len(train_prim) < 60 or len(test_prim_raw) < 10:
                logger.warning("Finestra %d saltata: dati insufficienti", i + 1)
                continue

            # Grid search su training window
            best_params, train_sharpe = await grid_search_window(
                train_prim, train_conf,
                self.symbol, self.timeframe, self.capital, self.param_grid,
            )

            # Valuta OOS su test window con i best params
            oos_metrics = await run_backtest_on_candles(
                test_prim, test_conf,
                self.symbol, self.timeframe, self.capital, best_params,
            )

            wr = WindowResult(
                window_idx=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_sharpe=train_sharpe,
                oos_sharpe=oos_metrics["sharpe"],
                oos_trades=oos_metrics["n_trades"],
                oos_pnl=oos_metrics["total_pnl"],
                oos_win_rate=oos_metrics["win_rate"],
                oos_max_dd_pct=oos_metrics["max_dd_pct"],
            )
            results.append(wr)
            all_best_params.append(best_params)

            logger.info(
                "Finestra %d | Train Sharpe=%.2f | OOS Sharpe=%.2f | "
                "OOS trade=%d | OOS PnL=%.2f USDT | OOS WR=%.1f%%",
                i + 1, train_sharpe, wr.oos_sharpe,
                wr.oos_trades, wr.oos_pnl, wr.oos_win_rate,
            )

        if not results:
            return OptimizationReport(
                symbol=self.symbol, timeframe=self.timeframe,
                total_months=self.total_months, n_windows=0,
                avg_oos_sharpe=0.0, best_params={}, windows=[],
            )

        avg_oos = sum(r.oos_sharpe for r in results) / len(results)
        robust_params = self._aggregate_params(all_best_params)

        return OptimizationReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            total_months=self.total_months,
            n_windows=len(results),
            avg_oos_sharpe=avg_oos,
            best_params=robust_params,
            windows=results,
        )

    def _build_windows(
        self, candles: List[Candle]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Costruisce le finestre (train_start, train_end, test_start, test_end).
        Scorre in avanti di test_months ad ogni iterazione.
        """
        if not candles:
            return []

        data_start = candles[0].timestamp
        data_end   = candles[-1].timestamp
        train_secs = self.train_months * self.SECS_PER_MONTH
        test_secs  = self.test_months  * self.SECS_PER_MONTH
        slide_secs = test_secs

        windows: List[Tuple[float, float, float, float]] = []
        cur_start = data_start

        while True:
            train_start = cur_start
            train_end   = train_start + train_secs
            test_start  = train_end
            test_end    = test_start + test_secs

            if test_end > data_end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            cur_start += slide_secs

        return windows

    @staticmethod
    def _aggregate_params(param_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggrega i best params di ogni finestra usando la mediana (robusta agli outlier).
        Per parametri numerici: mediana. Per stringhe: moda.
        """
        if not param_list:
            return {}

        keys = param_list[0].keys()
        result: Dict[str, Any] = {}

        for key in keys:
            values = [p[key] for p in param_list if key in p]
            if not values:
                continue
            if isinstance(values[0], (int, float, Decimal)):
                sorted_v = sorted(float(v) for v in values)
                mid = len(sorted_v) // 2
                if len(sorted_v) % 2 == 1:
                    median = sorted_v[mid]
                else:
                    median = (sorted_v[mid - 1] + sorted_v[mid]) / 2
                # Arrotonda all'intero se tutti i valori originali erano interi
                if all(isinstance(v, int) for v in values):
                    result[key] = int(round(median))
                else:
                    result[key] = round(median, 4)
            else:
                # Moda per non-numerici
                from collections import Counter
                result[key] = Counter(values).most_common(1)[0][0]

        return result


# ---------------------------------------------------------------------------
# Report e export
# ---------------------------------------------------------------------------

def print_report(report: OptimizationReport) -> None:
    w = 65
    print("\n" + "=" * w)
    print(f"  WALK-FORWARD OPTIMIZATION — {report.symbol} {report.timeframe}")
    print("=" * w)
    print(f"  Finestre totali    : {report.n_windows}")
    print(f"  Sharpe OOS medio   : {report.avg_oos_sharpe:.2f}")
    print("-" * w)
    print("  Parametri robusti (mediana su tutte le finestre):")
    for k, v in report.best_params.items():
        print(f"    {k:<26}: {v}")
    print("-" * w)
    print("  Dettaglio finestre:")
    print(f"  {'#':>2}  {'Train Sharpe':>12}  {'OOS Sharpe':>10}  {'OOS Trade':>9}  {'OOS PnL':>8}  {'OOS WR':>7}  {'Max DD':>7}")
    for r in report.windows:
        print(
            f"  {r.window_idx + 1:>2}  "
            f"{r.train_sharpe:>12.2f}  "
            f"{r.oos_sharpe:>10.2f}  "
            f"{r.oos_trades:>9d}  "
            f"{r.oos_pnl:>+8.2f}  "
            f"{r.oos_win_rate:>6.1f}%  "
            f"{r.oos_max_dd_pct:>6.1f}%"
        )
    print("=" * w + "\n")


def export_results(report: OptimizationReport, json_path: str = "optimization_results.json") -> None:
    data = {
        "symbol":         report.symbol,
        "timeframe":      report.timeframe,
        "total_months":   report.total_months,
        "n_windows":      report.n_windows,
        "avg_oos_sharpe": report.avg_oos_sharpe,
        "best_params":    report.best_params,
        "windows": [
            {
                "window_idx":    r.window_idx,
                "train_start":   r.train_start,
                "train_end":     r.train_end,
                "test_start":    r.test_start,
                "test_end":      r.test_end,
                "best_params":   r.best_params,
                "train_sharpe":  r.train_sharpe,
                "oos_sharpe":    r.oos_sharpe,
                "oos_trades":    r.oos_trades,
                "oos_pnl":       r.oos_pnl,
                "oos_win_rate":  r.oos_win_rate,
                "oos_max_dd_pct": r.oos_max_dd_pct,
            }
            for r in report.windows
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Risultati esportati: %s", json_path)


def export_best_params_yaml(report: OptimizationReport, yaml_path: str = "best_params.yaml") -> None:
    """
    Esporta i parametri ottimali in formato YAML pronto per essere incollato in config.yaml.
    """
    lines = [
        "# Parametri ottimali da walk-forward optimization",
        f"# Symbol: {report.symbol} | Timeframe: {report.timeframe}",
        f"# Finestre: {report.n_windows} | Sharpe OOS medio: {report.avg_oos_sharpe:.2f}",
        "#",
        "# Inserire questi valori nella sezione 'strategy:' di config.yaml",
        "strategy:",
    ]
    for k, v in report.best_params.items():
        lines.append(f"  {k}: {v}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Parametri ottimali esportati: %s", yaml_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Walk-Forward Optimizer — Loky Bot")
    parser.add_argument("--symbol",     nargs="+", default=["BTCUSDT"], help="Symbol/i Bybit (es. BTCUSDT ETHUSDT)")
    parser.add_argument("--timeframe",  default="15m",  help="Timeframe primario (default: 15m)")
    parser.add_argument("--months",     type=int, default=8, help="Mesi totali di dati (default: 8)")
    parser.add_argument("--train",      type=int, default=5, help="Mesi di training per finestra (default: 5)")
    parser.add_argument("--test",       type=int, default=1, help="Mesi di test OOS per finestra (default: 1)")
    parser.add_argument("--capital",    type=float, default=500.0, help="Capitale USDT simulato (default: 500)")
    args = parser.parse_args()

    async def _main():
        all_reports = []
        for sym in args.symbol:
            optimizer = WalkForwardOptimizer(
                symbol=sym,
                timeframe=args.timeframe,
                total_months=args.months,
                train_months=args.train,
                test_months=args.test,
                capital=Decimal(str(args.capital)),
            )
            report = await optimizer.run()
            print_report(report)
            export_results(report, f"opt_{sym}_{args.timeframe}.json")
            export_best_params_yaml(report, f"best_params_{sym}.yaml")
            all_reports.append(report)

        if len(all_reports) > 1:
            avg_all = sum(r.avg_oos_sharpe for r in all_reports) / len(all_reports)
            print(f"\nSharpe OOS medio (tutti i symbol): {avg_all:.2f}")

    asyncio.run(_main())


# ---------------------------------------------------------------------------
# RollingOptimizer — adattamento parametri in tempo reale
# ---------------------------------------------------------------------------

class RollingOptimizer:
    """
    Ottimizzatore rolling che adatta i parametri di trading in tempo reale
    basandosi sulla performance degli ultimi N trade.

    Traccia il win rate per ogni valore di parametro usato e suggerisce
    aggiustamenti graduali (max 10% per iterazione).

    Args:
        param_ranges — dict {nome_param: (min, max, step)}
        window — numero di trade per finestra rolling (default 50)
        adjust_pct — max aggiustamento per iterazione (default 0.10 = 10%)
    """

    def __init__(
        self,
        param_ranges: dict[str, tuple[float, float, float]] | None = None,
        window: int = 50,
        adjust_pct: float = 0.10,
    ) -> None:
        from decimal import Decimal
        self._ranges = param_ranges or {
            "tp_atr_mult": (1.5, 4.0, 0.25),
            "sl_atr_mult": (0.5, 1.5, 0.25),
            "mr_adx_max": (15.0, 30.0, 2.0),
            "tf_adx_min": (20.0, 35.0, 2.0),
            "breakout_lookback": (10.0, 50.0, 5.0),
        }
        self._window = window
        self._adjust_pct = adjust_pct

        # Trade results per parametro: {param: [(value_used, pnl), ...]}
        self._history: dict[str, list[tuple[float, float]]] = {
            k: [] for k in self._ranges
        }

    def record_trade(self, params_used: dict[str, float], pnl: float) -> None:
        """Registra il risultato di un trade con i parametri usati."""
        for name in self._ranges:
            if name in params_used:
                self._history[name].append((params_used[name], pnl))
                # Mantieni solo gli ultimi N trade
                if len(self._history[name]) > self._window:
                    self._history[name] = self._history[name][-self._window:]

    def suggest(self, current_params: dict[str, float]) -> dict[str, float]:
        """
        Suggerisce parametri ottimizzati basandosi sulla performance rolling.

        Per ogni parametro:
          1. Calcola il PnL medio per il valore corrente
          2. Calcola il PnL medio per valori adiacenti (±step)
          3. Se un valore adiacente è significativamente migliore, sposta verso quello
          4. Max spostamento: adjust_pct del range

        Returns:
            dict con parametri suggeriti (stessi nomi, valori potenzialmente diversi)
        """
        suggested = dict(current_params)

        for name, (lo, hi, step) in self._ranges.items():
            history = self._history.get(name, [])
            if len(history) < 20:
                continue  # dati insufficienti

            current_val = current_params.get(name)
            if current_val is None:
                continue

            # Calcola PnL medio per il valore corrente
            current_pnls = [pnl for val, pnl in history if abs(val - current_val) < step * 0.5]
            if not current_pnls:
                continue
            current_avg = sum(current_pnls) / len(current_pnls)

            # Prova valore superiore
            up_val = min(current_val + step, hi)
            up_pnls = [pnl for val, pnl in history if abs(val - up_val) < step * 0.5]
            up_avg = sum(up_pnls) / len(up_pnls) if up_pnls else current_avg

            # Prova valore inferiore
            down_val = max(current_val - step, lo)
            down_pnls = [pnl for val, pnl in history if abs(val - down_val) < step * 0.5]
            down_avg = sum(down_pnls) / len(down_pnls) if down_pnls else current_avg

            # Sposta verso il migliore (max adjust_pct del range)
            max_move = (hi - lo) * self._adjust_pct
            best_avg = max(current_avg, up_avg, down_avg)

            if best_avg == up_avg and up_avg > current_avg * 1.05:
                suggested[name] = min(current_val + min(step, max_move), hi)
            elif best_avg == down_avg and down_avg > current_avg * 1.05:
                suggested[name] = max(current_val - min(step, max_move), lo)

        return suggested

    @property
    def has_enough_data(self) -> bool:
        """True se almeno un parametro ha abbastanza dati per suggerire."""
        return any(len(h) >= 20 for h in self._history.values())
