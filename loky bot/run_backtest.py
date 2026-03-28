#!/usr/bin/env python3
"""
Loky Bot — Script di Backtest Standalone.

Lancia dal terminale:
    python run_backtest.py                    # default: BTCUSDT 15m 180gg $500
    python run_backtest.py ETHUSDT 15m 90     # ETHUSDT 3 mesi
    python run_backtest.py BTCUSDT 15m 180 1000  # capitale $1000

Requisiti: pip install -e ".[dev]"
"""

import asyncio
import sys
import os

# Assicura che src/ sia nel path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decimal import Decimal
from src.backtest import CandleBacktestEngine
from src.backtest_advanced import MonteCarloSimulator, WalkForwardAnalyzer, BootstrapValidator


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15m"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
    capital = Decimal(sys.argv[4]) if len(sys.argv) > 4 else Decimal("500")

    print(f"\n{'='*60}")
    print(f"  LOKY BOT — BACKTEST")
    print(f"  {symbol} | {timeframe} | {days} giorni | ${capital} USDT")
    print(f"{'='*60}\n")

    # Backtest principale
    engine = CandleBacktestEngine(
        symbol=symbol, timeframe=timeframe, days=days, capital=capital,
    )
    asyncio.run(engine.run())
    engine.print_report()

    # Raccogli PnL per analisi avanzata
    trades = engine._build_backtest_trades()
    if not trades:
        print("Nessun trade eseguito. Verifica i parametri.\n")
        return

    pnls = [float(t.pnl) for t in trades]

    # Monte Carlo
    print("\n" + "="*60)
    mc = MonteCarloSimulator(pnls, capital=float(capital), n_simulations=1000)
    result = mc.run()
    mc.print_report(result)

    # Walk-Forward
    if len(pnls) >= 20:
        wf = WalkForwardAnalyzer(pnls, n_windows=5)
        result = wf.run()
        wf.print_report(result)

    # Bootstrap CI
    bv = BootstrapValidator(pnls, capital=float(capital), n_iterations=1000)
    result = bv.run()
    bv.print_report(result)

    # Export CSV
    csv_file = f"backtest_{symbol}_{timeframe}_{days}d.csv"
    engine.export_csv(csv_file)
    print(f"Trade-by-trade esportati in: {csv_file}\n")


if __name__ == "__main__":
    main()
