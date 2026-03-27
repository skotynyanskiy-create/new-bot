"""
Ottimizzazione parametri con Optuna + BacktestEngine reale.

Uso:
    python -m src.optimization --symbol BTCUSDT --trials 200

Il processo:
1. Genera prezzi storici sintetici (Geometric Brownian Motion)
   oppure carica da file CSV se disponibile.
2. Per ogni trial Optuna, esegue un backtest completo.
3. Massimizza lo Sharpe ratio come funzione obiettivo.
4. Stampa i migliori parametri trovati.
"""

import asyncio
import math
import random
import argparse
import logging
from decimal import Decimal
from typing import List

import optuna
from optuna.samplers import TPESampler

from src.backtest import BacktestEngine
from src.config import BotSettings

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ #
#  Generazione dati storici sintetici (GBM)                           #
# ------------------------------------------------------------------ #
def generate_gbm_prices(
    n_ticks: int = 10_000,
    s0: float = 40_000.0,
    mu: float = 0.0,
    sigma: float = 0.0003,  # ~0.03% volatilità per tick (100ms)
    seed: int = 42,
) -> List[Decimal]:
    """Geometric Brownian Motion per simulare prezzi BTC/USDT."""
    random.seed(seed)
    prices = [s0]
    dt = 0.1  # 100ms

    import statistics as st
    for _ in range(n_ticks - 1):
        z = random.gauss(0, 1)
        drift = (mu - 0.5 * sigma ** 2) * dt
        shock = sigma * math.sqrt(dt) * z
        prices.append(prices[-1] * math.exp(drift + shock))

    return [Decimal(str(round(p, 2))) for p in prices]


# ------------------------------------------------------------------ #
#  Funzione obiettivo Optuna                                           #
# ------------------------------------------------------------------ #
def make_objective(symbol: str, prices: List[Decimal]):
    def objective(trial: optuna.Trial) -> float:
        cfg = BotSettings(
            tokens=[symbol],
            base_spread=Decimal(str(trial.suggest_float('base_spread', 0.0010, 0.0100))),
            skew_factor=Decimal(str(trial.suggest_float('skew_factor', 0.1, 3.0))),
            quote_size=Decimal(str(trial.suggest_float('quote_size', 0.0005, 0.005))),
            max_inventory=Decimal(str(trial.suggest_float('max_inventory', 0.005, 0.05))),
            fee_maker=Decimal('0.001'),
            fee_taker=Decimal('0.001'),
            rate_limit_rps=8,
            live_trading_enabled=False,
            max_daily_loss=Decimal('-100'),
        )

        engine = BacktestEngine(symbol=symbol, historical_prices=prices, cfg=cfg)
        result = asyncio.run(engine.run())

        sharpe = result.get('sharpe', 0.0)
        pnl    = result.get('pnl', 0.0)

        # Penalizza inventario residuo (non ha chiuso posizioni)
        residual_inventory = abs(result.get('final_inventory', 0.0))
        penalty = residual_inventory * float(cfg.quote_size) * 100

        return sharpe + (pnl * 0.01) - penalty

    return objective


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
def optimize_parameters(symbol: str = "BTCUSDT", n_trials: int = 100, n_ticks: int = 10_000):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"🔬 Ottimizzazione parametri per {symbol} — {n_trials} trial, {n_ticks} tick")

    prices = generate_gbm_prices(n_ticks=n_ticks)
    logger.info(f"📊 Prezzi GBM generati: {len(prices)} tick | range [{min(prices):.2f}, {max(prices):.2f}]")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f"polymm_{symbol}",
    )
    study.optimize(make_objective(symbol, prices), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print("\n" + "="*60)
    print(f"🏆 MIGLIORI PARAMETRI per {symbol}:")
    print(f"   base_spread:   {best['base_spread']:.4f}  ({best['base_spread']*100:.2f} bps)")
    print(f"   skew_factor:   {best['skew_factor']:.4f}")
    print(f"   quote_size:    {best['quote_size']:.5f} BTC")
    print(f"   max_inventory: {best['max_inventory']:.5f} BTC")
    print(f"   Sharpe:        {study.best_value:.3f}")
    print("="*60)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ottimizzazione parametri PolyMM-Pro")
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--ticks',  type=int, default=10_000)
    args = parser.parse_args()

    optimize_parameters(symbol=args.symbol, n_trials=args.trials, n_ticks=args.ticks)
