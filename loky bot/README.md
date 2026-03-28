# Loky Bot v6.7

Multi-strategy directional trading bot for Bybit USDT-M Linear Perpetual Futures.

## Architecture

```
Candle (15m) ──> IndicatorEngine (10 indicators)
                      |
              ┌───────┴────────┐
              v                v
    4 Strategy Engines    SignalAggregator
    - Breakout             - Confluence scoring (50/50)
    - Mean Reversion       - Adaptive strategy weights (expectancy)
    - Trend Following      - MACD + StochRSI bonus
    - Funding Rate         - Multi-TF confirmation (15m/1h/4h)
              |                |
              └───────┬────────┘
                      v
              25 Entry Filters
              - Crisis mode, CHOPPY, cooldown, circuit breaker
              - Daily stop, time-of-day, gap protection
              - Sentiment, macro trend, HTF trend
              - Kelly/Optimal-f sizing, dynamic leverage
              - Portfolio heat, slippage estimation
                      |
                      v
              Position Management
              - 3-level partial TP (50/25/25)
              - Dynamic TP extension
              - Trailing stop (post-TP1, ATR-adaptive)
              - MACD + Order Flow divergence exit
              - Liquidation monitor + gap protection
              - TTL exit + max hold time
```

## Symbols

BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT (configurable in `config.yaml`)

## Indicators

EMA (fast/slow + ribbon 8/13/21/34/55), RSI, ATR, ADX, Bollinger Bands, VWAP, Keltner Channel, MACD (12/26/9), Stochastic RSI (14/3/3)

## Protections

| Protection | Trigger | Action |
|-----------|---------|--------|
| Daily stop | Realized + unrealized > 5% capital | Block all entries |
| Peak drawdown | Equity < peak x 85% | Halt (permanent) |
| BTC crash | BTC -3% in 1h | Block altcoin entries 30min |
| Liquidation monitor | Margin < 15% | Emergency market close |
| Gap protection | Loss > 2x SL distance | Force close |
| Circuit breaker | 3 consecutive losses | Pause 15 candles |
| Correlation block | Spearman > 0.7 | Block correlated pair |
| Portfolio heat | Net exposure > 80% | Size x0.5 |
| Modifier cap | Cumulative modifiers | Floor at 40% of original size |
| Time-of-day | UTC 21-23, 04 | Skip entries (low liquidity) |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Backtest (requires internet for Bybit data)
python run_backtest.py                        # BTCUSDT 6 months $500
python run_backtest.py ETHUSDT 15m 90 1000    # ETHUSDT 3 months $1000

# Paper trading
python src/main.py --capital 500 --log-level INFO

# Run tests
python -m pytest tests/test_loky.py -v
```

## Configuration

All parameters in `config.yaml`. Key settings:

```yaml
tokens: [BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT]
primary_timeframe: "15m"
confirmation_timeframe: "1h"
macro_timeframe: "4h"

risk_per_trade_pct: 0.015    # 1.5% per trade
max_concurrent_positions: 2
leverage: 3
max_daily_loss_pct: 0.05     # 5% daily stop
max_peak_drawdown_pct: 0.15  # 15% halt

kelly_sizing_enabled: true
dynamic_leverage_enabled: true
live_trading_enabled: false   # paper by default
```

## Telegram Commands

`/status` `/equity` `/drawdown` `/strategy` `/pause` `/resume` `/help`

## Project Structure

```
src/
├── bot.py                 # Core state machine (1800 lines)
├── orchestrator.py        # Multi-symbol coordinator
├── config.py              # Pydantic settings (50+ params)
├── strategy/
│   ├── indicator_engine.py    # 10 technical indicators
│   ├── signal_aggregator.py   # Confluence scoring + adaptive weights
│   ├── breakout_engine.py     # Volume breakout detection
│   ├── mean_reversion_engine.py # BB touch + RSI extremes
│   ├── trend_following_engine.py # EMA ribbon pullback
│   ├── funding_rate_engine.py  # Funding rate harvesting
│   ├── orderflow_engine.py    # CVD + liquidation levels
│   ├── volatility_engine.py   # 4-regime classification
│   └── sizing.py              # Shared position sizing
├── core/
│   ├── account_risk.py        # Daily stop + drawdown
│   ├── portfolio_risk.py      # Correlation + heat management
│   ├── kelly_sizing.py        # Kelly + Optimal-f
│   └── liquidation_monitor.py # Margin monitoring
├── gateways/
│   ├── bybit_futures_*.py     # Bybit API integration
│   └── smart_execution.py     # Slippage estimation
├── backtest.py                # Backtest engine + SQLite cache
├── backtest_advanced.py       # Monte Carlo + Walk-Forward + Bootstrap
└── notifications/telegram.py  # 7 commands + alerts
```

## Stats

- 29 commits of improvements from v3.0 to v6.7
- 130 tests (127 unit/integration + 3 end-to-end)
- 30 bug fixes, 30+ features
- 0.22s E2E test runtime (300x optimization)
- 11,600 lines of clean, tested code

## License

Private. Not for redistribution.
