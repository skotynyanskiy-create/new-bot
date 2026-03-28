# Loky Bot — Guida Rapida: Setup e Primo Backtest

## Requisiti

- Python 3.11+
- Internet (per scaricare dati storici da Bybit)

## Setup (2 minuti)

```bash
# 1. Clona il repo
git clone https://github.com/skotynyanskiy-create/new-bot.git
cd new-bot/loky\ bot

# 2. Installa dipendenze
pip install -e ".[dev]"
# oppure con uv:
uv sync

# 3. Copia il file .env (opzionale per backtest, obbligatorio per live)
cp .env.example .env
```

## Backtest (1 minuto)

```bash
# BTCUSDT, 6 mesi, capitale $500
python -m src.backtest BTCUSDT 15m 180

# BTCUSDT, 3 mesi, capitale $1000
python -m src.backtest BTCUSDT 15m 90 1000

# ETHUSDT, 6 mesi
python -m src.backtest ETHUSDT 15m 180
```

Il backtest scarica i dati da Bybit (gratuito, no API key), li cache in SQLite
per run successivi, e stampa il report con:
- PnL totale e % return
- Sharpe ratio (annualizzato)
- Sortino ratio
- Max drawdown %
- Win rate e profit factor
- Trade per strategia

### Interpretare i Risultati

| Metrica | Buono | Ottimo | Preoccupante |
|---------|-------|--------|-------------|
| Sharpe | > 1.0 | > 2.0 | < 0.5 |
| Max DD | < 15% | < 8% | > 25% |
| Win Rate | > 45% | > 55% | < 35% |
| Profit Factor | > 1.3 | > 2.0 | < 1.0 |

## Validazione Avanzata (opzionale)

```python
# Monte Carlo (1000 permutazioni)
from src.backtest_advanced import MonteCarloSimulator
mc = MonteCarloSimulator(trade_pnls, capital=500, n_simulations=1000)
result = mc.run()
mc.print_report(result)

# Walk-Forward (verifica overfitting)
from src.backtest_advanced import WalkForwardAnalyzer
wf = WalkForwardAnalyzer(trade_pnls, n_windows=5)
result = wf.run()
wf.print_report(result)

# Bootstrap CI (confidence interval)
from src.backtest_advanced import BootstrapValidator
bv = BootstrapValidator(trade_pnls, capital=500)
result = bv.run()
bv.print_report(result)
```

## Paper Trading (dopo backtest positivo)

```bash
# 1. Configura .env con API key Bybit (testnet o mainnet)
#    BYBIT_API_KEY=...
#    BYBIT_SECRET_KEY=...
#    TELEGRAM_TOKEN=...  (opzionale)
#    TELEGRAM_CHAT_ID=... (opzionale)

# 2. Lancia in paper mode (default)
python src/main.py --capital 500 --log-level INFO

# 3. Monitora su Telegram: /status, /equity, /drawdown, /strategy
```

## Live Trading (dopo 2+ settimane di paper positivo)

```yaml
# config.yaml — cambia:
live_trading_enabled: true
testnet: false  # o true per testnet prima
```

⚠️ **NON attivare il live trading senza aver fatto backtest + paper trading.**
