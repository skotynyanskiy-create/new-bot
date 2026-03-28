"""
Entry point — Loky Multi-Strategy Futures Bot su Bybit USDT-M Perpetuals.

Avvio:
    uv run src/main.py                 # paper trading (default)
    uv run src/main.py --live          # live trading (richiede .env con chiavi reali)
    uv run src/main.py --capital 1000  # capitale custom USDT

Monitoring:
    http://localhost:9090/metrics   (Prometheus)
    http://localhost:8080/health    (Health check)
    loky.log                        (Log JSON strutturato con rotation)
"""

import argparse
import asyncio
import logging
import sys
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

from src.config import config
from src.logging_setup import setup_logging
from src.orchestrator import FuturesOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loky — Multi-Strategy Bybit Futures Bot")
    p.add_argument("--live",    action="store_true", help="Abilita live trading (default: paper)")
    p.add_argument("--capital", type=float, default=500.0, help="Capitale USDT (default: 500)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--json-console", action="store_true", help="Output log JSON su console (per produzione)")
    p.add_argument("--no-logfile", action="store_true", help="Disabilita log su file")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Setup logging strutturato
    setup_logging(
        log_file=None if args.no_logfile else "loky.log",
        level=getattr(logging, args.log_level),
        json_console=args.json_console,
    )

    logger = logging.getLogger("MAIN")
    capital = Decimal(str(args.capital))

    logger.info(
        "Avvio Loky Bot | symbols=%s | capital=%.0f USDT | mode=%s",
        ", ".join(config.tokens),
        capital,
        "LIVE" if config.live_trading_enabled else "PAPER",
    )

    orchestrator = FuturesOrchestrator(symbols=config.tokens, capital=capital)
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
