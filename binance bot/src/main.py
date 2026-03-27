"""
Entry point — Breakout/Momentum Bot su Binance USDT-M Futures.

Avvio:
    uv run src/main.py            # paper trading (default)
    uv run src/main.py --live     # live trading (richiede .env con chiavi reali)

Monitoring:
    http://localhost:8000/metrics  (Prometheus)
"""

import asyncio
import logging
import sys
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

from src.config import config
from src.orchestrator import FuturesOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MAIN")
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


async def main() -> None:
    logger.info(
        "Avvio Breakout/Momentum Futures Bot | %s | %s trading",
        ", ".join(config.tokens),
        "LIVE" if config.live_trading_enabled else "PAPER",
    )

    # Avvia Prometheus metrics server (opzionale)
    try:
        from prometheus_client import start_http_server
        start_http_server(8000)
        logger.info("Prometheus metrics: http://localhost:8000/metrics")
    except Exception:
        logger.info("prometheus_client non disponibile — metriche disabilitate")

    # Stima capitale: 500 USDT di default (aggiornare per live)
    capital = Decimal("500")

    orchestrator = FuturesOrchestrator(symbols=config.tokens, capital=capital)
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
