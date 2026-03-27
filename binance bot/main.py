import asyncio
import logging
from prometheus_client import start_http_server
from src.orchestrator import MultiMarketOrchestrator
from src.config import config

logging.basicConfig(level=logging.INFO)

def main():
    print("🚀 Avvio Polymm-Pro Trading Bot")
    
    # Avvia server metriche Prometheus
    start_http_server(8000)
    print("📊 Metriche disponibili su http://localhost:8000")
    
    # Avvia orchestrator
    orchestrator = MultiMarketOrchestrator(config.tokens)
    
    # Run event loop
    asyncio.run(orchestrator.run())

if __name__ == "__main__":
    main()
