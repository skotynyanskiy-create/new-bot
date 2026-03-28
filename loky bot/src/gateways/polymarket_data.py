import asyncio
import json
import logging
import websockets
from decimal import Decimal
from typing import Callable, Coroutine, Any
import time

from src.gateways.interfaces import MarketDataGateway

logger = logging.getLogger("PolymarketWS")

class PolymarketDataGateway(MarketDataGateway):
    def __init__(self, asset_ids: list[str]):
        self.asset_ids = asset_ids
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.on_market_data: Callable[[str, Decimal, float], Coroutine[Any, Any, None]] = None

    def set_on_market_data_callback(self, callback: Callable[[str, Decimal, float], Coroutine[Any, Any, None]]):
        self.on_market_data = callback

    async def run_forever(self):
        logger.info(f"🔗 Connessione Polymarket Multi-Asset (Top {len(self.asset_ids)} mercati)...")
        
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    subscribe_msg = {
                        "assets_ids": self.asset_ids,
                        "type": "market"
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("✅ Sottoscrizione Multi-Market Inviata.")
                    
                    async for message in ws:
                        await self._process_message(message)
                        
            except websockets.ConnectionClosed:
                logger.warning("⚠️ Connessione WSS chiusa. Reconnect in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Errore WSS: {e}. Reconnect in 5s...")
                await asyncio.sleep(5)

    async def _process_message(self, message: str):
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for event in data:
                    asset_id = event.get("asset_id")
                    if asset_id in self.asset_ids:
                        bids = event.get("bids", [])
                        asks = event.get("asks", [])
                        
                        # Calcolo VWAP L2 Depth sui primi 3 livelli
                        if bids and asks:
                            vwap_bid_sum = Decimal('0')
                            vwap_bid_vol = Decimal('0')
                            for b in bids[:3]:
                                b_p = Decimal(str(b["price"]))
                                b_s = Decimal(str(b["size"]))
                                vwap_bid_sum += b_p * b_s
                                vwap_bid_vol += b_s
                                
                            vwap_ask_sum = Decimal('0')
                            vwap_ask_vol = Decimal('0')
                            for a in asks[:3]:
                                a_p = Decimal(str(a["price"]))
                                a_s = Decimal(str(a["size"]))
                                vwap_ask_sum += a_p * a_s
                                vwap_ask_vol += a_s
                                
                            if vwap_bid_vol > Decimal('0') and vwap_ask_vol > Decimal('0'):
                                avg_bid = vwap_bid_sum / vwap_bid_vol
                                avg_ask = vwap_ask_sum / vwap_ask_vol
                                imbalance = vwap_bid_vol / (vwap_bid_vol + vwap_ask_vol)
                                
                                micro_price = (avg_ask * imbalance) + (avg_bid * (Decimal('1') - imbalance))
                                
                                timestamp = time.monotonic()
                                if self.on_market_data:
                                    asyncio.create_task(self.on_market_data(asset_id, Decimal(str(round(micro_price, 4))), timestamp))
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Payload parse fail: {e}")
