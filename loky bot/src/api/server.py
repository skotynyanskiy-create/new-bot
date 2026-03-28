import asyncio
import json
import logging
import time

from aiohttp import web

logger = logging.getLogger("TelemetryAPI")


class TelemetryServer:
    def __init__(self, orchestrator, port: int = 8080):
        self.orchestrator = orchestrator
        self.port         = port
        self.runner       = None
        self._start_time  = time.time()

    async def metrics_handler(self, request: web.Request) -> web.Response:
        total_pnl    = 0.0
        total_fills  = 0
        total_quotes = 0
        total_volume = 0.0
        markets      = {}

        for asset_id, bot in self.orchestrator.bots.items():
            pnl       = float(bot.pnl)
            inventory = float(bot.net_inventory)
            fills     = bot.fills_total
            quotes    = bot.quotes_sent
            volume    = float(bot.total_volume)

            fill_rate = fills / quotes if quotes > 0 else 0.0

            # Spread medio catturato per fill (approssimato)
            spread_per_fill = pnl / fills if fills > 0 else 0.0

            markets[asset_id] = {
                "pnl_usdt":         round(pnl, 4),
                "inventory":        round(inventory, 6),
                "avg_entry_price":  round(float(bot.avg_entry_price), 2),
                "open_orders":      bot._count_local_open_orders(),
                "fills_total":      fills,
                "quotes_sent":      quotes,
                "fill_rate":        round(fill_rate, 4),
                "volume_usdt":      round(volume, 2),
                "spread_per_fill":  round(spread_per_fill, 6),
            }

            total_pnl    += pnl
            total_fills  += fills
            total_quotes += quotes
            total_volume += volume

        uptime_s = int(time.time() - self._start_time)

        payload = {
            "status":        "online",
            "mode":          "LIVE" if self.orchestrator.execution_gateway.__class__.__name__ == "BinanceExecutionGateway" else "PAPER",
            "uptime_s":      uptime_s,
            "system_pnl":    round(total_pnl, 4),
            "total_fills":   total_fills,
            "total_quotes":  total_quotes,
            "system_fill_rate": round(total_fills / total_quotes, 4) if total_quotes > 0 else 0.0,
            "system_volume_usdt": round(total_volume, 2),
            "active_bots":   len(self.orchestrator.bots),
            "markets":       markets,
        }

        return web.Response(
            text=json.dumps(payload, indent=2),
            content_type='application/json',
        )

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.Response(text='{"status":"ok"}', content_type='application/json')

    async def run_server(self):
        try:
            app = web.Application()
            logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

            app.add_routes([
                web.get('/metrics', self.metrics_handler),
                web.get('/health',  self.health_handler),
            ])

            self.runner = web.AppRunner(app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()

            logger.info(f"🌐 Telemetry API: http://localhost:{self.port}/metrics")
            logger.info(f"🏥 Health check:  http://localhost:{self.port}/health")

            while True:
                await asyncio.sleep(3600)

        except asyncio.CancelledError:
            if self.runner:
                await self.runner.cleanup()
