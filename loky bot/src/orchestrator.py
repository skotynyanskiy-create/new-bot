"""
Loky — FuturesOrchestrator, coordinatore multi-asset per Bybit Futures.

Responsabilità:
  • Istanzia gateway dati ed esecuzione (Bybit/Binance)
  • Crea un LokyBot per ogni symbol
  • Imposta la leva all'avvio (live) o salta in paper trading
  • Fa il routing degli eventi candle e order_update ai bot corretti
  • Gestisce shutdown graceful (SIGINT/SIGTERM su Linux/Mac, Ctrl-C su Windows)
"""

import asyncio
import json
import logging
import signal
import sys
import time
from decimal import Decimal
from typing import Optional

from src.bot import LokyBot, _start_prometheus_server
from src.config import config
from src.core.account_risk import AccountRiskManager
from src.core.portfolio_risk import PortfolioRiskManager
from src.models import Candle
from src.notifications.telegram import TelegramNotifier
from src.state.persistency import StateManager

logger = logging.getLogger("FuturesOrchestrator")

# Soglia dead-man switch: se non arrivano candle entro N secondi → alert
_DEAD_MAN_THRESHOLD_S = 600   # 10 minuti


class FuturesOrchestrator:
    """
    Loky — coordina N symbol su Bybit Futures con multi-strategia.

    Args:
        symbols  — es. ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
        capital  — USDT totali disponibili (divisi equamente tra i symbol)
    """

    def __init__(self, symbols: list[str], capital: Decimal = Decimal("500")) -> None:
        self.symbols  = [s.upper() for s in symbols]
        self._capital = capital
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._notify_tasks: set[asyncio.Task] = set()
        self._start_time = time.time()
        self._last_candle_time = time.time()   # per dead-man switch

        # Capital per-symbol
        capital_per_symbol = capital / Decimal(len(symbols))

        # Gateway esecuzione
        if config.live_trading_enabled:
            net_label = "TESTNET" if config.testnet else "LIVE — VERRANNO MOSSI FONDI REALI"
            logger.warning("TRADING ATTIVATO — %s | exchange=%s", net_label, config.exchange)
            self._exec_gw = _build_execution_gateway(config)
        else:
            logger.info("Paper trading — nessun ordine reale verrà inviato")
            self._exec_gw = _PaperExecutionGateway()

        # Gateway dati
        self._data_gw = _build_data_gateway(config, self._route_candle)

        # Notifiche Telegram (shared, no-op se env vars mancanti)
        self._notifier = TelegramNotifier()

        # Risk manager account-level (condiviso tra tutti i bot)
        self._account_risk = AccountRiskManager(
            max_daily_loss_account=config.max_daily_loss * Decimal(len(self.symbols)),
            max_concurrent_positions=config.max_concurrent_positions,
            max_peak_drawdown_pct=getattr(config, 'max_peak_drawdown_pct', Decimal('0.15')),
            initial_capital=capital,
        )

        # Callback drawdown alert → Telegram
        async def _on_drawdown_stop(equity: float, peak: float, dd_pct: float) -> None:
            await self._notifier.info(
                f"🛑 <b>PEAK DRAWDOWN STOP</b>\n"
                f"Equity: {equity:.2f} USDT | Picco: {peak:.2f} USDT\n"
                f"Drawdown: {dd_pct:.1f}% — Trading sospeso.\n"
                f"Riavvio manuale necessario."
            )
        self._account_risk.set_on_drawdown_stop(_on_drawdown_stop)

        # Portfolio risk manager (condiviso tra tutti i bot — gestisce notional + correlazione)
        # Legge i gruppi di correlazione da config.yaml invece di hardcoded
        self._portfolio_risk = PortfolioRiskManager(
            capital=capital,
            max_leverage=config.max_leverage,
            max_single_position_pct=config.max_position_per_asset,
            correlation_groups=config.correlation_groups_as_dict(),
        )

        # Bot e state manager per ogni symbol
        self._bots: dict[str, LokyBot] = {}
        self._state_managers: list[StateManager] = []

        for sym in self.symbols:
            sm  = StateManager(db_path=f"data/state_{sym}.db")
            bot = LokyBot(
                symbol=sym,
                config=config,
                execution_gw=self._exec_gw,
                state_manager=sm,
                capital=capital_per_symbol,
                account_risk=self._account_risk,
                portfolio_risk=self._portfolio_risk,
                notifier=self._notifier,
            )
            self._bots[sym]          = bot
            self._state_managers.append(sm)
            logger.info("[Loky] Bot inizializzato: %s (capitale=%.2f USDT)", sym, capital_per_symbol)

        # Collega callback fill
        self._exec_gw.set_on_order_update_callback(self._route_order_update)

    # ------------------------------------------------------------------
    # Fire-and-forget task tracking
    # ------------------------------------------------------------------

    def _fire_and_forget(self, coro) -> None:
        """Crea un task asincrono tracciato con cleanup automatico e error logging."""
        task = asyncio.create_task(coro)
        self._notify_tasks.add(task)
        def _on_done(t: asyncio.Task) -> None:
            self._notify_tasks.discard(t)
            if not t.cancelled() and t.exception():
                logger.warning("Notifica async fallita: %s", t.exception())
        task.add_done_callback(_on_done)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _route_candle(self, symbol: str, candle: Candle) -> None:
        self._last_candle_time = time.time()   # aggiorna dead-man switch
        bot = self._bots.get(symbol)
        if bot:
            await bot.on_candle(candle)

    async def _route_order_update(self, order) -> None:
        bot = self._bots.get(order.symbol)
        if bot:
            await bot.on_order_update(order)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Avvia tutti i task e attende lo shutdown."""
        # Log configurazione attiva a startup
        config.log_startup_config()

        # Imposta leva su Bybit (solo live)
        # Usa max_leverage per avere margine; la size dinamica controlla l'esposizione reale
        if config.live_trading_enabled:
            lev = config.max_leverage if config.dynamic_leverage_enabled else config.leverage
            for sym in self.symbols:
                await self._exec_gw.set_leverage(sym, lev)
            logger.info(
                "Leva impostata: %dx su tutti i symbol (%s)",
                lev, "dinamica — size adattata in base ATR" if config.dynamic_leverage_enabled else "statica",
            )

        self._register_signal_handlers()

        # UserStream (live) — nessun matching engine per Futures
        if config.live_trading_enabled:
            self._tasks.append(asyncio.create_task(
                self._exec_gw.start_userstream(), name="userstream"
            ))

            # Watchdog posizioni (solo live): riconcilia bot vs exchange ogni 60s
            if hasattr(self._exec_gw, 'start_position_watchdog'):
                if hasattr(self._exec_gw, 'set_notifier'):
                    self._exec_gw.set_notifier(self._notifier)
                self._tasks.append(asyncio.create_task(
                    self._exec_gw.start_position_watchdog(self.symbols, interval_s=60.0),
                    name="position_watchdog"
                ))

        # Dati di mercato (start avvia bootstrap + WS)
        # Include il macro timeframe (4h) per il filtro tendenza macro
        macro_tf = getattr(config, 'macro_timeframe', '4h')
        timeframes = list({config.primary_timeframe, config.confirmation_timeframe, macro_tf})
        self._tasks.append(asyncio.create_task(
            self._data_gw.start(self.symbols, timeframes),
            name="data_feed"
        ))

        # Auto-save state
        for sm in self._state_managers:
            self._tasks.append(asyncio.create_task(sm.auto_save_loop(30.0)))

        logger.info(
            "[Loky] AVVIATO — %d symbol | %s trading | timeframe: %s/%s | exchange: %s",
            len(self.symbols),
            "LIVE" if config.live_trading_enabled else "PAPER",
            config.primary_timeframe,
            config.confirmation_timeframe,
            config.exchange,
        )

        mode = "LIVE" if config.live_trading_enabled else "PAPER"
        self._fire_and_forget(self._notifier.info(
            f"🤖 <b>Loky</b> avviato — {mode} | {', '.join(self.symbols)} | "
            f"{config.primary_timeframe}/{config.confirmation_timeframe} | {config.exchange}"
        ))

        # Prometheus metrics server (porta 9090)
        _start_prometheus_server(port=9090)

        # Health check HTTP (porta 8080)
        self._tasks.append(asyncio.create_task(
            self._health_server(), name="health_server"
        ))

        # Dead-man switch: controlla che arrivino candle regolarmente
        self._tasks.append(asyncio.create_task(
            self._dead_man_switch(), name="dead_man_switch"
        ))

        # Telegram: polling comandi + daily summary
        if hasattr(self._notifier, 'start_command_polling'):
            # Inietta callback per /status
            self._notifier.set_status_callback(self._status_text)
            self._tasks.append(asyncio.create_task(
                self._notifier.start_command_polling(), name="tg_commands"
            ))
        self._tasks.append(asyncio.create_task(
            self._daily_summary_loop(), name="daily_summary"
        ))

        try:
            await self._shutdown_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Interruzione ricevuta...")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        logger.info("Shutdown — cancellazione ordini in corso...")
        await self._notifier.info("Bot in shutdown — cancellazione ordini in corso.")
        try:
            await self._exec_gw.cancel_all_orders()
        except Exception as e:
            logger.error("Errore cancel-all: %s", e)

        await self._data_gw.stop()

        # Chiudi sessioni HTTP dei bot (funding rate engine + sentiment engine)
        for bot in self._bots.values():
            try:
                await bot.close()
            except Exception as e:
                logger.debug("Errore chiusura bot %s: %s", bot.symbol, e)

        for task in self._tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown: %d task non terminati entro 10s — forzatura.", len(self._tasks))

        # Cancella notifiche fire-and-forget pendenti
        for task in self._notify_tasks:
            task.cancel()
        if self._notify_tasks:
            await asyncio.gather(*self._notify_tasks, return_exceptions=True)
        self._notify_tasks.clear()

        logger.info("Shutdown completato.")

    def _register_signal_handlers(self) -> None:
        if sys.platform == "win32":
            return  # Windows: usa KeyboardInterrupt
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(self._graceful_shutdown())
            )

    async def _graceful_shutdown(self) -> None:
        logger.info("Segnale shutdown ricevuto.")
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Health check HTTP server (porta 8080)
    # ------------------------------------------------------------------

    async def _health_server(self) -> None:
        """
        Espone GET /health su porta 8080 per monitoraggio uptime.
        Risponde con JSON: {"status": "ok", "uptime_s": N, "positions": N, "symbols": [...]}
        """
        from aiohttp import web

        async def health_handler(request):
            positions = [
                sym for sym, bot in self._bots.items()
                if bot._position_side is not None
            ]
            body = json.dumps({
                "status":   "ok",
                "uptime_s": int(time.time() - self._start_time),
                "positions": len(positions),
                "open_symbols": positions,
                "paper_mode": not config.live_trading_enabled,
            })
            return web.Response(text=body, content_type="application/json")

        app = web.Application()
        app.router.add_get("/health", health_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8080)
        try:
            await site.start()
            logger.info("Health check server avviato su :8080/health")
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()

    # ------------------------------------------------------------------
    # Dead-man switch: alert se nessun candle per 10 minuti
    # ------------------------------------------------------------------

    async def _dead_man_switch(self) -> None:
        """Controlla che il feed dati sia attivo. Alert se silenzio > 10 min."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)
            silence = time.time() - self._last_candle_time
            if silence > _DEAD_MAN_THRESHOLD_S:
                msg = (
                    f"💀 DEAD-MAN SWITCH: nessun candle ricevuto da {silence:.0f}s. "
                    f"Feed dati probabilmente interrotto."
                )
                logger.warning(msg)
                self._fire_and_forget(self._notifier.info(msg))

    # ------------------------------------------------------------------
    # Daily summary automatico a mezzanotte UTC
    # ------------------------------------------------------------------

    async def _daily_summary_loop(self) -> None:
        """Invia il sommario giornaliero ogni giorno a mezzanotte UTC."""
        import datetime
        while not self._shutdown_event.is_set():
            now = datetime.datetime.utcnow()
            # Attendi fino a mezzanotte UTC
            tomorrow = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=5, microsecond=0
            )
            wait_s = (tomorrow - now).total_seconds()
            await asyncio.sleep(min(wait_s, 3600))  # controlla ogni ora max

            if datetime.datetime.utcnow().hour != 0:
                continue

            total_pnl = sum(bot.realized_pnl for bot in self._bots.values())
            n_trades  = sum(bot.total_trades for bot in self._bots.values())
            n_wins    = sum(bot.total_wins   for bot in self._bots.values())
            n_losses  = sum(bot.total_losses for bot in self._bots.values())
            self._fire_and_forget(self._notifier.daily_summary(
                date_str=now.strftime("%Y-%m-%d"),
                total_pnl=total_pnl,
                trades=n_trades,
                wins=n_wins,
                losses=n_losses,
            ))

    # ------------------------------------------------------------------
    # Status text per comando /status
    # ------------------------------------------------------------------

    async def _status_text(self) -> str:
        """Genera il testo di status per comando Telegram /status."""
        uptime_h  = (time.time() - self._start_time) / 3600
        total_pnl = sum(bot.realized_pnl for bot in self._bots.values())
        n_trades  = sum(bot.total_trades  for bot in self._bots.values())
        n_wins    = sum(bot.total_wins    for bot in self._bots.values())
        n_losses  = sum(bot.total_losses  for bot in self._bots.values())
        wr_pct    = (n_wins / n_trades * 100) if n_trades > 0 else 0.0
        positions = [
            f"{sym}: {bot._position_side.name if bot._position_side else '—'}"
            for sym, bot in self._bots.items()
        ]
        mode = "🔴 LIVE" if config.live_trading_enabled else "📄 PAPER"

        # Aggrega PnL attribution per engine (tutti i bot)
        from collections import defaultdict
        pnl_attr: dict[str, float] = defaultdict(float)
        for bot in self._bots.values():
            for strat, pnl in bot.pnl_by_strategy.items():
                pnl_attr[strat] += float(pnl)

        attr_lines = "  ".join(
            f"{k[:12]}:{v:+.2f}" for k, v in sorted(pnl_attr.items())
        ) if pnl_attr else "n/d"

        return (
            f"Uptime   : {uptime_h:.1f}h | {mode}\n"
            f"PnL      : {float(total_pnl):+.4f} USDT\n"
            f"Trade    : {n_trades} ({n_wins}W/{n_losses}L — {wr_pct:.0f}% WR)\n"
            f"Posizioni: {', '.join(positions)}\n"
            f"Engines  : {attr_lines}"
        )


# ---------------------------------------------------------------------------
# Factory: sceglie il gateway giusto in base a config.exchange
# ---------------------------------------------------------------------------

def _build_execution_gateway(cfg):
    if cfg.exchange == "bybit":
        from src.gateways.bybit_futures_execution import BybitFuturesExecutionGateway
        return BybitFuturesExecutionGateway(testnet=cfg.testnet, rate_limit_rps=cfg.rate_limit_rps)
    else:
        from src.gateways.binance_futures_execution import BinanceFuturesExecutionGateway
        return BinanceFuturesExecutionGateway(testnet=cfg.testnet)


def _build_data_gateway(cfg, on_candle_close):
    if cfg.exchange == "bybit":
        from src.gateways.bybit_futures_data import BybitFuturesDataGateway
        return BybitFuturesDataGateway(
            on_candle_close=on_candle_close,
            bootstrap_bars=100,
            testnet=cfg.testnet,
        )
    else:
        from src.gateways.binance_futures_data import BinanceFuturesDataGateway
        return BinanceFuturesDataGateway(
            on_candle_close=on_candle_close,
            bootstrap_bars=100,
            testnet=cfg.testnet,
        )


# ---------------------------------------------------------------------------
# Paper gateway minimale (nessun ordine reale, fill simulato lato bot)
# ---------------------------------------------------------------------------

class _PaperExecutionGateway:
    """Gateway no-op per paper trading: tutti i metodi sono stub."""

    def __init__(self) -> None:
        self._callback = None

    def set_on_order_update_callback(self, cb) -> None:
        self._callback = cb

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        logger.info("[PAPER] set_leverage %s x%d (ignorato)", symbol, leverage)

    async def submit_market_order(self, symbol, side, size) -> None:
        logger.info("[PAPER] market order %s %s %s (ignorato — fill simulato)", symbol, side.name, size)
        return None

    async def submit_limit_with_fallback(self, symbol, side, size, limit_price, timeout_s=5.0) -> None:
        logger.info("[PAPER] limit+fallback %s %s %s @ %s (ignorato)", symbol, side.name, size, limit_price)
        return None

    async def submit_tp_sl(self, symbol, side, size, tp_price, sl_price) -> None:
        logger.info("[PAPER] TP=%.4f SL=%.4f (ignorati)", tp_price, sl_price)

    async def submit_order(self, order) -> None:
        pass

    async def cancel_order(self, order) -> None:
        pass

    async def cancel_all_orders(self) -> None:
        pass

    async def fetch_open_orders_count(self) -> int:
        return 0

    async def fetch_real_inventory(self, asset_id: str) -> Decimal:
        return Decimal("0")

    async def match_engine_tick(self) -> None:
        pass

    async def start_userstream(self) -> None:
        pass
