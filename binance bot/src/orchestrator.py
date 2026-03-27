"""
FuturesOrchestrator — coordinatore multi-asset per Binance Futures.

Responsabilità:
  • Istanzia gateway dati (BinanceFuturesDataGateway) e esecuzione (BinanceFuturesExecutionGateway)
  • Crea un DirectionalBot per ogni symbol
  • Imposta la leva all'avvio (live) o salta in paper trading
  • Fa il routing degli eventi candle e order_update ai bot corretti
  • Gestisce shutdown graceful (SIGINT/SIGTERM su Linux/Mac, Ctrl-C su Windows)
"""

import asyncio
import logging
import signal
import sys
from decimal import Decimal
from typing import Optional

from src.bot import DirectionalBot
from src.config import config
from src.core.account_risk import AccountRiskManager
from src.models import Candle
from src.notifications.telegram import TelegramNotifier
from src.state.persistency import StateManager

logger = logging.getLogger("FuturesOrchestrator")


class FuturesOrchestrator:
    """
    Coordina N symbol su Binance Futures con strategia Breakout/Momentum.

    Args:
        symbols  — es. ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        capital  — USDT totali disponibili (divisi equamente tra i symbol)
    """

    def __init__(self, symbols: list[str], capital: Decimal = Decimal("500")) -> None:
        self.symbols  = [s.upper() for s in symbols]
        self._capital = capital
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

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
        )

        # Bot e state manager per ogni symbol
        self._bots: dict[str, DirectionalBot] = {}
        self._state_managers: list[StateManager] = []

        for sym in self.symbols:
            sm  = StateManager(db_path=f"data/state_{sym}.db")
            bot = DirectionalBot(
                symbol=sym,
                config=config,
                execution_gw=self._exec_gw,
                state_manager=sm,
                capital=capital_per_symbol,
                account_risk=self._account_risk,
                notifier=self._notifier,
            )
            self._bots[sym]          = bot
            self._state_managers.append(sm)
            logger.info("Bot inizializzato: %s (capitale=%.2f USDT)", sym, capital_per_symbol)

        # Collega callback fill
        self._exec_gw.set_on_order_update_callback(self._route_order_update)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _route_candle(self, symbol: str, candle: Candle) -> None:
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
        # Imposta leva (solo live)
        if config.live_trading_enabled:
            for sym in self.symbols:
                await self._exec_gw.set_leverage(sym, config.leverage)
            logger.info("Leva impostata: %dx su tutti i symbol", config.leverage)

        self._register_signal_handlers()

        # UserStream (live) — nessun matching engine per Futures
        if config.live_trading_enabled:
            self._tasks.append(asyncio.create_task(
                self._exec_gw.start_userstream(), name="userstream"
            ))

        # Dati di mercato (start avvia bootstrap + WS)
        self._tasks.append(asyncio.create_task(
            self._data_gw.start(self.symbols, [config.primary_timeframe, config.confirmation_timeframe]),
            name="data_feed"
        ))

        # Auto-save state
        for sm in self._state_managers:
            self._tasks.append(asyncio.create_task(sm.auto_save_loop(30.0)))

        logger.info(
            "ORCHESTRATORE AVVIATO — %d symbol | %s trading | timeframe: %s/%s",
            len(self.symbols),
            "LIVE" if config.live_trading_enabled else "PAPER",
            config.primary_timeframe,
            config.confirmation_timeframe,
        )

        mode = "LIVE" if config.live_trading_enabled else "PAPER"
        asyncio.create_task(self._notifier.info(
            f"Bot avviato — {mode} | {', '.join(self.symbols)} | "
            f"{config.primary_timeframe}/{config.confirmation_timeframe}"
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

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
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


# ---------------------------------------------------------------------------
# Factory: sceglie il gateway giusto in base a config.exchange
# ---------------------------------------------------------------------------

def _build_execution_gateway(cfg):
    if cfg.exchange == "bybit":
        from src.gateways.bybit_futures_execution import BybitFuturesExecutionGateway
        return BybitFuturesExecutionGateway(testnet=cfg.testnet)
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
