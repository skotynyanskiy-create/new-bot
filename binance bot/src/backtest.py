"""
CandleBacktestEngine — backtesting della strategia Breakout/Momentum su dati storici.

Funzionalità:
  • Download candele storiche da Binance Futures REST (/fapi/v1/klines)
  • Feed candle-by-candle a DirectionalBot (stessa logica del live)
  • TP/SL simulati su high/low della candela (gestiti da bot.py)
  • Fee taker realistiche
  • Metriche: PnL, Sharpe, max drawdown, win rate, profit factor

Utilizzo:
    engine = CandleBacktestEngine(symbol="BTCUSDT", timeframe="15m", days=180)
    await engine.run()
    engine.print_report()
    engine.export_csv("btc_backtest.csv")

CLI:
    python -m src.backtest BTCUSDT 15m 180
"""

import asyncio
import csv
import logging
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

import aiohttp

from src.bot import BotState, DirectionalBot
from src.config import BotSettings
from src.models import Candle, Side

logger = logging.getLogger(__name__)

_REST_BASE = "https://fapi.binance.com"


# ---------------------------------------------------------------------------
# Struttura risultato trade
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    pnl: Decimal
    fee: Decimal
    entry_time: float
    exit_time: float
    reason: str

    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time) / 3600


# ---------------------------------------------------------------------------
# Gateway e StateManager minimali per backtest (in-memory, no rete)
# ---------------------------------------------------------------------------

class _BacktestGateway:
    """No-op gateway: il paper mode è gestito interamente in bot.py."""

    def __init__(self) -> None:
        self._callback = None
        self.trades: List[BacktestTrade] = []

    def set_on_order_update_callback(self, cb) -> None:
        self._callback = cb

    async def set_leverage(self, symbol, leverage) -> None: pass
    async def submit_market_order(self, symbol, side, size): return None
    async def submit_tp_sl(self, symbol, side, size, tp, sl) -> None: pass
    async def submit_order(self, order) -> None: pass
    async def cancel_order(self, order) -> None: pass
    async def cancel_all_orders(self) -> None: pass
    async def fetch_open_orders_count(self) -> int: return 0
    async def fetch_real_inventory(self, asset_id) -> Decimal: return Decimal("0")
    async def match_engine_tick(self) -> None: pass
    async def start_userstream(self) -> None: pass

    def record(
        self,
        symbol: str,
        side: str,
        entry: Decimal,
        exit_: Decimal,
        size: Decimal,
        fee_rate: Decimal,
        entry_time: float,
        exit_time: float,
        reason: str,
    ) -> None:
        fee = (entry + exit_) * size * fee_rate
        pnl = ((exit_ - entry) if side == "BUY" else (entry - exit_)) * size - fee
        self.trades.append(BacktestTrade(
            symbol=symbol, side=side,
            entry_price=entry, exit_price=exit_,
            size=size, pnl=pnl, fee=fee,
            entry_time=entry_time, exit_time=exit_time,
            reason=reason,
        ))


class _InMemoryStateManager:
    def load_state(self) -> None: return None
    def update_snapshot(self, **kwargs) -> None: pass
    async def save_trade(self, trade) -> None: pass
    async def auto_save_loop(self, interval=30.0) -> None: pass


# ---------------------------------------------------------------------------
# Engine principale
# ---------------------------------------------------------------------------

class CandleBacktestEngine:
    """
    Backtest candle-based della strategia Breakout/Momentum.

    Args:
        symbol    — coppia Futures (es. "BTCUSDT")
        timeframe — timeframe candele (es. "15m")
        days      — giorni di storico da Binance (default 180)
        capital   — USDT simulati (default 500)
        config    — BotSettings opzionale (usa config.yaml altrimenti)
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        days: int = 180,
        capital: Decimal = Decimal("500"),
        config: Optional[BotSettings] = None,
    ) -> None:
        self.symbol    = symbol
        self.timeframe = timeframe
        self.days      = days
        self.capital   = capital
        self.cfg       = config or BotSettings.load()
        self._gw       = _BacktestGateway()
        self._sm       = _InMemoryStateManager()
        self._candles: List[Candle] = []

    # ------------------------------------------------------------------
    # Esecuzione
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("Backtest %s %s | %d giorni | capitale=%.0f USDT",
                    self.symbol, self.timeframe, self.days, self.capital)

        self._candles = await self._fetch_candles()
        if len(self._candles) < 60:
            logger.error("Dati insufficienti: solo %d candele.", len(self._candles))
            return
        logger.info("Candele: %d", len(self._candles))

        bot = DirectionalBot(
            symbol=self.symbol,
            config=self.cfg,
            execution_gw=self._gw,
            state_manager=self._sm,
            capital=self.capital,
        )

        # Intercetta _close_position per registrare ogni trade chiuso
        original_close = bot._close_position

        async def patched_close(exit_price: Decimal, reason: str) -> None:
            if bot._position_side is not None and bot._position_size > Decimal("0"):
                self._gw.record(
                    symbol=self.symbol,
                    side=bot._position_side.name,
                    entry=bot._entry_price,
                    exit_=exit_price,
                    size=bot._position_size,
                    fee_rate=self.cfg.fee_taker,
                    entry_time=bot._entry_time,
                    exit_time=time.time(),
                    reason=reason,
                )
            await original_close(exit_price, reason)

        bot._close_position = patched_close

        # Feed candele al bot
        for candle in self._candles:
            await bot.on_candle(candle)

        # Chiudi posizione eventualmente aperta alla fine dei dati
        if bot._state == BotState.POSITION_OPEN and self._candles:
            last = self._candles[-1]
            self._gw.record(
                symbol=self.symbol,
                side=bot._position_side.name if bot._position_side else "?",
                entry=bot._entry_price,
                exit_=last.close,
                size=bot._position_size,
                fee_rate=self.cfg.fee_taker,
                entry_time=bot._entry_time,
                exit_time=time.time(),
                reason="end_of_data",
            )

        logger.info("Backtest completato: %d trade eseguiti.", len(self._gw.trades))

    # ------------------------------------------------------------------
    # Download dati storici
    # ------------------------------------------------------------------

    async def _fetch_candles(self) -> List[Candle]:
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - self.days * 86400 * 1000
        limit    = 1500
        candles: List[Candle] = []
        cur      = start_ms

        async with aiohttp.ClientSession() as session:
            while cur < end_ms:
                params = {
                    "symbol":    self.symbol,
                    "interval":  self.timeframe,
                    "startTime": cur,
                    "endTime":   end_ms,
                    "limit":     limit,
                }
                try:
                    async with session.get(
                        f"{_REST_BASE}/fapi/v1/klines",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                except Exception as e:
                    logger.error("Fetch candele fallito: %s", e)
                    break

                if not data:
                    break

                for row in data:
                    candles.append(Candle(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        open=Decimal(str(row[1])),
                        high=Decimal(str(row[2])),
                        low=Decimal(str(row[3])),
                        close=Decimal(str(row[4])),
                        volume=Decimal(str(row[5])),
                        timestamp=row[0] / 1000.0,
                        is_closed=True,
                    ))

                last_ts = data[-1][0]
                if len(data) < limit or last_ts >= end_ms:
                    break
                cur = last_ts + 1
                await asyncio.sleep(0.15)

        # Deduplicazione e ordinamento
        seen: set = set()
        result: List[Candle] = []
        for c in sorted(candles, key=lambda x: x.timestamp):
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                result.append(c)
        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        trades = self._gw.trades
        if not trades:
            print(f"\nNessun trade eseguito su {self.symbol} {self.timeframe}.\n")
            return

        total_pnl = sum(t.pnl for t in trades)
        wins      = [t for t in trades if t.pnl > Decimal("0")]
        losses    = [t for t in trades if t.pnl <= Decimal("0")]
        win_rate  = len(wins) / len(trades) * 100
        avg_win   = sum(t.pnl for t in wins)   / Decimal(max(len(wins), 1))
        avg_loss  = sum(t.pnl for t in losses) / Decimal(max(len(losses), 1))

        gross_win  = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = float(gross_win / gross_loss) if gross_loss > 0 else 999.0

        # Max drawdown
        cum, peak, max_dd = Decimal("0"), Decimal("0"), Decimal("0")
        for t in trades:
            cum += t.pnl
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # Sharpe annualizzato (periodi per anno: 365*24*60/tf_minutes)
        tf_min = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                  "1h": 60, "4h": 240, "1d": 1440}.get(self.timeframe, 15)
        periods_per_year = 365 * 24 * 60 // tf_min
        pnl_list = [float(t.pnl) for t in trades]
        if len(pnl_list) > 1:
            mu    = sum(pnl_list) / len(pnl_list)
            sigma = math.sqrt(sum((r - mu) ** 2 for r in pnl_list) / len(pnl_list))
            sharpe = (mu / sigma * math.sqrt(periods_per_year)) if sigma > 0 else 0.0
        else:
            sharpe = 0.0

        by_reason: dict = {}
        for t in trades:
            by_reason[t.reason] = by_reason.get(t.reason, 0) + 1

        w = 52
        print("\n" + "=" * w)
        print(f"  BACKTEST — {self.symbol} {self.timeframe} | {self.days} giorni")
        print("=" * w)
        print(f"  Capitale iniziale : {float(self.capital):.2f} USDT")
        print(f"  PnL totale        : {float(total_pnl):+.4f} USDT")
        print(f"  Rendimento        : {float(total_pnl / self.capital * 100):+.2f}%")
        print(f"  Trade totali      : {len(trades)}")
        print(f"  Win rate          : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg win           : {float(avg_win):+.4f} USDT")
        print(f"  Avg loss          : {float(avg_loss):+.4f} USDT")
        print(f"  Profit factor     : {pf:.2f}")
        print(f"  Max drawdown      : -{float(max_dd):.4f} USDT")
        print(f"  Sharpe (annuo)    : {sharpe:.2f}")
        print("-" * w)
        for reason, count in sorted(by_reason.items()):
            print(f"  {reason:<22} : {count}")
        print("=" * w + "\n")

    def export_csv(self, path: str = "backtest_trades.csv") -> None:
        if not self._gw.trades:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "side", "entry_price", "exit_price",
                        "size", "pnl", "fee", "entry_time", "exit_time",
                        "duration_h", "reason"])
            for t in self._gw.trades:
                w.writerow([
                    t.symbol, t.side,
                    float(t.entry_price), float(t.exit_price),
                    float(t.size), float(t.pnl), float(t.fee),
                    t.entry_time, t.exit_time,
                    f"{t.duration_hours:.2f}", t.reason,
                ])
        logger.info("Trade esportati su: %s", path)


# ---------------------------------------------------------------------------
# CLI: python -m src.backtest [SYMBOL] [TIMEFRAME] [DAYS]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    symbol    = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15m"
    days      = int(sys.argv[3]) if len(sys.argv) > 3 else 180

    async def _main():
        engine = CandleBacktestEngine(symbol=symbol, timeframe=timeframe, days=days)
        await engine.run()
        engine.print_report()
        engine.export_csv(f"backtest_{symbol}_{timeframe}.csv")

    asyncio.run(_main())
