"""
CandleBacktestEngine — backtest futures-native su Bybit per Loky Bot.

Fix rispetto alla versione precedente:
  • Sorgente dati: Bybit V5 REST /v5/market/kline (era Binance)
  • Trade recording: intercetta save_trade() dal StateManager (era _close_position
    che non esiste) → registra correttamente ogni chiusura parziale e totale
  • Multi-timeframe: feed sia 15m (primary) sia 1h (confirmation) in ordine cronologico
  • Equity curve: calcola capital + cumulative PnL ad ogni step
  • Metriche: PnL, Sharpe, Calmar, Sortino, max drawdown %, win rate, profit factor,
    avg trade duration, avg R:R
  • Export: CSV (trade-by-trade) + JSON (report completo)

Utilizzo:
    engine = CandleBacktestEngine(symbol="BTCUSDT", timeframe="15m", days=180)
    await engine.run()
    engine.print_report()
    engine.export_csv("btc_backtest.csv")
    engine.export_json("btc_backtest.json")

CLI:
    python -m src.backtest BTCUSDT 15m 180
    python -m src.backtest BTCUSDT 15m 180 1000   # capitale iniziale
"""

import asyncio
import csv
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

import aiohttp

from src.bot import BotState, LokyBot
from src.config import BotSettings
from src.models import Candle, Side, Trade

logger = logging.getLogger(__name__)

# Bybit V5 REST endpoint
_BYBIT_REST = "https://api.bybit.com"
_BYBIT_KLINE_PATH = "/v5/market/kline"

# Mappa timeframe → Bybit interval string
_TF_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W",
}

# Minuti per timeframe (per calcoli annualizzazione)
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1w": 10080,
}


# ---------------------------------------------------------------------------
# Risultato singolo trade
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

    @property
    def is_win(self) -> bool:
        return self.pnl > Decimal("0")


# ---------------------------------------------------------------------------
# StateManager in-memory — registra i trade via save_trade()
# ---------------------------------------------------------------------------

class _InMemoryStateManager:
    """
    StateManager che registra i trade in memoria.
    save_trade() viene chiamato da bot.py ad ogni chiusura (parziale o totale).
    """

    def __init__(self) -> None:
        self._trades: List[Trade] = []
        # Contesto entry corrente, impostato dal backtest engine ad ogni apertura
        self._current_entry_price: Decimal = Decimal("0")
        self._current_entry_time: float = 0.0

    # Interface richiesta da LokyBot
    def load_state(self) -> None:
        return None

    async def update_snapshot(self, **kwargs) -> None:
        # Cattura entry_price dal bot state ad ogni snapshot update
        avg_entry = kwargs.get("avg_entry", None)
        if avg_entry is not None and avg_entry != Decimal("0"):
            self._current_entry_price = avg_entry

    async def save_trade(self, trade: Trade) -> None:
        # Annota entry info al momento del salvataggio
        trade._bt_entry_price = self._current_entry_price  # type: ignore[attr-defined]
        trade._bt_entry_time = self._current_entry_time     # type: ignore[attr-defined]
        self._trades.append(trade)

    async def auto_save_loop(self, interval: float = 30.0) -> None:
        pass

    @property
    def trades(self) -> List[Trade]:
        return list(self._trades)


# ---------------------------------------------------------------------------
# Backtest Gateway — no-op (paper mode gestito in bot.py)
# ---------------------------------------------------------------------------

class _BacktestGateway:
    """Gateway no-op: il paper trading è gestito interamente in bot.py."""

    def set_on_order_update_callback(self, cb) -> None:
        pass

    async def set_leverage(self, symbol, leverage) -> None:
        pass

    async def submit_market_order(self, symbol, side, size):
        return None

    async def submit_tp_sl(self, symbol, side, size, tp, sl) -> None:
        pass

    async def submit_order(self, order) -> None:
        pass

    async def cancel_order(self, order) -> None:
        pass

    async def cancel_all_orders(self) -> None:
        pass

    async def fetch_open_orders_count(self) -> int:
        return 0

    async def fetch_real_inventory(self, asset_id) -> Decimal:
        return Decimal("0")

    async def match_engine_tick(self) -> None:
        pass

    async def start_userstream(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Engine principale
# ---------------------------------------------------------------------------

class CandleBacktestEngine:
    """
    Backtest candle-based della strategia multi-engine Loky su dati Bybit storici.

    Args:
        symbol     — coppia Futures (es. "BTCUSDT")
        timeframe  — timeframe primario (es. "15m")
        days       — giorni di storico (default 180)
        capital    — USDT simulati (default 500)
        config     — BotSettings opzionale (usa config.yaml altrimenti)
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

        self._gw = _BacktestGateway()
        self._sm = _InMemoryStateManager()

        self._primary_candles: List[Candle] = []
        self._confirm_candles: List[Candle] = []
        self._equity_curve: List[float]     = []   # equity ad ogni candle primario

    # ------------------------------------------------------------------
    # Esecuzione principale
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info(
            "Backtest %s %s | %d giorni | capitale=%.0f USDT",
            self.symbol, self.timeframe, self.days, self.capital,
        )

        # Fetch candles per tutti e 3 i timeframe
        self._primary_candles = await self._fetch_candles(self.timeframe)
        confirm_tf = self.cfg.confirmation_timeframe
        macro_tf   = getattr(self.cfg, 'macro_timeframe', '4h')
        self._confirm_candles = await self._fetch_candles(confirm_tf)
        self._macro_candles: List[Candle] = await self._fetch_candles(macro_tf)

        if len(self._primary_candles) < 60:
            logger.error(
                "Dati insufficienti: solo %d candele %s.",
                len(self._primary_candles), self.timeframe,
            )
            return

        logger.info(
            "Candele: %d × %s + %d × %s + %d × %s",
            len(self._primary_candles), self.timeframe,
            len(self._confirm_candles), confirm_tf,
            len(self._macro_candles), macro_tf,
        )

        # Crea bot in modalità paper (live_trading_enabled = False)
        bot = LokyBot(
            symbol=self.symbol,
            config=self.cfg,
            execution_gw=self._gw,
            state_manager=self._sm,
            capital=self.capital,
        )

        # Merge e ordina tutti i candles cronologicamente (primary + confirmation + macro)
        all_candles = sorted(
            self._primary_candles + self._confirm_candles + self._macro_candles,
            key=lambda c: c.timestamp,
        )

        # Feed al bot
        running_pnl = Decimal("0")
        prev_state = BotState.FLAT
        for candle in all_candles:
            await bot.on_candle(candle)
            # Traccia entry_time nella state manager quando il bot apre una posizione
            if prev_state == BotState.FLAT and bot._state != BotState.FLAT:
                self._sm._current_entry_time = bot._entry_time
                self._sm._current_entry_price = bot._entry_price
            prev_state = bot._state
            # Traccia equity solo sui candle primari
            if candle.timeframe == self.timeframe:
                self._equity_curve.append(
                    float(self.capital + bot.realized_pnl)
                )

        # Forza chiusura posizione aperta a fine dati
        if bot._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            last = self._primary_candles[-1] if self._primary_candles else None
            if last:
                await bot._close_remaining(last.close, "end_of_data")

        logger.info(
            "Backtest completato: %d trade eseguiti. PnL finale: %.4f USDT",
            bot.total_trades, bot.realized_pnl,
        )

    # ------------------------------------------------------------------
    # Fetch candles da Bybit V5
    # ------------------------------------------------------------------

    async def _fetch_candles(self, timeframe: str) -> List[Candle]:
        """
        Scarica dati storici da Bybit V5 /v5/market/kline con cache SQLite locale.
        Prima run: scarica da API e salva in cache.
        Run successive: carica da cache se dati recenti (< 24h).
        """
        # Prova cache locale
        cached = self._load_from_cache(timeframe)
        if cached:
            logger.info("Cache hit: %s %s → %d candele da cache locale", self.symbol, timeframe, len(cached))
            return cached

        interval = _TF_MAP.get(timeframe, "15")
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - self.days * 86_400 * 1000
        limit    = 1000
        candles: List[Candle] = []

        async with aiohttp.ClientSession() as session:
            cur_end = end_ms
            while True:
                params = {
                    "category": "linear",
                    "symbol":   self.symbol,
                    "interval": interval,
                    "start":    start_ms,
                    "end":      cur_end,
                    "limit":    limit,
                }
                try:
                    async with session.get(
                        f"{_BYBIT_REST}{_BYBIT_KLINE_PATH}",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                except Exception as e:
                    logger.error("Fetch candele %s %s fallito: %s", self.symbol, timeframe, e)
                    break

                if data.get("retCode") != 0:
                    logger.error(
                        "Bybit errore %s: %s", data.get("retCode"), data.get("retMsg")
                    )
                    break

                rows = data.get("result", {}).get("list", [])
                if not rows:
                    break

                # Bybit: [timestamp_ms, open, high, low, close, volume, turnover]
                # Ordine: dal più recente al più vecchio → invertiamo
                for row in reversed(rows):
                    ts_ms = int(row[0])
                    if ts_ms < start_ms:
                        continue
                    candles.append(Candle(
                        symbol=self.symbol,
                        timeframe=timeframe,
                        open=Decimal(str(row[1])),
                        high=Decimal(str(row[2])),
                        low=Decimal(str(row[3])),
                        close=Decimal(str(row[4])),
                        volume=Decimal(str(row[5])),
                        timestamp=ts_ms / 1000.0,
                        is_closed=True,
                    ))

                # Se abbiamo preso meno di limit, siamo arrivati all'inizio
                if len(rows) < limit:
                    break

                # Prossima iterazione: dal più vecchio timestamp trovato - 1ms
                oldest_ts = int(rows[-1][0])
                if oldest_ts <= start_ms:
                    break
                cur_end = oldest_ts - 1
                await asyncio.sleep(0.12)   # rispetta rate limit Bybit (non autenticato: 10 req/s)

        # Deduplica e ordina
        seen: set = set()
        result: List[Candle] = []
        for c in sorted(candles, key=lambda x: x.timestamp):
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                result.append(c)

        logger.info(
            "Bybit %s %s: %d candele scaricate.", self.symbol, timeframe, len(result)
        )

        # Salva in cache locale
        if result:
            self._save_to_cache(timeframe, result)

        return result

    # ------------------------------------------------------------------
    # Cache SQLite locale per dati storici
    # ------------------------------------------------------------------

    def _cache_db_path(self) -> str:
        return f"data/backtest_cache_{self.symbol}.db"

    def _init_cache_db(self) -> sqlite3.Connection:
        import os
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(self._cache_db_path())
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candle_cache (
                symbol TEXT, timeframe TEXT, timestamp REAL,
                open TEXT, high TEXT, low TEXT, close TEXT, volume TEXT,
                fetched_at REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        conn.commit()
        return conn

    def _load_from_cache(self, timeframe: str) -> Optional[List[Candle]]:
        """Carica candele dalla cache se presenti e recenti (< 24h)."""
        import os
        if not os.path.exists(self._cache_db_path()):
            return None
        try:
            conn = self._init_cache_db()
            end_ts = time.time()
            start_ts = end_ts - self.days * 86_400
            # Controlla se la cache è recente
            row = conn.execute(
                "SELECT MAX(fetched_at) FROM candle_cache WHERE symbol=? AND timeframe=?",
                (self.symbol, timeframe),
            ).fetchone()
            if not row or row[0] is None or (time.time() - row[0]) > 86_400:
                conn.close()
                return None  # cache scaduta o vuota
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close, volume FROM candle_cache "
                "WHERE symbol=? AND timeframe=? AND timestamp>=? AND timestamp<=? "
                "ORDER BY timestamp",
                (self.symbol, timeframe, start_ts, end_ts),
            ).fetchall()
            conn.close()
            if len(rows) < 60:
                return None  # dati insufficienti
            return [
                Candle(
                    symbol=self.symbol, timeframe=timeframe,
                    open=Decimal(r[1]), high=Decimal(r[2]),
                    low=Decimal(r[3]), close=Decimal(r[4]),
                    volume=Decimal(r[5]), timestamp=r[0], is_closed=True,
                )
                for r in rows
            ]
        except Exception as e:
            logger.warning("Cache load error: %s", e)
            return None

    def _save_to_cache(self, timeframe: str, candles: List[Candle]) -> None:
        """Salva candele nella cache SQLite locale."""
        try:
            conn = self._init_cache_db()
            now = time.time()
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO candle_cache "
                    "(symbol, timeframe, timestamp, open, high, low, close, volume, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (c.symbol, timeframe, c.timestamp,
                         str(c.open), str(c.high), str(c.low), str(c.close), str(c.volume), now)
                        for c in candles
                    ],
                )
            conn.close()
            logger.info("Cache saved: %s %s → %d candele", self.symbol, timeframe, len(candles))
        except Exception as e:
            logger.warning("Cache save error: %s", e)

    # ------------------------------------------------------------------
    # Aggregazione trade da StateManager
    # ------------------------------------------------------------------

    def _build_backtest_trades(self) -> List[BacktestTrade]:
        """
        Converte i Trade registrati dallo StateManager in BacktestTrade.
        Ogni Trade dal StateManager rappresenta una chiusura (parziale o totale).
        """
        result = []
        for t in self._sm.trades:
            entry_price = getattr(t, '_bt_entry_price', Decimal("0"))
            entry_time  = getattr(t, '_bt_entry_time', t.timestamp)
            bt = BacktestTrade(
                symbol=t.symbol,
                side=t.side.name,
                entry_price=entry_price,
                exit_price=t.price,
                size=t.size,
                pnl=t.realized_pnl,
                fee=t.commission,
                entry_time=entry_time,
                exit_time=t.timestamp,
                reason=t.order_id,
            )
            result.append(bt)
        return result

    # ------------------------------------------------------------------
    # Metriche avanzate
    # ------------------------------------------------------------------

    def _compute_metrics(self, trades: List[BacktestTrade]) -> dict:
        if not trades:
            return {}

        total_pnl  = sum(t.pnl for t in trades)
        wins       = [t for t in trades if t.is_win]
        losses     = [t for t in trades if not t.is_win]
        win_rate   = len(wins) / len(trades)
        avg_win    = sum(t.pnl for t in wins) / Decimal(max(len(wins), 1))
        avg_loss   = sum(t.pnl for t in losses) / Decimal(max(len(losses), 1))
        gross_win  = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else 999.0

        # Max drawdown assoluto e percentuale
        cum, peak, max_dd_abs = Decimal("0"), Decimal("0"), Decimal("0")
        for t in trades:
            cum += t.pnl
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd_abs:
                max_dd_abs = dd
        max_dd_pct = float(max_dd_abs / self.capital * 100)

        # Sharpe annualizzato (su base PnL per trade come "return")
        tf_min = _TF_MINUTES.get(self.timeframe, 15)
        periods_per_year = 365 * 24 * 60 // tf_min
        pnl_list = [float(t.pnl) for t in trades]
        mu    = sum(pnl_list) / len(pnl_list)
        sigma = math.sqrt(sum((r - mu) ** 2 for r in pnl_list) / len(pnl_list)) if len(pnl_list) > 1 else 0
        sharpe = (mu / sigma * math.sqrt(periods_per_year)) if sigma > 0 else 0.0

        # Sortino (solo downside deviation)
        neg_ret = [r for r in pnl_list if r < 0]
        if neg_ret:
            down_dev = math.sqrt(sum(r**2 for r in neg_ret) / len(neg_ret))
            sortino  = (mu / down_dev * math.sqrt(periods_per_year)) if down_dev > 0 else 0.0
        else:
            sortino = 0.0

        # Calmar = return annualizzato / max drawdown %
        if self._primary_candles:
            days_actual = (
                self._primary_candles[-1].timestamp - self._primary_candles[0].timestamp
            ) / 86_400
        else:
            days_actual = self.days
        annual_return = float(total_pnl / self.capital) * (365 / max(days_actual, 1))
        calmar = annual_return / (max_dd_pct / 100) if max_dd_pct > 0 else 0.0

        # Avg durata trade (approssimazione)
        avg_duration_h = sum(t.duration_hours for t in trades) / len(trades)

        # Fee totali
        total_fee = sum(t.fee for t in trades)

        # Breakeven by reason
        by_reason: dict = {}
        for t in trades:
            key = t.reason if t.reason else "unknown"
            if key.startswith("partial_"):
                key = "partial_tp"
            elif key.startswith("close_"):
                key = "full_close"
            by_reason[key] = by_reason.get(key, 0) + 1

        return {
            "total_pnl":     float(total_pnl),
            "return_pct":    float(total_pnl / self.capital * 100),
            "n_trades":      len(trades),
            "win_rate":      win_rate * 100,
            "n_wins":        len(wins),
            "n_losses":      len(losses),
            "avg_win":       float(avg_win),
            "avg_loss":      float(avg_loss),
            "profit_factor": profit_factor,
            "max_dd_abs":    float(max_dd_abs),
            "max_dd_pct":    max_dd_pct,
            "sharpe":        sharpe,
            "sortino":       sortino,
            "calmar":        calmar,
            "avg_duration_h": avg_duration_h,
            "total_fee":     float(total_fee),
            "annual_return": annual_return * 100,
            "by_reason":     by_reason,
        }

    # ------------------------------------------------------------------
    # Report testuale
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        trades = self._build_backtest_trades()

        if not trades:
            print(f"\nNessun trade eseguito su {self.symbol} {self.timeframe}.\n")
            return

        m = self._compute_metrics(trades)
        w = 58

        print("\n" + "=" * w)
        print(f"  BACKTEST — {self.symbol} {self.timeframe} | {self.days}gg")
        print("=" * w)
        print(f"  Capitale iniziale : {float(self.capital):.2f} USDT")
        print(f"  PnL totale        : {m['total_pnl']:+.4f} USDT")
        print(f"  Rendimento        : {m['return_pct']:+.2f}%")
        print(f"  Rendimento annuo  : {m['annual_return']:+.1f}%")
        print("-" * w)
        print(f"  Trade totali      : {m['n_trades']}")
        print(f"  Win rate          : {m['win_rate']:.1f}%  ({m['n_wins']}W / {m['n_losses']}L)")
        print(f"  Avg win           : {m['avg_win']:+.4f} USDT")
        print(f"  Avg loss          : {m['avg_loss']:+.4f} USDT")
        print(f"  Profit factor     : {m['profit_factor']:.2f}")
        print(f"  Fee totali        : {m['total_fee']:.4f} USDT")
        print(f"  Durata media      : {m['avg_duration_h']:.1f}h")
        print("-" * w)
        print(f"  Max Drawdown      : -{m['max_dd_abs']:.4f} USDT ({m['max_dd_pct']:.1f}%)")
        print(f"  Sharpe (annuo)    : {m['sharpe']:.2f}")
        print(f"  Sortino (annuo)   : {m['sortino']:.2f}")
        print(f"  Calmar ratio      : {m['calmar']:.2f}")
        print("-" * w)
        for reason, count in sorted(m["by_reason"].items()):
            print(f"  {reason:<26} : {count}")
        print("=" * w + "\n")

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------

    def export_csv(self, path: str = "backtest_trades.csv") -> None:
        trades = self._build_backtest_trades()
        if not trades:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol", "side", "exit_price", "size", "pnl", "fee",
                "exit_time", "duration_h", "reason",
            ])
            for t in trades:
                writer.writerow([
                    t.symbol, t.side,
                    float(t.exit_price), float(t.size),
                    float(t.pnl), float(t.fee),
                    t.exit_time,
                    f"{t.duration_hours:.2f}", t.reason,
                ])
        logger.info("Trade esportati su: %s", path)

    # ------------------------------------------------------------------
    # Export JSON (report completo)
    # ------------------------------------------------------------------

    def export_json(self, path: str = "backtest_report.json") -> None:
        trades = self._build_backtest_trades()
        metrics = self._compute_metrics(trades)
        report = {
            "symbol":    self.symbol,
            "timeframe": self.timeframe,
            "days":      self.days,
            "capital":   float(self.capital),
            "metrics":   metrics,
            "equity_curve": self._equity_curve,
            "trades": [
                {
                    "symbol":      t.symbol,
                    "side":        t.side,
                    "exit_price":  float(t.exit_price),
                    "size":        float(t.size),
                    "pnl":         float(t.pnl),
                    "fee":         float(t.fee),
                    "exit_time":   t.exit_time,
                    "duration_h":  t.duration_hours,
                    "reason":      t.reason,
                }
                for t in trades
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Report JSON esportato su: %s", path)


# ---------------------------------------------------------------------------
# CLI: python -m src.backtest [SYMBOL] [TIMEFRAME] [DAYS] [CAPITAL]
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
    days      = int(sys.argv[3])     if len(sys.argv) > 3 else 180
    capital   = Decimal(sys.argv[4]) if len(sys.argv) > 4 else Decimal("500")

    async def _main():
        engine = CandleBacktestEngine(
            symbol=symbol, timeframe=timeframe, days=days, capital=capital,
        )
        await engine.run()
        engine.print_report()

        base = f"backtest_{symbol}_{timeframe}"
        engine.export_csv(f"{base}.csv")
        engine.export_json(f"{base}.json")
        print(f"Output: {base}.csv + {base}.json")

    asyncio.run(_main())
