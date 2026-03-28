"""
StateManager — persistenza SQLite per bot direzionale.

Schema:
  • state  — snapshot periodico (inventory, PnL, entry price, ecc.)
  • trades — audit trail di ogni trade chiuso

Cambiamenti rispetto alla versione market-maker:
  • load_state() ritorna un dict invece di scrivere direttamente sul bot
  • update_snapshot() sostituisce il loop di auto-save accoppiato
  • save_trade() è ora async
  • auto_save_loop() accetta un callable per recuperare lo snapshot corrente
"""

import asyncio
import logging
import os
import sqlite3
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from src.models import Trade

logger = logging.getLogger("StateManager")


class StateManager:
    def __init__(self, db_path: str = "data/state.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._db_lock = asyncio.Lock()   # Protegge accessi concorrenti da task asyncio
        self._init_db()

        # Snapshot corrente (aggiornato via update_snapshot)
        self._snapshot: dict = {
            "net_inventory": Decimal("0"),
            "pnl":           Decimal("0"),
            "avg_entry":     Decimal("0"),
            "quotes_sent":   0,
            "fills_total":   0,
        }

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS state (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                net_inventory TEXT NOT NULL,
                pnl           TEXT NOT NULL,
                avg_entry     TEXT NOT NULL DEFAULT '0',
                quotes_sent   INTEGER NOT NULL DEFAULT 0,
                fills_total   INTEGER NOT NULL DEFAULT 0,
                timestamp     REAL NOT NULL
            )
        ''')
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT NOT NULL,
                side             TEXT NOT NULL,
                size             TEXT NOT NULL,
                price            TEXT NOT NULL,
                commission       TEXT NOT NULL DEFAULT '0',
                commission_asset TEXT NOT NULL DEFAULT 'USDT',
                order_id         TEXT,
                realized_pnl     TEXT NOT NULL DEFAULT '0',
                timestamp        REAL NOT NULL
            )
        ''')
        self._conn.commit()

    # ------------------------------------------------------------------
    # Caricamento stato
    # ------------------------------------------------------------------
    def load_state(self) -> Optional[dict]:
        """
        Ritorna l'ultimo snapshot salvato come dict, o None se nessuno esiste.
        Chiavi: net_inventory, pnl, avg_entry, quotes_sent, fills_total
        """
        try:
            row = self._conn.execute(
                'SELECT net_inventory, pnl, avg_entry, quotes_sent, fills_total '
                'FROM state ORDER BY timestamp DESC LIMIT 1'
            ).fetchone()
            if row:
                snap = {
                    "net_inventory": Decimal(str(row[0])),
                    "pnl":           Decimal(str(row[1])),
                    "avg_entry":     Decimal(str(row[2])),
                    "quotes_sent":   int(row[3]),
                    "fills_total":   int(row[4]),
                }
                self._snapshot = snap
                logger.info(
                    "Stato recuperato: inv=%s pnl=%.4f USDT",
                    snap["net_inventory"], float(snap["pnl"]),
                )
                return snap
            logger.info("Nessuno stato precedente. First run.")
            return None
        except Exception as e:
            logger.error("Errore caricamento stato: %s", e)
            return None

    # ------------------------------------------------------------------
    # Aggiornamento snapshot (chiamato dal bot)
    # ------------------------------------------------------------------
    def update_snapshot(
        self,
        net_inventory: Decimal,
        pnl: Decimal,
        avg_entry: Decimal,
        quotes_sent: int,
        fills_total: int,
    ) -> None:
        """Aggiorna lo snapshot in memoria e lo persiste immediatamente."""
        self._snapshot = {
            "net_inventory": net_inventory,
            "pnl":           pnl,
            "avg_entry":     avg_entry,
            "quotes_sent":   quotes_sent,
            "fills_total":   fills_total,
        }
        self._write_snapshot()

    def _write_snapshot(self) -> None:
        try:
            s = self._snapshot
            with self._conn:
                self._conn.execute(
                    '''INSERT INTO state
                       (net_inventory, pnl, avg_entry, quotes_sent, fills_total, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (
                        str(s["net_inventory"]),
                        str(s["pnl"]),
                        str(s["avg_entry"]),
                        s["quotes_sent"],
                        s["fills_total"],
                        time.time(),
                    ),
                )
        except Exception as e:
            logger.error("Errore scrittura snapshot: %s", e)

    # ------------------------------------------------------------------
    # Salvataggio trade (audit trail)
    # ------------------------------------------------------------------
    async def save_trade(self, trade: "Trade") -> None:
        async with self._db_lock:
            try:
                with self._conn:
                    self._conn.execute(
                        '''INSERT INTO trades
                           (symbol, side, size, price, commission, commission_asset,
                            order_id, realized_pnl, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            trade.symbol,
                            trade.side.name,
                            str(trade.size),
                            str(trade.price),
                            str(trade.commission),
                            trade.commission_asset,
                            str(trade.order_id),
                            str(trade.realized_pnl),
                            trade.timestamp,
                        ),
                    )
            except Exception as e:
                logger.error("Errore salvataggio trade: %s", e)

    # ------------------------------------------------------------------
    # Auto-save loop periodico
    # ------------------------------------------------------------------
    async def auto_save_loop(self, interval: float = 30.0) -> None:
        """Persiste lo snapshot corrente ogni `interval` secondi."""
        logger.info("State Manager attivo — auto-save ogni %.0fs", interval)
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    self._write_snapshot()
                except Exception as e:
                    logger.error("Auto-save error: %s", e)
        except asyncio.CancelledError:
            # Flush finale prima di spegnersi — non perdere fino a 30s di stato
            self._write_snapshot()
            logger.info("State Manager: flush finale completato.")
