"""
TelegramNotifier — notifiche asincrone via Telegram Bot API.

Invia messaggi su:
  • Apertura posizione (symbol, direzione, entry, TP, SL, size)
  • Chiusura posizione (PnL, motivo)
  • Daily stop account raggiunto
  • Sommario giornaliero (opzionale, su richiesta)

Configurazione .env:
    TELEGRAM_TOKEN=<bot_token>
    TELEGRAM_CHAT_ID=<chat_id>

Se le variabili non sono configurate, tutte le notifiche vengono silenziate (no-op).
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Optional

import aiohttp

logger = logging.getLogger("TelegramNotifier")

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Invia notifiche al bot Telegram in modo fire-and-forget.

    Se TELEGRAM_TOKEN o TELEGRAM_CHAT_ID non sono impostati,
    tutte le chiamate sono no-op silenzioso.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token   = token   or os.getenv("TELEGRAM_TOKEN",   "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)
        if not self._enabled:
            logger.info("TelegramNotifier disabilitato (TELEGRAM_TOKEN/CHAT_ID mancanti).")

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    async def trade_opened(
        self,
        symbol: str,
        side: str,           # "LONG" | "SHORT"
        entry: Decimal,
        tp: Decimal,
        sl: Decimal,
        size: Decimal,
        capital: Decimal,
    ) -> None:
        icon = "🟢" if side == "LONG" else "🔴"
        risk_pct = abs(entry - sl) / entry * 100
        text = (
            f"{icon} <b>POSIZIONE APERTA</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : <b>{side}</b>\n"
            f"Entry  : <code>{float(entry):.4f}</code>\n"
            f"TP     : <code>{float(tp):.4f}</code>\n"
            f"SL     : <code>{float(sl):.4f}</code>\n"
            f"Size   : <code>{float(size):.4f}</code>\n"
            f"Rischio: <code>~{float(risk_pct):.2f}%</code>"
        )
        await self._send(text)

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        entry: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        reason: str,
    ) -> None:
        icon = "✅" if pnl >= Decimal("0") else "❌"
        sign = "+" if pnl >= Decimal("0") else ""
        text = (
            f"{icon} <b>POSIZIONE CHIUSA</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : <b>{side}</b>\n"
            f"Entry  : <code>{float(entry):.4f}</code>\n"
            f"Exit   : <code>{float(exit_price):.4f}</code>\n"
            f"PnL    : <b>{sign}{float(pnl):.4f} USDT</b>\n"
            f"Motivo : <code>{reason}</code>"
        )
        await self._send(text)

    async def daily_stop_triggered(
        self,
        daily_pnl: Decimal,
        limit: Decimal,
    ) -> None:
        text = (
            f"🚨 <b>ACCOUNT DAILY STOP</b>\n"
            f"PnL giornaliero : <b>{float(daily_pnl):+.4f} USDT</b>\n"
            f"Limite          : <code>{float(limit):.4f} USDT</code>\n"
            f"Tutti i bot sospesi fino a mezzanotte UTC."
        )
        await self._send(text)

    async def daily_summary(
        self,
        date_str: str,
        total_pnl: Decimal,
        trades: int,
        wins: int,
        losses: int,
    ) -> None:
        win_rate = (wins / trades * 100) if trades else 0.0
        sign     = "+" if total_pnl >= Decimal("0") else ""
        icon     = "📈" if total_pnl >= Decimal("0") else "📉"
        text = (
            f"{icon} <b>SOMMARIO GIORNALIERO — {date_str}</b>\n"
            f"PnL     : <b>{sign}{float(total_pnl):.4f} USDT</b>\n"
            f"Trade   : <code>{trades}</code>  "
            f"({wins}W / {losses}L — {win_rate:.0f}% WR)"
        )
        await self._send(text)

    async def info(self, message: str) -> None:
        """Messaggio libero (es. avvio/shutdown orchestratore)."""
        await self._send(f"ℹ️ {message}")

    # ------------------------------------------------------------------
    # Invio HTTP (fire-and-forget, non propaga eccezioni)
    # ------------------------------------------------------------------

    async def _send(self, text: str) -> None:
        if not self._enabled:
            return
        url     = _API_BASE.format(token=self._token)
        payload = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram API error %d: %s", resp.status, body[:200])
        except Exception as exc:
            logger.warning("Telegram send fallito: %s", exc)
