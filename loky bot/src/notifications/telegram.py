"""
TelegramNotifier — notifiche asincrone via Telegram Bot API.

Invia messaggi su:
  • Apertura posizione (symbol, direzione, entry, TP, SL, size, regime)
  • Chiusura posizione (PnL, motivo, durata)
  • Daily stop account raggiunto
  • Sommario giornaliero automatico (ogni mezzanotte UTC)
  • Alert errori API (>3 errori in 5 min)
  • Alert candle lag (data stream rallentato)
  • Comandi: /status, /pause, /resume

Configurazione .env:
    TELEGRAM_TOKEN=<bot_token>
    TELEGRAM_CHAT_ID=<chat_id>

Se le variabili non sono configurate, tutte le notifiche vengono silenziate (no-op).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from decimal import Decimal
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger("TelegramNotifier")

_API_BASE   = "https://api.telegram.org/bot{token}/sendMessage"
_UPDATE_URL = "https://api.telegram.org/bot{token}/getUpdates"


class TelegramNotifier:
    """
    Invia notifiche al bot Telegram in modo fire-and-forget.
    Supporta polling dei comandi (/status, /pause, /resume).

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

        # Stato per comandi remoti
        self._paused: bool = False
        self._last_update_id: int = 0

        # Callback per status (iniettata dall'orchestratore)
        self._status_callback: Optional[Callable] = None

        # Contatore errori API per alert automatici
        self._api_errors: list[float] = []   # timestamp degli errori recenti

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def set_status_callback(self, callback: Callable) -> None:
        """Inietta callback per risposta a /status. Deve ritornare una stringa."""
        self._status_callback = callback

    @property
    def is_paused(self) -> bool:
        """True se il trading è in pausa per comando remoto."""
        return self._paused

    async def trade_opened(
        self,
        symbol: str,
        side: str,           # "LONG" | "SHORT"
        entry: Decimal,
        tp: Decimal,
        sl: Decimal,
        size: Decimal,
        capital: Decimal,
        regime: str = "",
        strategy: str = "",
    ) -> None:
        icon = "🟢" if side == "LONG" else "🔴"
        risk_pct = abs(entry - sl) / entry * 100
        extra = ""
        if regime:
            extra += f"\nRegime : <code>{regime}</code>"
        if strategy:
            extra += f"\nEngine : <code>{strategy}</code>"
        text = (
            f"{icon} <b>POSIZIONE APERTA</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : <b>{side}</b>\n"
            f"Entry  : <code>{float(entry):.4f}</code>\n"
            f"TP1    : <code>{float(tp):.4f}</code>\n"
            f"SL     : <code>{float(sl):.4f}</code>\n"
            f"Size   : <code>{float(size):.4f}</code>\n"
            f"Rischio: <code>~{float(risk_pct):.2f}%</code>"
            f"{extra}"
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
        duration_h: float = 0.0,
    ) -> None:
        icon = "✅" if pnl >= Decimal("0") else "❌"
        sign = "+" if pnl >= Decimal("0") else ""
        dur_str = f" | ⏱ {duration_h:.1f}h" if duration_h > 0 else ""
        text = (
            f"{icon} <b>POSIZIONE CHIUSA</b>\n"
            f"Symbol : <code>{symbol}</code>\n"
            f"Side   : <b>{side}</b>\n"
            f"Entry  : <code>{float(entry):.4f}</code>\n"
            f"Exit   : <code>{float(exit_price):.4f}</code>\n"
            f"PnL    : <b>{sign}{float(pnl):.4f} USDT</b>{dur_str}\n"
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

    async def api_error_alert(self, error_msg: str) -> None:
        """Alert automatico se si accumulano troppi errori API in poco tempo."""
        now = time.time()
        self._api_errors = [t for t in self._api_errors if now - t < 300]  # ultimi 5 min
        self._api_errors.append(now)
        if len(self._api_errors) >= 3:
            text = (
                f"⚠️ <b>ERRORI API FREQUENTI</b>\n"
                f"Errori ultimi 5min: <code>{len(self._api_errors)}</code>\n"
                f"Ultimo: <code>{error_msg[:100]}</code>"
            )
            await self._send(text)
            self._api_errors.clear()  # reset dopo alert

    async def candle_lag_alert(self, symbol: str, lag_seconds: float) -> None:
        """Alert se il feed dati rallenta oltre la soglia."""
        text = (
            f"⏱ <b>CANDLE LAG RILEVATO</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Ritardo: <b>{lag_seconds:.0f}s</b> (soglia: 120s)\n"
            f"Possibile interruzione del feed dati."
        )
        await self._send(text)

    async def start_command_polling(self) -> None:
        """
        Task continuo: polling dei messaggi Telegram per comandi /status /pause /resume.
        Da avviare come asyncio task nell'orchestratore.
        """
        if not self._enabled:
            return
        logger.info("Telegram command polling avviato.")
        while True:
            try:
                await asyncio.sleep(5)
                await self._poll_commands()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Telegram polling errore: %s", e)

    async def _poll_commands(self) -> None:
        """Legge i nuovi messaggi Telegram e gestisce i comandi."""
        url = _UPDATE_URL.format(token=self._token)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params={"offset": self._last_update_id + 1, "timeout": 1},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
        except Exception:
            return

        for update in data.get("result", []):
            uid = update.get("update_id", 0)
            if uid > self._last_update_id:
                self._last_update_id = uid
            msg = update.get("message", {})

            # Autorizzazione: accetta comandi solo dalla chat configurata
            msg_chat_id = str(msg.get("chat", {}).get("id", ""))
            if msg_chat_id != self._chat_id:
                logger.debug("Comando Telegram ignorato da chat non autorizzata: %s", msg_chat_id)
                continue

            text = msg.get("text", "").strip().lower()
            if text == "/pause":
                self._paused = True
                await self._send("⏸ Trading in <b>PAUSA</b>. Usa /resume per riprendere.")
            elif text == "/resume":
                self._paused = False
                await self._send("▶️ Trading <b>RIPRESO</b>.")
            elif text == "/status":
                if self._status_callback:
                    try:
                        status_text = await asyncio.wait_for(
                            self._status_callback(), timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        status_text = "Status query timeout."
                    except Exception:
                        status_text = "Errore recupero status."
                else:
                    status_text = "Status callback non configurato."
                await self._send(f"📊 <b>STATUS</b>\n{status_text}")

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
            "chat_id":                  self._chat_id,
            "text":                     text,
            "parse_mode":               "HTML",
            "disable_web_page_preview": True,
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
