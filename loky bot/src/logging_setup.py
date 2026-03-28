"""
logging_setup — configurazione log strutturato JSON per Loky Bot.

Funzionalità:
  • Output JSON su file (con rotation 10MB × 5 file)
  • Output human-readable su console (durante sviluppo)
  • Campi strutturati: timestamp ISO, level, module, message
  • Facile da ingestire con Grafana Loki, ELK, o qualsiasi log aggregator

Utilizzo:
    from src.logging_setup import setup_logging
    setup_logging(log_file="loky.log", level=logging.INFO, json_console=False)
"""

import json
import logging
import logging.handlers
import sys
import time
from typing import Optional


class JsonFormatter(logging.Formatter):
    """
    Formatter che serializza ogni log record come JSON su una riga.

    Esempio output:
        {"ts": "2026-03-27T10:00:00.123Z", "level": "INFO", "logger": "bot",
         "msg": "POSIZIONE APERTA", "symbol": "BTCUSDT"}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{record.msecs:03.0f}Z",
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Aggiungi eventuali campi extra (es. symbol, side, ecc.)
        for key in ("symbol", "side", "engine", "score", "pnl", "reason"):
            if hasattr(record, key):
                log_dict[key] = getattr(record, key)

        # Aggiungi exception info se presente
        if record.exc_info:
            log_dict["exc"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Formatter human-readable per la console durante sviluppo."""

    _LEVEL_COLORS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[41m",    # red background
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelname, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        name  = record.name[:20].ljust(20)
        msg   = record.getMessage()
        line  = f"{ts} | {color}{record.levelname:<8}{self._RESET} | {name} | {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(
    log_file: Optional[str] = "loky.log",
    level: int = logging.INFO,
    json_console: bool = False,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configura il logging dell'applicazione.

    Args:
        log_file     — percorso del file di log JSON (None = solo console)
        level        — livello minimo di log
        json_console — True = output JSON anche su console (per produzione/container)
        max_bytes    — dimensione massima file prima della rotazione
        backup_count — numero di file di backup da mantenere
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Rimuovi handler esistenti
    root.handlers.clear()

    # --- Handler console ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    if json_console:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(HumanFormatter())
    root.addHandler(console)

    # --- Handler file JSON (con rotation) ---
    if log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
        except Exception as e:
            logging.warning("Impossibile aprire log file %s: %s", log_file, e)

    # Silenzia librerie verbose
    for noisy in ("websockets", "aiohttp", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(
        "Logging configurato: level=%s, file=%s, json_console=%s",
        logging.getLevelName(level), log_file, json_console,
    )
