"""
MarketSentimentEngine — segnali basati su Open Interest e Long/Short Ratio.

Fonti dati Bybit V5:
  • Open Interest (OI): /v5/market/open-interest
    → Delta OI positivo + prezzo su = LONG confirmation
    → Delta OI positivo + prezzo giù = SHORT confirmation
    → Delta OI negativo = posizioni si chiudono → segnale debole

  • Long/Short Ratio (retail sentiment): /v5/market/account-ratio
    → >70% long → contrarian SHORT bias (leva retail eccessiva = squeeze imminente)
    → <30% long → contrarian LONG bias

Il motore non genera segnali standalone ma produce:
  1. Un Signal con score 55-75 se OI + L/S concordano con la direzione
  2. Un bonus/malus score usato dal SignalAggregator come correttore
  3. Un block flag (is_sentiment_negative) che il bot può usare per filtrare entry

Frequenza di polling: ogni 8 candle (chiamata REST, non blocca il loop).
Cache con TTL di 5 minuti per evitare troppi request.

Utilizzo nel bot:
    sentiment = MarketSentimentEngine(config, testnet=False)
    result = await sentiment.analyze(symbol, candle)
    if result.block_long:
        # Non aprire LONG
    score_adj = result.score_adjustment  # +/-10 da aggiungere allo score del segnale
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import aiohttp

from src.models import Candle, Signal, SignalType

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")

# Bybit V5 endpoints
_BASE_URL         = "https://api.bybit.com"
_BASE_TESTNET_URL = "https://api-testnet.bybit.com"
_OI_PATH          = "/v5/market/open-interest"
_LS_PATH          = "/v5/market/account-ratio"

# Soglie sentiment
_LS_EXTREME_LONG  = Decimal("0.70")   # >70% retail long → contrarian short bias
_LS_EXTREME_SHORT = Decimal("0.30")   # <30% retail long → contrarian long bias
_OI_CHANGE_THRESH = Decimal("0.005")  # 0.5% variazione OI minima per rilevamento

# TTL cache in secondi
_CACHE_TTL = 300   # 5 minuti


@dataclass
class SentimentResult:
    """Risultato dell'analisi di sentiment per un symbol."""
    symbol: str
    oi_current: Decimal          # Open Interest corrente (USD)
    oi_prev: Decimal             # Open Interest precedente
    oi_delta_pct: Decimal        # variazione % OI
    ls_ratio: Decimal            # Long/Short ratio (es. 0.55 = 55% long)
    block_long: bool             # True se il sentiment blocca i LONG
    block_short: bool            # True se il sentiment blocca gli SHORT
    score_adjustment: int        # Bonus/malus da aggiungere al signal score (-15 a +15)
    timestamp: float             # Unix timestamp dell'analisi

    @property
    def is_stale(self) -> bool:
        """True se il risultato supera il TTL della cache."""
        return time.time() - self.timestamp > _CACHE_TTL

    def summary(self) -> str:
        direction = "↑" if self.oi_delta_pct > _ZERO else "↓"
        ls_pct    = float(self.ls_ratio * 100)
        return (
            f"OI={direction}{abs(float(self.oi_delta_pct))*100:.2f}% | "
            f"L/S={ls_pct:.1f}%L | "
            f"adj={self.score_adjustment:+d} | "
            f"block_long={self.block_long} block_short={self.block_short}"
        )


class MarketSentimentEngine:
    """
    Analizza Open Interest e Long/Short Ratio per filtrare e rafforzare i segnali.

    Args:
        testnet  — True per usare endpoint testnet Bybit
        ls_period — periodo per il L/S ratio (Bybit: "5min", "15min", "30min", "1h", "4h", "1d")
    """

    def __init__(self, testnet: bool = False, ls_period: str = "5min", live_trading: bool = False) -> None:
        self._base_url   = _BASE_TESTNET_URL if testnet else _BASE_URL
        self._ls_period  = ls_period
        self._live_trading = live_trading
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._cache: dict[str, SentimentResult] = {}   # symbol → last result

    @staticmethod
    def _neutral_result(symbol: str, candle) -> 'SentimentResult':
        """Risultato neutro per paper/backtest mode (no HTTP calls)."""
        return SentimentResult(
            symbol=symbol,
            oi_current=_ZERO, oi_prev=_ZERO, oi_delta_pct=_ZERO,
            ls_ratio=Decimal('0.5'),  # 50/50 neutro
            block_long=False, block_short=False,
            score_adjustment=0, timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    async def analyze(self, symbol: str, candle: Candle) -> SentimentResult:
        """
        Analizza OI e L/S ratio per il symbol.
        In paper/backtest: ritorna neutro (no HTTP calls).
        In live: usa cache con TTL, fetch da Bybit API.
        """
        # Paper/backtest mode: skip HTTP, ritorna neutro
        if not self._live_trading:
            return self._neutral_result(symbol, candle)

        # Cache hit
        if symbol in self._cache and not self._cache[symbol].is_stale:
            return self._cache[symbol]

        oi_current, oi_prev = await self._fetch_oi(symbol)
        ls_ratio             = await self._fetch_ls_ratio(symbol)

        result = self._compute_sentiment(
            symbol, candle, oi_current, oi_prev, ls_ratio
        )
        self._cache[symbol] = result
        logger.debug("[Sentiment] %s: %s", symbol, result.summary())
        return result

    def score_adjustment_for(self, symbol: str, direction: SignalType) -> int:
        """
        Restituisce il bonus/malus score per la direzione del segnale.
        Usa la cache — non fa REST call (deve essere chiamato dopo analyze()).
        Ritorna 0 se la cache è stale (dati scaduti) per non applicare score obsoleto.
        """
        if symbol not in self._cache:
            return 0
        r = self._cache[symbol]
        if r.is_stale:
            logger.debug("Sentiment cache scaduta per %s, nessun aggiustamento applicato.", symbol)
            return 0
        if direction == SignalType.LONG:
            if r.block_long:
                return -15   # Blocco esplicito
            return r.score_adjustment if r.score_adjustment > 0 else 0
        if direction == SignalType.SHORT:
            if r.block_short:
                return -15   # Blocco esplicito
            return r.score_adjustment if r.score_adjustment < 0 else 0
        return 0

    def is_blocked(self, symbol: str, direction: SignalType) -> bool:
        """
        Controlla esplicitamente se il sentiment blocca la direzione.
        Usa i flag block_long/block_short direttamente, senza dipendere dallo score.
        Ritorna False se cache assente o stale (fail-open: non blocca in caso di dubbio).
        """
        if symbol not in self._cache:
            return False
        r = self._cache[symbol]
        if r.is_stale:
            return False
        if direction == SignalType.LONG:
            return r.block_long
        if direction == SignalType.SHORT:
            return r.block_short
        return False

    async def close(self) -> None:
        """Chiude la sessione HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Fetch dati Bybit
    # ------------------------------------------------------------------

    async def _fetch_oi(self, symbol: str) -> tuple[Decimal, Decimal]:
        """
        Scarica gli ultimi 2 valori di Open Interest da Bybit.
        Restituisce (oi_attuale, oi_precedente).
        """
        params = {
            "category":    "linear",
            "symbol":      symbol,
            "intervalTime": "5min",
            "limit":       "2",
        }
        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}{_OI_PATH}",
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    rows = data.get("result", {}).get("list", [])
                    if len(rows) >= 2:
                        # Bybit restituisce dal più recente → rows[0] = attuale, rows[1] = precedente
                        oi_cur  = Decimal(str(rows[0].get("openInterest", "0")))
                        oi_prev = Decimal(str(rows[1].get("openInterest", "0")))
                        return oi_cur, oi_prev
        except Exception as e:
            logger.debug("OI fetch fallito per %s: %s", symbol, e)
        return _ZERO, _ZERO

    async def _fetch_ls_ratio(self, symbol: str) -> Decimal:
        """
        Scarica il Long/Short ratio retail da Bybit.
        Restituisce la frazione di long (es. 0.58 = 58% long).
        """
        params = {
            "category": "linear",
            "symbol":   symbol,
            "period":   self._ls_period,
            "limit":    "1",
        }
        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}{_LS_PATH}",
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                data = await resp.json()
                if data.get("retCode") == 0:
                    rows = data.get("result", {}).get("list", [])
                    if rows:
                        buy_ratio = rows[0].get("buyRatio", "0.5")
                        return Decimal(str(buy_ratio))
        except Exception as e:
            logger.debug("L/S ratio fetch fallito per %s: %s", symbol, e)
        return Decimal("0.5")   # neutral fallback

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    # ------------------------------------------------------------------
    # Logica sentiment
    # ------------------------------------------------------------------

    def _compute_sentiment(
        self,
        symbol: str,
        candle: Candle,
        oi_current: Decimal,
        oi_prev: Decimal,
        ls_ratio: Decimal,
    ) -> SentimentResult:
        """
        Calcola il sentiment combinando OI delta e L/S ratio.

        Matrice di decisione:
        OI↑ + prezzo↑ + retail bearish  → LONG confirmation (+score, no block)
        OI↑ + prezzo↑ + retail bullish  → LONG debole (retail già dentro, squeeze meno probabile)
        OI↑ + prezzo↓ + retail bullish  → SHORT confirmation (trap dei long)
        OI↓              → liquidazioni in corso → segnale debole per entrambi
        Retail >70% long → block LONG (crowd troppo posizionata, squeeze risk)
        Retail <30% long → block SHORT (crowd troppo short, short squeeze risk)
        """
        # OI delta %
        if oi_prev > _ZERO:
            oi_delta_pct = (oi_current - oi_prev) / oi_prev
        else:
            oi_delta_pct = _ZERO

        oi_rising = oi_delta_pct > _OI_CHANGE_THRESH
        oi_falling = oi_delta_pct < -_OI_CHANGE_THRESH

        # Direzione prezzo
        price_up = candle.close > candle.open

        # Sentiment retail (contrarian)
        retail_extreme_long  = ls_ratio > _LS_EXTREME_LONG
        retail_extreme_short = ls_ratio < _LS_EXTREME_SHORT
        retail_neutral       = not retail_extreme_long and not retail_extreme_short

        # --- Block flags ---
        # Blocca LONG se retail è già eccessivamente lungo (>70%)
        block_long = retail_extreme_long

        # Blocca SHORT se retail è già eccessivamente corto (<30%)
        block_short = retail_extreme_short

        # OI in calo = posizioni si chiudono = mercato si restringe → non blocca ma riduce score
        # (non blocchiamo completamente perché potrebbe essere riduzione di posizioni short)

        # --- Score adjustment (−15 a +15) ---
        score_adj = 0

        # OI rising + direzione prezzo concorde = volume reale nella direzione
        if oi_rising and price_up and not retail_extreme_long:
            score_adj += 10   # LONG confermato da OI
        elif oi_rising and not price_up and not retail_extreme_short:
            score_adj -= 10   # SHORT confermato da OI
        elif oi_falling:
            score_adj -= 5    # Liquidazioni in corso → segnale meno affidabile

        # Retail estremo contrarian
        if retail_extreme_long:
            score_adj -= 8    # 70%+ retail long → pericoloso aprire long
        elif retail_extreme_short:
            score_adj += 8    # <30% retail long → pericoloso stare short

        # Clamp
        score_adj = max(-15, min(15, score_adj))

        return SentimentResult(
            symbol=symbol,
            oi_current=oi_current,
            oi_prev=oi_prev,
            oi_delta_pct=oi_delta_pct,
            ls_ratio=ls_ratio,
            block_long=block_long,
            block_short=block_short,
            score_adjustment=score_adj,
            timestamp=time.time(),
        )
