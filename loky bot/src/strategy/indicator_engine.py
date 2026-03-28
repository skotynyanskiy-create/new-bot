"""
IndicatorEngine — calcola EMA, RSI, ATR, ADX, Bollinger Bands, EMA Ribbon su buffer rolling.

Tutti i calcoli usano Decimal per coerenza con il resto del codebase.
Nessuna dipendenza da NumPy/Pandas.
"""

import itertools
from collections import deque
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from src.models import Candle


_ZERO = Decimal('0')
_ONE  = Decimal('1')
_TWO  = Decimal('2')

# Periodi EMA Ribbon fissi
_RIBBON_PERIODS = (8, 13, 21, 34, 55)


class IndicatorEngine:
    """
    Mantiene buffer rolling di candele e calcola indicatori tecnici incrementalmente.

    Uso tipico:
        engine = IndicatorEngine()
        engine.update(candle)       # chiamato ad ogni candela chiusa
        if engine.ready():
            rsi = engine.rsi()
            atr = engine.atr()
            adx_val = engine.adx()
            upper, lower = engine.bb_upper(), engine.bb_lower()
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        vol_period: int = 20,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: Decimal = Decimal('2'),
    ) -> None:
        self._ema_fast_period = ema_fast
        self._ema_slow_period = ema_slow
        self._rsi_period      = rsi_period
        self._atr_period      = atr_period
        self._vol_period      = vol_period
        self._adx_period      = adx_period
        self._bb_period       = bb_period
        self._bb_std          = bb_std

        # Buffer massimo = periodo più lungo + 10 per sicurezza
        max_buf = max(
            ema_fast, ema_slow,
            rsi_period + 1, atr_period, vol_period,
            adx_period * 2,   # ADX ha bisogno di più storia per lo smoothing Wilder
            bb_period,
            max(_RIBBON_PERIODS),
        ) + 10
        self._candles: deque[Candle] = deque(maxlen=max_buf)

        # --- EMA fast/slow ---
        self._ema_fast_val: Optional[Decimal] = None
        self._ema_slow_val: Optional[Decimal] = None
        self._k_fast = _TWO / (Decimal(ema_fast) + _ONE)
        self._k_slow = _TWO / (Decimal(ema_slow) + _ONE)

        # --- RSI ---
        self._gains: deque[Decimal] = deque(maxlen=rsi_period)
        self._losses: deque[Decimal] = deque(maxlen=rsi_period)
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None

        # --- ATR ---
        self._tr_values: deque[Decimal] = deque(maxlen=atr_period)
        self._atr_val: Optional[Decimal] = None

        # --- ADX (Wilder smoothing) ---
        self._adx_tr_smooth:   Optional[Decimal] = None   # smoothed TR
        self._adx_dm_plus_s:   Optional[Decimal] = None   # smoothed DM+
        self._adx_dm_minus_s:  Optional[Decimal] = None   # smoothed DM-
        self._di_plus_val:     Optional[Decimal] = None
        self._di_minus_val:    Optional[Decimal] = None
        self._dx_values: deque[Decimal] = deque(maxlen=adx_period)
        self._adx_val:         Optional[Decimal] = None
        self._seed_dm_plus:  list[Decimal] = []
        self._seed_dm_minus: list[Decimal] = []
        self._seed_tr:       list[Decimal] = []

        # --- Bollinger Bands (SMA + rolling std) ---
        self._bb_upper_val:  Optional[Decimal] = None
        self._bb_lower_val:  Optional[Decimal] = None
        self._bb_middle_val: Optional[Decimal] = None

        # --- EMA Ribbon (8, 13, 21, 34, 55) ---
        self._ribbon_vals:  dict[int, Optional[Decimal]] = {p: None for p in _RIBBON_PERIODS}
        self._ribbon_k:     dict[int, Decimal] = {
            p: _TWO / (Decimal(p) + _ONE) for p in _RIBBON_PERIODS
        }

        # --- VWAP intraday (reset ogni nuovo giorno UTC) ---
        self._vwap_cum_pv:  Decimal = _ZERO   # Σ(typical_price × volume)
        self._vwap_cum_vol: Decimal = _ZERO   # Σ(volume)
        self._vwap_val:     Optional[Decimal] = None
        self._vwap_date:    Optional[object]   = None  # datetime.date UTC corrente

        self._n = 0  # candele processate

        # --- Cache S/R levels (ricalcolati ogni _SR_CACHE_INTERVAL candele) ---
        self._sr_cache_interval: int = 5
        self._sr_support_cache: list[Decimal] = []
        self._sr_resistance_cache: list[Decimal] = []
        self._sr_cache_n: int = 0  # _n al momento dell'ultimo calcolo

    # ------------------------------------------------------------------
    # Metodo pubblico di aggiornamento
    # ------------------------------------------------------------------

    def update(self, candle: Candle) -> None:
        """Aggiorna tutti gli indicatori con la nuova candela chiusa."""
        prev = self._candles[-1] if self._candles else None
        self._candles.append(candle)
        self._n += 1

        self._update_ema(candle)
        self._update_rsi(candle, prev)
        self._update_atr(candle, prev)
        self._update_adx(candle, prev)
        self._update_bb()
        self._update_ribbon(candle)
        self._update_vwap(candle)

    # ------------------------------------------------------------------
    # Proprietà pubbliche — Esistenti
    # ------------------------------------------------------------------

    def ready(self) -> bool:
        """True quando tutti gli indicatori principali hanno abbastanza dati."""
        min_required = max(
            self._ema_slow_period,
            self._rsi_period + 1,
            self._atr_period,
            self._vol_period,
            self._adx_period * 2,
            self._bb_period,
            max(_RIBBON_PERIODS),
        )
        return self._n >= min_required

    def ema_fast(self) -> Decimal:
        if self._ema_fast_val is None:
            raise ValueError("EMA fast non ancora disponibile")
        return self._ema_fast_val

    def ema_slow(self) -> Decimal:
        if self._ema_slow_val is None:
            raise ValueError("EMA slow non ancora disponibile")
        return self._ema_slow_val

    def rsi(self) -> Decimal:
        """RSI 0-100. Usa Wilder smoothing (EMA con alpha=1/period)."""
        if self._avg_gain is None or self._avg_loss is None:
            raise ValueError("RSI non ancora disponibile")
        if self._avg_loss == _ZERO:
            return Decimal('100')
        rs = self._avg_gain / self._avg_loss
        return (Decimal('100') - (Decimal('100') / (_ONE + rs))).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )

    def atr(self) -> Decimal:
        """Average True Range."""
        if self._atr_val is None:
            raise ValueError("ATR non ancora disponibile")
        return self._atr_val

    def volume_ma(self) -> Decimal:
        """Media mobile semplice del volume sulle ultime vol_period candele."""
        if len(self._candles) < self._vol_period:
            raise ValueError("Volume MA non ancora disponibile")
        recent = list(self._candles)[-self._vol_period:]
        return sum(c.volume for c in recent) / Decimal(self._vol_period)

    def volume_ratio(self) -> Decimal:
        """Rapporto volume ultima candela / volume MA. >1 = volume sopra la media."""
        vol_ma = self.volume_ma()
        if vol_ma <= _ZERO or not self._candles:
            raise ValueError("Volume ratio non disponibile")
        return self._candles[-1].volume / vol_ma

    def highest_high(self, lookback: int) -> Decimal:
        """Massimo dei HIGH nelle ultime `lookback` candele (esclusa l'attuale)."""
        n = len(self._candles)
        if n < 2:
            raise ValueError("Buffer insufficiente per highest_high")
        start = max(0, n - 1 - lookback)
        return max(c.high for c in itertools.islice(self._candles, start, n - 1))

    def lowest_low(self, lookback: int) -> Decimal:
        """Minimo dei LOW nelle ultime `lookback` candele (esclusa l'attuale)."""
        n = len(self._candles)
        if n < 2:
            raise ValueError("Buffer insufficiente per lowest_low")
        start = max(0, n - 1 - lookback)
        return min(c.low for c in itertools.islice(self._candles, start, n - 1))

    def recent_swing_low(self, lookback: int = 5) -> Decimal:
        """
        Minimo recente dei LOW nelle ultime `lookback` candele inclusa l'attuale.
        Usato per posizionare lo stop loss su struttura di mercato (sotto il swing low).
        """
        n = len(self._candles)
        if n == 0:
            raise ValueError("Buffer vuoto")
        start = max(0, n - lookback)
        return min(c.low for c in itertools.islice(self._candles, start, n))

    def recent_swing_high(self, lookback: int = 5) -> Decimal:
        """
        Massimo recente dei HIGH nelle ultime `lookback` candele inclusa l'attuale.
        Usato per posizionare lo stop loss su struttura di mercato (sopra il swing high) per SHORT.
        """
        n = len(self._candles)
        if n == 0:
            raise ValueError("Buffer vuoto")
        start = max(0, n - lookback)
        return max(c.high for c in itertools.islice(self._candles, start, n))

    # ------------------------------------------------------------------
    # Proprietà pubbliche — ADX
    # ------------------------------------------------------------------

    def adx(self) -> Decimal:
        """ADX 0-100. >25 = trending forte, <20 = ranging."""
        if self._adx_val is None:
            raise ValueError("ADX non ancora disponibile")
        return self._adx_val

    def di_plus(self) -> Decimal:
        """Directional Index + (bullish pressure)."""
        if self._di_plus_val is None:
            raise ValueError("DI+ non ancora disponibile")
        return self._di_plus_val

    def di_minus(self) -> Decimal:
        """Directional Index - (bearish pressure)."""
        if self._di_minus_val is None:
            raise ValueError("DI- non ancora disponibile")
        return self._di_minus_val

    # ------------------------------------------------------------------
    # Proprietà pubbliche — Bollinger Bands
    # ------------------------------------------------------------------

    def bb_upper(self) -> Decimal:
        if self._bb_upper_val is None:
            raise ValueError("Bollinger Bands non ancora disponibili")
        return self._bb_upper_val

    def bb_lower(self) -> Decimal:
        if self._bb_lower_val is None:
            raise ValueError("Bollinger Bands non ancora disponibili")
        return self._bb_lower_val

    def bb_middle(self) -> Decimal:
        if self._bb_middle_val is None:
            raise ValueError("Bollinger Bands non ancora disponibili")
        return self._bb_middle_val

    def bb_width(self) -> Decimal:
        """BB Width = (upper - lower) / middle. Misura di volatilità."""
        if self._bb_upper_val is None or self._bb_middle_val is None:
            raise ValueError("Bollinger Bands non ancora disponibili")
        if self._bb_middle_val == _ZERO:
            return _ZERO
        return (self._bb_upper_val - self._bb_lower_val) / self._bb_middle_val

    # ------------------------------------------------------------------
    # Keltner Channel & Squeeze
    # ------------------------------------------------------------------

    def keltner_upper(self, mult: Decimal = Decimal('1.5')) -> Decimal:
        """Keltner upper = EMA20 + mult × ATR."""
        ema = self.bb_middle()  # EMA20 (stessa base delle BB)
        atr = self.atr()
        return ema + mult * atr

    def keltner_lower(self, mult: Decimal = Decimal('1.5')) -> Decimal:
        """Keltner lower = EMA20 - mult × ATR."""
        ema = self.bb_middle()
        atr = self.atr()
        return ema - mult * atr

    def is_squeeze(self) -> bool:
        """
        Bollinger Bands Squeeze: BB dentro il Keltner Channel.
        Quando BB upper < Keltner upper AND BB lower > Keltner lower,
        la volatilità è compressa → breakout imminente.

        Usato come trigger per entry su breakout strategy.
        """
        try:
            return (
                self.bb_upper() < self.keltner_upper()
                and self.bb_lower() > self.keltner_lower()
            )
        except ValueError:
            return False

    def squeeze_release(self) -> Optional[str]:
        """
        Detecta il rilascio dello squeeze (BB esce dal Keltner).
        Ritorna 'bullish' se il prezzo è sopra BB middle, 'bearish' se sotto.
        """
        try:
            if not self.is_squeeze():
                # Non in squeeze — controlla se appena uscito
                bb_w = self.bb_width()
                if bb_w < Decimal('0.02'):  # ancora compresso ma uscendo
                    return None
                # Direzione del breakout
                if self._candles and self._candles[-1].close > self.bb_middle():
                    return "bullish"
                elif self._candles and self._candles[-1].close < self.bb_middle():
                    return "bearish"
        except ValueError:
            pass
        return None

    # ------------------------------------------------------------------
    # Proprietà pubbliche — EMA Ribbon
    # ------------------------------------------------------------------

    def ema_ribbon(self) -> tuple[Decimal, ...]:
        """Restituisce (EMA8, EMA13, EMA21, EMA34, EMA55)."""
        vals = tuple(self._ribbon_vals[p] for p in _RIBBON_PERIODS)
        if any(v is None for v in vals):
            raise ValueError("EMA Ribbon non ancora disponibile")
        return vals  # type: ignore[return-value]

    def ribbon_aligned_bullish(self) -> bool:
        """True se EMA8 > EMA13 > EMA21 > EMA34 > EMA55 (trend rialzista forte)."""
        try:
            e8, e13, e21, e34, e55 = self.ema_ribbon()
            return e8 > e13 > e21 > e34 > e55
        except ValueError:
            return False

    def ribbon_aligned_bearish(self) -> bool:
        """True se EMA8 < EMA13 < EMA21 < EMA34 < EMA55 (trend ribassista forte)."""
        try:
            e8, e13, e21, e34, e55 = self.ema_ribbon()
            return e8 < e13 < e21 < e34 < e55
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Proprietà pubbliche — VWAP
    # ------------------------------------------------------------------

    def vwap(self) -> Decimal:
        """
        VWAP intraday (Volume Weighted Average Price).
        Si azzera ogni nuovo giorno UTC.
        Solleva ValueError se non ancora disponibile.
        """
        if self._vwap_val is None:
            raise ValueError("VWAP non ancora disponibile")
        return self._vwap_val

    def price_above_vwap(self, price: Decimal) -> bool:
        """True se il prezzo è sopra il VWAP (bullish volume bias)."""
        try:
            return price > self.vwap()
        except ValueError:
            return True  # fallback: no filter se VWAP non disponibile

    def price_below_vwap(self, price: Decimal) -> bool:
        """True se il prezzo è sotto il VWAP (bearish volume bias)."""
        try:
            return price < self.vwap()
        except ValueError:
            return True  # fallback: no filter se VWAP non disponibile

    # ------------------------------------------------------------------
    # Metodi privati di calcolo — Esistenti
    # ------------------------------------------------------------------

    def _update_ema(self, candle: Candle) -> None:
        price = candle.close
        if self._ema_fast_val is None:
            if self._n >= self._ema_fast_period:
                buf = list(self._candles)[-self._ema_fast_period:]
                self._ema_fast_val = sum(c.close for c in buf) / Decimal(self._ema_fast_period)
        else:
            self._ema_fast_val = (price - self._ema_fast_val) * self._k_fast + self._ema_fast_val

        if self._ema_slow_val is None:
            if self._n >= self._ema_slow_period:
                buf = list(self._candles)[-self._ema_slow_period:]
                self._ema_slow_val = sum(c.close for c in buf) / Decimal(self._ema_slow_period)
        else:
            self._ema_slow_val = (price - self._ema_slow_val) * self._k_slow + self._ema_slow_val

    def _update_rsi(self, candle: Candle, prev: Optional[Candle]) -> None:
        if prev is None:
            return
        delta = candle.close - prev.close
        gain  = delta if delta > _ZERO else _ZERO
        loss  = -delta if delta < _ZERO else _ZERO

        if self._avg_gain is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) == self._rsi_period:
                self._avg_gain = sum(self._gains) / Decimal(self._rsi_period)
                self._avg_loss = sum(self._losses) / Decimal(self._rsi_period)
        else:
            self._avg_gain = (self._avg_gain * (Decimal(self._rsi_period) - _ONE) + gain) / Decimal(self._rsi_period)
            self._avg_loss = (self._avg_loss * (Decimal(self._rsi_period) - _ONE) + loss) / Decimal(self._rsi_period)

    def _update_atr(self, candle: Candle, prev: Optional[Candle]) -> None:
        if prev is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev.close),
                abs(candle.low  - prev.close),
            )

        self._tr_values.append(tr)

        if self._atr_val is None:
            if len(self._tr_values) == self._atr_period:
                self._atr_val = sum(self._tr_values) / Decimal(self._atr_period)
        else:
            self._atr_val = (self._atr_val * Decimal(self._atr_period - 1) + tr) / Decimal(self._atr_period)

    # ------------------------------------------------------------------
    # Metodi privati di calcolo — ADX (Wilder smoothing)
    # ------------------------------------------------------------------

    def _update_adx(self, candle: Candle, prev: Optional[Candle]) -> None:
        if prev is None:
            return

        period = Decimal(self._adx_period)

        # True Range
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev.close),
            abs(candle.low  - prev.close),
        )

        # Directional Movement
        up_move   = candle.high - prev.high
        down_move = prev.low  - candle.low

        dm_plus  = up_move   if (up_move > down_move and up_move > _ZERO)   else _ZERO
        dm_minus = down_move if (down_move > up_move and down_move > _ZERO) else _ZERO

        if self._adx_tr_smooth is None:
            # Fase di seed: accumula adx_period valori poi inizializza con SMA
            self._dx_values.append(tr)           # riuso la deque temporaneamente per TR seed
            self._seed_dm_plus.append(dm_plus)
            self._seed_dm_minus.append(dm_minus)
            self._seed_tr.append(tr)

            if len(self._seed_tr) == self._adx_period:
                self._adx_tr_smooth  = sum(self._seed_tr)
                self._adx_dm_plus_s  = sum(self._seed_dm_plus)
                self._adx_dm_minus_s = sum(self._seed_dm_minus)
                self._dx_values.clear()  # reset per uso DX
                # Prima DI+/DI-
                if self._adx_tr_smooth > _ZERO:
                    self._di_plus_val  = (self._adx_dm_plus_s  / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
                    self._di_minus_val = (self._adx_dm_minus_s / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
                else:
                    self._di_plus_val  = _ZERO
                    self._di_minus_val = _ZERO
                # Prima DX → feed per ADX seed
                di_sum  = self._di_plus_val + self._di_minus_val
                dx = (abs(self._di_plus_val - self._di_minus_val) / di_sum * Decimal('100')) if di_sum != _ZERO else _ZERO
                self._dx_values.append(dx)
        else:
            # Wilder smoothing: smooth = smooth - smooth/period + current
            self._adx_tr_smooth  = self._adx_tr_smooth  - self._adx_tr_smooth  / period + tr
            self._adx_dm_plus_s  = self._adx_dm_plus_s  - self._adx_dm_plus_s  / period + dm_plus
            self._adx_dm_minus_s = self._adx_dm_minus_s - self._adx_dm_minus_s / period + dm_minus

            if self._adx_tr_smooth > _ZERO:
                self._di_plus_val  = (self._adx_dm_plus_s  / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
                self._di_minus_val = (self._adx_dm_minus_s / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
            else:
                self._di_plus_val  = _ZERO
                self._di_minus_val = _ZERO

            di_sum = self._di_plus_val + self._di_minus_val
            dx = (abs(self._di_plus_val - self._di_minus_val) / di_sum * Decimal('100')) if di_sum != _ZERO else _ZERO
            self._dx_values.append(dx)

            if self._adx_val is None:
                # Seed ADX: SMA dei primi adx_period DX values
                if len(self._dx_values) == self._adx_period:
                    self._adx_val = sum(self._dx_values) / period
            else:
                # Wilder smoothing ADX
                self._adx_val = (self._adx_val * (period - _ONE) + dx) / period
                self._adx_val = self._adx_val.quantize(Decimal('0.01'))

    # ------------------------------------------------------------------
    # Metodi privati di calcolo — Bollinger Bands
    # ------------------------------------------------------------------

    def _update_bb(self) -> None:
        if len(self._candles) < self._bb_period:
            return

        closes = [c.close for c in list(self._candles)[-self._bb_period:]]
        n = Decimal(self._bb_period)
        mean = sum(closes) / n

        # Deviazione standard (population std)
        variance = sum((c - mean) ** 2 for c in closes) / n
        std = variance.sqrt()

        self._bb_middle_val = mean
        self._bb_upper_val  = mean + self._bb_std * std
        self._bb_lower_val  = mean - self._bb_std * std

    # ------------------------------------------------------------------
    # Metodi privati di calcolo — EMA Ribbon
    # ------------------------------------------------------------------

    def _update_ribbon(self, candle: Candle) -> None:
        price = candle.close
        for period in _RIBBON_PERIODS:
            if self._ribbon_vals[period] is None:
                if self._n >= period:
                    buf = list(self._candles)[-period:]
                    self._ribbon_vals[period] = sum(c.close for c in buf) / Decimal(period)
            else:
                k = self._ribbon_k[period]
                self._ribbon_vals[period] = (price - self._ribbon_vals[period]) * k + self._ribbon_vals[period]

    # ------------------------------------------------------------------
    # Metodi privati di calcolo — VWAP intraday
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Proprietà pubbliche — Support/Resistance Pivot Clusters
    # ------------------------------------------------------------------

    def _refresh_sr_cache(self, lookback: int = 50, tolerance_pct: Decimal = Decimal('0.002')) -> None:
        """Ricalcola S/R cache solo se sono passate abbastanza candele dall'ultimo calcolo."""
        if self._n - self._sr_cache_n < self._sr_cache_interval and self._sr_cache_n > 0:
            return

        buf = list(self._candles)
        if len(buf) < 5:
            self._sr_support_cache = []
            self._sr_resistance_cache = []
            self._sr_cache_n = self._n
            return

        window = buf[-lookback:] if len(buf) >= lookback else buf

        # Support pivots
        sup_pivots: list[Decimal] = []
        for i in range(1, len(window) - 1):
            if window[i].low < window[i - 1].low and window[i].low < window[i + 1].low:
                sup_pivots.append(window[i].low)
        self._sr_support_cache = self._cluster_levels(sup_pivots, tolerance_pct)

        # Resistance pivots
        res_pivots: list[Decimal] = []
        for i in range(1, len(window) - 1):
            if window[i].high > window[i - 1].high and window[i].high > window[i + 1].high:
                res_pivots.append(window[i].high)
        self._sr_resistance_cache = self._cluster_levels(res_pivots, tolerance_pct)

        self._sr_cache_n = self._n

    def support_levels(self, lookback: int = 50, tolerance_pct: Decimal = Decimal('0.002')) -> list[Decimal]:
        """
        Identifica livelli di supporto come cluster di swing low recenti.
        Cached: ricalcolato ogni _sr_cache_interval candele.
        """
        self._refresh_sr_cache(lookback, tolerance_pct)
        return self._sr_support_cache

    def resistance_levels(self, lookback: int = 50, tolerance_pct: Decimal = Decimal('0.002')) -> list[Decimal]:
        """
        Identifica livelli di resistenza come cluster di swing high recenti.
        Cached: ricalcolato ogni _sr_cache_interval candele.
        """
        self._refresh_sr_cache(lookback, tolerance_pct)
        return self._sr_resistance_cache

    def nearest_resistance_above(self, price: Decimal, lookback: int = 50) -> Optional[Decimal]:
        """Restituisce il livello di resistenza più vicino sopra `price`, o None se non trovato."""
        levels = self.resistance_levels(lookback)
        candidates = [lvl for lvl in levels if lvl > price]
        return min(candidates) if candidates else None

    def nearest_support_below(self, price: Decimal, lookback: int = 50) -> Optional[Decimal]:
        """Restituisce il livello di supporto più vicino sotto `price`, o None se non trovato."""
        levels = self.support_levels(lookback)
        candidates = [lvl for lvl in levels if lvl < price]
        return max(candidates) if candidates else None

    @staticmethod
    def _cluster_levels(pivots: list[Decimal], tolerance_pct: Decimal) -> list[Decimal]:
        """
        Raggruppa i pivot in cluster. Ogni cluster è rappresentato dalla sua media.
        Due pivot appartengono allo stesso cluster se distano < tolerance_pct.
        """
        if not pivots:
            return []

        pivots_sorted = sorted(pivots)
        clusters: list[list[Decimal]] = []
        current_cluster = [pivots_sorted[0]]

        for price in pivots_sorted[1:]:
            ref = current_cluster[0]
            if ref == _ZERO:
                current_cluster.append(price)
                continue
            if abs(price - ref) / ref <= tolerance_pct:
                current_cluster.append(price)
            else:
                clusters.append(current_cluster)
                current_cluster = [price]
        clusters.append(current_cluster)

        # Media di ogni cluster, solo se ha almeno 1 tocco (tutti i cluster)
        result = [sum(c) / Decimal(len(c)) for c in clusters]
        return sorted(result)

    def _update_vwap(self, candle: Candle) -> None:
        """
        Aggiorna il VWAP intraday.
        Reset a mezzanotte UTC (quando il timestamp del candle cambia giorno).

        Calcolo: VWAP = Σ(typical_price × volume) / Σ(volume)
        dove typical_price = (high + low + close) / 3
        """
        import datetime
        candle_date = datetime.datetime.fromtimestamp(candle.timestamp, tz=datetime.timezone.utc).date()

        # Reset a ogni nuovo giorno UTC (usa date completa, non solo .day per evitare bug a cambio mese)
        if self._vwap_date is not None and candle_date != self._vwap_date:
            self._vwap_cum_pv  = _ZERO
            self._vwap_cum_vol = _ZERO

        self._vwap_date = candle_date

        # Typical price = (H + L + C) / 3
        typical = (candle.high + candle.low + candle.close) / Decimal('3')
        self._vwap_cum_pv  += typical * candle.volume
        self._vwap_cum_vol += candle.volume

        if self._vwap_cum_vol > _ZERO:
            self._vwap_val = self._vwap_cum_pv / self._vwap_cum_vol
