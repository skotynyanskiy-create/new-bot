"""
IndicatorEngine — calcola EMA, RSI, ATR, ADX, Bollinger Bands, EMA Ribbon su buffer rolling.

Tutti i calcoli usano Decimal per coerenza con il resto del codebase.
Nessuna dipendenza da NumPy/Pandas.
"""

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

        # --- Bollinger Bands (SMA + rolling std) ---
        self._bb_upper_val:  Optional[Decimal] = None
        self._bb_lower_val:  Optional[Decimal] = None
        self._bb_middle_val: Optional[Decimal] = None

        # --- EMA Ribbon (8, 13, 21, 34, 55) ---
        self._ribbon_vals:  dict[int, Optional[Decimal]] = {p: None for p in _RIBBON_PERIODS}
        self._ribbon_k:     dict[int, Decimal] = {
            p: _TWO / (Decimal(p) + _ONE) for p in _RIBBON_PERIODS
        }

        self._n = 0  # candele processate

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

    def highest_high(self, lookback: int) -> Decimal:
        """Massimo dei HIGH nelle ultime `lookback` candele (esclusa l'attuale)."""
        buf = list(self._candles)
        prior = buf[:-1]
        if not prior:
            raise ValueError("Buffer insufficiente per highest_high")
        window = prior[-lookback:] if len(prior) >= lookback else prior
        return max(c.high for c in window)

    def lowest_low(self, lookback: int) -> Decimal:
        """Minimo dei LOW nelle ultime `lookback` candele (esclusa l'attuale)."""
        buf = list(self._candles)
        prior = buf[:-1]
        if not prior:
            raise ValueError("Buffer insufficiente per lowest_low")
        window = prior[-lookback:] if len(prior) >= lookback else prior
        return min(c.low for c in window)

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
            # Usiamo un attributo temporaneo separato per i seed DM
            if not hasattr(self, '_seed_dm_plus'):
                self._seed_dm_plus:  list[Decimal] = []
                self._seed_dm_minus: list[Decimal] = []
                self._seed_tr:       list[Decimal] = []
            self._seed_dm_plus.append(dm_plus)
            self._seed_dm_minus.append(dm_minus)
            self._seed_tr.append(tr)

            if len(self._seed_tr) == self._adx_period:
                self._adx_tr_smooth  = sum(self._seed_tr)
                self._adx_dm_plus_s  = sum(self._seed_dm_plus)
                self._adx_dm_minus_s = sum(self._seed_dm_minus)
                self._dx_values.clear()  # reset per uso DX
                # Prima DI+/DI-
                self._di_plus_val  = (self._adx_dm_plus_s  / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
                self._di_minus_val = (self._adx_dm_minus_s / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
                # Prima DX → feed per ADX seed
                di_sum  = self._di_plus_val + self._di_minus_val
                dx = (abs(self._di_plus_val - self._di_minus_val) / di_sum * Decimal('100')) if di_sum != _ZERO else _ZERO
                self._dx_values.append(dx)
        else:
            # Wilder smoothing: smooth = smooth - smooth/period + current
            self._adx_tr_smooth  = self._adx_tr_smooth  - self._adx_tr_smooth  / period + tr
            self._adx_dm_plus_s  = self._adx_dm_plus_s  - self._adx_dm_plus_s  / period + dm_plus
            self._adx_dm_minus_s = self._adx_dm_minus_s - self._adx_dm_minus_s / period + dm_minus

            self._di_plus_val  = (self._adx_dm_plus_s  / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))
            self._di_minus_val = (self._adx_dm_minus_s / self._adx_tr_smooth * Decimal('100')).quantize(Decimal('0.01'))

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
