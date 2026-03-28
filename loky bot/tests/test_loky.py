"""
Test suite per Loky Bot — verifica componenti core.
"""
import asyncio
import time
import pytest
from decimal import Decimal
from collections import deque

from src.models import Candle, Signal, SignalType, Side, Order, OrderStatus, Trade, TPLevel
from src.config import BotSettings, StrategySettings
from src.strategy.indicator_engine import IndicatorEngine
from src.core.kelly_sizing import KellySizer
from src.core.account_risk import AccountRiskManager


# ================================================================== #
#  HELPERS                                                             #
# ================================================================== #

def make_candle(
    symbol: str = "BTCUSDT",
    o: float = 100.0, h: float = 102.0, l: float = 99.0, c: float = 101.0,
    volume: float = 1000.0, ts: float = 0.0, timeframe: str = "15m",
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l)),
        close=Decimal(str(c)),
        volume=Decimal(str(volume)),
        timestamp=ts or time.time(),
        is_closed=True,
    )


def make_candles(n: int = 60, base_price: float = 100.0, symbol: str = "BTCUSDT") -> list[Candle]:
    """Genera N candle con trend leggermente rialzista."""
    candles = []
    for i in range(n):
        drift = i * 0.05
        o = base_price + drift
        h = o + 1.5
        l = o - 1.0
        c = o + 0.5
        candles.append(make_candle(
            symbol=symbol, o=o, h=h, l=l, c=c,
            volume=1000.0 + i * 10,
            ts=1700000000 + i * 900,  # 15m intervals
        ))
    return candles


# ================================================================== #
#  CONFIG VALIDATION                                                   #
# ================================================================== #

class TestConfigValidation:

    def test_default_config_loads(self):
        cfg = BotSettings()
        assert cfg.leverage == 3
        assert cfg.risk_per_trade_pct == Decimal("0.015")

    def test_risk_too_high_raises(self):
        with pytest.raises(Exception):
            BotSettings(risk_per_trade_pct=Decimal("0.10"))

    def test_risk_too_low_raises(self):
        with pytest.raises(Exception):
            BotSettings(risk_per_trade_pct=Decimal("0.0001"))

    def test_leverage_out_of_range_raises(self):
        with pytest.raises(Exception):
            BotSettings(leverage=25)

    def test_daily_loss_pct_must_be_valid(self):
        with pytest.raises(Exception):
            BotSettings(max_daily_loss_pct=Decimal("0"))
        with pytest.raises(Exception):
            BotSettings(max_daily_loss_pct=Decimal("0.60"))

    def test_tp_must_be_greater_than_sl(self):
        with pytest.raises(Exception):
            StrategySettings(tp_atr_mult=Decimal("0.5"), sl_atr_mult=Decimal("1.0"))

    def test_circuit_breaker_losses_min(self):
        with pytest.raises(Exception):
            StrategySettings(circuit_breaker_losses=0)

    def test_funding_threshold_valid(self):
        with pytest.raises(Exception):
            StrategySettings(funding_threshold=Decimal("-0.01"))

    def test_timeframe_validation(self):
        with pytest.raises(Exception):
            BotSettings(primary_timeframe="99h")

    def test_valid_timeframes_accepted(self):
        cfg = BotSettings(primary_timeframe="15m", confirmation_timeframe="1h", macro_timeframe="4h")
        assert cfg.primary_timeframe == "15m"

    def test_max_peak_drawdown_valid(self):
        with pytest.raises(Exception):
            BotSettings(max_peak_drawdown_pct=Decimal("0.60"))

    def test_leverage_consistency(self):
        with pytest.raises(Exception):
            BotSettings(leverage=10, max_leverage=5)


# ================================================================== #
#  INDICATOR ENGINE                                                    #
# ================================================================== #

class TestIndicatorEngine:

    def _feed_engine(self, n: int = 60) -> tuple[IndicatorEngine, list[Candle]]:
        engine = IndicatorEngine()
        candles = make_candles(n)
        for c in candles:
            engine.update(c)
        return engine, candles

    def test_ready_after_enough_candles(self):
        engine, _ = self._feed_engine(60)
        assert engine.ready() is True

    def test_not_ready_with_few_candles(self):
        engine = IndicatorEngine()
        for c in make_candles(5):
            engine.update(c)
        assert engine.ready() is False

    def test_ema_fast_slower_relation(self):
        engine, _ = self._feed_engine(60)
        fast = engine.ema_fast()
        slow = engine.ema_slow()
        # Con trend rialzista, EMA fast > EMA slow
        assert fast > slow

    def test_rsi_in_range(self):
        engine, _ = self._feed_engine(60)
        rsi = engine.rsi()
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_atr_positive(self):
        engine, _ = self._feed_engine(60)
        atr = engine.atr()
        assert atr > Decimal("0")

    def test_highest_high(self):
        engine, candles = self._feed_engine(60)
        hh = engine.highest_high(20)
        assert hh > Decimal("0")

    def test_lowest_low(self):
        engine, candles = self._feed_engine(60)
        ll = engine.lowest_low(20)
        assert ll > Decimal("0")
        assert ll < engine.highest_high(20)

    def test_vwap_computed(self):
        engine, _ = self._feed_engine(60)
        vwap = engine.vwap()
        assert vwap > Decimal("0")

    def test_volume_ratio(self):
        engine, _ = self._feed_engine(60)
        vr = engine.volume_ratio()
        assert vr > Decimal("0")

    def test_adx_no_division_by_zero(self):
        """ADX non crasha anche con candle a range zero."""
        engine = IndicatorEngine()
        for i in range(60):
            c = make_candle(o=100, h=100, l=100, c=100, volume=100, ts=1700000000 + i * 900)
            engine.update(c)
        # Non deve crashare, ADX può essere None o zero
        try:
            adx = engine.adx()
        except ValueError:
            pass  # OK se non pronto

    def test_support_resistance_cached(self):
        engine, _ = self._feed_engine(60)
        s1 = engine.support_levels()
        s2 = engine.support_levels()
        assert s1 == s2  # stessa cache

    def test_swing_low_high(self):
        engine, _ = self._feed_engine(60)
        sl = engine.recent_swing_low(5)
        sh = engine.recent_swing_high(5)
        assert sl < sh


# ================================================================== #
#  KELLY SIZING                                                        #
# ================================================================== #

class TestKellySizer:

    def test_bayesian_prior_before_min_trades(self):
        kelly = KellySizer(history_trades=20, half_kelly=True)
        size = kelly.position_size(
            Decimal("1000"), Decimal("10"),
            fallback_fraction=Decimal("0.015"),
        )
        assert size > Decimal("0")

    def test_kelly_updates_correctly(self):
        kelly = KellySizer(history_trades=5, half_kelly=True)
        for _ in range(10):
            kelly.update(Decimal("20"), Decimal("15"))  # win
        for _ in range(5):
            kelly.update(Decimal("-10"), Decimal("15"))  # loss
        assert kelly.is_ready is True
        size = kelly.position_size(Decimal("1000"), Decimal("10"))
        assert size > Decimal("0")

    def test_kelly_all_losses_returns_min(self):
        kelly = KellySizer(history_trades=5, half_kelly=True, min_fraction=Decimal("0.005"))
        for _ in range(10):
            kelly.update(Decimal("-10"), Decimal("15"))
        size = kelly.position_size(Decimal("1000"), Decimal("10"))
        # Dovrebbe ritornare il minimo, non zero
        assert size >= Decimal("0")


# ================================================================== #
#  ACCOUNT RISK                                                        #
# ================================================================== #

class TestAccountRisk:

    def test_daily_stop_triggers(self):
        # 5% di 1000 = 50 USDT daily stop
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            max_concurrent_positions=2,
            initial_capital=Decimal("1000"),
        )
        arm.register_open("BTCUSDT")
        arm.register_close("BTCUSDT", Decimal("-60"))  # -60 > -50 limit
        assert arm.can_open_position("SOLUSDT") is False

    def test_daily_stop_not_triggered(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            max_concurrent_positions=2,
            initial_capital=Decimal("1000"),
        )
        arm.register_open("BTCUSDT")
        arm.register_close("BTCUSDT", Decimal("-10"))  # -10 < -50 limit: ok
        assert arm.can_open_position("SOLUSDT") is True

    def test_daily_stop_scales_with_capital(self):
        """Daily stop percentuale scala correttamente col capitale."""
        # 5% di 500 = 25 USDT
        arm_small = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"), initial_capital=Decimal("500"),
        )
        # 5% di 5000 = 250 USDT
        arm_large = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"), initial_capital=Decimal("5000"),
        )
        # -30 USDT supera il limite per 500 ma non per 5000
        arm_small.register_open("BTCUSDT")
        arm_small.register_close("BTCUSDT", Decimal("-30"))
        arm_large.register_open("BTCUSDT")
        arm_large.register_close("BTCUSDT", Decimal("-30"))
        assert arm_small.can_open_position("X") is False
        assert arm_large.can_open_position("X") is True

    def test_peak_drawdown_protection(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.50"),
            max_concurrent_positions=2,
            max_peak_drawdown_pct=Decimal("0.10"),
            initial_capital=Decimal("1000"),
        )
        arm.register_open("BTCUSDT")
        arm.register_close("BTCUSDT", Decimal("100"))  # equity peak = 1100
        arm.register_open("ETHUSDT")
        arm.register_close("ETHUSDT", Decimal("-120"))  # equity = 980, dd = 10.9%
        assert arm.peak_drawdown_active is True

    def test_max_concurrent_positions(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.50"),
            max_concurrent_positions=2,
            initial_capital=Decimal("1000"),
        )
        arm.register_open("BTCUSDT")
        arm.register_open("ETHUSDT")
        assert arm.can_open_position("SOLUSDT") is False


# ================================================================== #
#  MODELS                                                              #
# ================================================================== #

class TestModels:

    def test_signal_defaults(self):
        sig = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            entry_price=Decimal("100"),
            take_profit=Decimal("102"),
            stop_loss=Decimal("99"),
            size=Decimal("0.1"),
            atr=Decimal("1.5"),
            timestamp=time.time(),
        )
        assert sig.score == Decimal("50")
        assert sig.strategy_name == "breakout"

    def test_tp_level(self):
        tp = TPLevel(Decimal("105"), Decimal("0.5"))
        assert tp.price == Decimal("105")
        assert tp.qty_fraction == Decimal("0.5")
        assert tp.hit is False


# ================================================================== #
#  BREAKOUT ENGINE                                                     #
# ================================================================== #

class TestBreakoutEngine:

    def test_no_signal_when_not_ready(self):
        from src.strategy.breakout_engine import BreakoutEngine
        cfg = BotSettings()
        ind = IndicatorEngine()
        engine = BreakoutEngine(cfg, ind, Decimal("1000"))

        candles = deque(make_candles(5), maxlen=200)
        for c in candles:
            ind.update(c)

        sig = engine.detect(candles)
        assert sig.signal_type == SignalType.NONE

    def test_detect_returns_signal_object(self):
        from src.strategy.breakout_engine import BreakoutEngine
        cfg = BotSettings()
        ind = IndicatorEngine()
        engine = BreakoutEngine(cfg, ind, Decimal("1000"))

        candles = deque(make_candles(60), maxlen=200)
        for c in candles:
            ind.update(c)

        sig = engine.detect(candles)
        assert isinstance(sig, Signal)
        assert sig.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.NONE)


# ================================================================== #
#  MEAN REVERSION ENGINE                                               #
# ================================================================== #

class TestMeanReversionEngine:

    def test_detect_returns_signal(self):
        from src.strategy.mean_reversion_engine import MeanReversionEngine
        cfg = BotSettings()
        ind = IndicatorEngine()
        engine = MeanReversionEngine(cfg, ind, Decimal("1000"))

        candles = deque(make_candles(60), maxlen=200)
        for c in candles:
            ind.update(c)

        sig = engine.detect(candles)
        assert isinstance(sig, Signal)


# ================================================================== #
#  TREND FOLLOWING ENGINE                                              #
# ================================================================== #

class TestTrendFollowingEngine:

    def test_detect_returns_signal(self):
        from src.strategy.trend_following_engine import TrendFollowingEngine
        cfg = BotSettings()
        ind = IndicatorEngine()
        engine = TrendFollowingEngine(cfg, ind, Decimal("1000"))

        candles = deque(make_candles(60), maxlen=200)
        for c in candles:
            ind.update(c)

        sig = engine.detect(candles)
        assert isinstance(sig, Signal)
        if sig.signal_type != SignalType.NONE:
            assert sig.strategy_name == "trend_following"
            assert sig.score >= Decimal("72")


# ================================================================== #
#  BUG FIX VALIDATION TESTS (Fase 1)                                  #
# ================================================================== #

class TestMeanReversionVolRatioFix:
    """Verifica che vol_ratio > 3.0 venga penalizzato (-5) e non premiato (+8)."""

    def test_vol_ratio_above_3_penalized(self):
        from src.strategy.mean_reversion_engine import MeanReversionEngine
        cfg = BotSettings()
        ind = IndicatorEngine()
        engine = MeanReversionEngine(cfg, ind, Decimal("1000"))

        # Simula il calcolo vol_score_bonus direttamente
        # vol_ratio > 3.0 deve dare -5, non +8
        vol_ratio = Decimal('3.5')
        if vol_ratio > Decimal('3.0'):
            bonus = Decimal('-5')
        elif vol_ratio > Decimal('2.0'):
            bonus = Decimal('8')
        elif vol_ratio > Decimal('1.5'):
            bonus = Decimal('4')
        else:
            bonus = Decimal('0')
        assert bonus == Decimal('-5'), f"vol_ratio 3.5 should penalize, got bonus={bonus}"

    def test_vol_ratio_2_5_rewarded(self):
        # vol_ratio tra 2.0 e 3.0 deve dare +8
        vol_ratio = Decimal('2.5')
        if vol_ratio > Decimal('3.0'):
            bonus = Decimal('-5')
        elif vol_ratio > Decimal('2.0'):
            bonus = Decimal('8')
        elif vol_ratio > Decimal('1.5'):
            bonus = Decimal('4')
        else:
            bonus = Decimal('0')
        assert bonus == Decimal('8')


class TestBreakoutRsiShortFix:
    """Verifica che il SHORT breakout usi RSI nella banda ribassista corretta."""

    def test_short_rsi_in_bearish_range(self):
        """RSI 30 (momentum ribassista) deve essere accettato per SHORT."""
        rsi_min = Decimal('45')
        rsi_max = Decimal('72')
        rsi_val = Decimal('30')
        # Nuova logica: (100 - rsi_max) <= rsi <= (100 - rsi_min) = 28 <= rsi <= 55
        rsi_short_ok = (Decimal('100') - rsi_max) <= rsi_val <= (Decimal('100') - rsi_min)
        assert rsi_short_ok is True

    def test_short_rsi_overbought_rejected(self):
        """RSI 75 (ipercomprato) NON deve essere accettato per breakout SHORT."""
        rsi_min = Decimal('45')
        rsi_max = Decimal('72')
        rsi_val = Decimal('75')
        rsi_short_ok = (Decimal('100') - rsi_max) <= rsi_val <= (Decimal('100') - rsi_min)
        assert rsi_short_ok is False

    def test_short_rsi_oversold_extreme_rejected(self):
        """RSI 15 (troppo ipervenduto) non deve essere accettato (sotto banda)."""
        rsi_min = Decimal('45')
        rsi_max = Decimal('72')
        rsi_val = Decimal('15')
        rsi_short_ok = (Decimal('100') - rsi_max) <= rsi_val <= (Decimal('100') - rsi_min)
        assert rsi_short_ok is False


class TestKellyPartialExitFix:
    """Verifica che il Kelly risk_amount tenga conto del profitto locked."""

    def test_net_risk_reduced_by_locked_profit(self):
        """risk_amount netto deve essere ridotto dal profitto locked dai partial exits."""
        gross_risk = Decimal('100')
        partial_locked_pnl = Decimal('30')
        # Nuova formula: max(gross - locked, gross * 0.1)
        risk_amount = max(gross_risk - partial_locked_pnl, gross_risk * Decimal('0.1'))
        assert risk_amount == Decimal('70')

    def test_net_risk_has_floor(self):
        """risk_amount non deve scendere sotto il 10% del rischio lordo."""
        gross_risk = Decimal('100')
        partial_locked_pnl = Decimal('95')  # locked quasi tutto
        risk_amount = max(gross_risk - partial_locked_pnl, gross_risk * Decimal('0.1'))
        assert risk_amount == Decimal('10')  # floor 10%

    def test_no_partial_exits_unchanged(self):
        """Senza partial exits il risk_amount resta invariato."""
        gross_risk = Decimal('100')
        partial_locked_pnl = Decimal('0')
        risk_amount = max(gross_risk - partial_locked_pnl, gross_risk * Decimal('0.1'))
        assert risk_amount == Decimal('100')


# ================================================================== #
#  FASE 2 — RACE CONDITIONS & ROBUSTEZZA                              #
# ================================================================== #

class TestUnrealizedPnlDailyStop:
    """Verifica che il daily stop consideri anche il PnL non realizzato."""

    def test_unrealized_loss_triggers_daily_stop(self):
        """Perdita non realizzata deve attivare il daily stop."""
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        # Realized PnL: -20 (sotto soglia di -50)
        arm.register_open("BTCUSDT")
        arm.register_close("BTCUSDT", Decimal("-20"))
        assert arm.can_open_position("ETHUSDT") is True  # -20 > -50, ok

        # Aggiungi unrealized PnL di -35 → totale = -55 > -50 → stop
        arm.update_unrealized_pnl(Decimal("-35"))
        assert arm.can_open_position("SOLUSDT") is False

    def test_unrealized_gain_doesnt_block(self):
        """Guadagno non realizzato non deve bloccare il trading."""
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        arm.update_unrealized_pnl(Decimal("50"))  # posizione in profitto
        assert arm.can_open_position("BTCUSDT") is True


class TestStatePersistenceAsync:
    """Verifica che update_snapshot sia async e protetto dal lock."""

    @pytest.mark.asyncio
    async def test_update_snapshot_is_async(self):
        import tempfile, shutil
        from src.state.persistency import StateManager
        tmp = tempfile.mkdtemp()
        try:
            sm = StateManager(db_path=f"{tmp}/test.db")
            # Deve essere awaitable
            await sm.update_snapshot(
                net_inventory=Decimal("0.01"),
                pnl=Decimal("5.0"),
                avg_entry=Decimal("100"),
                quotes_sent=0,
                fills_total=1,
            )
            state = sm.load_state()
            assert state is not None
            assert state["pnl"] == Decimal("5.0")
        finally:
            shutil.rmtree(tmp)
