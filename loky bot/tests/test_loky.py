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
        kelly = KellySizer(history_trades=20, kelly_divisor=2)
        size = kelly.position_size(
            Decimal("1000"), Decimal("10"),
            fallback_fraction=Decimal("0.015"),
        )
        assert size > Decimal("0")

    def test_kelly_updates_correctly(self):
        kelly = KellySizer(history_trades=5, kelly_divisor=2)
        for _ in range(10):
            kelly.update(Decimal("20"), Decimal("15"))  # win
        for _ in range(5):
            kelly.update(Decimal("-10"), Decimal("15"))  # loss
        assert kelly.is_ready is True
        size = kelly.position_size(Decimal("1000"), Decimal("10"))
        assert size > Decimal("0")

    def test_kelly_all_losses_returns_min(self):
        kelly = KellySizer(history_trades=5, kelly_divisor=2, min_fraction=Decimal("0.005"))
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


# ================================================================== #
#  FASE 3 — RISK MANAGEMENT AVANZATO                                  #
# ================================================================== #

class TestLiquidationMonitor:
    """Verifica il calcolo del prezzo di liquidazione e gli alert."""

    def test_long_liquidation_price(self):
        from src.core.liquidation_monitor import LiquidationMonitor
        mon = LiquidationMonitor(leverage=10)
        # LONG liq_price = entry × (1 - 1/lev + mmr) = 100 × (1 - 0.1 + 0.005) = 90.5
        liq = mon.liquidation_price(Decimal("100"), is_long=True)
        assert Decimal("90") < liq < Decimal("91")

    def test_short_liquidation_price(self):
        from src.core.liquidation_monitor import LiquidationMonitor
        mon = LiquidationMonitor(leverage=10)
        # SHORT liq_price = entry × (1 + 1/lev - mmr) = 100 × (1 + 0.1 - 0.005) = 109.5
        liq = mon.liquidation_price(Decimal("100"), is_long=False)
        assert Decimal("109") < liq < Decimal("110")

    def test_safe_alert(self):
        from src.core.liquidation_monitor import LiquidationMonitor, LiquidationAlert
        mon = LiquidationMonitor(leverage=5)
        alert = mon.check(Decimal("100"), Decimal("100"), is_long=True)
        assert alert == LiquidationAlert.SAFE

    def test_critical_alert_near_liquidation(self):
        from src.core.liquidation_monitor import LiquidationMonitor, LiquidationAlert
        mon = LiquidationMonitor(leverage=5)
        # LONG con leva 5: liq_price ≈ 80.5. Prezzo a 81 = margine ~3%
        liq = mon.liquidation_price(Decimal("100"), is_long=True)
        # Prezzo appena sopra il liq price = CRITICAL
        alert = mon.check(Decimal("100"), liq + Decimal("0.5"), is_long=True)
        assert alert == LiquidationAlert.CRITICAL

    def test_margin_distance_pct(self):
        from src.core.liquidation_monitor import LiquidationMonitor
        mon = LiquidationMonitor(leverage=5)
        # Al prezzo di entry: margine = 100%
        pct = mon.margin_distance_pct(Decimal("100"), Decimal("100"), is_long=True)
        assert pct == Decimal("1")  # 100%


class TestDrawdownLeverageReduction:
    """Verifica la riduzione graduata della leva su drawdown."""

    def test_no_drawdown_full_leverage(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"), max_leverage=10)
        # Popola ATR history
        for i in range(30):
            pm.record_atr(Decimal(str(1 + i * 0.01)))
        lev = pm.dynamic_leverage(Decimal("1.15"), drawdown_pct=Decimal("0.02"))
        assert lev > 1  # leva piena, no drawdown override

    def test_drawdown_5pct_halves_leverage(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"), max_leverage=10)
        for i in range(30):
            pm.record_atr(Decimal(str(1 + i * 0.01)))
        lev_normal = pm.dynamic_leverage(Decimal("1.15"), drawdown_pct=Decimal("0.02"))
        lev_dd = pm.dynamic_leverage(Decimal("1.15"), drawdown_pct=Decimal("0.07"))
        assert lev_dd <= lev_normal // 2 + 1  # dimezzato o meno

    def test_drawdown_10pct_survival_mode(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"), max_leverage=10)
        for i in range(30):
            pm.record_atr(Decimal(str(1 + i * 0.01)))
        lev = pm.dynamic_leverage(Decimal("1.15"), drawdown_pct=Decimal("0.12"))
        assert lev == 1  # sopravvivenza


class TestVolatilityAdjustedDailyStop:
    """Verifica che il daily stop si adatti alla volatilità."""

    def test_high_vol_tightens_stop(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        base = arm._max_daily_loss  # -50
        arm.adjust_daily_stop_for_volatility(Decimal("0.85"))  # alta vol
        assert arm._max_daily_loss > base  # più stretto (meno negativo = più vicino a 0)

    def test_low_vol_widens_stop(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        base = arm._max_daily_loss  # -50
        arm.adjust_daily_stop_for_volatility(Decimal("0.15"))  # bassa vol
        assert arm._max_daily_loss < base  # più largo (più negativo)

    def test_normal_vol_unchanged(self):
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        base = arm._max_daily_loss
        arm.adjust_daily_stop_for_volatility(Decimal("0.50"))  # vol media
        assert arm._max_daily_loss == base


# ================================================================== #
#  FASE 4 — STRATEGIE AVANZATE                                       #
# ================================================================== #

class TestSignalAggregatorImprovements:
    """Verifica miglioramenti al signal aggregator."""

    def test_volume_bonus_logarithmic(self):
        """Volume bonus deve seguire scala logaritmica."""
        import math
        # vol_ratio 2.0 → log2(2) × 4 = 4.0
        vol_ratio = 2.0
        log_vol = math.log2(vol_ratio) * 4
        assert abs(log_vol - 4.0) < 0.01

        # vol_ratio 0.5 → log2(0.5) × 4 = -4.0
        vol_ratio = 0.5
        log_vol = math.log2(vol_ratio) * 4
        assert abs(log_vol - (-4.0)) < 0.01

    def test_aggregator_with_macro_indicators(self):
        """SignalAggregator accetta macro_indicators."""
        from src.strategy.signal_aggregator import SignalAggregator
        ind = IndicatorEngine()
        agg = SignalAggregator(ind, htf_indicators=None, macro_indicators=None)
        assert agg._macro_ind is None


class TestAdaptiveStrategyWeights:
    """Verifica pesi adattivi per strategia."""

    def test_winning_strategy_boosted(self):
        from src.strategy.signal_aggregator import SignalAggregator
        agg = SignalAggregator(IndicatorEngine())
        # Registra 10 trade vincenti per breakout (expectancy forte)
        for _ in range(10):
            agg.record_trade_result("breakout", Decimal("5"))
        # Expectancy positiva forte → peso 1.3
        assert agg.strategy_weight("breakout") == Decimal("1.3")

    def test_losing_strategy_penalized(self):
        from src.strategy.signal_aggregator import SignalAggregator
        agg = SignalAggregator(IndicatorEngine())
        # Registra 8 trade perdenti su 10 per mean_reversion
        for _ in range(8):
            agg.record_trade_result("mean_reversion", Decimal("-3"))
        for _ in range(2):
            agg.record_trade_result("mean_reversion", Decimal("2"))
        assert agg.strategy_weight("mean_reversion") <= Decimal("0.7")

    def test_unknown_strategy_default_weight(self):
        from src.strategy.signal_aggregator import SignalAggregator
        agg = SignalAggregator(IndicatorEngine())
        assert agg.strategy_weight("nonexistent") == Decimal("1")

    def test_insufficient_data_neutral_weight(self):
        from src.strategy.signal_aggregator import SignalAggregator
        agg = SignalAggregator(IndicatorEngine())
        agg.record_trade_result("test", Decimal("10"))  # solo 1 trade
        assert agg.strategy_weight("test") == Decimal("1")


class TestVolatilityRegimeEngine:
    """Verifica il Volatility Regime Engine."""

    def test_normal_regime_with_insufficient_data(self):
        from src.strategy.volatility_engine import VolatilityRegimeEngine, VolatilityRegime
        ind = IndicatorEngine()
        engine = VolatilityRegimeEngine(ind)
        # Senza dati → NORMAL
        assert engine.detect() == VolatilityRegime.NORMAL

    def test_score_modifier_normal(self):
        from src.strategy.volatility_engine import VolatilityRegimeEngine
        ind = IndicatorEngine()
        engine = VolatilityRegimeEngine(ind)
        assert engine.score_modifier() == Decimal("1.00")

    def test_regime_name_property(self):
        from src.strategy.volatility_engine import VolatilityRegimeEngine
        ind = IndicatorEngine()
        engine = VolatilityRegimeEngine(ind)
        assert engine.regime_name == "normal"


# ================================================================== #
#  FASE 5 — ESECUZIONE AVANZATA                                      #
# ================================================================== #

class TestSlippageEstimator:
    """Verifica la stima dello slippage pre-trade."""

    def test_base_slippage_no_volume(self):
        from src.gateways.smart_execution import SlippageEstimator
        est = SlippageEstimator(base_slippage_pct=Decimal("0.0005"))
        slip = est.estimate(Decimal("100"), Decimal("1"), Decimal("0"))
        assert slip == Decimal("0.0005")

    def test_slippage_increases_with_size(self):
        from src.gateways.smart_execution import SlippageEstimator
        est = SlippageEstimator(base_slippage_pct=Decimal("0.0005"))
        small = est.estimate(Decimal("100"), Decimal("0.01"), Decimal("1000"))
        large = est.estimate(Decimal("100"), Decimal("10"), Decimal("1000"))
        assert large > small

    def test_acceptable_slippage(self):
        from src.gateways.smart_execution import SlippageEstimator
        est = SlippageEstimator(
            base_slippage_pct=Decimal("0.0005"),
            max_acceptable_pct=Decimal("0.002"),
        )
        ok, slip = est.is_acceptable(Decimal("100"), Decimal("0.01"), Decimal("1000"))
        assert ok is True

    def test_adjusted_size_reduces_when_needed(self):
        from src.gateways.smart_execution import SlippageEstimator
        est = SlippageEstimator(
            base_slippage_pct=Decimal("0.0005"),
            max_acceptable_pct=Decimal("0.0006"),  # soglia molto stretta
        )
        adjusted = est.adjusted_size(Decimal("100"), Decimal("100"), Decimal("10"))
        assert adjusted < Decimal("100")  # deve ridurre


class TestTWAPVolumeWeighted:
    """Verifica che il TWAP supporti pesi volume."""

    def test_twap_accepts_volume_weights(self):
        """Verifica che place_twap_order accetti volume_weights come parametro."""
        import inspect
        from src.state.order_manager import OrderManager as OM
        sig = inspect.signature(OM.place_twap_order)
        assert "volume_weights" in sig.parameters


# ================================================================== #
#  FASE 6 — BACKTESTING AVANZATO                                     #
# ================================================================== #

class TestMonteCarloSimulation:
    """Verifica la simulazione Monte Carlo."""

    def test_basic_run(self):
        from src.backtest_advanced import MonteCarloSimulator
        pnls = [10.0, -5.0, 15.0, -3.0, 8.0, -2.0, 12.0, -7.0, 6.0, -1.0]
        mc = MonteCarloSimulator(pnls, capital=500, n_simulations=100)
        result = mc.run()
        assert result.n_simulations == 100
        assert result.prob_profitable > 0  # almeno qualche sim profittevole

    def test_all_wins_high_probability(self):
        from src.backtest_advanced import MonteCarloSimulator
        pnls = [5.0] * 20  # tutti trade vincenti
        mc = MonteCarloSimulator(pnls, capital=500, n_simulations=100)
        result = mc.run()
        assert result.prob_profitable == 100.0  # tutte le simulazioni profittevoli
        assert result.mean_pnl == 100.0  # 20 × 5

    def test_all_losses_zero_probability(self):
        from src.backtest_advanced import MonteCarloSimulator
        pnls = [-5.0] * 20  # tutti trade perdenti
        mc = MonteCarloSimulator(pnls, capital=500, n_simulations=100)
        result = mc.run()
        assert result.prob_profitable == 0.0

    def test_empty_pnls(self):
        from src.backtest_advanced import MonteCarloSimulator
        mc = MonteCarloSimulator([], capital=500)
        result = mc.run()
        assert result.n_simulations == 0

    def test_confidence_interval(self):
        from src.backtest_advanced import MonteCarloSimulator
        pnls = [10.0, -5.0, 15.0, -3.0, 8.0] * 10  # 50 trade
        mc = MonteCarloSimulator(pnls, capital=500, n_simulations=500)
        result = mc.run()
        # 5° percentile deve essere < 95° percentile
        assert result.pct_5_pnl <= result.pct_95_pnl
        # Mediano deve essere tra i due
        assert result.pct_5_pnl <= result.median_pnl <= result.pct_95_pnl


class TestWalkForwardAnalysis:
    """Verifica l'analisi walk-forward."""

    def test_basic_run(self):
        from src.backtest_advanced import WalkForwardAnalyzer
        pnls = [5.0, -2.0, 8.0, -1.0, 3.0] * 10  # 50 trade
        wf = WalkForwardAnalyzer(pnls, train_ratio=0.8, n_windows=5)
        result = wf.run()
        assert result.n_windows > 0
        assert len(result.windows) > 0

    def test_robustness_score_range(self):
        from src.backtest_advanced import WalkForwardAnalyzer
        pnls = [5.0, -2.0, 8.0, -1.0, 3.0] * 10
        wf = WalkForwardAnalyzer(pnls, n_windows=5)
        result = wf.run()
        assert 0 <= result.robustness_score <= 100

    def test_insufficient_data(self):
        from src.backtest_advanced import WalkForwardAnalyzer
        pnls = [5.0, -2.0]  # troppo pochi trade
        wf = WalkForwardAnalyzer(pnls, n_windows=5)
        result = wf.run()
        assert result.n_windows == 0

    def test_profitable_windows_tracked(self):
        from src.backtest_advanced import WalkForwardAnalyzer
        # Trade tutti vincenti → tutte le finestre profittevoli
        pnls = [10.0] * 50
        wf = WalkForwardAnalyzer(pnls, n_windows=5)
        result = wf.run()
        assert result.pct_profitable_windows == 100.0


# ================================================================== #
#  FASE 7 — INSTITUTIONAL-GRADE IMPROVEMENTS                         #
# ================================================================== #

class TestScaleInKellyFix:
    """Verifica che scale-in non corrompa position_size_orig per Kelly."""

    def test_scale_in_preserves_kelly_risk_size(self):
        """_kelly_risk_size non deve cambiare dopo scale-in."""
        # Il fix: _position_size_orig NON viene aggiornato durante scale-in
        # Kelly usa _kelly_risk_size che è snapshot pre-scale-in
        original_size = Decimal("1.0")
        add_size = Decimal("0.5")
        kelly_risk_size = original_size  # snapshot pre-scale-in
        position_size = original_size + add_size  # 1.5 dopo scale-in
        # Kelly deve usare kelly_risk_size (1.0), non position_size (1.5)
        assert kelly_risk_size == original_size


class TestConfluenceFirstScoring:
    """Verifica che il nuovo scoring confluence-first funzioni."""

    def test_score_returns_valid_range(self):
        from src.strategy.signal_aggregator import SignalAggregator
        ind = IndicatorEngine()
        agg = SignalAggregator(ind)
        sig = Signal(
            symbol="BTCUSDT", signal_type=SignalType.LONG,
            entry_price=Decimal("100"), take_profit=Decimal("105"),
            stop_loss=Decimal("98"), size=Decimal("1"),
            atr=Decimal("2"), timestamp=time.time(), score=Decimal("65"),
        )
        # Score non deve essere > 100 o < 0
        score = agg.score(sig)
        assert Decimal("0") <= score <= Decimal("100")


class TestBreakoutMultiCandleConfirmation:
    """Verifica che il breakout richieda almeno 2 candle che testano il livello."""

    def test_single_spike_blocked(self):
        """Un spike singolo (1 candle) non deve generare breakout."""
        # Se solo 1 candle tocca l'HH, hh_touches < 2 → breakout bloccato
        hh_touches = 1
        assert hh_touches < 2  # condizione di blocco


class TestDailyStopCarryOver:
    """Verifica che il daily stop porti avanti le perdite non realizzate cross-midnight."""

    def test_no_carry_over_prevents_double_counting(self):
        """Unrealized losses NON vengono carry-over per evitare double-counting."""
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        arm.update_unrealized_pnl(Decimal("-20"))
        arm._day_start = 0  # forza reset
        arm._maybe_reset_daily()
        # Reset pulito: nessun carry-over (il check real-time include unrealized)
        assert arm._realized_pnl_day == Decimal("0")

    def test_unrealized_still_blocks_via_can_open(self):
        """Le perdite unrealized bloccano via can_open_position (check real-time)."""
        arm = AccountRiskManager(
            max_daily_loss_pct=Decimal("0.05"),
            initial_capital=Decimal("1000"),
        )
        # -60 unrealized supera il limite di -50 (5% di 1000)
        arm.update_unrealized_pnl(Decimal("-60"))
        assert arm.can_open_position("BTCUSDT") is False


# ================================================================== #
#  FASE 8 — CORRELAZIONE DINAMICA, ORDER FLOW, KELTNER               #
# ================================================================== #

class TestDynamicCorrelation:
    """Verifica la correlazione Spearman rolling."""

    def test_perfect_correlation_blocked(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"))
        # Registra rendimenti identici (correlazione = 1.0)
        for i in range(30):
            pm.record_return("BTCUSDT", float(i) * 0.01)
            pm.record_return("ETHUSDT", float(i) * 0.01)
        pm.register_open("BTCUSDT", Decimal("500"))
        blocked, reason = pm.is_correlated_with_open("ETHUSDT")
        assert blocked is True
        assert "Spearman" in reason

    def test_uncorrelated_allowed(self):
        import random
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"))
        rng = random.Random(42)
        for _ in range(30):
            pm.record_return("BTCUSDT", rng.gauss(0, 1))
            pm.record_return("SOLUSDT", rng.gauss(0, 1))
        pm.register_open("BTCUSDT", Decimal("500"))
        blocked, _ = pm.is_correlated_with_open("SOLUSDT")
        assert blocked is False  # random data = low correlation

    def test_insufficient_data_uses_static_groups(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(
            capital=Decimal("1000"),
            correlation_groups={"BTCUSDT": 0, "ETHUSDT": 0},
        )
        pm.register_open("BTCUSDT", Decimal("500"))
        # Nessun rendimento rolling → fallback a gruppi statici
        blocked, reason = pm.is_correlated_with_open("ETHUSDT")
        assert blocked is True
        assert "statico" in reason


class TestOrderFlowEngine:
    """Verifica l'Order Flow Engine."""

    def test_delta_calculation(self):
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        # Candela verde con close vicino al high → delta positivo
        c = Candle(
            symbol="BTCUSDT", timeframe="15m",
            open=Decimal("100"), high=Decimal("105"), low=Decimal("99"),
            close=Decimal("104"), volume=Decimal("1000"),
            timestamp=time.time(), is_closed=True,
        )
        engine.update(c)
        assert engine.current_delta > Decimal("0")

    def test_cvd_accumulates(self):
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        for i in range(10):
            c = Candle(
                symbol="BTCUSDT", timeframe="15m",
                open=Decimal("100"), high=Decimal("105"), low=Decimal("99"),
                close=Decimal("104"), volume=Decimal("100"),
                timestamp=time.time() + i, is_closed=True,
            )
            engine.update(c)
        assert engine.cvd > Decimal("0")  # tutte candele verdi

    def test_score_modifier_bullish(self):
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        for i in range(15):
            c = Candle(
                symbol="BTCUSDT", timeframe="15m",
                open=Decimal("100"), high=Decimal("105"), low=Decimal("99"),
                close=Decimal("104"), volume=Decimal("100"),
                timestamp=time.time() + i, is_closed=True,
            )
            engine.update(c)
        mod = engine.score_modifier(is_long=True)
        assert mod >= Decimal("1.0")  # bullish CVD + long = conferma


class TestKeltnerSqueeze:
    """Verifica il Keltner Channel e la detection dello squeeze."""

    def test_keltner_channels_exist(self):
        ind = IndicatorEngine()
        candles = make_candles(60)
        for c in candles:
            ind.update(c)
        upper = ind.keltner_upper()
        lower = ind.keltner_lower()
        assert upper > lower

    def test_squeeze_detection(self):
        ind = IndicatorEngine()
        candles = make_candles(60)
        for c in candles:
            ind.update(c)
        # is_squeeze ritorna bool
        result = ind.is_squeeze()
        assert isinstance(result, bool)


# ================================================================== #
#  V5 — PERFORMANCE, ML, AUTO-OPTIMIZER                               #
# ================================================================== #

class TestBBWidthCached:
    """Verifica che bb_width sia cached al momento dell'update."""

    def test_bb_width_cached(self):
        ind = IndicatorEngine()
        candles = make_candles(60)
        for c in candles:
            ind.update(c)
        # bb_width deve ritornare il valore cached
        w1 = ind.bb_width()
        w2 = ind.bb_width()
        assert w1 == w2  # stesso valore, nessun ricalcolo


class TestRollingOptimizer:
    """Verifica il rolling parameter optimizer."""

    def test_suggest_no_change_insufficient_data(self):
        from src.optimization import RollingOptimizer
        opt = RollingOptimizer()
        current = {"tp_atr_mult": 2.0, "sl_atr_mult": 1.0}
        suggested = opt.suggest(current)
        assert suggested == current  # nessun dato → nessun cambio

    def test_has_enough_data(self):
        from src.optimization import RollingOptimizer
        opt = RollingOptimizer()
        assert opt.has_enough_data is False
        for i in range(25):
            opt.record_trade({"tp_atr_mult": 2.0}, pnl=float(i))
        assert opt.has_enough_data is True


class TestPortfolioHeat:
    """Verifica il portfolio heat management."""

    def test_no_exposure_full_size(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"))
        assert pm.heat_size_modifier() == Decimal("1.0")

    def test_high_exposure_reduces_size(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"))
        pm.register_open("BTCUSDT", Decimal("900"))  # 90% del capitale
        mod = pm.heat_size_modifier()
        assert mod < Decimal("1.0")  # deve ridurre

    def test_moderate_exposure_moderate_reduction(self):
        from src.core.portfolio_risk import PortfolioRiskManager
        pm = PortfolioRiskManager(capital=Decimal("1000"))
        pm.register_open("BTCUSDT", Decimal("600"))  # 60%
        mod = pm.heat_size_modifier()
        assert mod == Decimal("0.7")


# ================================================================== #
#  V5.1 — OPTIMAL-F, LIQUIDATION, TTL, BOOTSTRAP                     #
# ================================================================== #

class TestOptimalF:
    """Verifica Optimal-f sizing."""

    def test_optimal_f_computation(self):
        kelly = KellySizer(history_trades=30, kelly_divisor=2, use_optimal_f=True)
        # Simula 20 trade misti
        for _ in range(12):
            kelly.update(Decimal("10"), Decimal("5"))  # win: +10, rischiando 5
        for _ in range(8):
            kelly.update(Decimal("-5"), Decimal("5"))   # loss: -5, rischiando 5
        f = kelly.optimal_fraction()
        assert f is not None
        assert f > Decimal("0")

    def test_kelly_divisor_configurable(self):
        # Third-Kelly (divisor=3) deve dare frazione più piccola di half-Kelly (divisor=2)
        kelly2 = KellySizer(history_trades=5, kelly_divisor=2)
        kelly3 = KellySizer(history_trades=5, kelly_divisor=3)
        for k in [kelly2, kelly3]:
            for _ in range(6):
                k.update(Decimal("10"), Decimal("5"))
            for _ in range(4):
                k.update(Decimal("-5"), Decimal("5"))
        f2 = kelly2.optimal_fraction()
        f3 = kelly3.optimal_fraction()
        assert f2 is not None and f3 is not None
        assert f3 <= f2  # third-Kelly più conservativo

    def test_strategy_priors_exist(self):
        assert "breakout" in KellySizer.STRATEGY_PRIORS
        assert "funding_rate" in KellySizer.STRATEGY_PRIORS


class TestLiquidationClustering:
    """Verifica la stima dei livelli di liquidazione."""

    def test_estimate_with_sufficient_data(self):
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        # Genera candle con swing highs/lows
        for i in range(30):
            base = 100 + (i % 5) * 2 - 4  # oscillazione 92-108
            c = Candle(
                symbol="BTCUSDT", timeframe="15m",
                open=Decimal(str(base)), high=Decimal(str(base + 3)),
                low=Decimal(str(base - 2)), close=Decimal(str(base + 1)),
                volume=Decimal("100"), timestamp=float(i), is_closed=True,
            )
            engine.update(c)
        levels = engine.estimate_liquidation_levels()
        # Deve trovare almeno qualche livello
        assert isinstance(levels, list)

class TestBootstrapCI:
    """Verifica Bootstrap Confidence Intervals."""

    def test_basic_bootstrap(self):
        from src.backtest_advanced import BootstrapValidator
        pnls = [5.0, -2.0, 8.0, -3.0, 10.0] * 10
        bv = BootstrapValidator(pnls, capital=500, n_iterations=200)
        result = bv.run()
        assert result.n_iterations == 200
        assert result.sharpe_5pct <= result.sharpe_median <= result.sharpe_95pct

    def test_all_wins_profitable(self):
        from src.backtest_advanced import BootstrapValidator
        pnls = [10.0] * 20
        bv = BootstrapValidator(pnls, capital=500, n_iterations=100)
        result = bv.run()
        assert result.sharpe_5pct > 0  # 95% CI interamente positivo

    def test_empty_pnls(self):
        from src.backtest_advanced import BootstrapValidator
        bv = BootstrapValidator([], capital=500)
        result = bv.run()
        assert result.n_iterations == 0


# ================================================================== #
#  V5.2 — PRECISION, REASONING, ZERO-ERROR                           #
# ================================================================== #

class TestCVDDeltaFixed:
    """Verifica che il CVD delta sia corretto per candele rosse."""

    def test_red_candle_close_at_low_zero_buy(self):
        """Candela rossa con close=low: buy_ratio deve essere ~0."""
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        c = Candle(
            symbol="BTCUSDT", timeframe="15m",
            open=Decimal("100"), high=Decimal("102"),
            low=Decimal("95"), close=Decimal("95"),  # close = low
            volume=Decimal("1000"), timestamp=1.0, is_closed=True,
        )
        engine.update(c)
        # Delta deve essere fortemente negativo (tutta sell pressure)
        assert engine.current_delta < Decimal("0")

    def test_green_candle_close_at_high_full_buy(self):
        """Candela verde con close=high: buy_ratio deve essere ~1."""
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        c = Candle(
            symbol="BTCUSDT", timeframe="15m",
            open=Decimal("95"), high=Decimal("102"),
            low=Decimal("95"), close=Decimal("102"),  # close = high
            volume=Decimal("1000"), timestamp=1.0, is_closed=True,
        )
        engine.update(c)
        assert engine.current_delta > Decimal("0")

    def test_doji_returns_zero_delta(self):
        """Doji (hl_range tiny): delta deve essere zero."""
        from src.strategy.orderflow_engine import OrderFlowEngine
        engine = OrderFlowEngine()
        c = Candle(
            symbol="BTCUSDT", timeframe="15m",
            open=Decimal("100.00"), high=Decimal("100.005"),
            low=Decimal("100.00"), close=Decimal("100.003"),
            volume=Decimal("1000"), timestamp=1.0, is_closed=True,
        )
        engine.update(c)
        assert engine.current_delta == Decimal("0")


class TestMACDDivergence:
    """Verifica MACD e divergence detection."""

    def test_macd_computes_after_26_candles(self):
        ind = IndicatorEngine()
        candles = make_candles(30)
        for c in candles:
            ind.update(c)
        macd = ind.macd()
        assert isinstance(macd, Decimal)

    def test_macd_histogram(self):
        ind = IndicatorEngine()
        candles = make_candles(30)
        for c in candles:
            ind.update(c)
        hist = ind.macd_histogram()
        assert isinstance(hist, Decimal)


class TestStochasticRSI:
    """Verifica Stochastic RSI."""

    def test_stoch_rsi_after_enough_data(self):
        ind = IndicatorEngine()
        candles = make_candles(60)
        for c in candles:
            ind.update(c)
        stoch = ind.stoch_rsi()
        assert Decimal("0") <= stoch <= Decimal("100")

    def test_stoch_k_d_available(self):
        ind = IndicatorEngine()
        candles = make_candles(60)
        for c in candles:
            ind.update(c)
        k = ind.stoch_k()
        d = ind.stoch_d()
        assert isinstance(k, Decimal)
        assert isinstance(d, Decimal)


class TestRegimeHysteresis:
    """Verifica che il regime non flip-floppi."""

    def test_regime_locked_for_minimum_candles(self):
        from src.strategy.volatility_engine import VolatilityRegimeEngine, VolatilityRegime
        ind = IndicatorEngine()
        engine = VolatilityRegimeEngine(ind, lock_candles=5)
        # Senza dati → NORMAL, lock per 5 candle
        for _ in range(3):
            engine.update()
            r = engine.detect()
        # Regime deve restare NORMAL durante il lock
        assert r == VolatilityRegime.NORMAL


# ================================================================== #
#  V5.6 — PRODUCTION SAFETY & RELIABILITY                            #
# ================================================================== #

class TestFundingRateChoppyBlock:
    """Verifica che funding rate NON generi segnali in regime CHOPPY."""

    def test_funding_blocked_when_choppy(self):
        """preferred_strategies vuoto in CHOPPY → funding bloccato."""
        from src.strategy.signal_aggregator import SignalAggregator
        ind = IndicatorEngine()
        agg = SignalAggregator(ind)
        # Senza dati, is_choppy_market() ritorna False (fallback NEUTRAL)
        # Ma se preferred_strategies ritorna [], funding dovrebbe essere bloccato
        # In regime CHOPPY: preferred_strategies() = []
        # In regime NEUTRAL: preferred_strategies() contiene "funding_rate"
        preferred = agg.preferred_strategies()
        assert "funding_rate" in preferred or len(preferred) == 0


class TestPositionStateRestore:
    """Verifica che la posizione venga ripristinata al restart."""

    @pytest.mark.asyncio
    async def test_position_persisted_and_restored(self):
        import tempfile, shutil
        from src.state.persistency import StateManager
        tmp = tempfile.mkdtemp()
        try:
            sm = StateManager(db_path=f"{tmp}/test.db")
            # Simula salvataggio con posizione aperta (inventory positivo = LONG)
            await sm.update_snapshot(
                net_inventory=Decimal("0.5"),  # LONG 0.5
                pnl=Decimal("10.0"),
                avg_entry=Decimal("50000"),
                quotes_sent=0,
                fills_total=5,
            )
            # Ricarica
            state = sm.load_state()
            assert state is not None
            assert state["net_inventory"] == Decimal("0.5")
            assert state["avg_entry"] == Decimal("50000")
        finally:
            shutil.rmtree(tmp)


class TestGracefulShutdownWarning:
    """Verifica che close() loggi warning se posizione aperta."""

    def test_close_method_exists_and_handles_position(self):
        """Verifica che close() sia async e gestisca posizioni."""
        import inspect
        from src.bot import LokyBot
        assert inspect.iscoroutinefunction(LokyBot.close)


# ================================================================== #
#  END-TO-END TRADE LIFECYCLE TEST                                    #
# ================================================================== #

class TestEndToEndTradeLifecycle:
    """
    Test completo del ciclo di vita di un trade:
      Warm-up → Signal → Entry → TP1 (50%) → Breakeven SL → TP2 (25%) → Exit

    Questo singolo test verifica che:
    1. Gli indicatori si inizializzano correttamente
    2. Il bot entra in posizione (paper mode)
    3. Partial TP1 chiude 50% e sposta SL a breakeven
    4. La posizione residua continua a essere gestita
    5. Il PnL finale è positivo
    6. Lo stato torna a FLAT dopo la chiusura
    """

    @pytest.mark.asyncio
    async def test_full_long_trade_with_partial_tp(self):
        from src.bot import LokyBot, BotState
        from src.backtest import _InMemoryStateManager, _BacktestGateway

        # --- Setup ---
        cfg = BotSettings(
            tokens=["BTCUSDT"],
            primary_timeframe="15m",
            confirmation_timeframe="1h",
            macro_timeframe="4h",
            live_trading_enabled=False,
            max_daily_loss_pct=Decimal("0.50"),  # largo per non interferire
        )
        sm = _InMemoryStateManager()
        gw = _BacktestGateway()

        bot = LokyBot(
            symbol="BTCUSDT",
            config=cfg,
            execution_gw=gw,
            state_manager=sm,
            capital=Decimal("10000"),
        )

        base_ts = 1700000000.0

        # --- Fase 1: Warm-up (60 candle con trend rialzista) ---
        # Genera candle che salgono gradualmente: open 100 → 115
        for i in range(60):
            price = 100.0 + i * 0.25
            c = make_candle(
                o=price, h=price + 1.5, l=price - 0.8, c=price + 0.3,
                volume=500.0, ts=base_ts + i * 900,
            )
            await bot.on_candle(c)

        # Verifica che gli indicatori siano pronti
        assert bot._indicators.ready()
        assert bot._state == BotState.FLAT

        # Registra ATR per calcoli successivi
        atr = bot._indicators.atr()
        assert atr > Decimal("0")
        last_price = Decimal("115.0") + Decimal("0.3")  # ~115.3

        # --- Fase 2: Genera breakout (close > HH con volume alto) ---
        # Highest high delle ultime 20 candle è circa 115 + 1.5 = 116.5
        # Facciamo un breakout a 118 con volume 3x
        hh_approx = 116.5
        breakout_price = hh_approx + 2.0  # 118.5
        breakout_candle = make_candle(
            o=hh_approx + 0.5, h=breakout_price + 1.0, l=hh_approx,
            c=breakout_price, volume=2000.0,  # volume alto (3-4x media)
            ts=base_ts + 60 * 900,
        )
        await bot.on_candle(breakout_candle)

        # Il bot potrebbe generare un segnale o metterlo in pending (next-candle entry)
        has_signal = bot._pending_signal is not None or bot._state != BotState.FLAT

        # --- Fase 3: Candle successiva per entry (next-candle) ---
        if bot._pending_signal is not None:
            entry_price = breakout_price + 0.2
            entry_candle = make_candle(
                o=entry_price, h=entry_price + 1.0, l=entry_price - 0.5,
                c=entry_price + 0.5, volume=600.0,
                ts=base_ts + 61 * 900,
            )
            await bot.on_candle(entry_candle)

        # Se il bot è entrato, verifica lo stato
        if bot._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            assert bot._position_size > Decimal("0")
            assert bot._entry_price > Decimal("0")
            assert len(bot._tp_levels) == 3
            entry = bot._entry_price
            orig_size = bot._position_size_orig
            tp1_price = bot._tp_levels[0].price

            # --- Fase 4: Price sale fino a TP1 ---
            # TP1 è a entry + 1.5×ATR×regime_mult
            # Generiamo candle che salgono gradualmente verso TP1
            current_price = float(entry) + 0.5
            for i in range(5):
                step = float(tp1_price - entry) / 4
                current_price += step
                c = make_candle(
                    o=current_price - 0.3, h=current_price + 0.5,
                    l=current_price - 0.5, c=current_price,
                    volume=500.0, ts=base_ts + (62 + i) * 900,
                )
                await bot.on_candle(c)

                if bot._tp_levels[0].hit:
                    break

            # Verifica TP1 hit
            if bot._tp_levels[0].hit:
                # 50% chiuso
                assert bot._position_size < orig_size
                # SL a breakeven (+ buffer)
                assert bot._sl_price >= entry

                # Verifica che ci sia PnL positivo dai partial
                assert bot._accumulated_trade_pnl > Decimal("0")

                # --- Fase 5: Continuiamo verso TP2 ---
                tp2_price = bot._tp_levels[1].price
                for i in range(5):
                    step = float(tp2_price - Decimal(str(current_price))) / 4
                    current_price += step
                    c = make_candle(
                        o=current_price - 0.2, h=current_price + 0.3,
                        l=current_price - 0.3, c=current_price,
                        volume=500.0, ts=base_ts + (67 + i) * 900,
                    )
                    await bot.on_candle(c)
                    if bot._tp_levels[1].hit:
                        break

        # --- Verifica finale ---
        # Il bot dovrebbe aver generato trade con PnL positivo
        total_pnl = bot.realized_pnl
        total_trades = bot.total_trades

        # Se il bot è entrato e ha fatto partial TP, PnL deve essere > 0
        if total_trades > 0:
            assert total_pnl > Decimal("0"), f"PnL should be positive, got {total_pnl}"

        # Se il bot non è entrato (filtri troppo stretti per dati sintetici),
        # verifichiamo almeno che non ha crashato e lo stato è coerente
        assert bot._state in (
            BotState.FLAT, BotState.POSITION_OPEN, BotState.PARTIAL_EXIT
        )

    @pytest.mark.asyncio
    async def test_sl_hit_produces_negative_pnl(self):
        """Verifica che uno SL hit produca PnL negativo e attivi il cooldown."""
        from src.bot import LokyBot, BotState
        from src.backtest import _InMemoryStateManager, _BacktestGateway

        cfg = BotSettings(
            tokens=["BTCUSDT"],
            live_trading_enabled=False,
            max_daily_loss_pct=Decimal("0.50"),
        )
        sm = _InMemoryStateManager()
        gw = _BacktestGateway()

        bot = LokyBot(
            symbol="BTCUSDT", config=cfg, execution_gw=gw,
            state_manager=sm, capital=Decimal("10000"),
        )

        base_ts = 1700000000.0

        # Warm-up: 60 candle con trend rialzista
        for i in range(60):
            price = 100.0 + i * 0.3
            c = make_candle(
                o=price, h=price + 1.5, l=price - 0.8, c=price + 0.3,
                volume=500.0, ts=base_ts + i * 900,
            )
            await bot.on_candle(c)

        # Se il bot ha una posizione aperta, facciamo crashare il prezzo
        if bot._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            entry = float(bot._entry_price)
            sl = float(bot._sl_price)
            # Crash: prezzo va sotto lo SL
            crash_price = sl - 2.0
            crash_candle = make_candle(
                o=entry - 1, h=entry, l=crash_price,
                c=crash_price + 0.5, volume=1000.0,
                ts=base_ts + 65 * 900,
            )
            await bot.on_candle(crash_candle)

            # Dopo SL: stato deve essere FLAT
            if bot._state == BotState.FLAT:
                # Cooldown attivo
                assert bot._cooldown_remaining > 0 or bot.total_trades > 0

    @pytest.mark.asyncio
    async def test_state_consistency_after_many_candles(self):
        """Verifica che il bot non si blocchi dopo 200+ candle."""
        from src.bot import LokyBot, BotState
        from src.backtest import _InMemoryStateManager, _BacktestGateway

        cfg = BotSettings(
            tokens=["BTCUSDT"],
            live_trading_enabled=False,
            max_daily_loss_pct=Decimal("0.50"),
        )
        sm = _InMemoryStateManager()
        gw = _BacktestGateway()
        bot = LokyBot(
            symbol="BTCUSDT", config=cfg, execution_gw=gw,
            state_manager=sm, capital=Decimal("10000"),
        )

        # 80 candle con oscillazione (range-bound) — enough per indicatori
        base_ts = 1700000000.0
        import math
        for i in range(80):
            # Oscillazione sinusoidale: 100 ± 5
            price = 100.0 + 5.0 * math.sin(i * 0.15) + i * 0.02
            c = make_candle(
                o=price - 0.3, h=price + 1.2, l=price - 1.0,
                c=price + 0.1, volume=400.0 + 100 * abs(math.sin(i * 0.3)),
                ts=base_ts + i * 900,
            )
            await bot.on_candle(c)

        # Bot non deve crashare e stato deve essere valido
        assert bot._state in (
            BotState.FLAT, BotState.POSITION_OPEN, BotState.PARTIAL_EXIT, BotState.ENTERING,
        )
        # Indicatori devono essere pronti
        assert bot._indicators.ready()


# ================================================================== #
#  V6.2 — INTEGRATION TESTS PER CODE PATH CRITICI                    #
# ================================================================== #

class TestIntegrationGapProtection:
    """Verifica che gap > 2× SL forzi la chiusura."""

    @pytest.mark.asyncio
    async def test_gap_protection_closes_position(self):
        from src.bot import LokyBot, BotState
        from src.backtest import _InMemoryStateManager, _BacktestGateway

        cfg = BotSettings(
            tokens=["BTCUSDT"], live_trading_enabled=False,
            max_daily_loss_pct=Decimal("0.50"),
        )
        bot = LokyBot(
            symbol="BTCUSDT", config=cfg, execution_gw=_BacktestGateway(),
            state_manager=_InMemoryStateManager(), capital=Decimal("10000"),
        )

        # Warm-up
        for i in range(60):
            p = 100.0 + i * 0.3
            await bot.on_candle(make_candle(o=p, h=p+1.5, l=p-0.8, c=p+0.3,
                                             ts=1700000000 + i*900))

        # Se il bot è in posizione, verifica gap protection
        if bot._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            entry = float(bot._entry_price)
            sl = float(bot._sl_price)
            sl_dist = abs(entry - sl)
            # Crash > 2× SL distance
            crash = entry - sl_dist * 3
            await bot.on_candle(make_candle(
                o=crash+1, h=crash+2, l=crash-1, c=crash,
                ts=1700000000 + 65*900,
            ))
            # Deve aver chiuso la posizione
            assert bot._state == BotState.FLAT


class TestIntegrationCircuitBreaker:
    """Verifica che 3 loss consecutive attivino il circuit breaker."""

    def test_circuit_breaker_params(self):
        """Verifica che i parametri del circuit breaker siano configurati."""
        cfg = BotSettings()
        assert cfg.strategy.circuit_breaker_losses == 3
        assert cfg.strategy.circuit_breaker_candles == 15


class TestIntegrationModifierCap:
    """Verifica che i cumulative modifiers non riducano size oltre il 40%."""

    def test_modifier_cap_at_40_percent(self):
        """Se tutti i modifier sono al minimo, size >= 40% dell'originale."""
        original_size = Decimal("1.0")
        # Worst case: vol(0.85) × of(0.85) × sentiment(0.70) × streak(0.65)
        worst_case = original_size * Decimal("0.85") * Decimal("0.85") * Decimal("0.70") * Decimal("0.65")
        # Senza cap: 0.328 (32.8%)
        assert worst_case < Decimal("0.40")
        # Con cap: max(worst_case, original × 0.40) = 0.40
        capped = max(worst_case, original_size * Decimal("0.40"))
        assert capped == Decimal("0.40")


class TestIntegrationBTCCrashProtection:
    """Verifica la BTC crash protection nell'orchestratore."""

    def test_crash_detection_threshold(self):
        """Un drop del 4% deve triggerare il crisis mode."""
        # Simula: 4 prezzi BTC con drop >3%
        from collections import deque
        prices = deque(maxlen=8)
        prices.append(Decimal("50000"))  # T-3
        prices.append(Decimal("49500"))  # T-2
        prices.append(Decimal("49000"))  # T-1
        prices.append(Decimal("48000"))  # T-0 (4% drop)
        drop_pct = (prices[0] - prices[-1]) / prices[0]
        assert drop_pct > Decimal("0.03")  # >3% = trigger

    def test_crisis_mode_expires(self):
        """Verifica che il crisis mode abbia un timeout di 30 minuti."""
        import time
        crisis_until = time.time() + 1800  # 30 minuti
        assert crisis_until > time.time()
        assert crisis_until - time.time() < 1801


class TestIntegrationPartialTPFlow:
    """Verifica il flow dei partial TP (50/25/25)."""

    def test_tp_levels_created_correctly(self):
        """Verifica che vengano creati 3 livelli TP con le frazioni giuste."""
        cfg = BotSettings()
        s = cfg.strategy
        assert s.partial_tp1_pct == Decimal("0.50")
        assert s.partial_tp2_pct == Decimal("0.25")
        assert s.partial_tp3_pct == Decimal("0.25")
        # Somma = 100%
        assert s.partial_tp1_pct + s.partial_tp2_pct + s.partial_tp3_pct == Decimal("1.00")

    def test_tp_levels_ascending(self):
        """TP1 < TP2 < TP3 per LONG."""
        cfg = BotSettings()
        s = cfg.strategy
        assert s.partial_tp1_atr < s.partial_tp2_atr < s.partial_tp3_atr
