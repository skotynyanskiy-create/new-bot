"""
Loky — bot multi-strategia ad alta performance per Bybit Futures.

Macchina a stati per symbol:
    FLAT → ENTERING → POSITION_OPEN → PARTIAL_EXIT → EXITING → FLAT

Feature:
  • 4 engine paralleli: Breakout, MeanReversion, TrendFollowing, FundingRate
  • Signal scoring 0-100 con selezione automatica del miglior segnale
  • Partial TP: 50% @ TP1, 25% @ TP2, 25% trail @ TP3
  • Kelly Criterion sizing dinamico
  • PortfolioRiskManager: notional cap, leva dinamica, filtro correlazione
  • Multi-timeframe: segnali su 15m filtrati dal trend 1h
  • Trailing stop, cooldown post-loss, next-candle entry
  • Notifiche Telegram su ogni evento rilevante
"""

import asyncio
import datetime
import logging
import time
from collections import deque
from decimal import Decimal
from enum import Enum, auto
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _prom = True
except ImportError:
    _prom = False

from src.config import BotSettings
from src.core.account_risk import AccountRiskManager
from src.core.kelly_sizing import KellySizer
from src.core.liquidation_monitor import LiquidationAlert, LiquidationMonitor
from src.core.portfolio_risk import PortfolioRiskManager
from src.models import Candle, Order, OrderStatus, Side, Signal, SignalType, TPLevel, Trade
from src.notifications.telegram import TelegramNotifier
from src.state.persistency import StateManager
from src.strategy.breakout_engine import BreakoutEngine
from src.strategy.funding_rate_engine import FundingRateEngine
from src.strategy.indicator_engine import IndicatorEngine
from src.strategy.market_sentiment_engine import MarketSentimentEngine
from src.strategy.mean_reversion_engine import MeanReversionEngine
from src.gateways.smart_execution import ExecutionAnalytics, SlippageEstimator
from src.strategy.orderflow_engine import OrderFlowEngine
from src.strategy.signal_aggregator import SignalAggregator
from src.strategy.volatility_engine import VolatilityRegime, VolatilityRegimeEngine
from src.strategy.trend_following_engine import TrendFollowingEngine

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE  = Decimal("1")


class BotState(Enum):
    FLAT          = auto()
    ENTERING      = auto()
    POSITION_OPEN = auto()
    PARTIAL_EXIT  = auto()   # parzialmente chiuso, ancora aperto
    EXITING       = auto()


# ---------------------------------------------------------------------------
# Prometheus metrics (opzionali — richiedono prometheus-client)
# ---------------------------------------------------------------------------
if _prom:
    _fills_total    = Counter(  "loky_trades_total",      "Trade chiusi",         ["symbol", "side", "result"])
    _pnl_gauge      = Gauge(    "loky_pnl_realized_usdt", "PnL realizzato",       ["symbol"])
    _position_gauge = Gauge(    "loky_position_open",     "Posizione aperta 0/1", ["symbol"])
    _signals_total  = Counter(  "loky_signals_total",     "Segnali rilevati",     ["symbol", "direction"])
    _signal_score   = Histogram("loky_signal_score",      "Score segnale 0-100",  ["symbol", "engine"],
                                buckets=[40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    _daily_loss     = Gauge(    "loky_daily_loss_usdt",   "PnL giornaliero",      ["symbol"])
    _regime_gauge   = Gauge(    "loky_market_regime",     "Regime (0=CHOPPY,1=RANGING,2=NEUTRAL,3=TRENDING,4=STRONG)", ["symbol"])

def _start_prometheus_server(port: int = 9090) -> None:
    """Avvia il server HTTP per Prometheus su porta 9090 (idempotente)."""
    if not _prom:
        return
    try:
        start_http_server(port)
        import logging as _log
        _log.getLogger(__name__).info("Prometheus metrics server avviato su :%d/metrics", port)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning("Prometheus server non avviato: %s", e)


class LokyBot:
    """
    Loky — bot multi-strategia per un singolo symbol su Bybit Futures.

    Args:
        symbol         — es. "BTCUSDT"
        config         — BotSettings caricato da config.yaml
        execution_gw   — gateway di esecuzione (live o paper)
        state_manager  — SQLite state manager
        capital        — USDT disponibile per questo bot
        account_risk   — AccountRiskManager condiviso tra symbol
        portfolio_risk — PortfolioRiskManager condiviso tra symbol
        notifier       — TelegramNotifier condiviso
    """

    def __init__(
        self,
        symbol: str,
        config: BotSettings,
        execution_gw,
        state_manager: StateManager,
        capital: Decimal = Decimal("500"),
        account_risk: Optional[AccountRiskManager] = None,
        portfolio_risk: Optional[PortfolioRiskManager] = None,
        notifier: Optional[TelegramNotifier] = None,
    ) -> None:
        self.symbol         = symbol
        self._cfg           = config
        self._gw            = execution_gw
        self._state_mgr     = state_manager
        self._capital       = capital
        self._account_risk  = account_risk
        self._portfolio_risk = portfolio_risk
        self._notifier      = notifier

        s = config.strategy
        self._primary_tf = config.primary_timeframe
        self._confirm_tf = config.confirmation_timeframe
        self._macro_tf   = getattr(config, 'macro_timeframe', '4h')

        # --- Indicatori primari (15m) ---
        self._indicators = IndicatorEngine(
            ema_fast   = s.ema_fast,
            ema_slow   = s.ema_slow,
            rsi_period = 14,
            atr_period = s.atr_period,
            vol_period = s.vol_period,
            adx_period = s.adx_period,
            bb_period  = s.bb_period,
            bb_std     = s.bb_std,
        )
        # --- Indicatori HTF (1h) per filtro trend ---
        self._htf_indicators = IndicatorEngine(
            ema_fast   = s.ema_fast,
            ema_slow   = s.ema_slow,
            rsi_period = 14,
            atr_period = s.atr_period,
            vol_period = s.vol_period,
            adx_period = s.adx_period,
            bb_period  = s.bb_period,
            bb_std     = s.bb_std,
        )
        # --- Indicatori macro (4h) per filtro tendenza macro ---
        self._macro_indicators = IndicatorEngine(
            ema_fast   = s.ema_fast,
            ema_slow   = s.ema_slow,
            rsi_period = 14,
            atr_period = s.atr_period,
            vol_period = s.vol_period,
            adx_period = s.adx_period,
            bb_period  = s.bb_period,
            bb_std     = s.bb_std,
        )

        # --- Strategy engines ---
        self._breakout      = BreakoutEngine(config, self._indicators, capital)
        self._mean_rev      = MeanReversionEngine(config, self._indicators, capital)
        self._trend_follow  = TrendFollowingEngine(config, self._indicators, capital)
        self._funding_rate  = FundingRateEngine(config, self._indicators, capital, testnet=config.testnet)
        self._sentiment     = MarketSentimentEngine(testnet=config.testnet)

        # --- Signal scoring ---
        self._aggregator = SignalAggregator(
            self._indicators, self._htf_indicators, self._macro_indicators
        )

        # --- Volatility regime ---
        self._vol_engine = VolatilityRegimeEngine(self._indicators)

        # --- Order flow ---
        self._orderflow = OrderFlowEngine()

        # --- Smart execution ---
        self._slippage_est = SlippageEstimator(
            base_slippage_pct=config.slippage_pct,
        )
        self._exec_analytics = ExecutionAnalytics()

        # --- Kelly Criterion sizing ---
        self._kelly = KellySizer(
            history_trades = config.kelly_min_trades,
            kelly_divisor  = 2,           # half-Kelly (configurabile: 1=full, 3=third, 4=quarter)
            min_fraction   = Decimal('0.005'),
            max_fraction   = Decimal('0.03'),
            use_optimal_f  = True,        # Optimal-f per fat tails crypto
        )

        # --- Liquidation monitor ---
        self._liq_monitor = LiquidationMonitor(leverage=config.leverage)

        self._candles: deque[Candle] = deque(maxlen=200)
        self._candle_count: int = 0       # contatore candle per polling funding rate
        self._last_price: Decimal = _ZERO  # ultimo prezzo (close dell'ultima candle)

        # --- Macchina a stati ---
        self._state: BotState               = BotState.FLAT
        self._current_signal: Optional[Signal] = None
        self._entry_order:    Optional[Order]  = None
        self._position_side:  Optional[Side]   = None
        self._position_size:  Decimal          = _ZERO   # size residua
        self._position_size_orig: Decimal      = _ZERO   # size originale (per parziali)
        self._entry_price:    Decimal          = _ZERO
        self._entry_time:     float            = 0.0
        self._sl_price:       Decimal          = _ZERO
        self._sl_price_orig:  Decimal          = _ZERO   # SL originale per Kelly update

        # --- Partial TP ---
        self._tp_levels: list[TPLevel] = []
        self._accumulated_trade_pnl: Decimal = _ZERO  # PnL totale del trade (somma dei parziali)
        self._partial_locked_pnl: Decimal = _ZERO     # PnL locked da partial exits (per Kelly)

        # --- Pyramid (scaling-in) ---
        self._scale_in_count: int = 0    # numero di add-on effettuati (max 1)
        self._scale_in_size:  Decimal = _ZERO  # size totale aggiunta in scaling
        self._kelly_risk_size: Decimal = _ZERO  # size originale pre-scale-in per Kelly update
        self._kelly_risk_entry: Decimal = _ZERO  # entry price originale per Kelly update

        # --- Win/Loss streak (anti-martingale sizing) ---
        self._win_streak:  int = 0   # trade vincenti consecutivi
        self._loss_streak: int = 0   # trade perdenti consecutivi

        # --- Circuit breaker (consecutive loss protection) ---
        self._consecutive_losses:    int = 0
        self._circuit_breaker_candles: int = 0   # candle rimanenti di pausa

        # --- Flag e controlli ---
        self._daily_stop_triggered: bool = False
        self._cooldown_remaining: int    = 0
        self._pending_signal: Optional[Signal] = None

        # --- PnL tracking ---
        self.realized_pnl: Decimal = _ZERO
        self.total_trades: int     = 0
        self.total_wins:   int     = 0
        self.total_losses: int     = 0

        # --- PnL attribution per strategy engine ---
        # Chiave: strategy_name (es. "TrendFollowing", "Breakout", "MeanReversion")
        self.pnl_by_strategy: dict[str, Decimal]  = {}
        self.trades_by_strategy: dict[str, int]   = {}

        self._hold_time_task: Optional[asyncio.Task] = None

        # Lock per on_order_update: previene race condition su double-fill
        self._order_lock = asyncio.Lock()

        # Tracking fire-and-forget async tasks (notifiche Telegram, ecc.)
        self._pending_tasks: set[asyncio.Task] = set()

        self._load_state()

    # ------------------------------------------------------------------
    # Evento principale: nuova candela chiusa
    # ------------------------------------------------------------------

    async def on_candle(self, candle: Candle) -> None:
        if candle.symbol != self.symbol:
            return

        # Smista per timeframe
        if candle.timeframe == self._confirm_tf:
            self._htf_indicators.update(candle)
            return

        if candle.timeframe == self._macro_tf:
            self._macro_indicators.update(candle)
            return

        if candle.timeframe != self._primary_tf:
            return

        # --- Timeframe primario ---
        self._candles.append(candle)
        self._indicators.update(candle)
        self._candle_count += 1
        self._last_price = candle.close
        self._vol_engine.update()   # aggiorna volatility regime
        self._orderflow.update(candle)  # aggiorna order flow

        # Registra ATR per leva dinamica
        if self._portfolio_risk is not None:
            try:
                self._portfolio_risk.record_atr(self._indicators.atr())
            except ValueError:
                pass

        # Posizione aperta: liquidation check + trailing SL + partial TP/SL + scaling-in
        if self._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            # Liquidation monitor: chiudi se margine critico
            is_long = self._position_side == Side.BUY
            liq_alert = self._liq_monitor.check(
                self._entry_price, candle.close, is_long, symbol=self.symbol,
            )
            if liq_alert == LiquidationAlert.CRITICAL:
                logger.warning(
                    "%s LIQUIDATION CRITICAL — chiusura emergenza a mercato.", self.symbol,
                )
                if self._notifier:
                    self._fire_and_forget(self._notifier.info(
                        f"🚨 <b>LIQUIDAZIONE IMMINENTE</b> {self.symbol} — "
                        f"chiusura forzata a mercato."
                    ))
                await self._close_remaining(candle.close, "liquidation_critical")
                return

            # Gap protection: se la perdita supera 2x lo SL originale, chiudi forzatamente.
            # Protegge da gap di prezzo e slippage estremo che oltrepassano lo SL.
            if self._sl_price_orig > _ZERO and self._position_size > _ZERO:
                sl_distance_orig = abs(self._entry_price - self._sl_price_orig)
                if is_long:
                    actual_loss_distance = self._entry_price - candle.close
                else:
                    actual_loss_distance = candle.close - self._entry_price
                if actual_loss_distance > sl_distance_orig * Decimal('2'):
                    logger.warning(
                        "%s GAP PROTECTION — perdita %.4f > 2x SL originale %.4f. Chiusura forzata.",
                        self.symbol, actual_loss_distance, sl_distance_orig,
                    )
                    if self._notifier:
                        self._fire_and_forget(self._notifier.info(
                            f"⚠️ <b>GAP PROTECTION</b> {self.symbol} — "
                            f"perdita > 2x SL. Chiusura forzata."
                        ))
                    await self._close_remaining(candle.close, "gap_protection_2x_sl")
                    return

            if not self._cfg.live_trading_enabled:
                self._update_trailing_sl(candle)
                await self._check_paper_tp_sl(candle)
            # MACD divergence exit: se MACD diverge, chiudi anticipatamente (post-TP1)
            if self._tp_levels and self._tp_levels[0].hit:
                try:
                    if is_long and self._indicators.macd_bearish_divergence():
                        logger.info("%s MACD bearish divergence — exit anticipato", self.symbol)
                        await self._close_remaining(candle.close, "macd_divergence")
                        return
                    if not is_long and self._indicators.macd_bullish_divergence():
                        logger.info("%s MACD bullish divergence — exit anticipato", self.symbol)
                        await self._close_remaining(candle.close, "macd_divergence")
                        return
                except ValueError:
                    pass

            # Order Flow divergence exit: se CVD diverge dal prezzo, chiudi anticipatamente
            if self._tp_levels and self._tp_levels[0].hit:
                price_rising = candle.close > self._entry_price if is_long else candle.close < self._entry_price
                div = self._orderflow.divergence_signal(price_rising)
                if div == "bearish_divergence" and is_long:
                    logger.info("%s Order Flow bearish divergence — exit anticipato", self.symbol)
                    await self._close_remaining(candle.close, "orderflow_divergence")
                    return
                if div == "bullish_divergence" and not is_long:
                    logger.info("%s Order Flow bullish divergence — exit anticipato", self.symbol)
                    await self._close_remaining(candle.close, "orderflow_divergence")
                    return

            # Scaling-in: valuta add-on anche in live (solo POSITION_OPEN, non PARTIAL_EXIT)
            if self._state == BotState.POSITION_OPEN:
                await self._check_scale_in(candle)
            return

        # Entry next-candle: entra al open della candela attuale
        if self._pending_signal is not None and self._state == BotState.FLAT:
            sig = self._pending_signal
            self._pending_signal = None
            # Gap validation: se il prezzo di apertura è troppo lontano dall'entry previsto,
            # scarta il segnale (il mercato è gappato, il setup non è più valido)
            gap = abs(candle.open - sig.entry_price)
            max_gap = sig.atr if sig.atr > _ZERO else sig.entry_price * Decimal('0.01')
            if gap > max_gap:
                logger.info(
                    "%s Next-candle entry scartata: gap %.4f > 1×ATR %.4f.",
                    self.symbol, gap, max_gap,
                )
                return
            # Recalcola TP/SL relativi al prezzo di entry reale (candle.open)
            # per mantenere lo stesso R:R del segnale originale
            price_shift = candle.open - sig.entry_price
            sig.take_profit += price_shift
            sig.stop_loss += price_shift
            sig.entry_price = candle.open
            await self._enter_position(sig, fill_price_override=candle.open)
            return

        if self._state != BotState.FLAT:
            return

        # Blocco esplicito regime CHOPPY: nessun trade ammesso (ADX < 15)
        if self._aggregator.is_choppy_market():
            return

        # Cooldown post-loss
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            logger.debug("%s Cooldown: %d candele rimaste.", self.symbol, self._cooldown_remaining)
            return

        # Circuit breaker: pausa dopo N perdite consecutive
        if self._circuit_breaker_candles > 0:
            self._circuit_breaker_candles -= 1
            if self._circuit_breaker_candles == 0:
                logger.info(
                    "%s Circuit breaker SCADUTO — trading ripreso.",
                    self.symbol,
                )
                if self._notifier:
                    self._fire_and_forget(self._notifier.info(
                        f"✅ Circuit breaker {self.symbol} scaduto — trading ripreso."
                    ))
            else:
                logger.debug(
                    "%s Circuit breaker attivo: %d candle rimaste.",
                    self.symbol, self._circuit_breaker_candles,
                )
            return

        if self._is_daily_stop_active():
            return

        # --- Time-of-day filter: evita orari a bassa liquidità ---
        utc_hour = datetime.datetime.utcnow().hour
        # Evita trading tra 21-23 UTC (bassa liquidità, spread alti)
        # e tra 4-5 UTC (gap di liquidità Asia → Europa)
        if utc_hour in (21, 22, 23, 4):
            logger.debug("%s Skip segnali: ora UTC %d (bassa liquidità)", self.symbol, utc_hour)
            return

        # --- Volatility regime filter: blocca entry in squeeze/gap ---
        vol_regime = self._vol_engine.detect()
        self._current_vol_regime = vol_regime  # cache per uso downstream
        # Gap protection pre-entry: se l'ultima candle ha un gap > 2×ATR, skip
        if len(self._candles) >= 2:
            try:
                atr = self._indicators.atr()
                prev_close = self._candles[-2].close
                gap = abs(candle.open - prev_close)
                if gap > atr * Decimal('2'):
                    logger.info(
                        "%s Gap %.4f > 2×ATR — skip segnali (rischio slippage).",
                        self.symbol, gap,
                    )
                    return
            except ValueError:
                pass

        # --- Detect regime e raccoglie segnali ---
        signals = self._collect_signals()

        # Funding rate engine: sondaggio ogni 4 candle
        if self._candle_count % 4 == 0:
            funding_sig = await self._collect_funding_signal()
            if funding_sig is not None:
                signals.append(funding_sig)

        # Market sentiment: aggiorna OI + L/S ratio ogni 8 candle
        if self._candle_count % 8 == 0:
            await self._update_sentiment(candle)

        if not signals:
            return

        # Volatility contraction filter: solo trend_following con score alto
        if hasattr(self, '_current_vol_regime') and self._current_vol_regime == VolatilityRegime.CONTRACTION:
            signals = [s for s in signals if s.strategy_name == "trend_following" and s.score >= Decimal('75')]
            if not signals:
                logger.debug("%s Contraction regime: no high-confidence TF signals, skip.", self.symbol)
                return

        # Seleziona il miglior segnale con scoring
        best = self._aggregator.select_best(signals)
        if best is None:
            return

        # Volatility regime modifier: penalizza in CONTRACTION, premia in COMPRESSION
        vol_mod = self._vol_engine.score_modifier()
        if vol_mod != Decimal('1'):
            best.score = min(best.score * vol_mod, Decimal('100')).quantize(Decimal('0.1'))

        # Order flow modifier: CVD conferma/contraddice il segnale
        is_long = best.signal_type == SignalType.LONG
        of_mod = self._orderflow.score_modifier(is_long)
        if of_mod != Decimal('1'):
            best.score = min(best.score * of_mod, Decimal('100')).quantize(Decimal('0.1'))
            best.size = (best.size * of_mod).quantize(Decimal('0.001'))

        # --- Filtro sentiment (OI + L/S ratio) ---
        # 1. Blocco esplicito basato su flag block_long/block_short (contrarian estremo)
        if self._sentiment.is_blocked(self.symbol, best.signal_type):
            logger.info(
                "%s Segnale %s bloccato da sentiment (retail overextended).",
                self.symbol, best.signal_type.name,
            )
            return
        # 2. Aggiustamento score basato su OI delta e ratio (±15 punti)
        sent_adj = self._sentiment.score_adjustment_for(self.symbol, best.signal_type)
        if sent_adj != 0:
            old_score = best.score
            best.score = max(_ZERO, best.score + Decimal(str(sent_adj)))
            logger.debug(
                "%s Sentiment adjustment: %+d punti (%s) score %.0f→%.0f",
                self.symbol, sent_adj, best.signal_type.name, old_score, best.score,
            )
        # 3. Sizing basato su sentiment: OI in linea con la direzione → size boost
        if sent_adj > 10:
            best.size = (best.size * Decimal('1.2')).quantize(Decimal('0.001'))
            logger.debug("%s Sentiment boost: size +20%% (forte allineamento OI)", self.symbol)
        elif sent_adj < -5:
            best.size = (best.size * Decimal('0.7')).quantize(Decimal('0.001'))
            logger.debug("%s Sentiment cautela: size -30%% (OI contrario)", self.symbol)

        # --- Filtro macro trend (4h) ---
        if not self._macro_trend_aligned(best.signal_type):
            logger.debug(
                "%s Segnale %s (%s) scartato: trend 4h non allineato.",
                self.symbol, best.signal_type.name, best.strategy_name,
            )
            return

        # --- Filtro multi-timeframe (1h) ---
        if not self._htf_trend_aligned(best.signal_type):
            logger.debug(
                "%s Segnale %s (%s) scartato: trend 1h non allineato.",
                self.symbol, best.signal_type.name, best.strategy_name,
            )
            return

        # Score minimo dopo tutti gli aggiustamenti
        if best.score < self._cfg.strategy.min_signal_score:
            logger.debug(
                "%s Segnale %s score insufficiente post-sentiment: %.0f < %.0f",
                self.symbol, best.signal_type.name, best.score, self._cfg.strategy.min_signal_score,
            )
            return

        # Portfolio risk check spostato in _enter_position (DOPO Kelly/leverage adjustments)
        # per verificare il notional EFFETTIVO, non quello pre-adjustment

        # --- Anti-martingale: aggiusta size in base al streak ---
        best.size = self._apply_streak_sizing(best.size)

        if _prom:
            _signals_total.labels(symbol=self.symbol, direction=best.signal_type.name).inc()
            _signal_score.labels(symbol=self.symbol, engine=best.strategy_name).observe(float(best.score))
            # Regime metric
            regime_map = {"CHOPPY": 0, "RANGING": 1, "NEUTRAL": 2, "TRENDING": 3, "STRONG_TREND": 4}
            regime_val = regime_map.get(self._aggregator.detect_regime(), 2)
            _regime_gauge.labels(symbol=self.symbol).set(regime_val)

        if self._cfg.next_candle_entry:
            self._pending_signal = best
        else:
            await self._enter_position(best)

    # ------------------------------------------------------------------
    # Raccolta segnali da tutti gli engine
    # ------------------------------------------------------------------

    def _collect_signals(self) -> list[Signal]:
        """Esegue gli engine sincroni al regime corrente (funding rate escluso — è asincrono)."""
        if not self._indicators.ready():
            return []

        regime    = self._aggregator.detect_regime()
        preferred = self._aggregator.preferred_strategies()
        signals: list[Signal] = []

        if "breakout" in preferred:
            sig = self._breakout.detect(self._candles)
            if sig.signal_type != SignalType.NONE:
                signals.append(sig)

        if "mean_reversion" in preferred:
            sig = self._mean_rev.detect(self._candles)
            if sig.signal_type != SignalType.NONE:
                signals.append(sig)

        if "trend_following" in preferred:
            sig = self._trend_follow.detect(self._candles)
            if sig.signal_type != SignalType.NONE:
                signals.append(sig)

        logger.debug("%s Regime=%s segnali trovati=%d", self.symbol, regime, len(signals))
        return signals

    async def _update_sentiment(self, candle: Candle) -> None:
        """
        Aggiorna il MarketSentimentEngine per il symbol corrente.
        Fire-and-forget: errori loggati, non propagati.
        """
        try:
            result = await self._sentiment.analyze(self.symbol, candle)
            logger.debug("%s Sentiment: %s", self.symbol, result.summary())
        except Exception as e:
            logger.debug("%s Sentiment update errore: %s", self.symbol, e)

    async def _collect_funding_signal(self) -> Optional[Signal]:
        """
        Raccoglie il segnale dal FundingRateEngine (async — richiede chiamata REST).
        Chiamato separatamente da on_candle ogni N candle per non rallentare il loop.
        """
        if not self._indicators.ready():
            return None
        try:
            sig = await self._funding_rate.detect(self._candles)
            if sig.signal_type != SignalType.NONE:
                return sig
        except Exception as e:
            logger.warning("%s FundingRateEngine errore: %s", self.symbol, e)
        return None

    # ------------------------------------------------------------------
    # HTF filter
    # ------------------------------------------------------------------

    def _htf_trend_aligned(self, signal_type: SignalType) -> bool:
        if not self._htf_indicators.ready():
            return True
        try:
            ema_f = self._htf_indicators.ema_fast()
            ema_s = self._htf_indicators.ema_slow()
        except ValueError:
            return True

        if signal_type == SignalType.LONG:
            return ema_f > ema_s
        if signal_type == SignalType.SHORT:
            return ema_f < ema_s
        return False

    def _macro_trend_aligned(self, signal_type: SignalType) -> bool:
        """
        Filtro macro (4h): blocca trades contro il trend su timeframe superiore.
        Fallback True se il timeframe 4h non ha ancora abbastanza dati.
        """
        if not self._macro_indicators.ready():
            return True
        try:
            ema_f = self._macro_indicators.ema_fast()
            ema_s = self._macro_indicators.ema_slow()
        except ValueError:
            return True

        # Threshold minimo: se spread < 0.2% del prezzo, macro è neutro → blocca
        if ema_s > _ZERO:
            spread_pct = abs(ema_f - ema_s) / ema_s
            if spread_pct < Decimal('0.002'):
                logger.debug(
                    "%s Macro trend neutro (spread=%.3f%% < 0.2%%), segnale bloccato.",
                    self.symbol, float(spread_pct * 100),
                )
                return False

        if signal_type == SignalType.LONG:
            aligned = ema_f > ema_s
        elif signal_type == SignalType.SHORT:
            aligned = ema_f < ema_s
        else:
            return False

        if not aligned:
            logger.debug(
                "%s Segnale %s bloccato dal filtro macro 4h (EMA_fast=%.4f %s EMA_slow=%.4f).",
                self.symbol, signal_type.name, ema_f,
                "<" if signal_type == SignalType.LONG else ">", ema_s,
            )
        return aligned

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def _update_trailing_sl(self, candle: Candle) -> None:
        """
        Trailing stop adattivo: la distanza del trail si adatta alla volatilità ATR.

        Logica Chandelier Exit adattiva:
        - Se ATR è nel percentile alto (>80°): trail = 1.0×ATR (respiro maggiore in alta vol)
        - Se ATR è nel percentile medio (40-80°): trail = 0.5×ATR (default)
        - Se ATR è nel percentile basso (<40°): trail = 0.3×ATR (più stretto in bassa vol)

        Attiva solo dopo che il profit supera 1×ATR (evita trigger prematuro).
        """
        if not self._cfg.strategy.trailing_stop_enabled:
            return
        if self._position_side is None or self._position_size == _ZERO:
            return
        # Attiva trail solo DOPO almeno 1 partial TP (TP1 50%).
        # Prima di TP1, lo SL è fisso e il prezzo deve raggiungere il primo target.
        # Dopo TP1, lo SL è a breakeven e il trail protegge il residuo.
        tp1_hit = self._tp_levels and self._tp_levels[0].hit
        if not tp1_hit:
            return
        try:
            atr = self._indicators.atr()
        except ValueError:
            return

        # Calcola il trail multiplier adattivo basato sul percentile ATR (dal portfolio risk)
        trail_mult = self._cfg.strategy.trail_atr_mult  # default (es. 0.5)
        if self._portfolio_risk is not None:
            atr_pct = self._portfolio_risk.atr_percentile(atr)
            if atr_pct > Decimal('0.80'):
                trail_mult = Decimal('1.0')   # alta volatilità → trail più largo
            elif atr_pct < Decimal('0.40'):
                trail_mult = Decimal('0.3')   # bassa volatilità → trail più stretto

        trail_dist = trail_mult * atr
        price      = candle.close

        if self._position_side == Side.BUY:
            if (price - self._entry_price) < atr:
                return
            new_sl = price - trail_dist
            if new_sl > self._sl_price:
                logger.debug("%s Trail SL adattivo (mult=%.1f): %.4f → %.4f",
                             self.symbol, trail_mult, self._sl_price, new_sl)
                self._sl_price = new_sl
        else:
            if (self._entry_price - price) < atr:
                return
            new_sl = price + trail_dist
            if new_sl < self._sl_price:
                logger.debug("%s Trail SL adattivo (mult=%.1f): %.4f → %.4f",
                             self.symbol, trail_mult, self._sl_price, new_sl)
                self._sl_price = new_sl

    # ------------------------------------------------------------------
    # Partial TP + SL check (paper trading)
    # ------------------------------------------------------------------

    async def _check_paper_tp_sl(self, candle: Candle) -> None:
        """
        Simula partial TP e SL su high/low della candela.
        Ordine di priorità: SL prima, poi TP dal più basso al più alto.
        """
        if self._position_side is None:
            return

        is_long = self._position_side == Side.BUY

        # --- SL ---
        sl = self._sl_price
        if sl != _ZERO:
            sl_hit = candle.low <= sl if is_long else candle.high >= sl
            if sl_hit:
                # Slippage realistico su SL: exit leggermente peggiore del livello SL
                exit_side = Side.SELL if is_long else Side.BUY
                sl_fill = self._calc_paper_fill_price(sl, exit_side, self._position_size)
                await self._close_remaining(sl_fill, "SL hit (paper)")
                return

        # --- TTL-based TP scaling: target si stringono col tempo ---
        # Dopo 2/3 del max hold time, chiudi qualsiasi profitto > 0.5×ATR
        if self._entry_time > 0:
            hold_seconds = time.time() - self._entry_time
            max_hold_s = self._cfg.strategy.max_hold_hours * 3600
            hold_ratio = hold_seconds / max_hold_s if max_hold_s > 0 else 0

            if hold_ratio > 0.66:  # ultimi 33% del tempo
                try:
                    atr = self._indicators.atr()
                    min_profit_to_exit = atr * Decimal('0.5')
                    if is_long:
                        profit = candle.close - self._entry_price
                    else:
                        profit = self._entry_price - candle.close
                    if profit > min_profit_to_exit:
                        logger.info(
                            "%s TTL exit: hold %.0f%%, profit %.4f > 0.5×ATR. Chiusura.",
                            self.symbol, hold_ratio * 100, profit,
                        )
                        await self._close_remaining(candle.close, "ttl_profit_exit")
                        return
                except ValueError:
                    pass

        # --- Partial TP levels ---
        for i, tp_lvl in enumerate(self._tp_levels):
            if tp_lvl.hit:
                continue
            tp_hit = candle.high >= tp_lvl.price if is_long else candle.low <= tp_lvl.price
            if not tp_hit:
                continue

            close_size = (self._position_size_orig * tp_lvl.qty_fraction).quantize(Decimal('0.001'))
            close_size = min(close_size, self._position_size)
            if close_size <= _ZERO:
                tp_lvl.hit = True
                continue

            # Applica slippage realistico anche al TP in paper trading:
            # in live, il fill avviene a prezzo di mercato (quasi sempre leggermente peggiore del TP)
            tp_fill = self._calc_paper_fill_price(tp_lvl.price, self._position_side, close_size) \
                if self._position_side is not None else tp_lvl.price
            await self._close_partial(tp_fill, close_size, f"TP{i+1} (paper)")
            tp_lvl.hit = True

            # Dopo TP1 → SL a breakeven
            if i == 0:
                if is_long:
                    self._sl_price = max(self._sl_price, self._entry_price)
                else:
                    self._sl_price = min(self._sl_price, self._entry_price)
                logger.info("%s SL spostato a breakeven: %.4f", self.symbol, self._sl_price)

            if self._position_size <= _ZERO:
                break

    # ------------------------------------------------------------------
    # Daily stop
    # ------------------------------------------------------------------

    def _is_daily_stop_active(self) -> bool:
        if self.realized_pnl <= -(self._capital * self._cfg.max_daily_loss_pct):
            if not self._daily_stop_triggered:
                logger.warning(
                    "%s Daily loss limit raggiunto (%.2f USDT). Trading sospeso.",
                    self.symbol, self.realized_pnl,
                )
                self._daily_stop_triggered = True
                if self._notifier:
                    self._fire_and_forget(self._notifier.daily_stop_triggered(
                        symbol=self.symbol, pnl=self.realized_pnl,
                    ))
            return True
        self._daily_stop_triggered = False
        return False

    # ------------------------------------------------------------------
    # Evento fill ordine
    # ------------------------------------------------------------------

    async def on_order_update(self, order: Order) -> None:
        if order.symbol != self.symbol:
            return
        # Lock: previene double-fill se due update WebSocket arrivano in parallelo
        async with self._order_lock:
            if order.status == OrderStatus.FILLED:
                await self._handle_fill(order)
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                await self._handle_partial_fill(order)
            elif order.status == OrderStatus.REJECTED:
                logger.error("%s Ordine rifiutato: %s", self.symbol, order.id)
                if self._state == BotState.ENTERING:
                    self._state = BotState.FLAT

    def _fire_and_forget(self, coro) -> None:
        """Crea un task asincrono tracciato con cleanup automatico al completamento."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        def _on_done(t: asyncio.Task) -> None:
            self._pending_tasks.discard(t)
            if not t.cancelled() and t.exception():
                logger.warning("%s Notifica async fallita: %s", self.symbol, t.exception())
        task.add_done_callback(_on_done)

    async def close(self) -> None:
        """Chiude le sessioni HTTP interne e cancella task pending."""
        for task in self._pending_tasks:
            task.cancel()
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()
        await self._funding_rate.close()
        await self._sentiment.close()

    # ------------------------------------------------------------------
    # Ingresso posizione
    # ------------------------------------------------------------------

    async def _enter_position(
        self,
        signal: Signal,
        fill_price_override: Optional[Decimal] = None,
    ) -> None:
        # Doppia guardia: circuit breaker può essere attivo se segnale era pending
        if self._circuit_breaker_candles > 0:
            logger.warning(
                "%s Entry bloccata: circuit breaker ancora attivo (%d candle).",
                self.symbol, self._circuit_breaker_candles,
            )
            return

        if self._account_risk and not self._account_risk.can_open_position(self.symbol):
            return

        self._state          = BotState.ENTERING
        self._current_signal = signal
        side = Side.BUY if signal.signal_type == SignalType.LONG else Side.SELL

        # Kelly sizing: override size se abilitato e pronto
        if self._cfg.kelly_sizing_enabled and self._kelly.is_ready:
            sl_distance = abs(signal.entry_price - signal.stop_loss)
            kelly_size  = self._kelly.position_size(
                self._capital, sl_distance,
                fallback_fraction=self._cfg.risk_per_trade_pct,
            ).quantize(Decimal('0.001'))
            if kelly_size > _ZERO:
                signal.size = kelly_size

        # Dynamic leverage: cappa la size al max notional con leva dinamica
        if self._cfg.dynamic_leverage_enabled and self._portfolio_risk is not None:
            try:
                atr = self._indicators.atr()
                dd_pct = self._account_risk.current_drawdown_pct if self._account_risk else _ZERO
                dyn_lev = self._portfolio_risk.dynamic_leverage(atr, drawdown_pct=dd_pct)
                max_notional = self._capital * Decimal(str(dyn_lev))
                if signal.entry_price > _ZERO:
                    max_dyn_size = (max_notional / signal.entry_price).quantize(Decimal('0.001'))
                    if signal.size > max_dyn_size:
                        logger.debug(
                            "%s Dynamic leverage %dx: size cappata %.3f → %.3f",
                            self.symbol, dyn_lev, signal.size, max_dyn_size,
                        )
                        signal.size = max_dyn_size
            except ValueError:
                pass  # ATR non pronto, usa leva statica

        # Portfolio heat: riduce size se portafoglio troppo esposto in una direzione
        if self._portfolio_risk is not None:
            heat_mod = self._portfolio_risk.heat_size_modifier()
            if heat_mod < Decimal('1'):
                signal.size = (signal.size * heat_mod).quantize(Decimal('0.001'))
                logger.debug("%s Portfolio heat: size ×%.1f", self.symbol, float(heat_mod))

        # Portfolio risk check (POST size adjustments): verifica notional effettivo
        if self._portfolio_risk is not None:
            final_notional = signal.entry_price * signal.size
            ok, reason = self._portfolio_risk.can_open(self.symbol, final_notional)
            if not ok:
                logger.info("%s Entry bloccata da PortfolioRisk (post-adj): %s", self.symbol, reason)
                self._state = BotState.FLAT
                return

        # Validazione size finale: blocca size invalide prima che arrivino al gateway
        min_notional = Decimal('6')   # Bybit minimum
        max_size     = Decimal('100000')
        if signal.size <= _ZERO or signal.size > max_size:
            logger.error(
                "%s Size invalida (%.6f) — entry annullata.", self.symbol, signal.size,
            )
            self._state = BotState.FLAT
            return
        if signal.entry_price * signal.size < min_notional:
            logger.debug(
                "%s Notional troppo basso (%.2f USDT < %.0f) — entry annullata.",
                self.symbol, float(signal.entry_price * signal.size), float(min_notional),
            )
            self._state = BotState.FLAT
            return

        # Slippage estimation: verifica e adatta la size se slippage eccessivo
        try:
            vol_ma = self._indicators.volume_ma()
            ok, est_slip = self._slippage_est.is_acceptable(signal.entry_price, signal.size, vol_ma)
            if not ok:
                adjusted = self._slippage_est.adjusted_size(signal.entry_price, signal.size, vol_ma)
                if adjusted <= _ZERO:
                    logger.warning(
                        "%s Slippage troppo alto (%.2f%%) — entry annullata.",
                        self.symbol, float(est_slip * 100),
                    )
                    self._state = BotState.FLAT
                    return
                signal.size = adjusted
        except ValueError:
            est_slip = self._cfg.slippage_pct

        logger.info(
            "%s [Loky] ENTRATA %s (%s) | entry≈%.4f tp=%.4f sl=%.4f size=%.3f score=%.0f slip≈%.2f%%",
            self.symbol, signal.signal_type.name, signal.strategy_name,
            signal.entry_price, signal.take_profit, signal.stop_loss,
            signal.size, signal.score, float(est_slip * 100),
        )

        if not self._cfg.live_trading_enabled:
            base_price = fill_price_override or (
                self._candles[-1].close if self._candles else signal.entry_price
            )
            fill_price = self._calc_paper_fill_price(base_price, side, signal.size)
            await self._open_position(side, signal.size, fill_price, signal)
            return

        # Limit entry con market fallback: risparmia ~50% fee (maker vs taker)
        if self._cfg.limit_entry_enabled and hasattr(self._gw, 'submit_limit_with_fallback'):
            order = await self._gw.submit_limit_with_fallback(
                self.symbol, side, signal.size,
                limit_price=signal.entry_price,
                timeout_s=self._cfg.limit_entry_timeout_s,
            )
        else:
            order = await self._gw.submit_market_order(self.symbol, side, signal.size)
        if order is None:
            logger.error("%s submit order fallito", self.symbol)
            self._state = BotState.FLAT
            return
        self._entry_order = order

    async def _open_position(
        self,
        side: Side,
        size: Decimal,
        fill_price: Decimal,
        signal: Signal,
    ) -> None:
        """Apre la posizione, calcola 3 livelli di TP parziali e imposta lo stato."""
        self._position_side      = side
        self._position_size      = size
        self._position_size_orig = size
        self._entry_price        = fill_price
        self._entry_time         = time.time()
        self._state              = BotState.POSITION_OPEN
        self._accumulated_trade_pnl = _ZERO
        self._partial_locked_pnl   = _ZERO
        self._kelly_risk_size    = size        # snapshot pre-scale-in per Kelly
        self._kelly_risk_entry   = fill_price  # entry originale per Kelly

        atr = signal.atr
        s   = self._cfg.strategy

        # SL su struttura di mercato: il più lontano tra ATR-based e swing structure.
        # Questo riduce i whipsaw posizionando lo SL sotto/sopra minimi/massimi recenti.
        atr_sl_distance = s.sl_atr_mult * atr

        if side == Side.BUY:
            atr_sl = fill_price - atr_sl_distance
            try:
                swing_sl = self._indicators.recent_swing_low(5) * Decimal('0.999')  # -0.1% buffer
                # Usa il più lontano (conservativo) tra i due SL
                self._sl_price = min(atr_sl, swing_sl)
            except ValueError:
                self._sl_price = atr_sl
        else:
            atr_sl = fill_price + atr_sl_distance
            try:
                swing_sl = self._indicators.recent_swing_high(5) * Decimal('1.001')  # +0.1% buffer
                self._sl_price = max(atr_sl, swing_sl)
            except ValueError:
                self._sl_price = atr_sl
        self._sl_price_orig = self._sl_price

        # Regime-specific TP multiplier: allarga/stringe i target in base al regime
        regime = self._aggregator.detect_regime()
        if regime == "STRONG_TREND":
            tp_mult = Decimal('1.5')  # trend forte: target più lontani
        elif regime == "RANGING":
            tp_mult = Decimal('0.7')  # ranging: target più stretti
        elif regime == "CHOPPY":
            tp_mult = Decimal('0.5')  # choppy: exit rapido
        else:
            tp_mult = Decimal('1.0')

        # 3 livelli di Partial TP (adattati al regime)
        # TP3 viene affinato usando il livello S/R più vicino nella direzione del trade.
        if side == Side.BUY:
            tp1 = fill_price + s.partial_tp1_atr * atr * tp_mult
            tp2 = fill_price + s.partial_tp2_atr * atr * tp_mult
            tp3_atr = fill_price + s.partial_tp3_atr * atr * tp_mult
            # Usa la resistenza strutturale più vicina sopra fill_price come TP3 se disponibile
            # e se cade tra TP2 e tp3_atr × 1.5 (non troppo vicino né troppo lontano)
            try:
                nearest_r = self._indicators.nearest_resistance_above(fill_price)
                if nearest_r is not None and tp2 < nearest_r < tp3_atr * Decimal('1.5'):
                    tp3 = nearest_r * Decimal('0.999')  # -0.1% buffer sotto la resistenza
                    logger.debug("%s TP3 affinato su resistenza strutturale: %.4f (ATR era %.4f)",
                                 self.symbol, tp3, tp3_atr)
                else:
                    tp3 = tp3_atr
                    logger.debug("%s Nessuna resistenza valida sopra %.4f — TP3 da ATR: %.4f",
                                 self.symbol, fill_price, tp3_atr)
            except Exception:
                tp3 = tp3_atr
            self._tp_levels = [
                TPLevel(tp1, s.partial_tp1_pct),
                TPLevel(tp2, s.partial_tp2_pct),
                TPLevel(tp3, s.partial_tp3_pct),
            ]
        else:
            tp1 = fill_price - s.partial_tp1_atr * atr * tp_mult
            tp2 = fill_price - s.partial_tp2_atr * atr * tp_mult
            tp3_atr = fill_price - s.partial_tp3_atr * atr * tp_mult
            # Usa il supporto strutturale più vicino sotto fill_price come TP3 per SHORT
            try:
                nearest_s = self._indicators.nearest_support_below(fill_price)
                if nearest_s is not None and tp3_atr * Decimal('0.666') < nearest_s < tp2:
                    tp3 = nearest_s * Decimal('1.001')  # +0.1% buffer sopra il supporto
                    logger.debug("%s TP3 SHORT affinato su supporto strutturale: %.4f (ATR era %.4f)",
                                 self.symbol, tp3, tp3_atr)
                else:
                    tp3 = tp3_atr
                    logger.debug("%s Nessun supporto valido sotto %.4f — TP3 SHORT da ATR: %.4f",
                                 self.symbol, fill_price, tp3_atr)
            except Exception:
                tp3 = tp3_atr
            self._tp_levels = [
                TPLevel(tp1, s.partial_tp1_pct),
                TPLevel(tp2, s.partial_tp2_pct),
                TPLevel(tp3, s.partial_tp3_pct),
            ]

        tp1_price = self._tp_levels[0].price if self._tp_levels else _ZERO
        tp2_price = self._tp_levels[1].price if len(self._tp_levels) > 1 else _ZERO
        tp3_price = self._tp_levels[2].price if len(self._tp_levels) > 2 else _ZERO

        logger.info(
            "%s [Loky] POSIZIONE APERTA: %s %.3f @ %.4f | SL=%.4f | TP1=%.4f TP2=%.4f TP3=%.4f",
            self.symbol, side.name, size, fill_price, self._sl_price,
            tp1_price, tp2_price, tp3_price,
        )

        # Notifica account risk
        if self._account_risk:
            self._account_risk.register_open(self.symbol)

        # Notifica portfolio risk
        if self._portfolio_risk:
            self._portfolio_risk.register_open(self.symbol, fill_price * size)

        # Telegram
        if self._notifier:
            self._fire_and_forget(self._notifier.trade_opened(
                symbol=self.symbol,
                side="LONG" if side == Side.BUY else "SHORT",
                entry=fill_price,
                tp=tp1_price,
                sl=self._sl_price,
                size=size,
                capital=self._capital,
            ))

        self._start_hold_time_check()
        await self._save_state()

        if _prom:
            _position_gauge.labels(symbol=self.symbol).set(1)

    async def _handle_fill(self, order: Order) -> None:
        if self._entry_order is None and self._state == BotState.ENTERING:
            logger.warning("%s Fill ricevuto prima di _entry_order — ignorato.", self.symbol)
            return
        if self._state == BotState.ENTERING and self._entry_order and order.id == self._entry_order.id:
            await self._open_position(order.side, order.filled_size, order.price, self._current_signal)
            if self._cfg.live_trading_enabled and self._tp_levels:
                await self._gw.submit_tp_sl(
                    self.symbol,
                    order.side,
                    self._position_size,
                    self._tp_levels[0].price,
                    self._sl_price,
                )
            return

        if self._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            await self._close_remaining_locked(order.price, "TP/SL hit")

    async def _handle_partial_fill(self, order: Order) -> None:
        """Gestisce fill parziali (PartiallyFilled) da Bybit.

        - In ENTERING: logga e aspetta il FILLED finale (Bybit invierà un evento Filled
          quando l'ordine è completamente eseguito).
        - In POSITION_OPEN/PARTIAL_EXIT: tratta come chiusura parziale della posizione
          (es. TP parziale eseguito dal exchange).
        """
        if self._state == BotState.ENTERING:
            logger.info(
                "%s Partial fill in ENTERING: %.3f/%s filled @ %.4f — attendo fill completo.",
                self.symbol, order.filled_size,
                self._current_signal.size if self._current_signal else "?",
                order.price,
            )
            return

        if self._state in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            # Fill parziale su ordine di chiusura (TP/SL parziale dal exchange)
            partial_size = order.filled_size
            if partial_size > _ZERO and partial_size < self._position_size:
                logger.info(
                    "%s Partial exit fill: %.3f @ %.4f (residuo: %.3f)",
                    self.symbol, partial_size, order.price,
                    self._position_size - partial_size,
                )
                await self._close_partial(order.price, partial_size, "partial_fill_exchange")

    # ------------------------------------------------------------------
    # Chiusura parziale (partial TP)
    # ------------------------------------------------------------------

    async def _close_partial(
        self, exit_price: Decimal, close_size: Decimal, reason: str
    ) -> None:
        """Chiude una frazione della posizione. Lascia il resto aperto."""
        pnl = self._calc_pnl_for_size(exit_price, close_size)
        self.realized_pnl += pnl
        self.total_trades  += 1
        self._position_size -= close_size
        self._accumulated_trade_pnl += pnl
        self._partial_locked_pnl += pnl  # traccia profitto locked per Kelly
        self._state = BotState.PARTIAL_EXIT

        logger.info(
            "%s [Loky] PARTIAL EXIT (%s) | exit=%.4f size=%.3f pnl=%.4f USDT | residuo=%.3f",
            self.symbol, reason, exit_price, close_size, pnl, self._position_size,
        )

        # Salva trade parziale
        if self._position_side:
            fee = exit_price * close_size * self._cfg.fee_taker
            trade = Trade(
                symbol=self.symbol,
                side=self._position_side,
                size=close_size,
                price=exit_price,
                commission=fee,
                commission_asset="USDT",
                order_id=f"partial_{int(time.time())}",
                timestamp=time.time(),
                realized_pnl=pnl,
            )
            await self._state_mgr.save_trade(trade)

        if _prom:
            _pnl_gauge.labels(symbol=self.symbol).set(float(self.realized_pnl))

        # Se tutta la posizione è chiusa dopo i parziali, finalizza
        if self._position_size <= _ZERO:
            await self._finalize_close(exit_price, reason)

    # ------------------------------------------------------------------
    # Chiusura totale della posizione residua
    # ------------------------------------------------------------------

    async def _close_remaining(self, exit_price: Decimal, reason: str) -> None:
        """Chiude la posizione residua (SL hit, max hold, o forza)."""
        async with self._order_lock:
            await self._close_remaining_locked(exit_price, reason)

    async def _close_remaining_locked(self, exit_price: Decimal, reason: str) -> None:
        """Implementazione interna protetta dal lock."""
        if self._state not in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT, BotState.EXITING):
            return
        if self._position_size <= _ZERO:
            await self._finalize_close(exit_price, reason)
            return
        self._state = BotState.EXITING

        close_size = self._position_size
        pnl = self._calc_pnl_for_size(exit_price, close_size)
        self.realized_pnl += pnl
        self.total_trades  += 1
        self._accumulated_trade_pnl += pnl

        logger.info(
            "%s [Loky] USCITA (%s) | exit=%.4f size=%.3f pnl=%.4f USDT | PnL totale=%.4f",
            self.symbol, reason, exit_price, close_size, pnl, self.realized_pnl,
        )

        # Salva trade
        if self._position_side:
            fee = exit_price * close_size * self._cfg.fee_taker
            trade = Trade(
                symbol=self.symbol,
                side=self._position_side,
                size=close_size,
                price=exit_price,
                commission=fee,
                commission_asset="USDT",
                order_id=f"close_{int(time.time())}",
                timestamp=time.time(),
                realized_pnl=pnl,
            )
            await self._state_mgr.save_trade(trade)

        await self._finalize_close(exit_price, reason)

    async def _finalize_close(self, exit_price: Decimal, reason: str) -> None:
        """Aggiorna Kelly, metriche, attribution PnL, notifiche e resetta lo stato."""

        # Usa il PnL accumulato reale (somma di tutti i partial exit + chiusura finale)
        # invece di ricalcolare su position_size_orig × ultimo exit_price (errato con partial TP)
        total_pnl_trade = self._accumulated_trade_pnl

        # Aggiorna Kelly con il risultato del trade completo.
        # risk_amount netto: rischio iniziale meno profitto già locked da partial exits.
        # Dopo TP1 lo SL va a breakeven → il rischio residuo è effettivamente ridotto.
        # Questo evita di sovrastimare il win rate nel Kelly Criterion.
        kelly_entry = self._kelly_risk_entry if self._kelly_risk_entry > _ZERO else self._entry_price
        kelly_size  = self._kelly_risk_size  if self._kelly_risk_size  > _ZERO else self._position_size_orig
        sl_distance = abs(kelly_entry - self._sl_price_orig)
        gross_risk  = sl_distance * kelly_size
        # Rischio netto = max(rischio lordo - profitto locked, piccolo floor per evitare div/0)
        risk_amount = max(gross_risk - self._partial_locked_pnl, gross_risk * Decimal('0.1'))
        self._kelly.update(total_pnl_trade, risk_amount)

        # PnL attribution per strategy engine (per analytics/performance review)
        strategy_name = self._current_signal.strategy_name if self._current_signal else "unknown"
        self.pnl_by_strategy[strategy_name] = (
            self.pnl_by_strategy.get(strategy_name, _ZERO) + total_pnl_trade
        )
        self.trades_by_strategy[strategy_name] = (
            self.trades_by_strategy.get(strategy_name, 0) + 1
        )
        # Aggiorna pesi adattivi per strategia nell'aggregator
        self._aggregator.record_trade_result(strategy_name, total_pnl_trade)

        if _prom:
            result_label = "win" if total_pnl_trade > _ZERO else "loss"
            side_name = self._position_side.name if self._position_side else "UNKNOWN"
            _fills_total.labels(symbol=self.symbol, side=side_name, result=result_label).inc()
            _pnl_gauge.labels(symbol=self.symbol).set(float(self.realized_pnl))
            _position_gauge.labels(symbol=self.symbol).set(0)
            _daily_loss.labels(symbol=self.symbol).set(float(self.realized_pnl))

        # Telegram — invia PnL del singolo trade, non il cumulativo
        if self._notifier:
            duration_h = (time.time() - self._entry_time) / 3600 if self._entry_time > 0 else 0.0
            self._fire_and_forget(self._notifier.trade_closed(
                symbol=self.symbol,
                side="LONG" if self._position_side == Side.BUY else "SHORT",
                entry=self._entry_price,
                exit_price=exit_price,
                pnl=total_pnl_trade,
                reason=reason,
                duration_h=duration_h,
            ))

        # Cooldown post-loss: solo su perdite reali, non su TP o timeout neutro
        if total_pnl_trade < _ZERO:
            self._cooldown_remaining = self._cfg.strategy.loss_cooldown_candles
        else:
            self._cooldown_remaining = 0  # Reset su vincita (non bloccare opportunità successive)

        # Conteggio wins/losses
        if total_pnl_trade > _ZERO:
            self.total_wins += 1
        elif total_pnl_trade < _ZERO:
            self.total_losses += 1

        # Aggiorna streak e circuit breaker
        self._update_streak(total_pnl_trade)

        # Account risk manager: usa PnL accumulato reale
        if self._account_risk:
            self._account_risk.register_close(self.symbol, total_pnl_trade)

        # Portfolio risk manager
        if self._portfolio_risk:
            self._portfolio_risk.register_close(self.symbol)

        self._reset_position_state()
        await self._save_state()

    async def _exit_position_market(self, reason: str) -> None:
        """Chiude la posizione con ordine market (es. max hold time)."""
        if self._state not in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
            return

        if self._cfg.live_trading_enabled:
            try:
                await self._gw.cancel_all_orders()
            except Exception as e:
                logger.warning("%s Errore cancel TP/SL: %s", self.symbol, e)
            close_side = Side.SELL if self._position_side == Side.BUY else Side.BUY
            order = await self._gw.submit_market_order(self.symbol, close_side, self._position_size)
            if order:
                return  # fill arriverà via on_order_update

        last_candle = self._candles[-1] if self._candles else None
        base_price  = last_candle.close if last_candle else self._entry_price
        # Slippage realistico su market exit in paper
        if self._position_side is not None:
            exit_side  = Side.SELL if self._position_side == Side.BUY else Side.BUY
            exit_price = self._calc_paper_fill_price(base_price, exit_side, self._position_size)
        else:
            exit_price = base_price
        await self._close_remaining(exit_price, reason)

    # ------------------------------------------------------------------
    # Max hold time
    # ------------------------------------------------------------------

    def _start_hold_time_check(self) -> None:
        if self._hold_time_task and not self._hold_time_task.done():
            self._hold_time_task.cancel()
        self._hold_time_task = asyncio.create_task(self._hold_time_loop())

    async def _hold_time_loop(self) -> None:
        max_seconds = self._cfg.strategy.max_hold_hours * 3600
        try:
            while True:
                await asyncio.sleep(60)
                if self._state not in (BotState.POSITION_OPEN, BotState.PARTIAL_EXIT):
                    break
                if time.time() - self._entry_time >= max_seconds:
                    logger.warning("%s Max hold time raggiunto. Chiusura forzata.", self.symbol)
                    await self._exit_position_market("max_hold_time")
                    break
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Calcolo PnL
    # ------------------------------------------------------------------

    def _calc_pnl_for_size(self, exit_price: Decimal, size: Decimal) -> Decimal:
        """PnL netto per una data size, con fee maker (entry) + taker (exit)."""
        if size == _ZERO or self._position_side is None:
            return _ZERO
        gross = (exit_price - self._entry_price) * size
        if self._position_side == Side.SELL:
            gross = -gross
        entry_fee_rate = self._cfg.fee_maker if self._cfg.limit_entry_enabled else self._cfg.fee_taker
        fee = self._entry_price * size * entry_fee_rate + exit_price * size * self._cfg.fee_taker
        return gross - fee

    @property
    def unrealized_pnl(self) -> Decimal:
        """PnL non realizzato della posizione aperta corrente."""
        if self._position_size <= _ZERO or self._position_side is None:
            return _ZERO
        return self._calc_pnl_for_size(self._last_price, self._position_size)

    def _calc_paper_fill_price(
        self, base_price: Decimal, side: Side, size: Decimal
    ) -> Decimal:
        """
        Calcola il prezzo di fill simulato per paper trading.

        Include:
        1. Slippage base (config.slippage_pct)
        2. Market impact: se il notional supera lo 0.1% del volume medio USDT,
           aggiunge slippage proporzionale (cap 3× slippage base).

        Questo rende il paper trading molto più realistico rispetto a fill
        a prezzo fisso.
        """
        slip = self._cfg.slippage_pct

        # Market impact basato su dimensione ordine vs volume medio
        try:
            vol_ma = self._indicators.volume_ma()
            notional = size * base_price
            vol_usdt = vol_ma * base_price
            if vol_usdt > _ZERO:
                impact_ratio = notional / vol_usdt
                # 0.1% del volume → slippage addizionale proporzionale, cap 3×
                if impact_ratio > Decimal('0.001'):
                    extra_slip = min(
                        impact_ratio * self._cfg.slippage_pct * Decimal('10'),
                        self._cfg.slippage_pct * Decimal('2'),  # cap: 2× extra = 3× totale
                    )
                    slip = slip + extra_slip
        except ValueError:
            pass

        if side == Side.BUY:
            return base_price * (_ONE + slip)
        return base_price * (_ONE - slip)

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    async def _save_state(self) -> None:
        inv = self._position_size if self._position_side == Side.BUY else -self._position_size
        await self._state_mgr.update_snapshot(
            net_inventory=inv,
            pnl=self.realized_pnl,
            avg_entry=self._entry_price,
            quotes_sent=0,
            fills_total=self.total_trades,
        )

    def _load_state(self) -> None:
        snap = self._state_mgr.load_state()
        if snap:
            self.realized_pnl = snap.get("pnl", _ZERO)
            self.total_trades  = snap.get("fills_total", 0)
            logger.info("%s [Loky] Stato caricato: PnL=%.4f USDT, trade=%d",
                        self.symbol, self.realized_pnl, self.total_trades)

    # ------------------------------------------------------------------
    # Anti-martingale sizing
    # ------------------------------------------------------------------

    def _apply_streak_sizing(self, size: Decimal) -> Decimal:
        """
        Aggiusta la size in base al win/loss streak (anti-martingale).

        Logica:
          Win streak ≥ 3 → size × 1.20 (capitalizza momentum)
          Win streak ≥ 2 → size × 1.10
          Loss streak ≥ 3 → size × 0.65 (protegge il capitale)
          Loss streak ≥ 2 → size × 0.80
          Neutrale       → size × 1.00

        Il moltiplicatore si applica DOPO Kelly sizing, quindi il cap Kelly
        max_fraction rimane valido come limite superiore assoluto.
        Nessun cambiamento se size è già _ZERO.
        """
        if size == _ZERO:
            return size

        if self._win_streak >= 3:
            mult = Decimal("1.20")
        elif self._win_streak >= 2:
            mult = Decimal("1.10")
        elif self._loss_streak >= 3:
            mult = Decimal("0.65")
        elif self._loss_streak >= 2:
            mult = Decimal("0.80")
        else:
            return size

        adj = (size * mult).quantize(Decimal("0.001"))
        logger.debug(
            "%s Anti-martingale: win=%d loss=%d mult=%.2f → size %.3f→%.3f",
            self.symbol, self._win_streak, self._loss_streak, float(mult), size, adj,
        )
        return adj

    def _update_streak(self, pnl: Decimal) -> None:
        """
        Aggiorna win/loss streak e circuit breaker dopo la chiusura di un trade.

        Circuit breaker: se consecutive_losses >= 3, imposta pausa di 15 candle.
        Il numero di perdite consecutive si resetta su qualsiasi vincita.
        """
        if pnl > _ZERO:
            self._win_streak        += 1
            self._loss_streak        = 0
            self._consecutive_losses = 0
        else:
            self._loss_streak        += 1
            self._win_streak          = 0
            self._consecutive_losses += 1

            # Circuit breaker
            max_consec = getattr(self._cfg.strategy, "circuit_breaker_losses", 3)
            pause_candles = getattr(self._cfg.strategy, "circuit_breaker_candles", 15)
            if self._consecutive_losses >= max_consec and self._circuit_breaker_candles == 0:
                self._circuit_breaker_candles = pause_candles
                logger.warning(
                    "%s CIRCUIT BREAKER: %d perdite consecutive → pausa %d candle.",
                    self.symbol, self._consecutive_losses, pause_candles,
                )
                if self._notifier:
                    self._fire_and_forget(self._notifier.info(
                        f"⚡ CIRCUIT BREAKER {self.symbol}: {self._consecutive_losses} perdite consecutive "
                        f"→ pausa {pause_candles} candle."
                    ))

    # ------------------------------------------------------------------
    # Pyramid scaling-in (D2)
    # ------------------------------------------------------------------

    async def _check_scale_in(self, candle: Candle) -> None:
        """
        Pyramid entry per TrendFollowing: aggiunge fino a 1 posizione extra
        quando il trade è già in profitto di almeno 0.5×ATR.

        Condizioni di attivazione:
          • Strategia corrente: TrendFollowing
          • Numero di scale-in già effettuati < 1
          • Profitto attuale > 0.5×ATR (posizione già in guadagno)
          • Size totale post-scaling <= 2× size originale
          • Non in pausa Telegram e daily stop non attivo
          • Segnale TrendFollowing ancora nella stessa direzione

        All'esecuzione:
          • Aggiunge size = 0.5 × position_size_orig
          • Aggiorna entry_price alla media ponderata delle entry
          • SL aggiornato al SL corrente (già trailed) — nessuna modifica
          • Aggiorna position_size_orig per calcoli futuri Kelly/PnL
        """
        # Prerequisiti generali
        if self._scale_in_count >= 1:
            return
        if self._current_signal is None:
            return
        if self._current_signal.strategy_name != "trend_following":
            return
        if self._is_daily_stop_active():
            return
        if self._notifier and self._notifier.is_paused:
            return

        try:
            atr = self._indicators.atr()
        except ValueError:
            return

        # Verifica che la posizione sia già in profitto di almeno 0.5×ATR
        price = candle.close
        if self._position_side == Side.BUY:
            profit = price - self._entry_price
        else:
            profit = self._entry_price - price

        min_profit_threshold = self._cfg.strategy.scale_in_profit_atr_mult * atr
        if profit < min_profit_threshold:
            return

        # Verifica che il segnale TrendFollowing sia ancora nella stessa direzione
        try:
            tf_sig = self._trend_follow.detect(self._candles)
        except Exception:
            return

        expected_type = SignalType.LONG if self._position_side == Side.BUY else SignalType.SHORT
        if tf_sig.signal_type != expected_type:
            return

        # Calcola la size da aggiungere (50% della size originale)
        add_size = (self._position_size_orig * Decimal('0.5')).quantize(Decimal('0.001'))
        if add_size <= _ZERO:
            return

        # Verifica portfolio risk per il notional aggiuntivo
        if self._portfolio_risk is not None:
            notional_add = price * add_size
            ok, reason = self._portfolio_risk.can_open(self.symbol, notional_add)
            if not ok:
                logger.debug("%s Scale-in bloccato da PortfolioRisk: %s", self.symbol, reason)
                return

        # Esegui lo scale-in
        fill_price = self._calc_paper_fill_price(candle.open, self._position_side, add_size) \
            if not self._cfg.live_trading_enabled else price

        if self._cfg.live_trading_enabled:
            order = await self._gw.submit_market_order(self.symbol, self._position_side, add_size)
            if order is None:
                logger.warning("%s Scale-in: submit_market_order fallito", self.symbol)
                return
            fill_price = order.price if hasattr(order, 'price') and order.price else price

        # Aggiorna stato: weighted average entry price
        total_size = self._position_size + add_size
        if total_size > _ZERO:
            self._entry_price = (
                (self._entry_price * self._position_size + fill_price * add_size) / total_size
            )
        self._position_size  += add_size
        # NON aggiornare _position_size_orig: Kelly usa _kelly_risk_size (snapshot pre-scale-in)
        # per evitare di corrompere il calcolo del risk_amount.
        self._scale_in_count += 1
        self._scale_in_size  += add_size

        # Aggiorna portfolio risk: aggiorna il notional in-place (non registrare doppia apertura)
        if self._portfolio_risk is not None:
            current = self._portfolio_risk._open_notional.get(self.symbol, _ZERO)
            self._portfolio_risk._open_notional[self.symbol] = current + fill_price * add_size

        logger.info(
            "%s [Loky] SCALE-IN #%d (%s) | +%.3f @ %.4f | size totale=%.3f | "
            "entry avg=%.4f | profitto pre-add=%.4f USDT",
            self.symbol, self._scale_in_count,
            self._current_signal.strategy_name,
            add_size, fill_price, self._position_size,
            self._entry_price, float(profit),
        )

        if self._notifier:
            self._fire_and_forget(self._notifier.info(
                f"📈 SCALE-IN {self.symbol} +{float(add_size):.3f} @ {float(fill_price):.4f} "
                f"(tot={float(self._position_size):.3f}, avg={float(self._entry_price):.4f})"
            ))

    def _reset_position_state(self) -> None:
        self._state               = BotState.FLAT
        self._current_signal      = None
        self._entry_order         = None
        self._position_side       = None
        self._position_size       = _ZERO
        self._position_size_orig  = _ZERO
        self._entry_price         = _ZERO
        self._entry_time          = 0.0
        self._sl_price            = _ZERO
        self._sl_price_orig       = _ZERO
        self._tp_levels           = []
        self._pending_signal      = None
        self._scale_in_count      = 0
        self._scale_in_size       = _ZERO
        self._kelly_risk_size     = _ZERO
        self._kelly_risk_entry    = _ZERO
        if self._hold_time_task and not self._hold_time_task.done():
            self._hold_time_task.cancel()
