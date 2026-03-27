"""
DirectionalBot — bot direzionale Breakout/Momentum per Binance Futures.

Macchina a stati per symbol:
    FLAT → ENTERING → POSITION_OPEN → EXITING → FLAT

Feature:
  • Multi-timeframe: segnali su 15m filtrati dal trend 1h (EMA20/50)
  • TP/SL calcolati sul prezzo di fill reale
  • Cancellazione TP/SL prima di ogni chiusura forzata
  • Daily loss check ad ogni candela
  • Paper trading con simulazione TP/SL su high/low
  • Notifiche Telegram su apertura/chiusura posizione
"""

import asyncio
import logging
import time
from collections import deque
from decimal import Decimal
from enum import Enum, auto
from typing import Optional

try:
    from prometheus_client import Counter, Gauge
    _prom = True
except ImportError:
    _prom = False

from src.config import BotSettings
from src.core.account_risk import AccountRiskManager
from src.core.kelly_sizing import KellySizer
from src.models import Candle, Order, OrderStatus, Side, Signal, SignalType, Trade
from src.notifications.telegram import TelegramNotifier
from src.state.persistency import StateManager
from src.strategy.breakout_engine import BreakoutEngine
from src.strategy.indicator_engine import IndicatorEngine
from src.strategy.mean_reversion_engine import MeanReversionEngine
from src.strategy.trend_following_engine import TrendFollowingEngine
from src.strategy.signal_aggregator import SignalAggregator

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")


class BotState(Enum):
    FLAT          = auto()
    ENTERING      = auto()
    POSITION_OPEN = auto()
    EXITING       = auto()


# ---------------------------------------------------------------------------
# Prometheus metrics (opzionali)
# ---------------------------------------------------------------------------
if _prom:
    _fills_total    = Counter("directional_fills_total",   "Fill totali",      ["symbol", "side"])
    _pnl_gauge      = Gauge(  "directional_pnl_usdt",      "PnL corrente",     ["symbol"])
    _position_gauge = Gauge(  "directional_position_open", "Posizione aperta", ["symbol"])
    _signals_total  = Counter("directional_signals_total", "Segnali rilevati", ["symbol", "direction"])


class DirectionalBot:
    """
    Bot direzionale per un singolo symbol su Binance Futures.

    Args:
        symbol        — es. "BTCUSDT"
        config        — BotSettings caricato da config.yaml
        execution_gw  — gateway di esecuzione (live o paper)
        state_manager — SQLite state manager
        capital       — USDT disponibile per questo bot
    """

    def __init__(
        self,
        symbol: str,
        config: BotSettings,
        execution_gw,
        state_manager: StateManager,
        capital: Decimal = Decimal("500"),
        account_risk: Optional["AccountRiskManager"] = None,
        notifier: Optional["TelegramNotifier"] = None,
    ) -> None:
        self.symbol       = symbol
        self._cfg         = config
        self._gw          = execution_gw
        self._state_mgr   = state_manager
        self._capital     = capital
        self._account_risk = account_risk  # None = nessun controllo account-level
        self._notifier     = notifier       # None = notifiche disabilitate

        s = config.strategy
        self._primary_tf = config.primary_timeframe         # es. "15m"
        self._confirm_tf = config.confirmation_timeframe    # es. "1h"

        # Indicatori sul timeframe primario (15m) — per segnali
        self._indicators = IndicatorEngine(
            ema_fast   = s.ema_fast,
            ema_slow   = s.ema_slow,
            rsi_period = 14,
            atr_period = s.atr_period,
            vol_period = s.vol_period,
        )
        # Indicatori sul timeframe di conferma (1h) — per filtro trend HTF
        self._htf_indicators = IndicatorEngine(
            ema_fast   = s.ema_fast,
            ema_slow   = s.ema_slow,
            rsi_period = 14,
            atr_period = s.atr_period,
            vol_period = s.vol_period,
        )

        # Strategy engines
        self._breakout       = BreakoutEngine(config, self._indicators, capital)
        self._mean_reversion = MeanReversionEngine(config, self._indicators, capital)
        self._trend_following = TrendFollowingEngine(config, self._indicators, capital)

        # Signal aggregator con regime detection
        self._aggregator = SignalAggregator(
            indicators=self._indicators,
            htf_indicators=self._htf_indicators,
        )

        self._candles: deque[Candle] = deque(maxlen=200)        # candele 15m

        # Kelly Criterion sizing (attivo dopo kelly_min_trades trade)
        self._kelly = KellySizer(
            history_trades=config.kelly_min_trades,
            half_kelly=True,
            min_fraction=Decimal('0.005'),
            max_fraction=Decimal('0.03'),
        ) if config.kelly_sizing_enabled else None

        # Macchina a stati
        self._state: BotState               = BotState.FLAT
        self._current_signal: Optional[Signal] = None
        self._entry_order:    Optional[Order]  = None
        self._position_side:  Optional[Side]   = None
        self._position_size:  Decimal          = _ZERO
        self._entry_price:    Decimal          = _ZERO
        self._entry_time:     float            = 0.0
        self._tp_price:       Decimal          = _ZERO
        self._sl_price:       Decimal          = _ZERO
        # Flag: daily loss stop già scattato oggi (reset esterno a mezzanotte)
        self._daily_stop_triggered: bool       = False
        # Cooldown post-loss: N candele di pausa dopo SL hit
        self._cooldown_remaining: int          = 0
        # Segnale in attesa di entry alla prossima candela (next_candle_entry)
        self._pending_signal: Optional[Signal] = None

        # Partial TP tracking
        self._partial_tp1_hit: bool = False  # 50% chiuso a TP1
        self._partial_tp2_hit: bool = False  # 25% chiuso a TP2
        self._original_size: Decimal = _ZERO  # size iniziale prima dei partial close

        # Loss streak recovery
        self._consecutive_losses: int = 0

        # PnL tracking
        self.realized_pnl: Decimal = _ZERO
        self.total_trades: int     = 0

        self._hold_time_task: Optional[asyncio.Task] = None

        self._load_state()

    # ------------------------------------------------------------------
    # Evento principale: nuova candela chiusa
    # ------------------------------------------------------------------

    async def on_candle(self, candle: Candle) -> None:
        """
        Chiamato dal gateway dati ad ogni candela chiusa.
        Riceve sia candele 15m (primary) che 1h (confirmation).
        """
        if candle.symbol != self.symbol:
            return

        # Smista per timeframe
        if candle.timeframe == self._confirm_tf:
            self._htf_indicators.update(candle)
            return  # le candele 1h aggiornano solo il filtro HTF, non generano segnali

        if candle.timeframe != self._primary_tf:
            return  # timeframe non gestito

        # --- Timeframe primario (15m) ---
        self._candles.append(candle)
        self._indicators.update(candle)

        # Paper trading: controlla trailing stop e TP/SL sulla candela corrente
        if self._state == BotState.POSITION_OPEN and not self._cfg.live_trading_enabled:
            self._update_trailing_sl(candle)
            await self._check_paper_tp_sl(candle)
            return

        # Entry next-candle: se c'è un segnale pendente, entra ora (open candela attuale)
        if self._pending_signal is not None and self._state == BotState.FLAT:
            sig = self._pending_signal
            self._pending_signal = None
            await self._enter_position(sig, fill_price_override=candle.open)
            return

        if self._state != BotState.FLAT:
            return

        # Decrementa cooldown post-loss
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            logger.debug("%s Cooldown attivo: %d candele rimaste.", self.symbol, self._cooldown_remaining)
            return

        if self._is_daily_stop_active():
            return

        # Rileva regime e filtra engine per regime
        preferred = self._aggregator.preferred_strategies()

        # Raccogli segnali da tutti gli engine adatti al regime
        signals: list[Signal] = []
        if "breakout" in preferred:
            signals.append(self._breakout.detect(self._candles))
        if "mean_reversion" in preferred:
            signals.append(self._mean_reversion.detect(self._candles))
        if "trend_following" in preferred:
            signals.append(self._trend_following.detect(self._candles))

        # Seleziona il migliore tramite aggregator (scoring + soglia minima)
        signal = self._aggregator.select_best(signals)
        if signal is None:
            return

        # Filtro multi-timeframe: il trend 1h deve essere allineato
        if not self._htf_trend_aligned(signal.signal_type):
            logger.debug(
                "%s Segnale %s scartato: trend 1h non allineato.",
                self.symbol, signal.signal_type.name,
            )
            return

        if _prom:
            _signals_total.labels(symbol=self.symbol, direction=signal.signal_type.name).inc()

        # Applica Kelly sizing se disponibile
        if self._kelly and self._kelly.is_ready and signal.atr > _ZERO:
            sl_distance = self._cfg.strategy.sl_atr_mult * signal.atr
            kelly_size = self._kelly.position_size(
                self._capital, sl_distance, fallback_fraction=self._cfg.risk_per_trade_pct
            )
            if kelly_size > _ZERO:
                kelly_size = kelly_size.quantize(Decimal('0.001'))
                logger.info(
                    "%s Kelly sizing: %.3f → %.3f",
                    self.symbol, signal.size, kelly_size,
                )
                signal.size = kelly_size

        # Loss streak recovery: riduce size del 50% dopo 3+ loss consecutive
        if self._consecutive_losses >= 3:
            signal.size = (signal.size * Decimal('0.5')).quantize(Decimal('0.001'))
            logger.info(
                "%s Loss streak recovery (%d consecutive): size dimezzata a %.3f",
                self.symbol, self._consecutive_losses, signal.size,
            )

        if self._cfg.next_candle_entry:
            # Salva il segnale e aspetta la candela successiva
            self._pending_signal = signal
            logger.debug("%s Segnale %s in attesa: entry al prossimo open.", self.symbol, signal.signal_type.name)
        else:
            await self._enter_position(signal)

    def _htf_trend_aligned(self, signal_type: SignalType) -> bool:
        """
        Verifica che il trend su 1h sia allineato con il segnale 15m.
        LONG valido solo se EMA20(1h) > EMA50(1h).
        SHORT valido solo se EMA20(1h) < EMA50(1h).
        Se gli indicatori 1h non sono ancora pronti, lascia passare il segnale.
        """
        if not self._htf_indicators.ready():
            logger.debug(
                "%s HTF indicators non pronti — segnale bloccato fino a warmup completato.",
                self.symbol,
            )
            return False  # blocca segnali finché HTF non è pronto
        try:
            ema_fast_htf = self._htf_indicators.ema_fast()
            ema_slow_htf = self._htf_indicators.ema_slow()
        except ValueError:
            return True

        if signal_type == SignalType.LONG:
            return ema_fast_htf > ema_slow_htf
        if signal_type == SignalType.SHORT:
            return ema_fast_htf < ema_slow_htf
        return False

    def _update_trailing_sl(self, candle: Candle) -> None:
        """
        Aggiorna lo stop loss seguendo il prezzo (trailing stop).
        Attivo dopo che il profitto supera 1×ATR.
        Il nuovo SL è posizionato a trail_atr_mult×ATR dietro il prezzo corrente.
        """
        if not self._cfg.strategy.trailing_stop_enabled:
            return
        if self._position_side is None or self._position_size == _ZERO:
            return

        # Recupera ATR corrente
        try:
            atr = self._indicators.atr()
        except ValueError:
            return

        trail_dist = self._cfg.strategy.trail_atr_mult * atr
        price      = candle.close

        if self._position_side == Side.BUY:
            profit = price - self._entry_price
            if profit < atr:
                return  # non ancora in profitto sufficiente
            new_sl = price - trail_dist
            if new_sl > self._sl_price:
                logger.debug(
                    "%s Trailing SL aggiornato: %.4f → %.4f",
                    self.symbol, self._sl_price, new_sl,
                )
                self._sl_price = new_sl
        else:  # SELL / SHORT
            profit = self._entry_price - price
            if profit < atr:
                return
            new_sl = price + trail_dist
            if new_sl < self._sl_price:
                logger.debug(
                    "%s Trailing SL aggiornato: %.4f → %.4f",
                    self.symbol, self._sl_price, new_sl,
                )
                self._sl_price = new_sl

    async def _check_paper_tp_sl(self, candle: Candle) -> None:
        """
        Simula hit di TP/SL in paper trading con partial take-profit (50/25/25).
        Priorità: SL (protezione capitale) poi partial TP1 → TP2 → TP3 (full close).
        """
        sl = getattr(self, '_sl_price', _ZERO)
        if sl == _ZERO or self._position_size == _ZERO:
            return

        s = self._cfg.strategy
        atr = getattr(self, '_signal_atr', _ZERO)

        # Calcola livelli partial TP
        if self._position_side == Side.BUY:
            tp1 = self._entry_price + s.partial_tp1_atr * atr if atr > _ZERO else self._tp_price
            tp2 = self._entry_price + s.partial_tp2_atr * atr if atr > _ZERO else self._tp_price
            tp3 = self._entry_price + s.partial_tp3_atr * atr if atr > _ZERO else self._tp_price

            # SL check first
            if candle.low <= sl:
                await self._close_position(sl, "SL hit (paper)")
                return

            # Partial TP1: chiudi 50%
            if not self._partial_tp1_hit and candle.high >= tp1:
                close_size = (self._original_size * s.partial_tp1_pct).quantize(Decimal('0.001'))
                if close_size > _ZERO and close_size < self._position_size:
                    await self._partial_close(tp1, close_size, "TP1 (50%)")
                    self._partial_tp1_hit = True
                    # Sposta SL a breakeven dopo TP1
                    self._sl_price = self._entry_price
                    logger.info("%s SL spostato a breakeven dopo TP1", self.symbol)
                    return

            # Partial TP2: chiudi 25%
            if self._partial_tp1_hit and not self._partial_tp2_hit and candle.high >= tp2:
                close_size = (self._original_size * s.partial_tp2_pct).quantize(Decimal('0.001'))
                if close_size > _ZERO and close_size < self._position_size:
                    await self._partial_close(tp2, close_size, "TP2 (25%)")
                    self._partial_tp2_hit = True
                    return

            # TP3 / full TP: chiudi il restante con trailing
            if self._partial_tp2_hit and candle.high >= tp3:
                await self._close_position(tp3, "TP3 trail (paper)")
                return

            # Fallback: TP classico se partial TP non attivo (atr=0)
            if atr == _ZERO and candle.high >= self._tp_price:
                await self._close_position(self._tp_price, "TP hit (paper)")

        elif self._position_side == Side.SELL:
            tp1 = self._entry_price - s.partial_tp1_atr * atr if atr > _ZERO else self._tp_price
            tp2 = self._entry_price - s.partial_tp2_atr * atr if atr > _ZERO else self._tp_price
            tp3 = self._entry_price - s.partial_tp3_atr * atr if atr > _ZERO else self._tp_price

            if candle.high >= sl:
                await self._close_position(sl, "SL hit (paper)")
                return

            if not self._partial_tp1_hit and candle.low <= tp1:
                close_size = (self._original_size * s.partial_tp1_pct).quantize(Decimal('0.001'))
                if close_size > _ZERO and close_size < self._position_size:
                    await self._partial_close(tp1, close_size, "TP1 (50%)")
                    self._partial_tp1_hit = True
                    self._sl_price = self._entry_price
                    logger.info("%s SL spostato a breakeven dopo TP1", self.symbol)
                    return

            if self._partial_tp1_hit and not self._partial_tp2_hit and candle.low <= tp2:
                close_size = (self._original_size * s.partial_tp2_pct).quantize(Decimal('0.001'))
                if close_size > _ZERO and close_size < self._position_size:
                    await self._partial_close(tp2, close_size, "TP2 (25%)")
                    self._partial_tp2_hit = True
                    return

            if self._partial_tp2_hit and candle.low <= tp3:
                await self._close_position(tp3, "TP3 trail (paper)")
                return

            if atr == _ZERO and candle.low <= self._tp_price:
                await self._close_position(self._tp_price, "TP hit (paper)")

    async def _partial_close(self, exit_price: Decimal, close_size: Decimal, reason: str) -> None:
        """Chiude parzialmente la posizione (partial TP)."""
        pnl = self._calc_partial_pnl(exit_price, close_size)
        self.realized_pnl += pnl
        self._position_size -= close_size

        logger.info(
            "%s PARTIAL CLOSE (%s) | exit=%.4f size=%.3f pnl=%.4f | remaining=%.3f",
            self.symbol, reason, exit_price, close_size, pnl, self._position_size,
        )

        if self._notifier:
            asyncio.create_task(self._notifier.trade_closed(
                symbol=self.symbol,
                side="LONG" if self._position_side == Side.BUY else "SHORT",
                entry=self._entry_price,
                exit_price=exit_price,
                pnl=pnl,
                reason=reason,
            ))

        self._save_state()

    def _calc_partial_pnl(self, exit_price: Decimal, size: Decimal) -> Decimal:
        """PnL netto per chiusura parziale."""
        if size == _ZERO or self._position_side is None:
            return _ZERO
        gross = (exit_price - self._entry_price) * size
        if self._position_side == Side.SELL:
            gross = -gross
        fee = (self._entry_price + exit_price) * size * self._cfg.fee_taker
        return gross - fee

    def _is_daily_stop_active(self) -> bool:
        """Ritorna True se il daily loss limit è stato raggiunto."""
        if self.realized_pnl <= self._cfg.max_daily_loss:
            if not self._daily_stop_triggered:
                logger.warning(
                    "%s Daily loss limit raggiunto (%.2f USDT). Trading sospeso.",
                    self.symbol, self.realized_pnl,
                )
                self._daily_stop_triggered = True
            return True
        # Reset flag se PnL risale (possibile se si ricarica stato da db)
        self._daily_stop_triggered = False
        return False

    # ------------------------------------------------------------------
    # Evento fill ordine
    # ------------------------------------------------------------------

    async def on_order_update(self, order: Order) -> None:
        """Chiamato dal gateway esecuzione ad ogni aggiornamento ordine."""
        if order.symbol != self.symbol:
            return

        if order.status == OrderStatus.FILLED:
            await self._handle_fill(order)
        elif order.status == OrderStatus.REJECTED:
            logger.error("%s Ordine rifiutato: %s", self.symbol, order.id)
            if self._state == BotState.ENTERING:
                self._state = BotState.FLAT

    # ------------------------------------------------------------------
    # Ingresso posizione
    # ------------------------------------------------------------------

    async def _enter_position(
        self,
        signal: Signal,
        fill_price_override: Optional[Decimal] = None,
    ) -> None:
        # Verifica account-level risk prima di procedere
        if self._account_risk and not self._account_risk.can_open_position(self.symbol):
            return

        self._state          = BotState.ENTERING
        self._current_signal = signal
        side = Side.BUY if signal.signal_type == SignalType.LONG else Side.SELL

        logger.info(
            "%s ENTRATA %s | entry≈%.4f tp=%.4f sl=%.4f size=%.3f",
            self.symbol, signal.signal_type.name,
            signal.entry_price, signal.take_profit, signal.stop_loss, signal.size,
        )

        if not self._cfg.live_trading_enabled:
            # Paper trading: usa fill_price_override (next-candle open) o candle close
            base_price = fill_price_override or (
                self._candles[-1].close if self._candles else signal.entry_price
            )
            # Applica slippage simulato
            slip = self._cfg.slippage_pct
            fill_price = base_price * (1 + slip) if side == Side.BUY else base_price * (1 - slip)
            self._open_position(side, signal.size, fill_price, signal)
            return

        order = await self._gw.submit_market_order(self.symbol, side, signal.size)
        if order is None:
            logger.error("%s submit_market_order fallito", self.symbol)
            self._state = BotState.FLAT
            return
        self._entry_order = order

    def _open_position(
        self,
        side: Side,
        size: Decimal,
        fill_price: Decimal,
        signal: Signal,
    ) -> None:
        """Apre la posizione e ricalcola TP/SL sul prezzo di fill reale."""
        self._position_side = side
        self._position_size = size
        self._original_size = size  # per calcoli partial TP
        self._entry_price   = fill_price
        self._entry_time    = time.time()
        self._state         = BotState.POSITION_OPEN
        self._partial_tp1_hit = False
        self._partial_tp2_hit = False

        # Notifica account risk manager
        if self._account_risk:
            self._account_risk.register_open(self.symbol)

        # Ricalcola TP/SL dal fill reale; usa ATR corrente se disponibile, altrimenti dal segnale
        try:
            atr = self._indicators.atr()
        except ValueError:
            atr = signal.atr
        self._signal_atr = atr  # salva per partial TP
        if side == Side.BUY:
            self._tp_price = fill_price + self._cfg.strategy.tp_atr_mult * atr
            self._sl_price = fill_price - self._cfg.strategy.sl_atr_mult * atr
        else:
            self._tp_price = fill_price - self._cfg.strategy.tp_atr_mult * atr
            self._sl_price = fill_price + self._cfg.strategy.sl_atr_mult * atr

        logger.info(
            "%s POSIZIONE APERTA: %s %.3f @ %.4f | TP=%.4f SL=%.4f",
            self.symbol, side.name, size, fill_price, self._tp_price, self._sl_price,
        )

        if self._notifier:
            asyncio.create_task(self._notifier.trade_opened(
                symbol=self.symbol,
                side="LONG" if side == Side.BUY else "SHORT",
                entry=fill_price,
                tp=self._tp_price,
                sl=self._sl_price,
                size=size,
                capital=self._capital,
            ))

        self._start_hold_time_check()
        self._save_state()

        if _prom:
            _position_gauge.labels(symbol=self.symbol).set(1)

    async def _handle_fill(self, order: Order) -> None:
        """Gestisce fill di entrata o di uscita (TP/SL)."""
        # Fill di entrata
        if self._state == BotState.ENTERING and self._entry_order and order.id == self._entry_order.id:
            self._open_position(order.side, order.filled_size, order.price, self._current_signal)

            # Piazza TP/SL live con i prezzi ricalcolati sul fill reale
            if self._cfg.live_trading_enabled:
                tp_sl_ok = await self._gw.submit_tp_sl(
                    self.symbol,
                    order.side,
                    self._position_size,
                    self._tp_price,
                    self._sl_price,
                )
                if not tp_sl_ok:
                    logger.critical(
                        "%s TP/SL FALLITO — chiusura di emergenza della posizione!",
                        self.symbol,
                    )
                    await self._exit_position_market("tp_sl_placement_failed")
                    if self._notifier:
                        asyncio.create_task(self._notifier.info(
                            f"⚠️ {self.symbol}: TP/SL falliti, posizione chiusa di emergenza"
                        ))
            return

        # Fill di uscita (TP o SL ha fillato)
        if self._state == BotState.POSITION_OPEN:
            await self._close_position(order.price, "TP/SL hit")

    # ------------------------------------------------------------------
    # Uscita posizione
    # ------------------------------------------------------------------

    async def _close_position(self, exit_price: Decimal, reason: str) -> None:
        if self._state not in (BotState.POSITION_OPEN, BotState.EXITING):
            return
        self._state = BotState.EXITING

        pnl = self._calc_pnl(exit_price)
        self.realized_pnl += pnl
        self.total_trades  += 1

        logger.info(
            "%s USCITA (%s) | exit=%.4f pnl=%.4f USDT | PnL totale=%.4f USDT",
            self.symbol, reason, exit_price, pnl, self.realized_pnl,
        )

        if self._position_side:
            # Fee = entry fee + exit fee (taker su entrambi per market orders)
            fee = (self._entry_price * self._position_size + exit_price * self._position_size) \
                  * self._cfg.fee_taker
            trade = Trade(
                symbol=self.symbol,
                side=self._position_side,
                size=self._position_size,
                price=exit_price,
                commission=fee,
                commission_asset="USDT",
                order_id=f"close_{int(time.time())}",
                timestamp=time.time(),
                realized_pnl=pnl,
            )
            await self._state_mgr.save_trade(trade)

        if _prom:
            _pnl_gauge.labels(symbol=self.symbol).set(float(self.realized_pnl))
            _position_gauge.labels(symbol=self.symbol).set(0)
            side_name = self._position_side.name if self._position_side else "UNKNOWN"
            _fills_total.labels(symbol=self.symbol, side=side_name).inc()

        if self._notifier:
            asyncio.create_task(self._notifier.trade_closed(
                symbol=self.symbol,
                side="LONG" if self._position_side == Side.BUY else "SHORT",
                entry=self._entry_price,
                exit_price=exit_price,
                pnl=pnl,
                reason=reason,
            ))

        # Aggiorna Kelly sizer con il risultato del trade
        if self._kelly and self._position_side is not None:
            risk_amount = abs(self._entry_price - self._sl_price) * self._original_size
            self._kelly.update(pnl, risk_amount)
            if self._kelly.is_ready:
                logger.info(
                    "%s Kelly stats: %s", self.symbol, self._kelly.stats()
                )

        # Aggiorna loss streak
        if pnl < _ZERO:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Cooldown post-loss: blocca re-entry per N candele dopo SL hit
        if "SL" in reason or "sl" in reason.lower():
            self._cooldown_remaining = self._cfg.strategy.loss_cooldown_candles
            logger.info(
                "%s Cooldown attivato: %d candele di pausa post-loss.",
                self.symbol, self._cooldown_remaining,
            )

        # Notifica account risk manager della chiusura
        if self._account_risk:
            self._account_risk.register_close(self.symbol, pnl)

        self._reset_position_state()
        self._save_state()

    async def _exit_position_market(self, reason: str) -> None:
        """
        Chiude la posizione con ordine market (es. max hold time).
        PRIMA cancella eventuali TP/SL pendenti per evitare ordini fantasma.
        """
        if self._state != BotState.POSITION_OPEN:
            return

        # Cancella TP/SL prima di inviare la chiusura
        if self._cfg.live_trading_enabled:
            try:
                await self._gw.cancel_all_orders()
                logger.info("%s Ordini TP/SL cancellati prima della chiusura forzata.", self.symbol)
            except Exception as e:
                logger.warning("%s Errore cancel TP/SL: %s", self.symbol, e)

        close_side = Side.SELL if self._position_side == Side.BUY else Side.BUY

        if self._cfg.live_trading_enabled:
            order = await self._gw.submit_market_order(self.symbol, close_side, self._position_size)
            if order:
                logger.info("%s Ordine market chiusura inviato (%s).", self.symbol, reason)
                return  # Il fill arriverà via on_order_update

        # Paper trading o fallback: chiusura immediata al prezzo corrente
        last_candle = self._candles[-1] if self._candles else None
        exit_price  = last_candle.close if last_candle else self._entry_price
        await self._close_position(exit_price, reason)

    # ------------------------------------------------------------------
    # Check max hold time
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
                if self._state != BotState.POSITION_OPEN:
                    break
                elapsed = time.time() - self._entry_time
                if elapsed >= max_seconds:
                    logger.warning(
                        "%s Max hold time raggiunto (%.1fh). Chiusura forzata.",
                        self.symbol, elapsed / 3600,
                    )
                    await self._exit_position_market("max_hold_time")
                    break
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Calcolo PnL
    # ------------------------------------------------------------------

    def _calc_pnl(self, exit_price: Decimal) -> Decimal:
        """PnL netto dopo fee taker su entrata e uscita."""
        if self._position_size == _ZERO or self._position_side is None:
            return _ZERO
        gross = (exit_price - self._entry_price) * self._position_size
        if self._position_side == Side.SELL:
            gross = -gross
        fee = (self._entry_price + exit_price) * self._position_size * self._cfg.fee_taker
        return gross - fee

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        inv = self._position_size if self._position_side == Side.BUY else -self._position_size
        self._state_mgr.update_snapshot(
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
            logger.info("%s Stato caricato: PnL=%.4f USDT, trade=%d",
                        self.symbol, self.realized_pnl, self.total_trades)

    def _reset_position_state(self) -> None:
        self._state          = BotState.FLAT
        self._current_signal = None
        self._entry_order    = None
        self._position_side  = None
        self._position_size  = _ZERO
        self._original_size  = _ZERO
        self._entry_price    = _ZERO
        self._entry_time     = 0.0
        self._tp_price       = _ZERO
        self._sl_price       = _ZERO
        self._signal_atr     = _ZERO
        self._partial_tp1_hit = False
        self._partial_tp2_hit = False
        self._pending_signal = None
        if self._hold_time_task and not self._hold_time_task.done():
            self._hold_time_task.cancel()
