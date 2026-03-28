"""
Suite di test completa per PolyMM-Pro v5.
Verifica tutti i fix applicati + funzionalità core.
"""
import asyncio
import math
import os
import time
import tempfile
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.models import Order, Side, OrderStatus, Trade
from src.state.order_manager import OrderManager
from src.gateways.simulator import SimulatorGateway
from src.backtest import CandleBacktestEngine as BacktestEngine
from src.config import BotSettings
from src.bot import LokyBot as EventDrivenBot


# ================================================================== #
#  HELPER                                                              #
# ================================================================== #
def make_cfg(**overrides) -> BotSettings:
    defaults = dict(
        tokens=["BTCUSDT"],
        base_spread=Decimal('0.0025'),
        quote_size=Decimal('0.001'),
        skew_factor=Decimal('1.0'),
        max_inventory=Decimal('0.01'),
        fee_maker=Decimal('0.001'),
        fee_taker=Decimal('0.001'),
        rate_limit_rps=100,
        max_daily_loss_pct=Decimal('0.50'),
        max_position_per_asset=Decimal('1.0'),
        live_trading_enabled=False,
    )
    defaults.update(overrides)
    return BotSettings(**defaults)


# ================================================================== #
# (Legacy test classes removed: TestVolatilityNotDoubled,              #
#  TestFairValueEngine, TestQuoteEngineExtended, TestRiskEngineExtended)#
# ================================================================== #
class _Removed_Placeholder:
    def test_volatility_factor_1_unchanged(self):
        """Con volatility_factor=1.0, half_spread = base_spread * fair_value / 2."""
        engine = QuoteEngine(skew_factor=Decimal('0'))
        q = engine.calculate_quotes(
            symbol="TEST",
            fair_value=Decimal('100'),
            base_spread=Decimal('0.01'),   # 1% di $100
            net_inventory=Decimal('0'),
            quote_size=Decimal('1'),
            max_inventory=Decimal('10'),
            volatility_factor=Decimal('1.0'),
        )
        # half_spread = 0.01 * 100 / 2 = 0.50 → bid=99.50, ask=100.50
        assert q.bid_price == Decimal('99.50'), f"bid atteso 99.50, ottenuto {q.bid_price}"
        assert q.ask_price == Decimal('100.50'), f"ask atteso 100.50, ottenuto {q.ask_price}"

    def test_volatility_factor_2_linear_not_quadratic(self):
        """Con volatility_factor=2.0, lo spread deve raddoppiare (non quadruplicare)."""
        engine = QuoteEngine(skew_factor=Decimal('0'))
        q1 = engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('1000'),
            base_spread=Decimal('0.01'), net_inventory=Decimal('0'),
            quote_size=Decimal('1'), max_inventory=Decimal('10'),
            volatility_factor=Decimal('1.0'),
        )
        q2 = engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('1000'),
            base_spread=Decimal('0.02'),  # spread già raddoppiato in bot.py
            net_inventory=Decimal('0'),
            quote_size=Decimal('1'), max_inventory=Decimal('10'),
            volatility_factor=Decimal('1.0'),  # non applicato di nuovo
        )
        spread1 = q1.ask_price - q1.bid_price
        spread2 = q2.ask_price - q2.bid_price
        # spread2 deve essere ~2x spread1, non 4x
        assert spread2 == spread1 * 2, f"atteso 2x, ottenuto {spread2/spread1}x"

    def test_volatility_cap_5x(self):
        """Lo spread non deve superare 5x il base_spread."""
        engine = QuoteEngine(skew_factor=Decimal('0'))
        q = engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('1000'),
            base_spread=Decimal('0.01'), net_inventory=Decimal('0'),
            quote_size=Decimal('1'), max_inventory=Decimal('10'),
            volatility_factor=Decimal('5.0'),
        )
        spread = q.ask_price - q.bid_price
        max_spread = Decimal('1000') * Decimal('0.01') * Decimal('5') * 2  # 5x base su entrambi i lati
        assert spread <= max_spread


# ================================================================== #
#  FIX #2 — Backtest produce fill reali (kill switch non scatta)      #
# ================================================================== #
class TestBacktestWorks:
    @pytest.mark.asyncio
    async def test_backtest_produces_fills(self):
        """Il backtest deve produrre fill > 0 dopo i fix timestamp e callback.

        Strategia prezzi:
          - Tick 0: price=40000 → ordini BID@39980, ASK@40020 piazzati
          - Tick 1+: price=40100 → 40100 ≥ 40020 → ASK fills
          - Poi: price=39800 → 39800 ≤ 39980 → BID fills
        Prezzi costanti → quotes invariate → ordini NON cancellati → fill garantito.
        """
        prices = (
            [Decimal('40000')] * 2       # tick iniziali per piazzare gli ordini
            + [Decimal('40100')] * 30    # sopra ASK (40020) → SELL fills
            + [Decimal('39800')] * 30    # sotto BID (39980) → BUY fills
        )

        cfg = make_cfg(
            base_spread=Decimal('0.0010'),
            quote_size=Decimal('0.001'),
            max_inventory=Decimal('0.05'),
            rate_limit_rps=100,
        )
        engine = BacktestEngine(symbol="BTCUSDT", historical_prices=prices, cfg=cfg)
        result = await engine.run()

        assert result['fills'] > 0, "Backtest deve produrre almeno un fill"
        assert result['quotes_sent'] > 0, "Backtest deve piazzare almeno una quote"

    @pytest.mark.asyncio
    async def test_backtest_metrics_complete(self):
        """Il backtest deve restituire tutte le metriche richieste."""
        prices = [Decimal(str(40000 + i * 10)) for i in range(100)]
        cfg = make_cfg()
        engine = BacktestEngine(symbol="BTCUSDT", historical_prices=prices, cfg=cfg)
        result = await engine.run()

        required_keys = ['pnl', 'sharpe', 'max_drawdown', 'fills', 'quotes_sent',
                         'fill_rate', 'final_inventory']
        for key in required_keys:
            assert key in result, f"Chiave mancante: {key}"

    @pytest.mark.asyncio
    async def test_backtest_sharpe_finite(self):
        """Il Sharpe ratio non deve essere NaN o infinito."""
        prices = []
        p = 40000.0
        import random
        random.seed(123)
        for _ in range(500):
            p *= math.exp(random.gauss(0, 0.001))
            prices.append(Decimal(str(round(p, 2))))

        cfg = make_cfg(base_spread=Decimal('0.002'))
        engine = BacktestEngine(symbol="BTCUSDT", historical_prices=prices, cfg=cfg)
        result = await engine.run()

        sharpe = result['sharpe']
        assert math.isfinite(sharpe), f"Sharpe non finito: {sharpe}"


# ================================================================== #
#  FIX #3 — Commissioni BUY contabilizzate                           #
# ================================================================== #
class TestBuyCommission:
    @pytest.mark.asyncio
    async def test_buy_commission_deducted_from_pnl(self):
        """Ogni BUY deve ridurre il PnL della commissione taker."""
        gateway = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gateway)

        order = Order(
            id="test_buy",
            symbol="BTCUSDT",
            side=Side.BUY,
            price=Decimal('40000'),
            size=Decimal('0.001'),
            status=OrderStatus.FILLED,
            filled_size=Decimal('0.001'),
        )

        pnl_before = bot.pnl
        await bot.on_order_update(order)

        expected_commission = Decimal('40000') * Decimal('0.001') * Decimal('0.001')
        assert bot.pnl == pnl_before - expected_commission, (
            f"PnL dopo BUY: atteso -{expected_commission}, ottenuto {bot.pnl - pnl_before}"
        )

    @pytest.mark.asyncio
    async def test_roundtrip_pnl_accounts_both_commissions(self):
        """BUY + SELL simmetrico deve avere PnL negativo (commissioni su entrambi i lati)."""
        gateway = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gateway)

        buy_order = Order(
            id="buy1", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('40000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )
        sell_order = Order(
            id="sell1", symbol="BTCUSDT", side=Side.SELL,
            price=Decimal('40000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )

        await bot.on_order_update(buy_order)
        await bot.on_order_update(sell_order)

        # BUY a 40000, SELL a 40000: PnL = 0 - fee_taker - fee_maker < 0
        assert bot.pnl < Decimal('0'), (
            f"Roundtrip allo stesso prezzo deve avere PnL negativo per le fee, ottenuto {bot.pnl}"
        )

    @pytest.mark.asyncio
    async def test_sell_profit_exceeds_fees(self):
        """BUY a 39000, SELL a 41000: PnL deve essere positivo (spread cattura le fee)."""
        gateway = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gateway)

        buy_order = Order(
            id="buy2", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('39000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )
        sell_order = Order(
            id="sell2", symbol="BTCUSDT", side=Side.SELL,
            price=Decimal('41000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )

        await bot.on_order_update(buy_order)
        await bot.on_order_update(sell_order)

        assert bot.pnl > Decimal('0'), f"Trade profittevole deve avere PnL > 0, ottenuto {bot.pnl}"


# ================================================================== #
#  FIX #4 — quote_tolerance letto una volta sola                      #
# ================================================================== #
class TestOrderManagerTolerance:
    def test_tolerance_loaded_once_in_init(self):
        """OrderManager deve avere quote_tolerance come attributo (non ricaricarla)."""
        manager = OrderManager()
        assert hasattr(manager, 'quote_tolerance')
        assert isinstance(manager.quote_tolerance, Decimal)

    @pytest.mark.asyncio
    async def test_tolerance_prevents_unnecessary_cancel(self):
        """Differenze di size sotto la tolerance non devono triggerare cancel."""
        manager = OrderManager()
        gateway = MagicMock()
        gateway.submit_order = AsyncMock()
        gateway.cancel_order = AsyncMock()

        # Piazza ordine iniziale
        await manager.sync_target_quote(
            symbol="BTCUSDT", gateway=gateway, side=Side.BUY,
            target_price=Decimal('40000'), target_size=Decimal('0.001'),
        )

        # Cambia size di meno della tolerance (0.00001 < default 0.0001)
        tiny_diff = manager.quote_tolerance / Decimal('10')
        await manager.sync_target_quote(
            symbol="BTCUSDT", gateway=gateway, side=Side.BUY,
            target_price=Decimal('40000'),
            target_size=Decimal('0.001') + tiny_diff,
        )

        # Nessun cancel deve essere stato inviato
        gateway.cancel_order.assert_not_called()


# ================================================================== #
#  FIX #7 — Order ID univoco                                          #
# ================================================================== #
class TestOrderIdUniqueness:
    @pytest.mark.asyncio
    async def test_order_ids_are_unique(self):
        """Ogni ordine deve avere un ID diverso."""
        manager = OrderManager()
        gateway = MagicMock()
        gateway.submit_order = AsyncMock()
        gateway.cancel_order = AsyncMock()

        ids = set()
        for i in range(20):
            # Forza nuovo ordine: metti active_order a None
            manager.active_orders[Side.BUY] = None
            await manager.sync_target_quote(
                symbol="BTCUSDT", gateway=gateway, side=Side.BUY,
                target_price=Decimal(str(40000 + i)),
                target_size=Decimal('0.001'),
            )
            order = manager.active_orders[Side.BUY]
            if order:
                ids.add(order.id)

        assert len(ids) == 20, f"ID duplicati trovati: {20 - len(ids)} duplicati su 20"

    @pytest.mark.asyncio
    async def test_order_id_is_valid_uuid(self):
        """L'ID dell'ordine deve essere un UUID valido."""
        import uuid
        manager = OrderManager()
        gateway = MagicMock()
        gateway.submit_order = AsyncMock()

        await manager.sync_target_quote(
            symbol="BTCUSDT", gateway=gateway, side=Side.BUY,
            target_price=Decimal('40000'), target_size=Decimal('0.001'),
        )
        order_id = manager.active_orders[Side.BUY].id
        # Deve essere parsabile come UUID
        parsed = uuid.UUID(order_id)
        assert str(parsed) == order_id


# (TestFairValueEngine, TestQuoteEngineExtended, TestRiskEngineExtended
#  rimossi — moduli legacy cancellati in v6.0)
class TestFairValueEngine:
    def setup_method(self):
        self.engine = FairValueEngine()

    def test_vwap_basic(self):
        book = {
            'bids': [(100, 10), (99, 20)],
            'asks': [(101, 10), (102, 20)],
        }
        vwap = self.engine.calculate_vwap(book, depth_levels=5)
        # (100*10 + 99*20 + 101*10 + 102*20) / 60
        expected = Decimal(str((100*10 + 99*20 + 101*10 + 102*20) / 60))
        assert abs(vwap - expected) < Decimal('0.01')

    def test_microprice_balanced(self):
        """Con volumi uguali su bid e ask, microprice = midprice."""
        book = {
            'bids': [(100, 10)],
            'asks': [(102, 10)],
        }
        mp = self.engine.calculate_microprice(book)
        assert mp == Decimal('101')

    def test_microprice_bid_heavy(self):
        """Con più volume sul bid, microprice si sposta verso il bid."""
        book = {
            'bids': [(100, 90)],
            'asks': [(102, 10)],
        }
        mp = self.engine.calculate_microprice(book)
        # Pesa più bid → microprice vicino a 100
        assert mp < Decimal('101')
        assert mp > Decimal('100')

    def test_imbalance_neutral(self):
        book = {
            'bids': [(100, 50)],
            'asks': [(101, 50)],
        }
        imb = self.engine.detect_order_imbalance(book)
        assert imb == Decimal('0.5')

    def test_imbalance_bullish(self):
        book = {
            'bids': [(100, 80)],
            'asks': [(101, 20)],
        }
        imb = self.engine.detect_order_imbalance(book)
        assert imb == Decimal('0.8')

    def test_logistic_fair_value_at_strike(self):
        """Al prezzo dello strike, FV deve essere ~0.5."""
        fv = self.engine.calculate_fair_value(Decimal('100'), Decimal('100'))
        assert abs(fv - Decimal('0.5')) < Decimal('0.01')

    def test_logistic_fair_value_bounds(self):
        """FV deve sempre essere nel range [0.01, 0.99]."""
        fv_high = self.engine.calculate_fair_value(Decimal('1000'), Decimal('100'))
        fv_low  = self.engine.calculate_fair_value(Decimal('1'), Decimal('100'))
        assert fv_high <= Decimal('0.99')
        assert fv_low  >= Decimal('0.01')


# ================================================================== #
#  QuoteEngine                                                         #
# ================================================================== #
class TestQuoteEngineExtended:
    def setup_method(self):
        self.engine = QuoteEngine(
            skew_factor=Decimal('1.0'),
            fee_maker=Decimal('0.001'),
            fee_taker=Decimal('0.001'),
        )

    def test_min_spread_enforced(self):
        """Lo spread non deve scendere sotto min_spread * fair_value (meno 1 tick per arrotondamento)."""
        q = self.engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('100'),
            base_spread=Decimal('0.00001'),  # spread minuscolo, sostituito da min_spread
            net_inventory=Decimal('0'), quote_size=Decimal('1'),
            max_inventory=Decimal('10'),
        )
        spread = q.ask_price - q.bid_price
        # min_spread_rate = (0.001+0.001)*1.05 = 0.0021
        # min_spread_abs = 0.0021 * 100 = 0.21 USDT
        # Tolleranza: -1 tick ($0.01) per Decimal ROUND_HALF_EVEN su half_spread
        min_expected_abs = Decimal('100') * Decimal('0.0021') - Decimal('0.01')
        assert spread >= min_expected_abs, (
            f"spread {spread} < min_expected (with rounding tolerance) {min_expected_abs}"
        )

    def test_bid_always_below_ask(self):
        """bid_price deve sempre essere strettamente inferiore ad ask_price."""
        for inv in [-1, -0.5, 0, 0.5, 1]:
            q = self.engine.calculate_quotes(
                symbol="TEST", fair_value=Decimal('1000'),
                base_spread=Decimal('0.005'),
                net_inventory=Decimal(str(inv * 0.01)),
                quote_size=Decimal('0.001'),
                max_inventory=Decimal('0.01'),
            )
            assert q.bid_price < q.ask_price, f"bid >= ask con inventory={inv}"

    def test_zero_size_at_max_inventory_buy(self):
        """A inventory massima, bid_size deve essere 0."""
        q = self.engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('1000'),
            base_spread=Decimal('0.005'),
            net_inventory=Decimal('0.01'),   # = max_inventory
            quote_size=Decimal('0.001'),
            max_inventory=Decimal('0.01'),
        )
        assert q.bid_size == Decimal('0')

    def test_zero_size_at_max_inventory_sell(self):
        """A inventory minima (-max), ask_size deve essere 0."""
        q = self.engine.calculate_quotes(
            symbol="TEST", fair_value=Decimal('1000'),
            base_spread=Decimal('0.005'),
            net_inventory=Decimal('-0.01'),  # = -max_inventory
            quote_size=Decimal('0.001'),
            max_inventory=Decimal('0.01'),
        )
        assert q.ask_size == Decimal('0')


# ================================================================== #
#  RiskEngine                                                          #
# ================================================================== #
class TestRiskEngineExtended:
    def test_rate_limiter_allows_within_limit(self):
        """Il rate limiter deve permettere n richieste entro il limite."""
        engine = RiskEngine(rate_limit_rps=5)
        results = [engine.check_rate_limit() for _ in range(5)]
        assert all(results), "Le prime 5 richieste devono passare"

    def test_rate_limiter_blocks_excess(self):
        """Il rate limiter deve bloccare la richiesta oltre il limite."""
        engine = RiskEngine(rate_limit_rps=3)
        for _ in range(3):
            engine.check_rate_limit()
        assert not engine.check_rate_limit(), "La 4a richiesta deve essere bloccata"

    def test_daily_pnl_stop_triggers(self):
        """Il daily stop deve triggerare quando PnL scende sotto il limite."""
        engine = RiskEngine(max_daily_loss=Decimal('-10'))
        with pytest.raises(KillSwitchException, match="Daily loss"):
            engine.check_daily_pnl_stop(Decimal('-15'))

    def test_daily_pnl_ok(self):
        """Il daily stop non deve triggerare con PnL sopra il limite."""
        engine = RiskEngine(max_daily_loss=Decimal('-50'))
        assert engine.check_daily_pnl_stop(Decimal('-10')) is True

    def test_position_guard_blocks_excess(self):
        """validate_open_position deve bloccare ordini che eccedono il limite."""
        engine = RiskEngine(max_position_per_asset=Decimal('0.01'))
        result = engine.validate_open_position("BTCUSDT", Decimal('0.005'), Decimal('0.008'))
        assert not result  # 0.008 + 0.005 = 0.013 > 0.01

    def test_orderbook_staleness_triggers(self):
        """Kill switch deve triggerare per orderbook scaduto."""
        engine = RiskEngine()
        t = time.monotonic()
        state = SystemState(
            net_inventory=Decimal('0'), pnl=Decimal('0'),
            last_market_data_ts=t,
            last_orderbook_ts=t - 1.0,  # 1 secondo fa
            local_open_orders=0, remote_open_orders=0
        )
        with pytest.raises(KillSwitchException, match="Orderbook staleness"):
            engine.validate_system_health(state)


# ================================================================== #
#  SimulatorGateway                                                    #
# ================================================================== #
class TestSimulatorGateway:
    @pytest.mark.asyncio
    async def test_buy_fills_when_price_drops(self):
        """BUY deve essere eseguito quando il mercato scende sotto il bid."""
        gw = SimulatorGateway()
        fill_count = 0

        async def on_update(o):
            nonlocal fill_count
            # Contiamo solo i FILLED, evitando il problema di oggetti mutabili
            if o.status == OrderStatus.FILLED:
                fill_count += 1

        gw.set_on_order_update_callback(on_update)

        order = Order(
            id="b1", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('40000'), size=Decimal('0.001'),
            status=OrderStatus.PENDING, filled_size=Decimal('0'),
        )
        await gw.submit_order(order)
        gw.update_market_price("BTCUSDT", Decimal('39900'))  # sotto il bid
        await gw.match_engine_tick()

        assert fill_count == 1

    @pytest.mark.asyncio
    async def test_sell_does_not_fill_below_price(self):
        """SELL NON deve essere eseguito se il mercato è sotto l'ask."""
        gw = SimulatorGateway()
        fills = []

        async def on_update(o):
            if o.status == OrderStatus.FILLED:
                fills.append(o)
        gw.set_on_order_update_callback(on_update)

        order = Order(
            id="s1", symbol="BTCUSDT", side=Side.SELL,
            price=Decimal('40100'), size=Decimal('0.001'),
            status=OrderStatus.PENDING, filled_size=Decimal('0'),
        )
        await gw.submit_order(order)
        gw.update_market_price("BTCUSDT", Decimal('40050'))  # sotto l'ask
        await gw.match_engine_tick()

        assert len(fills) == 0


# ================================================================== #
#  Persistency                                                         #
# ================================================================== #
class TestPersistency:
    def test_save_and_load_state(self):
        """save_trade + load_state devono sopravvivere a un ciclo completo."""
        from src.state.persistency import StateManager

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test_state.db")
            gateway = SimulatorGateway()
            bot = EventDrivenBot(symbol="BTCUSDT", gateway=gateway)

            sm = StateManager(bot, db_path=db_path)

            trade = Trade(
                symbol="BTCUSDT", side=Side.BUY,
                size=Decimal('0.001'), price=Decimal('40000'),
                commission=Decimal('0.04'), commission_asset='USDT',
                order_id="test-uuid-123", timestamp=time.time(),
                realized_pnl=Decimal('0'),
            )
            sm.save_trade(trade)

            sm._conn.execute(
                '''INSERT INTO state (net_inventory, pnl, avg_entry, quotes_sent, fills_total, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                ('0.005', '12.34', '39500', 10, 2, time.time())
            )
            sm._conn.commit()

            # Chiude connessione PRIMA di riaprire (fix Windows file lock)
            sm._conn.close()

            bot2 = EventDrivenBot(symbol="BTCUSDT", gateway=gateway)
            sm2 = StateManager(bot2, db_path=db_path)
            sm2.load_state()

            assert bot2.net_inventory == Decimal('0.005')
            assert bot2.pnl == Decimal('12.34')
            assert bot2.avg_entry_price == Decimal('39500')

            sm2._conn.close()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ================================================================== #
#  Bot — inventory tracking                                            #
# ================================================================== #
class TestBotInventoryTracking:
    @pytest.mark.asyncio
    async def test_inventory_increases_on_buy(self):
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)

        order = Order(
            id="b1", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('40000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )
        await bot.on_order_update(order)
        assert bot.net_inventory == Decimal('0.001')

    @pytest.mark.asyncio
    async def test_inventory_decreases_on_sell(self):
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)
        bot.net_inventory = Decimal('0.002')
        bot.avg_entry_price = Decimal('39000')

        order = Order(
            id="s1", symbol="BTCUSDT", side=Side.SELL,
            price=Decimal('40000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        )
        await bot.on_order_update(order)
        assert bot.net_inventory == Decimal('0.001')

    @pytest.mark.asyncio
    async def test_avg_entry_price_weighted_average(self):
        """Prezzo medio di carico deve essere corretto su acquisti multipli."""
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)

        await bot.on_order_update(Order(
            id="b1", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('38000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        ))
        await bot.on_order_update(Order(
            id="b2", symbol="BTCUSDT", side=Side.BUY,
            price=Decimal('42000'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        ))

        # Avg entry = 40000 (più fee incorporata → leggermente più alta)
        assert bot.avg_entry_price > Decimal('39999')
        assert bot.avg_entry_price < Decimal('40100')

    @pytest.mark.asyncio
    async def test_inventory_resets_to_zero_after_full_sell(self):
        """Dopo una vendita completa, inventory = 0 e avg_entry = 0."""
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)
        bot.net_inventory = Decimal('0.001')
        bot.avg_entry_price = Decimal('40000')

        await bot.on_order_update(Order(
            id="s1", symbol="BTCUSDT", side=Side.SELL,
            price=Decimal('40500'), size=Decimal('0.001'),
            status=OrderStatus.FILLED, filled_size=Decimal('0.001'),
        ))

        assert bot.net_inventory == Decimal('0')
        assert bot.avg_entry_price == Decimal('0')


# ================================================================== #
#  Bot — volatility log_returns incrementali (FIX #5)                 #
# ================================================================== #
class TestBotVolatilityIncremental:
    @pytest.mark.asyncio
    async def test_log_returns_populated_incrementally(self):
        """log_returns deve crescere tick per tick senza ricalcolare tutto."""
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)

        prices = [Decimal(str(40000 + i * 10)) for i in range(10)]
        for p in prices:
            await bot.on_market_data_event(p, time.monotonic())

        # Dopo 10 prezzi, ci sono 9 log-returns
        assert len(bot.log_returns) == 9

    @pytest.mark.asyncio
    async def test_log_returns_window_capped(self):
        """log_returns non deve superare 119 elementi (maxlen)."""
        gw = SimulatorGateway()
        bot = EventDrivenBot(symbol="BTCUSDT", gateway=gw)

        for i in range(150):
            await bot.on_market_data_event(
                Decimal(str(40000 + i)), time.monotonic()
            )

        assert len(bot.log_returns) <= 119
