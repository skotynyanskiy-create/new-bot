import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from decimal import Decimal
import os


class StrategySettings(BaseModel):
    """Parametri della strategia Breakout/Momentum + Multi-Strategy."""

    # --- Breakout ---
    breakout_lookback: int = 20
    volume_multiplier: Decimal = Decimal('1.5')
    rsi_min: Decimal = Decimal('45')
    rsi_max: Decimal = Decimal('72')
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    vol_period: int = 20
    tp_atr_mult: Decimal = Decimal('2.0')
    sl_atr_mult: Decimal = Decimal('1.0')
    max_hold_hours: int = 8
    trailing_stop_enabled: bool = True
    trail_atr_mult: Decimal = Decimal('0.5')
    loss_cooldown_candles: int = 3

    # --- ADX ---
    adx_period: int = 14
    adx_trending_threshold: Decimal = Decimal('25')   # ADX > 25 = trending
    adx_ranging_threshold: Decimal = Decimal('20')    # ADX < 20 = ranging

    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: Decimal = Decimal('2.0')

    # --- Mean Reversion ---
    mr_rsi_oversold: Decimal = Decimal('32')
    mr_rsi_overbought: Decimal = Decimal('68')
    mr_adx_max: Decimal = Decimal('22')

    # --- Trend Following ---
    tf_adx_min: Decimal = Decimal('28')
    tf_tp_atr_mult: Decimal = Decimal('3.0')
    tf_sl_atr_mult: Decimal = Decimal('1.2')

    # --- Funding Rate Harvesting ---
    funding_threshold: Decimal = Decimal('0.0008')    # 0.08% per 8h

    # --- Partial TP (50/25/25) ---
    partial_tp1_atr: Decimal = Decimal('1.5')   # TP1 a 1.5×ATR → chiudi 50%
    partial_tp2_atr: Decimal = Decimal('2.5')   # TP2 a 2.5×ATR → chiudi 25%
    partial_tp3_atr: Decimal = Decimal('4.0')   # TP3 a 4.0×ATR → trail 25%
    partial_tp1_pct: Decimal = Decimal('0.50')
    partial_tp2_pct: Decimal = Decimal('0.25')
    partial_tp3_pct: Decimal = Decimal('0.25')

    # --- Signal Scoring ---
    min_signal_score: Decimal = Decimal('50')   # score minimo per entrare


class BotSettings(BaseSettings):
    tokens: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"

    # Strategia (nested)
    strategy: StrategySettings = StrategySettings()

    # Risk management
    risk_per_trade_pct: Decimal = Decimal('0.015')
    max_concurrent_positions: int = 2
    leverage: int = 3
    max_daily_loss: Decimal = Decimal('-30')
    max_position_per_asset: Decimal = Decimal('0.05')

    # Portfolio & Leverage dinamica
    max_leverage: int = 5
    dynamic_leverage_enabled: bool = True
    kelly_sizing_enabled: bool = True
    kelly_min_trades: int = 20

    # Fees Futures (standard tier)
    fee_maker: Decimal = Decimal('0.0002')
    fee_taker: Decimal = Decimal('0.0004')

    # Operativo
    exchange: str = "bybit"
    live_trading_enabled: bool = False
    testnet: bool = False
    rate_limit_rps: int = 8
    next_candle_entry: bool = True
    slippage_pct: Decimal = Decimal('0.0005')

    @classmethod
    def load(cls) -> "BotSettings":
        config_path = "config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        if "strategy" in data and isinstance(data["strategy"], dict):
                            data["strategy"] = StrategySettings(**data["strategy"])
                        return cls(**data)
        except Exception as e:
            print(f"Errore caricamento config.yaml: {e}")
        return cls()


config = BotSettings.load()
