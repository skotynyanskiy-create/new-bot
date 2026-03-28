import logging
import os
from decimal import Decimal

import yaml
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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

    # --- Circuit Breaker ---
    circuit_breaker_losses:  int = 3    # perdite consecutive che attivano il circuit breaker
    circuit_breaker_candles: int = 15   # candle di pausa dopo attivazione

    # --- Anti-martingale ---
    # Size multipliers gestiti internamente nel bot (non configurabili via YAML per sicurezza)
    # Win streak ≥3 → ×1.20, ≥2 → ×1.10 | Loss streak ≥3 → ×0.65, ≥2 → ×0.80

    # --- Pyramid scaling-in ---
    scale_in_profit_atr_mult: Decimal = Decimal('0.5')  # profitto minimo (in ATR) per attivare scale-in

    # --- Regime avanzato ---
    # choppy_market_pause rimosso v6.5 — il CHOPPY block è gestito da is_choppy_market() nell'aggregator

    @field_validator('sl_atr_mult')
    @classmethod
    def sl_must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= Decimal('0'):
            raise ValueError(f"sl_atr_mult deve essere > 0, ricevuto: {v}")
        return v

    @field_validator('tp_atr_mult')
    @classmethod
    def tp_must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= Decimal('0'):
            raise ValueError(f"tp_atr_mult deve essere > 0, ricevuto: {v}")
        return v

    @field_validator('funding_threshold')
    @classmethod
    def funding_threshold_valid(cls, v: Decimal) -> Decimal:
        if v < Decimal('0') or v > Decimal('0.01'):
            raise ValueError(f"funding_threshold deve essere tra 0 e 1% (0.01), ricevuto: {v}")
        return v

    @field_validator('circuit_breaker_losses')
    @classmethod
    def circuit_breaker_losses_valid(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError(f"circuit_breaker_losses deve essere tra 1 e 20, ricevuto: {v}")
        return v

    @field_validator('circuit_breaker_candles')
    @classmethod
    def circuit_breaker_candles_valid(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"circuit_breaker_candles deve essere tra 1 e 100, ricevuto: {v}")
        return v

    @field_validator('min_signal_score')
    @classmethod
    def min_score_valid(cls, v: Decimal) -> Decimal:
        if v < Decimal('0') or v > Decimal('100'):
            raise ValueError(f"min_signal_score deve essere tra 0 e 100, ricevuto: {v}")
        return v

    @model_validator(mode='after')
    def tp_greater_than_sl(self) -> 'StrategySettings':
        if self.tp_atr_mult <= self.sl_atr_mult:
            raise ValueError(
                f"tp_atr_mult ({self.tp_atr_mult}) deve essere > sl_atr_mult ({self.sl_atr_mult})"
            )
        return self


class BotSettings(BaseSettings):
    tokens: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"
    macro_timeframe: str = "4h"   # Filtro macro trend (EMA fast/slow sul 4h)

    # Strategia (nested)
    strategy: StrategySettings = StrategySettings()

    # Risk management
    risk_per_trade_pct: Decimal = Decimal('0.015')
    max_concurrent_positions: int = 2
    leverage: int = 3
    max_daily_loss_pct: Decimal = Decimal('0.05')      # 5% del capitale = daily stop (scala col capitale)
    max_peak_drawdown_pct: Decimal = Decimal('0.15')   # halt se drawdown > 15% dal picco equity
    max_position_per_asset: Decimal = Decimal('0.05')

    # Portfolio & Leverage dinamica
    max_leverage: int = 5
    dynamic_leverage_enabled: bool = True
    kelly_sizing_enabled: bool = True
    kelly_min_trades: int = 20

    # Gruppi di correlazione: lista di liste di symbol correlati.
    # Se un symbol di un gruppo è già in posizione, gli altri sono bloccati.
    # Esempio: [["BTCUSDT", "ETHUSDT"], ["SOLUSDT", "AVAXUSDT"]]
    correlation_groups: list[list[str]] = [
        ["BTCUSDT", "ETHUSDT"],
        ["SOLUSDT", "AVAXUSDT"],
    ]

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
    limit_entry_enabled: bool = True     # usa limit order + market fallback (riduce fee del 50%)
    limit_entry_timeout_s: float = 5.0   # secondi di attesa prima di fallback a market

    def correlation_groups_as_dict(self) -> dict[str, int]:
        """Converte la lista di gruppi in {symbol: group_id} per PortfolioRiskManager."""
        result: dict[str, int] = {}
        for gid, group in enumerate(self.correlation_groups):
            for symbol in group:
                result[symbol] = gid
        return result

    @field_validator('risk_per_trade_pct')
    @classmethod
    def risk_in_range(cls, v: Decimal) -> Decimal:
        if not (Decimal('0.001') <= v <= Decimal('0.05')):
            raise ValueError(
                f"risk_per_trade_pct deve essere tra 0.1% e 5%, ricevuto: {float(v)*100:.2f}%"
            )
        return v

    @field_validator('leverage')
    @classmethod
    def leverage_in_range(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError(f"leverage deve essere tra 1 e 20, ricevuto: {v}")
        return v

    @field_validator('max_daily_loss_pct')
    @classmethod
    def daily_loss_pct_valid(cls, v: Decimal) -> Decimal:
        if v <= Decimal('0') or v > Decimal('0.50'):
            raise ValueError(f"max_daily_loss_pct deve essere tra 0 e 50% (0.50), ricevuto: {v}")
        return v

    @field_validator('max_leverage')
    @classmethod
    def max_leverage_in_range(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError(f"max_leverage deve essere tra 1 e 50, ricevuto: {v}")
        return v

    @field_validator('max_peak_drawdown_pct')
    @classmethod
    def drawdown_pct_valid(cls, v: Decimal) -> Decimal:
        if v <= Decimal('0') or v > Decimal('0.50'):
            raise ValueError(f"max_peak_drawdown_pct deve essere tra 0 e 50%, ricevuto: {v}")
        return v

    @field_validator('primary_timeframe', 'confirmation_timeframe', 'macro_timeframe')
    @classmethod
    def timeframe_valid(cls, v: str) -> str:
        _VALID_TF = {'1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'}
        if v not in _VALID_TF:
            raise ValueError(f"Timeframe '{v}' non valido. Valori ammessi: {sorted(_VALID_TF)}")
        return v

    @model_validator(mode='after')
    def validate_leverage_consistency(self) -> 'BotSettings':
        if self.leverage > self.max_leverage:
            raise ValueError(
                f"leverage ({self.leverage}) non può superare max_leverage ({self.max_leverage})"
            )
        return self

    @model_validator(mode='after')
    def validate_correlation_groups(self) -> 'BotSettings':
        token_set = set(t.upper() for t in self.tokens)
        for group in self.correlation_groups:
            invalid = [s for s in group if s.upper() not in token_set]
            if invalid:
                logger.warning(
                    "correlation_groups contiene simboli non in tokens: %s — verranno ignorati.",
                    invalid,
                )
        return self

    def log_startup_config(self) -> None:
        """Stampa la configurazione attiva a startup con warning per parametri aggressivi."""
        logger.info("=" * 55)
        logger.info("  Loky Bot — Configurazione Attiva")
        logger.info("=" * 55)
        logger.info("  Exchange       : %s (%s)", self.exchange,
                    "LIVE" if self.live_trading_enabled else ("TESTNET" if self.testnet else "PAPER"))
        logger.info("  Symbols        : %s", ", ".join(self.tokens))
        logger.info("  Timeframe      : %s + %s (HTF)", self.primary_timeframe, self.confirmation_timeframe)
        logger.info("  Leverage       : %dx (max %dx)", self.leverage, self.max_leverage)
        logger.info("  Risk/trade     : %.1f%%", float(self.risk_per_trade_pct) * 100)
        logger.info("  Daily stop     : %.1f%% del capitale", float(self.max_daily_loss_pct) * 100)
        logger.info("  Peak DD stop   : %.0f%%", float(self.max_peak_drawdown_pct) * 100)
        logger.info("  Kelly sizing   : %s (min %d trade)", self.kelly_sizing_enabled, self.kelly_min_trades)
        logger.info("=" * 55)

        # Warning parametri aggressivi
        if float(self.risk_per_trade_pct) > 0.03:
            logger.warning("⚠️  risk_per_trade_pct > 3%% è molto aggressivo!")
        if self.leverage >= 10:
            logger.warning("⚠️  Leverage %dx è molto elevato — rischio liquidazione!", self.leverage)
        if self.live_trading_enabled and not self.testnet:
            logger.warning("🔴 LIVE TRADING ATTIVO — ordini reali verranno inviati!")
        elif not self.live_trading_enabled:
            logger.warning(
                "📄 PAPER TRADING MODE — nessun ordine reale. "
                "Imposta live_trading_enabled: true in config.yaml per il live."
            )

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
