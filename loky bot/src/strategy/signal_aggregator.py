"""
SignalAggregator — scoring e selezione del miglior segnale.

Scoring formula (0-100):
  trend_strength_score  × 0.30   (ADX normalizzato 0-100)
  volume_confirm_score  × 0.20   (volume vs MA, cap 2x = 100)
  rsi_position_score    × 0.20   (distanza da zona neutrale 50)
  htf_alignment_score   × 0.20   (1h allineato = 100, no = 0)
  bb_position_score     × 0.10   (distanza da BB middle)

Regime detection (migliorato con CHOPPY):
  ADX > 35  → STRONG_TREND  (solo TrendFollowing, size × 1.5)
  ADX 25-35 → TRENDING      (Breakout + TrendFollowing)
  ADX 20-25 → NEUTRAL       (tutti gli engine)
  ADX 15-20 → RANGING       (solo MeanReversion)
  ADX < 15  → CHOPPY        (nessun trade — mercato senza direzione)

Soglie:
  score >= 70 → size × 1.5  (segnale forte)
  score 50-70 → size × 1.0
  score < 50  → skip
"""

import logging
from decimal import Decimal
from typing import Optional

from src.models import Signal, SignalType
from src.strategy.indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')
_ONE  = Decimal('1')
_HUNDRED = Decimal('100')

# Soglie regime (ADX)
_ADX_STRONG_TREND = Decimal('35')
_ADX_TRENDING     = Decimal('25')
_ADX_RANGING      = Decimal('20')
_ADX_CHOPPY       = Decimal('15')  # Nuovo: sotto questo → nessun trade

# Soglie score
_SCORE_STRONG = Decimal('70')
_SCORE_MIN    = Decimal('50')

# Moltiplicatori size per regime
_SIZE_MULT_STRONG_TREND = Decimal('1.5')   # Strong trend: size aumentata
_SIZE_MULT_NORMAL       = Decimal('1.0')
_SIZE_MULT_RANGING      = Decimal('0.8')   # Ranging: size ridotta (più conservativo)


class SignalAggregator:
    """
    Riceve segnali da tutte le strategie, li scora, seleziona il migliore.

    Args:
        indicators — IndicatorEngine già aggiornato con dati 15m
        htf_indicators — IndicatorEngine per timeframe di conferma (1h), opzionale
    """

    def __init__(
        self,
        indicators: IndicatorEngine,
        htf_indicators: Optional[IndicatorEngine] = None,
        macro_indicators: Optional[IndicatorEngine] = None,
    ) -> None:
        self._ind       = indicators
        self._htf_ind   = htf_indicators
        self._macro_ind = macro_indicators

        # Adaptive strategy weights: traccia performance rolling per strategia
        # {strategy_name: {"wins": int, "losses": int, "pnl": Decimal}}
        self._strategy_stats: dict[str, dict] = {}
        self._strategy_weight_cache: dict[str, Decimal] = {}

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self) -> str:
        """
        Ritorna il regime di mercato basato su ADX.

        STRONG_TREND → ADX > 35  (trend molto forte)
        TRENDING     → ADX 25-35 (trend)
        NEUTRAL      → ADX 20-25 (transizione)
        RANGING      → ADX 15-20 (ranging)
        CHOPPY       → ADX < 15  (nessun trade)

        Falls back to 'NEUTRAL' se ADX non disponibile.
        """
        try:
            adx_val = self._ind.adx()
            if adx_val >= _ADX_STRONG_TREND:
                return "STRONG_TREND"
            elif adx_val >= _ADX_TRENDING:
                return "TRENDING"
            elif adx_val >= _ADX_RANGING:
                return "NEUTRAL"
            elif adx_val >= _ADX_CHOPPY:
                return "RANGING"
            else:
                return "CHOPPY"
        except ValueError:
            return "NEUTRAL"

    def is_choppy_market(self) -> bool:
        """
        True se il mercato è in regime CHOPPY (ADX < 15).
        In questo caso il bot NON deve prendere trade di nessun tipo.
        """
        return self.detect_regime() == "CHOPPY"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, signal: Signal) -> Decimal:
        """
        Calcola score 0-100 per il segnale dato.
        Il segnale ha già uno score base impostato dall'engine.
        Qui applichiamo bonus/penalità contestuali.
        """
        if signal.signal_type == SignalType.NONE:
            return _ZERO

        base = signal.score  # score impostato dall'engine (es. 65, 72, 80)

        # Bonus ADX strength
        try:
            adx_val = self._ind.adx()
            adx_normalized = min(adx_val / Decimal('50'), _ONE) * Decimal('10')
        except ValueError:
            adx_normalized = _ZERO

        # Bonus volume: scala logaritmica continua (non step discreti)
        # log2(vol_ratio) × 4, capped [-4, +10]
        # 0.5x → -4, 1.0x → 0, 1.5x → +2.3, 2.0x → +4, 3.0x → +6.3, 4.0x → +8
        try:
            vol_ratio = self._ind.volume_ratio()
            if vol_ratio > _ZERO:
                import math
                log_vol = Decimal(str(math.log2(max(float(vol_ratio), 0.25))))
                vol_bonus = max(Decimal('-4'), min(log_vol * Decimal('4'), Decimal('10')))
            else:
                vol_bonus = Decimal('-4')
        except (ValueError, OverflowError):
            vol_bonus = _ZERO

        # HTF alignment: gradiente basato su distanza EMA (non binario)
        # Misura quanto forte è l'allineamento HTF: da -8 (forte contrario) a +10 (forte allineato)
        htf_bonus = _ZERO
        if self._htf_ind is not None:
            try:
                htf_fast = self._htf_ind.ema_fast()
                htf_slow = self._htf_ind.ema_slow()
                if htf_slow > _ZERO:
                    # Distanza normalizzata EMA fast vs slow (% del prezzo)
                    ema_spread = (htf_fast - htf_slow) / htf_slow
                    # Per LONG: spread positivo = allineato, negativo = contrario
                    # Per SHORT: invertito
                    if signal.signal_type == SignalType.SHORT:
                        ema_spread = -ema_spread
                    # Scala: 0.5% spread → +5, 1% → +8, 2%+ → +10, -0.5% → -5
                    htf_bonus = min(ema_spread * Decimal('1000'), Decimal('10'))
                    htf_bonus = max(htf_bonus, Decimal('-8'))
            except ValueError:
                pass

        # Multi-Timeframe Confluence: bonus se macro (4h) è allineato
        macro_bonus = _ZERO
        tf_aligned = 1  # primary TF è sempre "allineato" (ha generato il segnale)
        if htf_bonus > _ZERO:
            tf_aligned += 1  # HTF (1h) allineato
        if self._macro_ind is not None:
            try:
                macro_fast = self._macro_ind.ema_fast()
                macro_slow = self._macro_ind.ema_slow()
                if macro_slow > _ZERO:
                    macro_spread = (macro_fast - macro_slow) / macro_slow
                    if signal.signal_type == SignalType.SHORT:
                        macro_spread = -macro_spread
                    if macro_spread > Decimal('0.001'):  # >0.1% = allineato
                        macro_bonus = Decimal('5')
                        tf_aligned += 1
                    elif macro_spread < Decimal('-0.001'):  # contrario
                        macro_bonus = Decimal('-5')
            except ValueError:
                pass

        # Confluence multiplier: 3 TF allineati → bonus extra
        confluence_bonus = _ZERO
        if tf_aligned >= 3:
            confluence_bonus = Decimal('5')  # triple confluence bonus
        elif tf_aligned < 2:
            confluence_bonus = Decimal('-3')  # solo 1 TF: penalità

        # Adaptive strategy weight: penalizza/premia in base a performance storica
        strat_weight = self.strategy_weight(signal.strategy_name)
        raw_score = base + adx_normalized + vol_bonus + htf_bonus + macro_bonus + confluence_bonus
        final_score = min(raw_score * strat_weight, _HUNDRED)
        return max(_ZERO, final_score).quantize(Decimal('0.1'))

    # ------------------------------------------------------------------
    # Selezione migliore
    # ------------------------------------------------------------------

    def select_best(self, signals: list[Signal]) -> Optional[Signal]:
        """
        Aggiorna lo score di ogni segnale con il contesto corrente,
        filtra quelli sotto la soglia minima, ritorna il migliore.
        Applica size multiplier se segnale forte (score >= 70).
        """
        valid: list[tuple[Decimal, Signal]] = []

        for sig in signals:
            if sig.signal_type == SignalType.NONE:
                continue
            final_score = self.score(sig)
            if final_score < _SCORE_MIN:
                logger.debug(
                    "Segnale %s %s scartato (score=%.1f < %.0f)",
                    sig.strategy_name, sig.signal_type.name, final_score, _SCORE_MIN,
                )
                continue
            # Applica moltiplicatore size in base a regime + score
            regime = self.detect_regime()
            if regime == "STRONG_TREND" and final_score >= _SCORE_STRONG:
                sig.size = (sig.size * _SIZE_MULT_STRONG_TREND).quantize(Decimal('0.001'))
            elif final_score >= _SCORE_STRONG:
                sig.size = (sig.size * _SIZE_MULT_NORMAL).quantize(Decimal('0.001'))
            elif regime == "RANGING":
                sig.size = (sig.size * _SIZE_MULT_RANGING).quantize(Decimal('0.001'))
            sig.score = final_score
            valid.append((final_score, sig))

        if not valid:
            return None

        # Ordina per score decrescente, ritorna il migliore
        valid.sort(key=lambda x: x[0], reverse=True)
        best_score, best_signal = valid[0]
        logger.info(
            "Segnale selezionato: %s %s score=%.1f",
            best_signal.strategy_name, best_signal.signal_type.name, best_score,
        )
        return best_signal

    # ------------------------------------------------------------------
    # Helper regime → strategie consigliate
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Adaptive strategy weights
    # ------------------------------------------------------------------

    def record_trade_result(self, strategy_name: str, pnl: Decimal) -> None:
        """
        Registra il risultato di un trade per una strategia.
        Usato per calcolare i pesi adattivi delle strategie.
        """
        if strategy_name not in self._strategy_stats:
            self._strategy_stats[strategy_name] = {"wins": 0, "losses": 0, "pnl": _ZERO}

        stats = self._strategy_stats[strategy_name]
        stats["pnl"] += pnl
        if pnl > _ZERO:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Ricalcola pesi
        self._update_strategy_weights()

    def _update_strategy_weights(self) -> None:
        """
        Ricalcola i pesi adattivi per ogni strategia basandosi su performance.

        Logica:
          - Win rate > 55% → peso 1.2 (boost)
          - Win rate 40-55% → peso 1.0 (neutro)
          - Win rate < 40% → peso 0.7 (penalità)
          - PnL negativo cumulativo → peso 0.6 (forte penalità)
          - Meno di 5 trade → peso 1.0 (dati insufficienti)
        """
        for name, stats in self._strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            if total < 5:
                self._strategy_weight_cache[name] = _ONE
                continue

            win_rate = Decimal(stats["wins"]) / Decimal(total)

            if stats["pnl"] < _ZERO:
                weight = Decimal('0.6')
            elif win_rate > Decimal('0.55'):
                weight = Decimal('1.2')
            elif win_rate >= Decimal('0.40'):
                weight = _ONE
            else:
                weight = Decimal('0.7')

            self._strategy_weight_cache[name] = weight
            logger.debug(
                "Strategy weight: %s → %.1f (WR=%.0f%% PnL=%.2f trades=%d)",
                name, float(weight), float(win_rate * 100), float(stats["pnl"]), total,
            )

    def strategy_weight(self, strategy_name: str) -> Decimal:
        """Ritorna il peso adattivo per una strategia (default 1.0)."""
        return self._strategy_weight_cache.get(strategy_name, _ONE)

    def preferred_strategies(self) -> list[str]:
        """
        Ritorna la lista delle strategie preferite per il regime corrente.
        Usata dal bot per evitare di eseguire engine non adatti al regime.

        CHOPPY → lista vuota (nessun trade)
        RANGING → solo MeanReversion (mercato oscillante)
        NEUTRAL → tutti gli engine
        TRENDING → Breakout + TrendFollowing
        STRONG_TREND → solo TrendFollowing (trend fortissimo)
        """
        regime = self.detect_regime()
        if regime == "CHOPPY":
            logger.debug("Regime CHOPPY (ADX < 15): nessun trade.")
            return []                                           # nessun segnale ammesso
        elif regime == "STRONG_TREND":
            return ["trend_following", "funding_rate"]
        elif regime == "TRENDING":
            return ["breakout", "trend_following", "funding_rate"]
        elif regime == "RANGING":
            return ["mean_reversion", "funding_rate"]
        else:  # NEUTRAL
            return ["breakout", "mean_reversion", "trend_following", "funding_rate"]
