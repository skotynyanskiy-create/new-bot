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
    ) -> None:
        self._ind     = indicators
        self._htf_ind = htf_indicators

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

        # Bonus volume: volume sopra la media → conferma del segnale
        try:
            vol_ratio = self._ind.volume_ratio()
            # Volume > 1.5× media → +5, > 2× → +8, < 0.5× → -3
            if vol_ratio >= Decimal('2.0'):
                vol_bonus = Decimal('8')
            elif vol_ratio >= Decimal('1.5'):
                vol_bonus = Decimal('5')
            elif vol_ratio < Decimal('0.5'):
                vol_bonus = Decimal('-3')
            else:
                vol_bonus = _ZERO
        except ValueError:
            vol_bonus = _ZERO

        # HTF alignment bonus
        htf_bonus = _ZERO
        if self._htf_ind is not None:
            try:
                htf_fast = self._htf_ind.ema_fast()
                htf_slow = self._htf_ind.ema_slow()
                if signal.signal_type == SignalType.LONG and htf_fast > htf_slow:
                    htf_bonus = Decimal('8')
                elif signal.signal_type == SignalType.SHORT and htf_fast < htf_slow:
                    htf_bonus = Decimal('8')
                else:
                    htf_bonus = Decimal('-5')  # HTF contrario = penalità
            except ValueError:
                pass

        final_score = min(base + adx_normalized + vol_bonus + htf_bonus, _HUNDRED)
        return final_score.quantize(Decimal('0.1'))

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
