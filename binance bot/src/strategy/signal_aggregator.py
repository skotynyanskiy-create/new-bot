"""
SignalAggregator — scoring e selezione del miglior segnale.

Scoring formula (0-100):
  trend_strength_score  × 0.30   (ADX normalizzato 0-100)
  volume_confirm_score  × 0.20   (volume vs MA, cap 2x = 100)
  rsi_position_score    × 0.20   (distanza da zona neutrale 50)
  htf_alignment_score   × 0.20   (1h allineato = 100, no = 0)
  bb_position_score     × 0.10   (distanza da BB middle)

Regime detection:
  ADX > 25  → TRENDING   (priorità Breakout + TrendFollowing)
  ADX < 20  → RANGING    (priorità MeanReversion)
  altrimenti → NEUTRAL

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

# Soglie regime
_ADX_TRENDING = Decimal('25')
_ADX_RANGING  = Decimal('20')

# Soglie score
_SCORE_STRONG = Decimal('70')
_SCORE_MIN    = Decimal('50')

# Moltiplicatori size
_SIZE_MULT_STRONG = Decimal('1.5')
_SIZE_MULT_NORMAL = Decimal('1.0')


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
        Ritorna 'TRENDING', 'RANGING', o 'NEUTRAL' basato su ADX.
        Falls back to 'NEUTRAL' se ADX non disponibile.
        """
        try:
            adx_val = self._ind.adx()
            if adx_val > _ADX_TRENDING:
                return "TRENDING"
            elif adx_val < _ADX_RANGING:
                return "RANGING"
            else:
                return "NEUTRAL"
        except ValueError:
            return "NEUTRAL"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, signal: Signal) -> Decimal:
        """
        Calcola score 0-100 per il segnale dato.
        Il segnale ha già uno score base impostato dall'engine.
        Qui applichiamo bonus/penalità contestuali:
          - ADX strength:    0-10 punti
          - Volume confirm:  0-8  punti
          - RSI distance:    0-7  punti
          - BB position:     0-5  punti
          - HTF alignment:   +8 o -5 punti
        """
        if signal.signal_type == SignalType.NONE:
            return _ZERO

        base = signal.score  # score impostato dall'engine (es. 65, 72, 80)

        # Bonus ADX strength (0-10)
        try:
            adx_val = self._ind.adx()
            adx_normalized = min(adx_val / Decimal('50'), _ONE) * Decimal('10')
        except ValueError:
            adx_normalized = _ZERO

        # Bonus volume confirmation (0-8)
        vol_bonus = _ZERO
        try:
            vol_ma = self._ind.volume_ma()
            if vol_ma > _ZERO:
                # Recupera ultimo volume dal buffer indicatori
                candles = self._ind._candles
                if candles:
                    vol_ratio = candles[-1].volume / vol_ma
                    # vol_ratio >= 2.0 → 8 punti, 1.5 → 6, 1.0 → 4, <1.0 → 0
                    vol_bonus = min(max(vol_ratio - _ONE, _ZERO) * Decimal('8'), Decimal('8'))
        except (ValueError, ZeroDivisionError):
            pass

        # Bonus RSI distance from neutral (0-7)
        rsi_bonus = _ZERO
        try:
            rsi_val = self._ind.rsi()
            rsi_distance = abs(rsi_val - Decimal('50'))
            # RSI a 30 o 70 → distance=20 → 5.6 punti; RSI a 20 o 80 → 8.4 capped a 7
            rsi_bonus = min(rsi_distance / Decimal('25') * Decimal('7'), Decimal('7'))
        except ValueError:
            pass

        # Bonus BB position (0-5): premia segnali vicino alle bande
        bb_bonus = _ZERO
        try:
            bb_upper = self._ind.bb_upper()
            bb_lower = self._ind.bb_lower()
            bb_width = bb_upper - bb_lower
            if bb_width > _ZERO and self._ind._candles:
                price = self._ind._candles[-1].close
                if signal.signal_type == SignalType.LONG:
                    # Più vicino a BB lower → più punti
                    dist_from_lower = (price - bb_lower) / bb_width
                    bb_bonus = max(_ZERO, (_ONE - dist_from_lower) * Decimal('5'))
                elif signal.signal_type == SignalType.SHORT:
                    dist_from_upper = (bb_upper - price) / bb_width
                    bb_bonus = max(_ZERO, (_ONE - dist_from_upper) * Decimal('5'))
        except ValueError:
            pass

        # HTF alignment bonus (+8 / -5)
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

        final_score = min(
            base + adx_normalized + vol_bonus + rsi_bonus + bb_bonus + htf_bonus,
            _HUNDRED,
        )
        return max(final_score, _ZERO).quantize(Decimal('0.1'))

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
            # Applica moltiplicatore size
            if final_score >= _SCORE_STRONG:
                sig.size = (sig.size * _SIZE_MULT_STRONG).quantize(Decimal('0.001'))
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
        Usata dal bot per evitare di eseguire engine non adatti.
        """
        regime = self.detect_regime()
        if regime == "TRENDING":
            return ["breakout", "trend_following", "funding_rate"]
        elif regime == "RANGING":
            return ["mean_reversion", "funding_rate"]
        else:  # NEUTRAL
            return ["breakout", "mean_reversion", "trend_following", "funding_rate"]
