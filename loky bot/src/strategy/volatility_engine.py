"""
VolatilityRegimeEngine — identifica il regime di volatilità multi-indicatore.

Combina 3 metriche per una classificazione più robusta rispetto al solo ADX:
  1. ATR percentile (rolling 100 candle): alta/bassa volatilità storica
  2. Bollinger Width: ampiezza delle bande (squeeze = compressione)
  3. ADX: forza del trend

Regimi di volatilità:
  COMPRESSION  — BB width ai minimi + ATR basso → breakout imminente (entry preferenziale)
  EXPANSION    — BB width in crescita + ATR alto → trend in corso (follow trend)
  CONTRACTION  — BB width in calo dopo espansione → trend esaurito (cautela)
  NORMAL       — condizioni medie (tutto ammesso)

Questo engine NON genera segnali direttamente, ma fornisce un moltiplicatore
di score e size che il SignalAggregator può usare.
"""

import logging
from collections import deque
from decimal import Decimal
from enum import Enum
from typing import Optional

from src.strategy.indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)

_ZERO = Decimal('0')


class VolatilityRegime(Enum):
    COMPRESSION = "compression"   # squeeze: BB strette, ATR basso → breakout imminente
    EXPANSION = "expansion"       # volatilità in crescita → trend in corso
    CONTRACTION = "contraction"   # volatilità in calo → trend esaurito
    NORMAL = "normal"             # condizioni medie


class VolatilityRegimeEngine:
    """
    Classifica il regime di volatilità corrente usando ATR + BB Width.

    Args:
        indicators — IndicatorEngine per il timeframe primario
        bb_width_history_len — lunghezza rolling window per BB width
    """

    def __init__(
        self,
        indicators: IndicatorEngine,
        bb_width_history_len: int = 50,
    ) -> None:
        self._ind = indicators
        self._bb_width_history: deque[Decimal] = deque(maxlen=bb_width_history_len)
        self._atr_history: deque[Decimal] = deque(maxlen=100)

    def update(self) -> None:
        """Aggiorna le rolling history con i valori correnti."""
        try:
            bb_upper = self._ind.bb_upper()
            bb_lower = self._ind.bb_lower()
            bb_middle = self._ind.bb_middle()
            atr = self._ind.atr()

            if bb_middle > _ZERO:
                bb_width = (bb_upper - bb_lower) / bb_middle
                self._bb_width_history.append(bb_width)
            self._atr_history.append(atr)
        except ValueError:
            pass

    def detect(self) -> VolatilityRegime:
        """
        Classifica il regime di volatilità corrente.

        COMPRESSION: BB width nel 20° percentile inferiore E ATR nel 30° inferiore
        EXPANSION: BB width nel 70° percentile superiore E ATR in crescita
        CONTRACTION: BB width in calo (ultimi 5 < ultimi 20) E ATR in calo
        NORMAL: tutto il resto
        """
        if len(self._bb_width_history) < 20 or len(self._atr_history) < 20:
            return VolatilityRegime.NORMAL

        current_bb_width = self._bb_width_history[-1]
        current_atr = self._atr_history[-1]

        # Percentile BB width
        sorted_bb = sorted(self._bb_width_history)
        n = len(sorted_bb)
        bb_rank = sum(1 for w in sorted_bb if w < current_bb_width)
        bb_percentile = Decimal(bb_rank) / Decimal(n)

        # Percentile ATR
        sorted_atr = sorted(self._atr_history)
        n_atr = len(sorted_atr)
        atr_rank = sum(1 for a in sorted_atr if a < current_atr)
        atr_percentile = Decimal(atr_rank) / Decimal(n_atr)

        # Trend BB width: media ultimi 5 vs media ultimi 20
        recent_5 = list(self._bb_width_history)[-5:]
        recent_20 = list(self._bb_width_history)[-20:]
        avg_recent = sum(recent_5) / Decimal(len(recent_5))
        avg_older = sum(recent_20) / Decimal(len(recent_20))

        bb_expanding = avg_recent > avg_older * Decimal('1.1')
        bb_contracting = avg_recent < avg_older * Decimal('0.9')

        # COMPRESSION: bande strette + ATR basso → breakout imminente
        if bb_percentile < Decimal('0.20') and atr_percentile < Decimal('0.30'):
            return VolatilityRegime.COMPRESSION

        # EXPANSION: bande in crescita + ATR alto → trend in corso
        if bb_expanding and atr_percentile > Decimal('0.60'):
            return VolatilityRegime.EXPANSION

        # CONTRACTION: bande in calo dopo espansione → trend esaurito
        if bb_contracting and atr_percentile > Decimal('0.40'):
            return VolatilityRegime.CONTRACTION

        return VolatilityRegime.NORMAL

    def score_modifier(self) -> Decimal:
        """
        Ritorna un moltiplicatore di score basato sul regime di volatilità.

        COMPRESSION → ×1.15 (breakout imminente, segnali più affidabili)
        EXPANSION   → ×1.10 (trend in corso, conferma)
        CONTRACTION → ×0.85 (trend esaurito, cautela)
        NORMAL      → ×1.00
        """
        regime = self.detect()
        if regime == VolatilityRegime.COMPRESSION:
            return Decimal('1.15')
        elif regime == VolatilityRegime.EXPANSION:
            return Decimal('1.10')
        elif regime == VolatilityRegime.CONTRACTION:
            return Decimal('0.85')
        return Decimal('1.00')

    @property
    def regime_name(self) -> str:
        return self.detect().value
