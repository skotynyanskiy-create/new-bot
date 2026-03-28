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
import math
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
        Scoring 0-100 con approccio confluence-first.

        Pesi (rivisti per massima profittabilità):
          Trend strength (ADX)    : 25%  — trend forti hanno momentum
          HTF alignment (1h)      : 25%  — previene whipsaw
          Confluence TF (4h)      : 20%  — triple TF = segnale istituzionale
          Volume confirmation     : 15%  — valida il movimento
          Strategy weight         : 15%  — premia strategie vincenti

        Il base score dell'engine vale come floor (non viene ignorato).
        """
        if signal.signal_type == SignalType.NONE:
            return _ZERO

        base = signal.score  # score dall'engine (65, 72, 80)

        # --- 1. Trend Strength (25%) ---
        # ADX normalizzato su range 0-60 (più realistico di 0-50)
        try:
            adx_val = self._ind.adx()
            trend_score = min(adx_val / Decimal('60'), _ONE) * Decimal('25')
        except ValueError:
            trend_score = Decimal('12')  # neutro

        # --- 2. HTF Alignment (25%) ---
        # Gradiente: forte allineamento = 25pt, forte contrario = -15pt
        htf_score = _ZERO
        htf_aligned = False
        if self._htf_ind is not None:
            try:
                htf_fast = self._htf_ind.ema_fast()
                htf_slow = self._htf_ind.ema_slow()
                if htf_slow > _ZERO:
                    ema_spread = (htf_fast - htf_slow) / htf_slow
                    if signal.signal_type == SignalType.SHORT:
                        ema_spread = -ema_spread
                    # Simmetrico: ±20 (stessa penalità per contrario e premio per allineato)
                    htf_score = min(max(ema_spread * Decimal('2000'), Decimal('-20')), Decimal('20'))
                    if ema_spread > Decimal('0.001'):  # >0.1% spread = definitivamente allineato
                        htf_aligned = True
            except ValueError:
                pass

        # --- 3. Confluence TF (20%) ---
        # Macro (4h) allineamento + conteggio TF
        tf_aligned = 1  # primary sempre allineato
        if htf_aligned:
            tf_aligned += 1
        macro_aligned = False
        if self._macro_ind is not None:
            try:
                macro_fast = self._macro_ind.ema_fast()
                macro_slow = self._macro_ind.ema_slow()
                if macro_slow > _ZERO:
                    macro_spread = (macro_fast - macro_slow) / macro_slow
                    if signal.signal_type == SignalType.SHORT:
                        macro_spread = -macro_spread
                    if macro_spread > Decimal('0.001'):
                        tf_aligned += 1
                        macro_aligned = True
            except ValueError:
                pass

        # Confluence scaling: 3TF=20pt, 2TF=12pt, 1TF=4pt (penalità forte)
        if tf_aligned >= 3:
            confluence_score = Decimal('20')
        elif tf_aligned == 2:
            confluence_score = Decimal('12')
        else:
            confluence_score = Decimal('4')

        # --- 4. Volume Confirmation (15%) ---
        # Percentile-based: volume rank nella rolling window
        try:
            vol_ratio = self._ind.volume_ratio()
            if vol_ratio > _ZERO:
                log_vol = Decimal(str(math.log2(max(float(vol_ratio), 0.25))))
                vol_score = max(Decimal('0'), min((log_vol + Decimal('1')) * Decimal('7.5'), Decimal('15')))
            else:
                vol_score = _ZERO
        except (ValueError, OverflowError):
            vol_score = Decimal('7')  # neutro

        # --- 5. Strategy Weight (15%) ---
        strat_weight = self.strategy_weight(signal.strategy_name)
        # Converti peso (0.6-1.2) in punti (0-15)
        strat_score = min((strat_weight - Decimal('0.5')) * Decimal('21.4'), Decimal('15'))
        strat_score = max(_ZERO, strat_score)

        # Score contestuale (0-100)
        context_score = trend_score + htf_score + confluence_score + vol_score + strat_score

        # Blend: 75% engine base + 25% contesto (engine signals sono calibrati, contesto è rumore)
        blended = base * Decimal('0.75') + context_score * Decimal('0.25')
        final_score = max(_ZERO, min(blended, _HUNDRED))
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
