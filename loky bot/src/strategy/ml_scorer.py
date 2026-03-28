"""
MLScorer — online logistic regression per scoring dei segnali.

Pure Python, zero dipendenze esterne. Implementa:
  - Logistic regression con gradient descent online
  - Feature vector da indicatori tecnici
  - Weight decay per prevenire overfitting
  - Attivazione dopo min_trades per evitare sizing erratico

Feature vector (8 dimensioni):
  [ADX_norm, RSI_norm, ATR_pctile, vol_ratio_log, HTF_aligned,
   CVD_trend, regime_num, bb_width_norm]

Output: probabilità 0-1 che il trade sia vincente.
"""

import logging
import math
from collections import deque
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


class MLScorer:
    """
    Online logistic regression per win probability estimation.

    Args:
        n_features — dimensione del feature vector (default 8)
        learning_rate — tasso di apprendimento (default 0.01)
        weight_decay — regolarizzazione L2 (default 0.001)
        min_trades — trade minimi prima di attivare (default 30)
    """

    def __init__(
        self,
        n_features: int = 8,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        min_trades: int = 30,
    ) -> None:
        self._n_features = n_features
        self._lr = learning_rate
        self._wd = weight_decay
        self._min_trades = min_trades

        # Pesi inizializzati a zero (prior neutro)
        self._weights = [0.0] * n_features
        self._bias = 0.0
        self._n_updates = 0

        # History per normalizzazione running
        self._feature_means = [0.0] * n_features
        self._feature_vars = [1.0] * n_features
        self._n_samples = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid con clamp per evitare overflow."""
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _normalize(self, features: list[float]) -> list[float]:
        """Z-score normalization con running stats."""
        if self._n_samples < 5:
            return features
        return [
            (f - self._feature_means[i]) / max(math.sqrt(self._feature_vars[i]), 1e-8)
            for i, f in enumerate(features)
        ]

    def _update_running_stats(self, features: list[float]) -> None:
        """Welford's online algorithm per mean e variance."""
        self._n_samples += 1
        n = self._n_samples
        for i, f in enumerate(features):
            old_mean = self._feature_means[i]
            self._feature_means[i] += (f - old_mean) / n
            self._feature_vars[i] += (f - old_mean) * (f - self._feature_means[i])
            if n > 1:
                self._feature_vars[i] = self._feature_vars[i] / (n - 1) if n > 1 else 1.0

    def predict(self, features: list[float]) -> float:
        """
        Predice la probabilità di win (0-1) per il feature vector dato.

        Returns:
            float — probabilità (0.5 = neutro, >0.6 = segnale buono)
        """
        if self._n_updates < self._min_trades:
            return 0.5  # neutro fino a min_trades

        norm = self._normalize(features)
        z = self._bias + sum(w * f for w, f in zip(self._weights, norm))
        return self._sigmoid(z)

    def update(self, features: list[float], won: bool) -> None:
        """
        Aggiorna i pesi con un singolo esempio (online learning).

        Args:
            features — feature vector al momento dell'entry
            won — True se il trade è stato profittevole
        """
        self._update_running_stats(features)

        target = 1.0 if won else 0.0
        norm = self._normalize(features)
        pred = self._sigmoid(self._bias + sum(w * f for w, f in zip(self._weights, norm)))

        # Gradient: d_loss/d_w = (pred - target) * feature
        error = pred - target
        for i in range(self._n_features):
            grad = error * norm[i] + self._wd * self._weights[i]
            self._weights[i] -= self._lr * grad
        self._bias -= self._lr * error

        self._n_updates += 1

    def score_modifier(self, features: list[float]) -> Decimal:
        """
        Ritorna un moltiplicatore di score basato sulla probabilità ML.

        prob > 0.65 → ×1.15 (ML dice alto win probability)
        prob 0.45-0.65 → ×1.0 (neutro)
        prob < 0.45 → ×0.80 (ML dice basso win probability)
        """
        prob = self.predict(features)
        if prob > 0.65:
            return Decimal('1.15')
        elif prob < 0.45:
            return Decimal('0.80')
        return Decimal('1.0')

    @staticmethod
    def build_features(
        adx: float,
        rsi: float,
        atr_pctile: float,
        vol_ratio: float,
        htf_aligned: bool,
        cvd_bullish: bool,
        regime_num: int,
        bb_width: float,
    ) -> list[float]:
        """Costruisce il feature vector normalizzato."""
        return [
            adx / 60.0,                    # ADX normalizzato 0-1
            rsi / 100.0,                   # RSI normalizzato 0-1
            atr_pctile,                    # già 0-1
            math.log2(max(vol_ratio, 0.25)) / 3.0,  # log volume ratio
            1.0 if htf_aligned else 0.0,   # binario
            1.0 if cvd_bullish else 0.0,   # binario
            regime_num / 4.0,              # 0=CHOPPY, 4=STRONG_TREND
            min(bb_width * 50.0, 1.0),     # BB width normalizzato
        ]

    @property
    def is_active(self) -> bool:
        return self._n_updates >= self._min_trades

    def stats(self) -> dict:
        return {
            "n_updates": self._n_updates,
            "is_active": self.is_active,
            "weights": self._weights.copy(),
            "bias": self._bias,
        }
