# scripts/twas/models/elastic_net.py
"""
Elastic Net expression prediction model.

Combines L1 (LASSO) and L2 (Ridge) regularization for
sparse yet stable coefficient estimates.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV

from .base import ExpressionModel
from . import register_model


@register_model("elastic_net")
class ElasticNetModel(ExpressionModel):
    """
    Elastic Net regression for expression prediction.
    
    Uses cross-validation to select optimal regularization parameters.
    Good default choice for TWAS as it produces sparse models while
    handling correlated predictors (SNPs in LD).
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        n_alphas: int = 100,
        cv: int = 5,
        max_iter: int = 10000,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize ElasticNet model.
        
        Args:
            l1_ratio: Balance between L1 and L2 (0=Ridge, 1=LASSO)
            n_alphas: Number of alpha values to try
            cv: Cross-validation folds for alpha selection
            max_iter: Maximum iterations for optimization
            random_state: Random seed
        """
        super().__init__(**kwargs)
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self._model: ElasticNetCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetModel":
        """Fit ElasticNet with cross-validated alpha selection."""
        self._model = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            n_alphas=self.n_alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self.weights_ = self._model.coef_
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression values."""
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def _get_params(self) -> dict[str, Any]:
        """Get parameters for cloning."""
        return {
            "l1_ratio": self.l1_ratio,
            "n_alphas": self.n_alphas,
            "cv": self.cv,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
