# scripts/twas/models/base.py
"""
Base class for expression prediction models.

All models must implement this interface for consistent usage
in TWAS simulations.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.model_selection import KFold


class ExpressionModel(ABC):
    """
    Abstract base class for expression prediction models.
    
    All models predict gene expression from genotypes and provide
    weights for TWAS association testing.
    """

    def __init__(self, **kwargs: Any):
        """Initialize model with optional parameters."""
        self.weights_: np.ndarray | None = None
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExpressionModel":
        """
        Fit the model to training data.
        
        Args:
            X: Genotype matrix (n_samples, n_snps)
            y: Expression values (n_samples,)
            
        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expression from genotypes.
        
        Args:
            X: Genotype matrix (n_samples, n_snps)
            
        Returns:
            Predicted expression values
        """
        pass

    def get_weights(self) -> np.ndarray:
        """
        Get model weights for each SNP.
        
        Returns:
            Array of weights (n_snps,)
            
        Raises:
            ValueError: If model not fitted
        """
        if self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.weights_

    def get_nonzero_snps(self) -> np.ndarray:
        """
        Get indices of SNPs with non-zero weights.
        
        Returns:
            Array of indices
        """
        weights = self.get_weights()
        return np.where(np.abs(weights) > 1e-10)[0]

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model with cross-validation.
        
        Args:
            X: Genotype matrix
            y: Expression values
            cv: Number of folds
            seed: Random seed
            
        Returns:
            Dict with cv_r2, cv_corr, n_nonzero
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        # Get cross-validated predictions
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in kfold.split(X):
            # Clone and fit on fold
            model = self.__class__(**self._get_params())
            model.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X[test_idx])

        # Compute metrics
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        corr = np.corrcoef(y, y_pred)[0, 1] if np.std(y_pred) > 0 else 0.0

        # Fit on full data for n_nonzero
        self.fit(X, y)
        n_nonzero = len(self.get_nonzero_snps())

        return {
            "cv_r2": max(0, r2),  # Clip negative R2
            "cv_corr": corr if not np.isnan(corr) else 0.0,
            "n_nonzero": n_nonzero,
        }

    def _get_params(self) -> dict[str, Any]:
        """Get model parameters for cloning. Override in subclasses."""
        return {}
