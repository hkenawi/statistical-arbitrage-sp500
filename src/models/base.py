"""
Abstract base class for all statistical arbitrage models.

Every model — DNN, GBT, RAF, LSTM, CNN — must inherit from BaseModel
and implement fit() and predict_proba(). This ensures the backtest engine
and Streamlit app can swap models without any model-specific logic.

Usage
-----
from src.models.base import BaseModel

class MyModel(BaseModel):
    def fit(self, X_train, y_train):
        ...
    def predict_proba(self, X):
        ...
"""
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models in the statistical arbitrage pipeline.

    All models must implement:
        fit(X_train, y_train)  — train the model on one batch window
        predict_proba(X)       — return P(outperform) for each observation

    The backtest engine and Streamlit app interact exclusively through
    this interface and have no knowledge of model internals.
    """

    @abstractmethod
    def fit(self,
            X_train: pd.DataFrame | np.ndarray,
            y_train: pd.Series | np.ndarray) -> None:
        """
        Train the model on one sliding window training set.

        Parameters
        ----------
        X_train : shape (n_samples, n_features) for DNN/GBT/RAF
                  shape (n_samples, sequence_length) for LSTM/CNN
        y_train : shape (n_samples,) binary labels in {0, 1}
        """
        ...

    @abstractmethod
    def predict_proba(self,
                      X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return the probability that each observation outperforms the
        cross-sectional median on the next trading day.

        Parameters
        ----------
        X : shape (n_samples, n_features) for DNN/GBT/RAF
            shape (n_samples, sequence_length) for LSTM/CNN

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in (0, 1)
        """
        ...

    def tune(self, X_train, y_train) -> dict:
        """
        Optional hyperparameter tuning. Subclasses that support Optuna
        override this method. Returns best_params dict.

        Not an abstractmethod because it does not need to be implemented by OG models. 
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"