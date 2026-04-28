"""
Random Forest (RAF) model for statistical arbitrage on the S&P 500.
Replicates the RAF baseline from Krauss, Do & Huck (2016), Section 4.3.3.

The paper trains a Random Forest with 1000 trees, max depth 20, and feature
subsampling of mRAF = floor(sqrt(31)) ≈ 5 features per split — the standard
sklearn default for classification. The RAF achieves the highest standalone
Sharpe ratio (1.90 post-cost) among all three base models in the paper,
attributed to its immunity to overfitting via random feature selection and
its ability to capture high-order interactions through deep trees.

Key paper details reproduced here (Section 4.3.3):
    - 1000 decision trees (BRAF = 1000)
    - max_depth = 20 — default in H2O/sklearn; allows high-order interactions
    - max_features = "sqrt" — mRAF = floor(sqrt(31)) ≈ 5 features per split
    - seed = 1 for reproducibility (paper states seed fixed to 1)

Hyperparameter tuning:
    When use_tuner=True, fit() automatically runs an Optuna study on a
    chronological validation split of the training data before training the
    final model. Best parameters are saved to configs/random_forest_best_params.json
    for reproducibility. When use_tuner=False, fixed hyperparameters from
    the paper (or passed to the constructor) are used directly.

Input:
    X of shape (n_samples, 31) — 31 lagged return features per observation:
    R(1)–R(20) at daily resolution, then R(40), R(60), …, R(240) at monthly.

Output:
    predict_proba() returns P(outperform cross-sectional median) in (0, 1)
    for each observation — identical interface to all other models.
"""
import json
import numpy as np
import pandas as pd
import optuna

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from src.models.base import BaseModel

# Suppress Optuna's per-trial logging — we log our own summary instead
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Path to save best hyperparameters found by Optuna
root = Path(__file__).resolve().parents[2]
PARAMS_PATH = root / "configs" / "random_forest_best_params.json"

# Default hyperparameters matching the paper exactly (Section 4.3.3)
DEFAULTS = {
    "n_trees": 1000,
    "max_depth": 20,
    "col_sample": "sqrt",  # mRAF = floor(sqrt(31)) ≈ 5 features per split
    "seed": 1,             # paper explicitly sets seed to 1
}


class RandomForestModel(BaseModel):
    """
    Random Forest wrapper implementing the BaseModel interface.

    Replicates the RAF baseline from Krauss, Do & Huck (2016) trained on
    31 hand-crafted lag features. Among the three base models in the paper,
    the RAF achieves the highest Sharpe ratio (1.90 post-cost), attributed
    to its immunity to overfitting via rigid random feature selection and its
    ability to capture high-order interactions through deep trees (max_depth=20).

    Unlike boosting, random forests are not prone to overfitting, so a high
    BRAF of 1000 trees is safe — more trees strictly reduce variance without
    increasing bias (Breiman, 2001). The paper confirms this: doubling to 2000
    trees only marginally improves returns (0.44 vs 0.43 percent per day).

    Parameters
    ----------
    n_trees      : number of decision trees in the forest (paper: 1000)
    max_depth    : maximum tree depth (paper: 20 — allows high-order interactions)
    col_sample   : feature subsampling strategy per split. "sqrt" gives
                   mRAF = floor(sqrt(31)) ≈ 5 features (paper default).
                   Can also be "log2" or a float in (0, 1].
    seed         : random seed for reproducibility (paper: 1)
    use_tuner    : if True, run Optuna before training to find optimal
                   hyperparameters. Overrides all other hyperparameter
                   arguments. Best params saved to configs/random_forest_best_params.json
    n_trials     : number of Optuna trials (only used when use_tuner=True)
    val_fraction : fraction of training data held out for Optuna validation
                   (only used when use_tuner=True). Split is chronological
                   to avoid lookahead bias.
    n_jobs       : number of CPU cores to use for parallel tree building.
                   -1 uses all available cores. RAF is embarrassingly parallel
                   — each tree is fully independent — so parallelism is safe.

    Examples
    --------
    # Paper-exact configuration
    model = RandomForestModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # Optuna tuning — fit() handles everything automatically
    model = RandomForestModel(use_tuner=True, n_trials=50)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)
    """

    def __init__(self,
                 n_trees: int = DEFAULTS["n_trees"],
                 max_depth: int = DEFAULTS["max_depth"],
                 col_sample: str | float = DEFAULTS["col_sample"],
                 seed: int = DEFAULTS["seed"],
                 use_tuner: bool = False,
                 n_trials: int = 50,
                 val_fraction: float = 0.2,
                 n_jobs: int = -1):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.col_sample = col_sample
        self.seed = seed
        self.use_tuner = use_tuner
        self.n_trials = n_trials
        self.val_fraction = val_fraction
        self.n_jobs = n_jobs

        self.model = None        # RandomForestClassifier built in fit()
        self.best_params = None  # populated by tune() if use_tuner=True

    def _to_numpy(self,
                  X: pd.DataFrame | np.ndarray,
                  y: pd.Series | np.ndarray | None = None):
        """Convert pandas/numpy inputs to plain numpy arrays."""
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else np.array(y)
            return X_np, y_np
        return X_np

    def _build_model(self,
                     n_trees: int,
                     max_depth: int,
                     col_sample: str | float) -> RandomForestClassifier:
        """
        Instantiate a RandomForestClassifier with the given hyperparameters.

        Uses entropy criterion to match the classification behaviour described
        in the paper. n_jobs=-1 exploits all available cores — RAF tree
        building is embarrassingly parallel (each tree is fully independent).
        """
        return RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features=col_sample,
            criterion="entropy",
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

    def tune(self,
             X_train: pd.DataFrame | np.ndarray,
             y_train: pd.Series | np.ndarray) -> dict:
        """
        Run an Optuna study on a chronological validation split of the
        training data to find the best hyperparameters.

        Called automatically by fit() when use_tuner=True.
        Best parameters are saved to configs/random_forest_best_params.json.

        The paper notes that RAF is robust to hyperparameter choice — even
        doubling or halving n_trees produces near-identical results. Optuna
        can confirm this and identify any gains available beyond the defaults.

        Returns
        -------
        dict of best hyperparameters
        """
        X_np, y_np = self._to_numpy(X_train, y_train)

        # Chronological split — never shuffle financial time series
        n_val = int(len(X_np) * self.val_fraction)
        n_train = len(X_np) - n_val
        X_tr, X_val = X_np[:n_train], X_np[n_train:]
        y_tr, y_val = y_np[:n_train], y_np[n_train:]

        print(f"  Running Optuna study ({self.n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            n_trees = trial.suggest_int("n_trees", 100, 2000, step=100)
            max_depth = trial.suggest_int("max_depth", 10, 30)
            col_sample = trial.suggest_categorical(
                "col_sample", ["sqrt", "log2", 0.3, 0.5, 0.7]
            )

            model = self._build_model(n_trees, max_depth, col_sample)
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, probs)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best = study.best_params

        # Save for reproducibility
        PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_PATH, "w") as f:
            json.dump(best, f, indent=2)
        print(f"  Best params saved → {PARAMS_PATH}")
        print(f"  Best params: {best}")
        return best

    def fit(self,
            X_train: pd.DataFrame | np.ndarray,
            y_train: pd.Series | np.ndarray) -> None:
        """
        Train the Random Forest on one sliding window training set.

        If use_tuner=True, runs Optuna first to find optimal hyperparameters,
        then trains the final model with those parameters on the full
        training set. Otherwise uses the paper's fixed hyperparameters.

        RAF is not prone to overfitting (Breiman, 2001), so training is done
        on the full window without a held-out validation set — consistent with
        the paper's approach.

        Parameters
        ----------
        X_train : shape (n_samples, 31) — 31 lag features per observation
        y_train : shape (n_samples,) — binary labels in {0, 1}
        """
        X_np, y_np = self._to_numpy(X_train, y_train)

        # Resolve hyperparameters
        if self.use_tuner:
            self.best_params = self.tune(X_train, y_train)
            n_trees = self.best_params["n_trees"]
            max_depth = self.best_params["max_depth"]
            col_sample = self.best_params["col_sample"]
        else:
            n_trees = self.n_trees
            max_depth = self.max_depth
            col_sample = self.col_sample

        self.model = self._build_model(n_trees, max_depth, col_sample)

        print(f"  Training Random Forest — {n_trees} trees, "
              f"max_depth={max_depth}, "
              f"col_sample={col_sample} "
              f"(mRAF ≈ {int(31 ** 0.5)} features per split)")

        self.model.fit(X_np, y_np)

        print("  Training complete.")

    def predict_proba(self,
                      X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(outperform cross-sectional median) for each observation.

        Parameters
        ----------
        X : shape (n_samples, 31)

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in (0, 1)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        X_np = self._to_numpy(X)
        # RandomForestClassifier.predict_proba returns (n_samples, 2) — take class 1
        return self.model.predict_proba(X_np)[:, 1]

    def __repr__(self) -> str:
        return (f"RandomForestModel(n_trees={self.n_trees}, "
                f"max_depth={self.max_depth}, "
                f"col_sample={self.col_sample}, "
                f"use_tuner={self.use_tuner})")