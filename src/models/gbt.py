"""
Gradient-Boosted Trees (GBT) model for statistical arbitrage on the S&P 500.
Replicates the GBT baseline from Krauss, Do & Huck (2016), Section 4.4.

The paper trains XGBoost with 100 trees, max depth 3, learning rate 0.1,
and column subsampling of ~50% (mGBT ≈ 15 out of 31 features) on the same
31 hand-crafted lagged return features used by the DNN.

Key paper details reproduced here:
    - XGBoost with gradient boosting on binary cross-entropy loss
    - 100 boosting rounds (n_trees)
    - max_depth = 3 — shallow trees prevent overfitting on tabular financial data
    - learning_rate = 0.1
    - colsample_bytree = 0.5 — mGBT ≈ 15 features per tree (Section 4.4)
    - seed = 1 for reproducibility (paper explicitly states seed fixed to 1)

Checkpointing:
    fit() accepts checkpoint_dir and checkpoint_every arguments, matching
    the interface of LSTMModel and DNNModel. Because XGBoost is not a
    PyTorch model, checkpoints are written as joblib files (.joblib) rather
    than state_dicts. Each checkpoint contains the fully fitted XGBClassifier
    at the specified boosting round, trained via the num_boost_round API.

    Layout written to checkpoint_dir per batch:
        latest.joblib          — overwritten every checkpoint_every rounds
        round_NNN.joblib       — snapshot at round NNN
        final.joblib           — written once after all rounds complete
        metadata.json          — hyperparams + timestamp for reproducibility

    Reloading a checkpoint:
        import joblib
        model = GBTModel()
        model.model = joblib.load("checkpoints/gbt/batch_00/final.joblib")
        probs = model.predict_proba(X_trade)

Hyperparameter tuning:
    When use_tuner=True, fit() automatically runs an Optuna study on a
    chronological validation split of the training data before training the
    final model. Best parameters are saved to configs/gbt_best_params.json
    for reproducibility. When use_tuner=False, fixed hyperparameters from
    the paper (or passed to the constructor) are used directly.

Input:
    X of shape (n_samples, 31) — 31 lagged return features per observation.

Output:
    predict_proba() returns P(outperform cross-sectional median) in (0, 1)
    for each observation — identical interface to all other models.
"""
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from datetime import datetime
from pathlib import Path

from src.models.base import BaseModel

# Suppress Optuna's per-trial logging — we log our own summary instead
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Path to save best hyperparameters found by Optuna
root = Path(__file__).resolve().parents[2]
PARAMS_PATH = root/"config"/"gbt_best_params.json"

# Default hyperparameters matching the paper exactly (Section 4.4)
DEFAULTS = {
    "n_trees": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "col_sample": 0.5,   # colsample_bytree — mGBT ≈ 15 out of 31 features
    "seed": 1,   # paper explicitly sets seed to 1
}


class GBTModel(BaseModel):
    """
    Gradient-Boosted Trees wrapper implementing the BaseModel interface.

    Replicates the XGBoost baseline from Krauss, Do & Huck (2016) trained on
    31 hand-crafted lag features. Unlike the DNN, GBT is a non-parametric
    ensemble method that handles feature interactions and non-linearities
    natively without architectural choices.

    Parameters
    ----------
    n_trees       : number of boosting rounds / trees (paper: 100)
    max_depth     : maximum tree depth (paper: 3)
    learning_rate : shrinkage applied to each tree's contribution (paper: 0.1)
    col_sample    : fraction of features sampled per tree (paper: ~0.5)
    seed          : random seed for reproducibility (paper: 1)
    use_tuner     : if True, run Optuna before training to find optimal
                    hyperparameters. Overrides all other hyperparameter
                    arguments. Best params saved to configs/gbt_best_params.json
    n_trials      : number of Optuna trials (only used when use_tuner=True)
    val_fraction  : fraction of training data held out for Optuna validation
                    (only used when use_tuner=True). Split is chronological
                    to avoid lookahead bias.

    Examples
    --------
    # Paper-exact configuration
    model = GBTModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # Optuna tuning — fit() handles everything automatically
    model = GBTModel(use_tuner=True, n_trials=50)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # With checkpointing — same interface as LSTMModel / DNNModel
    model = GBTModel(n_trees=100)
    model.fit(X_train, y_train,
              checkpoint_dir=Path("checkpoints/gbt/batch_00"),
              checkpoint_every=5)
    probs = model.predict_proba(X_trade)
    """

    def __init__(self,
                 n_trees: int = DEFAULTS["n_trees"],
                 max_depth: int = DEFAULTS["max_depth"],
                 learning_rate: float = DEFAULTS["learning_rate"],
                 col_sample: float = DEFAULTS["col_sample"],
                 seed: int = DEFAULTS["seed"],
                 use_tuner: bool = False,
                 n_trials: int = 50,
                 val_fraction: float = 0.2):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.col_sample = col_sample
        self.seed = seed
        self.use_tuner = use_tuner
        self.n_trials = n_trials
        self.val_fraction = val_fraction

        self.model = None        # XGBClassifier built in fit()
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
                     learning_rate: float,
                     col_sample: float) -> xgb.XGBClassifier:
        """
        Instantiate an XGBClassifier with the given hyperparameters.

        Uses binary:logistic objective to match the paper's binary
        cross-sectional outperformance framing.
        """
        return xgb.XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=col_sample,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self.seed,
            verbosity=0,
        )

    def _save_checkpoint(self,
                         checkpoint_dir: Path,
                         round_num: int,
                         checkpoint_every: int,
                         params: dict,
                         final: bool = False) -> None:
        """
        Persist the current model to checkpoint_dir.

        XGBoost models are serialised with joblib rather than torch.save,
        but the naming convention mirrors LSTMModel / DNNModel exactly:
            latest.joblib      — always written (most recent state)
            round_NNN.joblib   — written every checkpoint_every rounds
            final.joblib       — written once after the last round

        A metadata.json sidecar is also kept in the directory so that
        checkpoints are self-describing without needing to unpickle them.

        Parameters
        ----------
        checkpoint_dir   : directory that must already exist
        round_num        : current boosting round (1-indexed)
        checkpoint_every : period for named round snapshots
        params           : hyperparameter dict — written to metadata.json
        final            : if True, write final.joblib
        """
        joblib.dump(self.model, checkpoint_dir / "latest.joblib")

        if final:
            joblib.dump(self.model, checkpoint_dir / "final.joblib")
            print(f"    ✓ Final checkpoint saved.")
        elif round_num % checkpoint_every == 0:
            name = f"round_{round_num:03d}.joblib"
            joblib.dump(self.model, checkpoint_dir / name)
            print(f"    ✓ Checkpoint → {name}")

        # Keep metadata current so the directory is self-describing
        metadata = {
            "model": "GBTModel",
            "round": round_num,
            "params": params,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def tune(self,
             X_train: pd.DataFrame | np.ndarray,
             y_train: pd.Series | np.ndarray) -> dict:
        """
        Run an Optuna study on a chronological validation split of the
        training data to find the best hyperparameters.

        Called automatically by fit() when use_tuner=True.
        Best parameters are saved to configs/gbt_best_params.json.

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
            n_trees = trial.suggest_int("n_trees", 50, 500)
            max_depth = trial.suggest_int("max_depth", 2, 6)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            col_sample = trial.suggest_float("col_sample", 0.3, 0.8)

            model = self._build_model(n_trees, max_depth, learning_rate, col_sample)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Use log-loss on the validation set as the Optuna objective
            from sklearn.metrics import log_loss
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
            y_train: pd.Series | np.ndarray,
            checkpoint_dir: Path | None = None,
            checkpoint_every: int = 5) -> None:
        """
        Train the GBT on one sliding window training set.

        Because XGBoost does not expose a per-round callback that lets us
        snapshot a partially-trained model, checkpointing is achieved by
        training in incremental blocks of checkpoint_every rounds using
        xgb.train with xgb_model warm-start. This is equivalent to a
        single training run but allows a checkpoint to be written after
        every block.

        If use_tuner=True, runs Optuna first to find optimal hyperparameters,
        then trains the final model with those parameters on the full
        training set. Otherwise uses the paper's fixed hyperparameters.

        Parameters
        ----------
        X_train          : shape (n_samples, 31) — 31 lag features per observation
        y_train          : shape (n_samples,) — binary labels in {0, 1}
        checkpoint_dir   : directory for checkpoint files. If None, no
                           checkpoints are written. Mirrors LSTMModel interface.
        checkpoint_every : save a named round_NNN.joblib every N boosting
                           rounds (default 5). latest.joblib written every block.
        """
        X_np, y_np = self._to_numpy(X_train, y_train)

        # Resolve hyperparameters
        if self.use_tuner:
            self.best_params = self.tune(X_train, y_train)
            n_trees = self.best_params["n_trees"]
            max_depth = self.best_params["max_depth"]
            learning_rate = self.best_params["learning_rate"]
            col_sample = self.best_params["col_sample"]
        else:
            n_trees = self.n_trees
            max_depth = self.max_depth
            learning_rate = self.learning_rate
            col_sample = self.col_sample

        params = dict(n_trees=n_trees, max_depth=max_depth,
                      learning_rate=learning_rate, col_sample=col_sample)

        print(f"  Training GBT — {n_trees} trees, "
              f"max_depth={max_depth}, "
              f"lr={learning_rate}, "
              f"col_sample={col_sample}")

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints → {checkpoint_dir} "
                  f"(every {checkpoint_every} rounds + latest.joblib)")

        if checkpoint_dir is None:
            # No checkpointing — train in a single call, simplest path
            self.model = self._build_model(n_trees, max_depth, learning_rate, col_sample)
            self.model.fit(X_np, y_np)
        else:
            # Incremental training in blocks of checkpoint_every rounds so
            # we can checkpoint after each block without any training overhead.
            # XGBClassifier supports warm-starting via xgb_model parameter.
            rounds_done = 0
            booster = None  # the underlying xgb.Booster for warm-start

            while rounds_done < n_trees:
                block = min(checkpoint_every, n_trees - rounds_done)

                clf = self._build_model(block, max_depth, learning_rate, col_sample)

                if booster is None:
                    clf.fit(X_np, y_np)
                else:
                    # Pass existing booster as warm-start so XGBoost continues
                    # adding trees rather than restarting from scratch
                    clf.fit(X_np, y_np, xgb_model=booster)

                booster = clf.get_booster()
                rounds_done += block
                self.model = clf

                self._save_checkpoint(
                    checkpoint_dir, rounds_done, checkpoint_every, params,
                    final=(rounds_done >= n_trees),
                )

                print(f"    Round {rounds_done:>4}/{n_trees}  "
                      f"(block of {block} trees complete)")

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
        # XGBClassifier.predict_proba returns (n_samples, 2) — take class 1
        return self.model.predict_proba(X_np)[:, 1]

    def __repr__(self) -> str:
        return (f"GBTModel(n_trees={self.n_trees}, "
                f"max_depth={self.max_depth}, "
                f"learning_rate={self.learning_rate}, "
                f"col_sample={self.col_sample}, "
                f"use_tuner={self.use_tuner})")