"""
LSTM model for statistical arbitrage on the S&P 500.
Extends Krauss, Do & Huck (2016) by replacing the 31 hand-crafted lag
features with a raw return sequence fed into a stacked LSTM.

Hypothesis:
    Learned temporal feature extraction over a raw return sequence captures
    non-linear sequential dependencies that hand-crafted lag aggregation misses.
    An LSTM maintaining hidden state across the full sequence should therefore
    outperform the 31-feature DNN baseline.

Input:
    X of shape (n_samples, SEQUENCE_LENGTH) — each row is one stock-day,
    columns are raw returns in chronological order [r_{t-T+1}, ..., r_t].
    Reshaped internally to (n_samples, SEQUENCE_LENGTH, 1) before feeding
    into the LSTM.

Output:
    predict_proba() returns P(outperform cross-sectional median) in (0, 1)
    for each observation — identical interface to all baseline models.

Hyperparameter tuning:
    When use_tuner=True, fit() automatically runs an Optuna study on a
    validation split of the training data before training the final model.
    Best parameters are saved to configs/lstm_best_params.json for
    reproducibility. When use_tuner=False, fixed hyperparameters passed
    to the constructor are used directly.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

# Suppress Optuna's per-trial logging — we log our own summary instead
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Path to save best hyperparameters found by Optuna
root = Path(__file__).resolve().parents[2]
PARAMS_PATH = root / "configs"/"lstm_best_params.json"

# Default fixed hyperparameters — used when use_tuner=False
# These are sensible starting points based on the time-series literature
DEFAULTS = {
    "hidden_size": 64,
    "n_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 512,
    "epochs": 20,
}


class _LSTMNetwork(nn.Module):
    """
    Internal PyTorch module — not used directly outside this file.
    Instantiated and managed by LSTMModel.

    Architecture:
        Input  (batch, seq_len, 1)
            ↓
        Stacked LSTM (n_layers, hidden_size)
            ↓
        Last hidden state h_T (batch, hidden_size)
            ↓
        Dropout
            ↓
        Linear (hidden_size → 1)
            ↓
        Sigmoid → P(outperform)
    """

    def __init__(self,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            # nn.LSTM applies dropout between layers only — not after the
            # final layer. When n_layers=1 this is correctly ignored.
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, 1)

        Returns
        -------
        (batch,) — P(outperform) for each observation
        """
        # lstm_out: (batch, seq_len, hidden_size) — all timestep outputs
        # h_n: (n_layers, batch, hidden_size) — final hidden states
        _, (h_n, _) = self.lstm(x)

        # Take the hidden state of the last LSTM layer only
        h_last = h_n[-1]             # (batch, hidden_size)
        h_last = self.dropout(h_last)
        logit = self.head(h_last)   # (batch, 1)
        prob = self.sigmoid(logit)  # (batch, 1)
        return prob.squeeze(1)      # (batch,)


class LSTMModel(BaseModel):
    """
    LSTM wrapper implementing the BaseModel interface.

    Parameters
    ----------
    hidden_size  : number of LSTM hidden units per layer
    n_layers     : number of stacked LSTM layers
    dropout      : dropout probability applied between LSTM layers
                   and before the linear head
    lr           : Adam learning rate
    batch_size   : mini-batch size for training
    epochs       : number of full passes over the training set
    use_tuner    : if True, run Optuna before training to find optimal
                   hyperparameters. Overrides all other hyperparameter
                   arguments. Best params saved to configs/lstm_best_params.json
    n_trials     : number of Optuna trials (only used when use_tuner=True)
    val_fraction : fraction of training data held out for Optuna validation
                   (only used when use_tuner=True)
    device       : "cuda" if GPU available, otherwise "cpu"

    Examples
    --------
    # Fixed hyperparameters
    model = LSTMModel(hidden_size=128, n_layers=2, use_tuner=False)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # Optuna tuning — fit() handles everything automatically
    model = LSTMModel(use_tuner=True, n_trials=50)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)
    """

    def __init__(self,
                 hidden_size: int = DEFAULTS["hidden_size"],
                 n_layers: int = DEFAULTS["n_layers"],
                 dropout: float = DEFAULTS["dropout"],
                 lr: float = DEFAULTS["lr"],
                 batch_size: int = DEFAULTS["batch_size"],
                 epochs: int = DEFAULTS["epochs"],
                 use_tuner: bool = False,
                 n_trials: int = 50,
                 val_fraction: float = 0.2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_tuner = use_tuner
        self.n_trials = n_trials
        self.val_fraction = val_fraction
        self.device = torch.device(device)

        self.network = None   # built in fit() once hyperparams are finalized
        self.best_params = None   # populated by tune() if use_tuner=True

    def _to_tensor(self, X: pd.DataFrame | np.ndarray,
                   y: pd.Series | np.ndarray | None = None):
        """
        Convert numpy/pandas inputs to PyTorch tensors on the correct device.
        Reshapes X from (n, seq_len) to (n, seq_len, 1) for the LSTM.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)  # (n, seq_len, 1)
        X_t = X_t.to(self.device)

        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else np.array(y)
            y_t = torch.tensor(y_np, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _train_epoch(self,
                     network: _LSTMNetwork,
                     loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module) -> float:
        """Run one epoch of training, return mean loss."""
        network.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            probs = network(X_batch)
            loss = criterion(probs, y_batch)
            loss.backward()
            # Gradient clipping — important for LSTM stability
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    def _evaluate(self,
                  network: _LSTMNetwork,
                  loader: DataLoader,
                  criterion: nn.Module) -> float:
        """Evaluate on a dataloader, return mean loss."""
        network.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                probs = network(X_batch)
                loss = criterion(probs, y_batch)
                total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    def _build_loaders(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: int,
                       val_fraction: float | None = None):
        """
        Build train (and optionally validation) DataLoaders.
        If val_fraction is provided, splits chronologically — not randomly —
        to avoid lookahead bias in the validation set.
        """
        X_t, y_t = self._to_tensor(X, y)

        if val_fraction is not None:
            n_val    = int(len(X_t) * val_fraction)
            n_train  = len(X_t) - n_val

            X_tr, X_val = X_t[:n_train], X_t[n_train:]
            y_tr, y_val = y_t[:n_train], y_t[n_train:]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                                      batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(X_val, y_val),
                                      batch_size=batch_size, shuffle=False)
            return train_loader, val_loader

        train_loader = DataLoader(TensorDataset(X_t, y_t),
                                  batch_size=batch_size, shuffle=True)
        return train_loader, None

    # ── BaseModel interface ────────────────────────────────────────────────────

    def tune(self,
             X_train: pd.DataFrame | np.ndarray,
             y_train: pd.Series | np.ndarray) -> dict:
        """
        Run an Optuna study on a chronological validation split of the
        training data to find the best hyperparameters.

        Called automatically by fit() when use_tuner=True.
        Best parameters are saved to configs/lstm_best_params.json.

        Returns
        -------
        dict of best hyperparameters
        """
        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_np = y_train.values if isinstance(y_train, pd.Series)    else np.array(y_train)

        print(f"  Running Optuna study ({self.n_trials} trials)...")

        criterion = nn.BCELoss()

        def objective(trial: optuna.Trial) -> float:
            hidden_size  = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
            n_layers     = trial.suggest_int("n_layers", 1, 3)
            dropout      = trial.suggest_float("dropout", 0.2, 0.5)
            lr           = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024])

            network   = _LSTMNetwork(hidden_size, n_layers, dropout).to(self.device)
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)

            train_loader, val_loader = self._build_loaders(
                X_np, y_np, batch_size, val_fraction=self.val_fraction
            )

            # Train for a reduced number of epochs during tuning for speed
            tune_epochs = max(5, self.epochs // 4)
            for _ in range(tune_epochs):
                self._train_epoch(network, train_loader, optimizer, criterion)

            val_loss = self._evaluate(network, val_loader, criterion)
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best = study.best_params
        # epochs is not tuned by Optuna — carry over from constructor
        best["epochs"] = self.epochs

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
        Train the LSTM on one sliding window training set.

        If use_tuner=True, runs Optuna first to find optimal hyperparameters,
        then trains the final model with those parameters on the full
        training set. Otherwise uses the hyperparameters passed to __init__.

        Parameters
        ----------
        X_train : shape (n_samples, sequence_length)
        y_train : shape (n_samples,) binary labels in {0, 1}
        """
        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_np = y_train.values if isinstance(y_train, pd.Series)    else np.array(y_train)

        # Step 1: resolve hyperparameters
        if self.use_tuner:
            self.best_params = self.tune(X_train, y_train)
            hidden_size = self.best_params["hidden_size"]
            n_layers    = self.best_params["n_layers"]
            dropout     = self.best_params["dropout"]
            lr          = self.best_params["lr"]
            batch_size  = self.best_params["batch_size"]
            epochs      = self.best_params["epochs"]
        else:
            hidden_size = self.hidden_size
            n_layers    = self.n_layers
            dropout     = self.dropout
            lr          = self.lr
            batch_size  = self.batch_size
            epochs      = self.epochs

        # Step 2: build network and optimizer
        self.network  = _LSTMNetwork(hidden_size, n_layers, dropout).to(self.device)
        optimizer     = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion     = nn.BCELoss()

        # Step 3: train on the full training set (no val split for final training)
        train_loader, _ = self._build_loaders(X_np, y_np, batch_size)

        print(f"  Training LSTM — {epochs} epochs, "
              f"hidden={hidden_size}, layers={n_layers}, "
              f"dropout={dropout}, lr={lr}, batch={batch_size}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(
                self.network, train_loader, optimizer, criterion
            )
            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:>3}/{epochs}  loss={train_loss:.4f}")

        print(f"  Training complete.")

    def predict_proba(self,
                      X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(outperform cross-sectional median) for each observation.

        Parameters
        ----------
        X : shape (n_samples, sequence_length)

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in (0, 1)
        """
        if self.network is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        self.network.eval()
        X_t = self._to_tensor(X)

        with torch.no_grad():
            probs = self.network(X_t)

        return probs.cpu().numpy()

    def __repr__(self) -> str:
        return (f"LSTMModel(hidden_size={self.hidden_size}, "
                f"n_layers={self.n_layers}, "
                f"dropout={self.dropout}, "
                f"use_tuner={self.use_tuner})")