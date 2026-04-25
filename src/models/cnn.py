"""
1D CNN model for statistical arbitrage on the S&P 500.
Extends Krauss, Do & Huck (2016) by replacing the 31 hand-crafted lag
features with a raw return sequence fed into stacked 1D convolutional layers.

Hypothesis:
    The paper's own variable importance analysis shows that R(1)-R(5) — the
    past 5 trading days — carry the highest predictive power across all models.
    If local short-window patterns are the primary signal, then a model whose
    inductive bias is explicitly designed to detect such patterns via learned
    convolutional filters should be well-matched to this problem and outperform
    a flat-feature DNN.

    1D convolution kernels act as learned, adaptive sliding windows — computing
    the dot product between a transformation matrix and a local window of the
    return sequence. Unlike a fixed moving average, kernel weights are optimised
    to detect patterns that are actually predictive of next-day relative
    performance.

Input:
    X of shape (n_samples, SEQUENCE_LENGTH) — each row is one stock-day,
    columns are raw returns in chronological order [r_{t-T+1}, ..., r_t].
    Reshaped internally to (n_samples, 1, SEQUENCE_LENGTH) for PyTorch's
    Conv1d which expects (batch, channels, length).

    Note: this is different from the LSTM which expects (batch, seq_len, 1).
    Conv1d treats the sequence as a signal with one input channel.

Output:
    predict_proba() returns P(outperform cross-sectional median) in (0, 1)
    for each observation — identical interface to all baseline models.

Hyperparameter tuning:
    When use_tuner=True, fit() automatically runs an Optuna study on a
    chronological validation split of the training data before training the
    final model. Best parameters are saved to configs/cnn_best_params.json
    for reproducibility. When use_tuner=False, fixed hyperparameters passed
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
PARAMS_PATH = root/"configs"/"cnn_best_params.json"

# Default fixed hyperparameters — used when use_tuner=False
DEFAULTS = {"num_filters": 64,
            "kernel_size": 5,
            "num_layers": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 512,
            "epochs": 20,}


class _CNNNetwork(nn.Module):
    """
    Internal PyTorch module. Not used directly outside this file.
    Instantiated and managed by CNNModel.

    Architecture:
        Input  (batch, 1, seq_len)
            ↓
        num_layers × [Conv1d + ReLU + MaxPool1d]
            ↓
        Global Average Pooling (batch, num_filters)
            ↓
        Dropout
            ↓
        Linear (num_filters → 1)
            ↓
        Sigmoid → P(outperform)

    Notes on Conv1d input format:
        PyTorch's Conv1d expects (batch, in_channels, length).
        We treat the return sequence as a 1-channel signal — analogous
        to a single-channel audio waveform. Each convolutional filter
        learns to detect a different local pattern in the return sequence.
    """

    def __init__(self,
                 seq_len: int,
                 num_filters: int,
                 kernel_size: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()

        self.seq_len = seq_len

        # Build convolutional layers dynamically based on num_layers
        # Each layer: Conv1d → ReLU → MaxPool1d(2)
        # MaxPool1d(2) halves the sequence length after each layer —
        # we use padding='same' on Conv1d to preserve length before pooling
        layers = []
        in_channels = 1  # single input channel — the raw return sequence

        for i in range(num_layers):
            layers += [
                nn.Conv1d(in_channels=in_channels,
                          out_channels=num_filters,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2,),  # preserve sequence length
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),  # halve sequence length
            ]
            in_channels = num_filters  # subsequent layers take num_filters as input

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling — averages across the entire remaining
        # sequence length after all conv layers, producing one value per filter.
        # AdaptiveAvgPool1d(1) handles any sequence length automatically.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (batch, 1, seq_len) — single channel return sequence

        Returns
        -------
        (batch,) — P(outperform) for each observation
        """
        # Apply stacked conv layers
        # out shape after each layer: (batch, num_filters, seq_len / 2^layer)
        out = self.conv_layers(x)           # (batch, num_filters, reduced_len)

        # Global Average Pooling
        # AdaptiveAvgPool1d(1) averages across the entire remaining length
        out = self.gap(out)                 # (batch, num_filters, 1)
        out = out.squeeze(-1)              # (batch, num_filters)

        # Dropout + linear head + sigmoid
        out = self.dropout(out)
        logit = self.head(out)             # (batch, 1)
        prob = self.sigmoid(logit)        # (batch, 1)
        return prob.squeeze(1)             # (batch,)


class CNNModel(BaseModel):
    """
    1D CNN wrapper implementing the BaseModel interface.

    Parameters
    ----------
    num_filters  : number of convolutional filters per layer
    kernel_size  : width of the local pattern detection window.
                   smaller values (3, 5) focus on very short-term patterns,
                   larger values (10, 20) capture medium-term patterns
    num_layers   : number of stacked convolutional layers
    dropout      : dropout probability applied before the linear head
    lr           : Adam learning rate
    batch_size   : mini-batch size for training
    epochs       : number of full passes over the training set
    use_tuner    : if True, run Optuna before training to find optimal
                   hyperparameters. Overrides all other hyperparameter
                   arguments. Best params saved to configs/cnn_best_params.json
    n_trials     : number of Optuna trials (only used when use_tuner=True)
    val_fraction : fraction of training data held out for Optuna validation
                   (only used when use_tuner=True)
    device       : "cuda" if GPU available, otherwise "cpu"

    Examples
    --------
    # Fixed hyperparameters
    model = CNNModel(num_filters=64, kernel_size=5, use_tuner=False)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # Optuna tuning — fit() handles everything automatically
    model = CNNModel(use_tuner=True, n_trials=50)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)
    """

    def __init__(self,
                 num_filters: int = DEFAULTS["num_filters"],
                 kernel_size: int = DEFAULTS["kernel_size"],
                 num_layers: int = DEFAULTS["num_layers"],
                 dropout: float = DEFAULTS["dropout"],
                 lr: float = DEFAULTS["lr"],
                 batch_size: int = DEFAULTS["batch_size"],
                 epochs: int = DEFAULTS["epochs"],
                 use_tuner: bool = False,
                 n_trials: int = 50,
                 val_fraction: float = 0.2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
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
        self.seq_len = None   # inferred from X_train in fit()

    def _to_tensor(self,
                   X: pd.DataFrame | np.ndarray,
                   y: pd.Series | np.ndarray | None = None):
        """
        Convert numpy/pandas inputs to PyTorch tensors on the correct device.
        Reshapes X from (n, seq_len) to (n, 1, seq_len) for Conv1d.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)  # (n, 1, seq_len)
        X_t = X_t.to(self.device)

        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else np.array(y)
            y_t = torch.tensor(y_np, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _train_epoch(self,
                     network: _CNNNetwork,
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
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss/len(loader.dataset)

    def _evaluate(self,
                  network: _CNNNetwork,
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
        return total_loss/len(loader.dataset)

    def _build_loaders(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       batch_size: int,
                       val_fraction: float | None = None):
        """
        Build train (and optionally validation) DataLoaders.
        Validation split is chronological to avoid lookahead bias.
        """
        X_t, y_t = self._to_tensor(X, y)

        if val_fraction is not None:
            n_val = int(len(X_t) * val_fraction)
            n_train = len(X_t) - n_val

            X_tr, X_val = X_t[:n_train], X_t[n_train:]
            y_tr, y_val = y_t[:n_train], y_t[n_train:]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                                      batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val),
                                      batch_size=batch_size, shuffle=False)
            return train_loader, val_loader

        train_loader = DataLoader(TensorDataset(X_t, y_t),
                                  batch_size=batch_size, shuffle=True)
        return train_loader, None

    def tune(self,
             X_train: pd.DataFrame | np.ndarray,
             y_train: pd.Series | np.ndarray) -> dict:
        """
        Run an Optuna study on a chronological validation split of the
        training data to find the best hyperparameters.

        Called automatically by fit() when use_tuner=True.
        Best parameters are saved to configs/cnn_best_params.json.

        Returns
        -------
        dict of best hyperparameters
        """
        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)

        print(f"  Running Optuna study ({self.n_trials} trials)...")

        criterion = nn.BCELoss()
        seq_len = X_np.shape[1]

        def objective(trial: optuna.Trial) -> float:
            num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
            kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 10, 20])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.2, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

            network = _CNNNetwork(seq_len, num_filters, kernel_size,
                                    num_layers, dropout).to(self.device)
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)

            train_loader, val_loader = self._build_loaders(X_np,
                                                           y_np,
                                                           batch_size,
                                                           val_fraction=self.val_fraction)

            # Train for reduced epochs during tuning for speed
            tune_epochs = max(5, self.epochs//4)
            for _ in range(tune_epochs):
                self._train_epoch(network, train_loader, optimizer, criterion)

            val_loss = self._evaluate(network, val_loader, criterion)
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best = study.best_params
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
        Train the CNN on one sliding window training set.

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

        # Infer sequence length from data
        self.seq_len = X_np.shape[1]

        # Step 1: resolve hyperparameters
        if self.use_tuner:
            self.best_params = self.tune(X_train, y_train)
            num_filters = self.best_params["num_filters"]
            kernel_size = self.best_params["kernel_size"]
            num_layers = self.best_params["num_layers"]
            dropout = self.best_params["dropout"]
            lr = self.best_params["lr"]
            batch_size = self.best_params["batch_size"]
            epochs = self.best_params["epochs"]
        else:
            num_filters = self.num_filters
            kernel_size = self.kernel_size
            num_layers = self.num_layers
            dropout = self.dropout
            lr = self.lr
            batch_size = self.batch_size
            epochs = self.epochs

        # Step 2: build network and optimizer
        self.network = _CNNNetwork(self.seq_len, num_filters, kernel_size,
                                   num_layers, dropout).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Step 3: train on the full training set
        train_loader, _ = self._build_loaders(X_np, y_np, batch_size)

        print(f"  Training CNN — {epochs} epochs, "
              f"filters={num_filters}, kernel={kernel_size}, "
              f"layers={num_layers}, dropout={dropout}, "
              f"lr={lr}, batch={batch_size}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(self.network,
                                           train_loader,
                                           optimizer,
                                           criterion)
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
        return (f"CNNModel(num_filters={self.num_filters}, "
                f"kernel_size={self.kernel_size}, "
                f"num_layers={self.num_layers}, "
                f"use_tuner={self.use_tuner})")