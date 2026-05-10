"""
Baseline DNN model replicating Krauss, Do & Huck (2016).

This file implements the exact DNN architecture described in Section 4.3 of
the paper: a fully-connected network with architecture 31-31-10-5-2, trained
on 31 hand-crafted lagged return features to predict which S&P 500 stocks
will outperform the cross-sectional median the following day.

Architecture (from paper, Section 4.3):
    Input (31 features)
        ↓
    Dense(31) + Maxout + Dropout(0.5)
        ↓
    Dense(10) + Maxout + Dropout(0.5)
        ↓
    Dense(5)  + Maxout + Dropout(0.5)
        ↓
    Dense(2)  + Softmax → P(outperform)

Key paper details reproduced here:
    - Maxout activation units (Goodfellow et al., 2013)
    - Dropout(0.5) on all hidden layers, Dropout(0.1) on input layer
    - L1 regularisation (lambda=1e-5) on all weight matrices
    - Adam optimiser (Kingma & Ba, 2015)
    - 400 training epochs
    - Seed = 5 for reproducibility

Checkpointing:
    fit() accepts checkpoint_dir and checkpoint_every arguments, matching
    the interface of LSTMModel. On each epoch, latest.pt is written so
    that training can be inspected at any point. Every checkpoint_every
    epochs an additional epoch_NNN.pt snapshot is written. At the end of
    training, final.pt is written. All files contain the raw network
    state_dict so they can be reloaded with:

        model = DNNModel(...)
        model.network = _DNNNetwork(...)
        model.network.load_state_dict(torch.load("epoch_005.pt"))

Input:
    X of shape (n_samples, 31) — 31 lagged return features per observation:
    R(1)–R(20) at daily resolution, then R(40), R(60), …, R(240) at monthly.

Output:
    predict_proba() returns P(outperform cross-sectional median) in (0, 1)
    for each observation — identical interface to all other models.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

# Default hyperparameters matching the paper exactly (Section 4.3)
DEFAULTS = {
    "architecture": [31, 31, 10, 5, 2],  # layer sizes including input & output
    "dropout_hidden": 0.5,               # dropout on all hidden layers
    "dropout_input": 0.1,                # dropout on input layer
    "l1_lambda": 1e-5,                   # L1 regularization strength
    "epochs": 400,
    "lr": 1e-3,                          # Adam default — not specified in paper
    "batch_size": 512,
    "seed": 1,               # paper explicitly sets seed to 1
}


class _MaxoutUnit(nn.Module):
    """
    Maxout activation unit (Goodfellow et al., 2013).

    Maxout computes k linear projections and returns the element-wise maximum.
    With k=2, this approximates any convex activation function (ReLU, tanh, etc.)
    and is particularly well-suited to dropout — the paper's primary regulariser.

    Parameters
    ----------
    in_features  : number of input features
    out_features : number of maxout units (output size)
    k            : number of linear pieces per unit (paper uses k=2)
    """

    def __init__(self, in_features: int, out_features: int, k: int = 2):
        super().__init__()
        # k parallel linear layers — maxout selects the largest activation
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(k)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, in_features)

        Returns
        -------
        (batch, out_features) — element-wise max over k linear projections
        """
        # Stack k projections along a new dimension, then take element-wise max
        activations = torch.stack([linear(x) for linear in self.linears], dim=-1)
        return activations.max(dim=-1).values  # (batch, out_features)


class _DNNNetwork(nn.Module):
    """
    Internal PyTorch module. Not used directly outside this file.
    Instantiated and managed by DNNModel.

    Implements the 31-31-10-5-2 architecture from Krauss et al. (2016)
    with Maxout activations and dropout regularisation.

    Architecture:
        Input (batch, 31)
            ↓
        Dropout(dropout_input)
            ↓
        MaxoutUnit(31 → 31) + Dropout(dropout_hidden)
            ↓
        MaxoutUnit(31 → 10) + Dropout(dropout_hidden)
            ↓
        MaxoutUnit(10 → 5)  + Dropout(dropout_hidden)
            ↓
        Linear(5 → 2) + Softmax
            ↓
        P(outperform) — the second softmax output (class = 1)
    """

    def __init__(self,
                 architecture: list[int],
                 dropout_hidden: float,
                 dropout_input: float):
        super().__init__()

        # Input dropout — applied directly to the feature vector
        self.input_dropout = nn.Dropout(dropout_input)

        # Hidden maxout layers — all layers except the first (input) and
        # last two (penultimate hidden and output) entries in architecture
        # architecture = [31, 31, 10, 5, 2]
        # hidden pairs: (31→31), (31→10), (10→5)
        self.hidden_layers = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()

        for i in range(len(architecture) - 2):
            in_size = architecture[i]
            out_size = architecture[i + 1]
            self.hidden_layers.append(_MaxoutUnit(in_size, out_size))
            self.hidden_dropouts.append(nn.Dropout(dropout_hidden))

        # Output layer — Linear + Softmax, no dropout
        # From the 5-unit penultimate layer to 2 output classes
        self.output_layer = nn.Linear(architecture[-2], architecture[-1])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_features)

        Returns
        -------
        (batch,) — P(outperform) = softmax class 1 probability
        """
        out = self.input_dropout(x)

        for layer, dropout in zip(self.hidden_layers, self.hidden_dropouts):
            out = layer(out)
            out = dropout(out)

        logits = self.output_layer(out)    # (batch, 2)
        probs = self.softmax(logits)       # (batch, 2)
        return probs[:, 1]                 # (batch,) — P(outperform)


class DNNModel(BaseModel):
    """
    Baseline DNN wrapper implementing the BaseModel interface.

    Replicates the exact DNN from Krauss, Do & Huck (2016) — a fully-connected
    network with Maxout activations and aggressive dropout regularisation,
    trained on 31 hand-crafted lag features.

    Parameters
    ----------
    architecture   : list of layer sizes [input, hidden..., output].
                     Default [31, 31, 10, 5, 2] matches the paper exactly.
    dropout_hidden : dropout probability on all hidden layers (paper: 0.5)
    dropout_input  : dropout probability on the input layer (paper: 0.1)
    l1_lambda      : L1 regularisation coefficient (paper: 1e-5)
    epochs         : number of full passes over the training set (paper: 400)
    lr             : Adam learning rate (paper does not specify; 1e-3 default)
    batch_size     : mini-batch size for training
    seed           : random seed for reproducibility (paper: 1)
    device         : "cuda" if GPU available, otherwise "cpu"

    Examples
    --------
    # Paper-exact configuration
    model = DNNModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)

    # With checkpointing — same interface as LSTMModel
    model = DNNModel(epochs=20)
    model.fit(X_train, y_train, checkpoint_dir=Path("checkpoints/dnn/batch_00"),
              checkpoint_every=5)
    probs = model.predict_proba(X_trade)

    # Custom architecture
    model = DNNModel(architecture=[31, 62, 10, 5, 2], dropout_hidden=0.3)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_trade)
    """

    def __init__(self,
                 architecture: list[int] = DEFAULTS["architecture"],
                 dropout_hidden: float = DEFAULTS["dropout_hidden"],
                 dropout_input: float = DEFAULTS["dropout_input"],
                 l1_lambda: float = DEFAULTS["l1_lambda"],
                 epochs: int = DEFAULTS["epochs"],
                 lr: float = DEFAULTS["lr"],
                 batch_size: int = DEFAULTS["batch_size"],
                 seed: int = DEFAULTS["seed"],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.architecture = architecture
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)

        self.network = None  # built in fit()

    def _set_seed(self) -> None:
        """Fix seeds for reproducibility as per the paper (seed=1)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _to_tensor(self,
                   X: pd.DataFrame | np.ndarray,
                   y: pd.Series | np.ndarray | None = None):
        """Convert numpy/pandas inputs to PyTorch tensors on the correct device."""
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)

        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else np.array(y)
            y_t = torch.tensor(y_np, dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def _l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 regularisation penalty over all weight matrices.

        The paper applies L1 regularisation (lambda=1e-5) to prevent
        overfitting on the 31-feature input, which is relatively low-dimensional
        compared to the training set size.
        """
        l1 = torch.tensor(0.0, device=self.device)
        for name, param in self.network.named_parameters():
            if "weight" in name:
                l1 = l1 + param.abs().sum()
        return self.l1_lambda * l1

    def _train_epoch(self,
                     loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module) -> float:
        """Run one training epoch, return mean loss (BCE + L1 penalty)."""
        self.network.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            probs = self.network(X_batch)
            loss = criterion(probs, y_batch) + self._l1_penalty()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        return total_loss / len(loader.dataset)

    def _save_checkpoint(self,
                         checkpoint_dir: Path,
                         epoch: int,
                         checkpoint_every: int,
                         final: bool = False) -> None:
        """
        Write checkpoint files to checkpoint_dir.

        Always writes latest.pt so training state is never more than one
        epoch stale. Writes epoch_NNN.pt every checkpoint_every epochs.
        Writes final.pt when final=True (called once after the last epoch).

        Parameters
        ----------
        checkpoint_dir   : directory that must already exist
        epoch            : current epoch number (1-indexed)
        checkpoint_every : period for named epoch snapshots
        final            : if True, write final.pt instead of epoch_NNN.pt
        """
        torch.save(self.network.state_dict(), checkpoint_dir / "latest.pt")

        if final:
            torch.save(self.network.state_dict(), checkpoint_dir / "final.pt")
            print(f"    ✓ Final checkpoint saved.")
        elif epoch % checkpoint_every == 0:
            name = f"epoch_{epoch:03d}.pt"
            torch.save(self.network.state_dict(), checkpoint_dir / name)
            print(f"    ✓ Checkpoint → {name}")

    def fit(self,
            X_train: pd.DataFrame | np.ndarray,
            y_train: pd.Series | np.ndarray,
            checkpoint_dir: Path | None = None,
            checkpoint_every: int = 5) -> None:
        """
        Train the DNN on one sliding window training set.

        Uses the exact paper configuration: 400 epochs, Maxout activations,
        dropout(0.5) on hidden layers, dropout(0.1) on input, L1(1e-5).

        Parameters
        ----------
        X_train          : shape (n_samples, 31) — 31 lag features per observation
        y_train          : shape (n_samples,) — binary labels in {0, 1}
        checkpoint_dir   : directory for checkpoint files. If None, no
                           checkpoints are written. Mirrors LSTMModel interface.
        checkpoint_every : write epoch_NNN.pt every N epochs (default 5).
                           latest.pt is always written each epoch.
        """
        self._set_seed()

        X_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)

        # Build network
        self.network = _DNNNetwork(
            self.architecture,
            self.dropout_hidden,
            self.dropout_input,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_t, y_t = self._to_tensor(X_np, y_np)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        print(f"  Training DNN — {self.epochs} epochs, "
              f"arch={self.architecture}, "
              f"dropout_hidden={self.dropout_hidden}, "
              f"dropout_input={self.dropout_input}, "
              f"l1={self.l1_lambda}")

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints → {checkpoint_dir} "
                  f"(every {checkpoint_every} epochs + latest.pt)")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(loader, optimizer, criterion)

            if epoch % 50 == 0 or epoch == 1:
                print(f"    Epoch {epoch:>3}/{self.epochs}  loss={train_loss:.4f}")

            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch, checkpoint_every)

        if checkpoint_dir is not None:
            self._save_checkpoint(checkpoint_dir, self.epochs, checkpoint_every,
                                  final=True)

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
        if self.network is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Disable dropout at inference — eval() switches all Dropout layers off
        self.network.eval()
        X_t = self._to_tensor(X)

        with torch.no_grad():
            probs = self.network(X_t)

        return probs.cpu().numpy()

    def __repr__(self) -> str:
        return (f"DNNModel(architecture={self.architecture}, "
                f"dropout_hidden={self.dropout_hidden}, "
                f"dropout_input={self.dropout_input}, "
                f"l1_lambda={self.l1_lambda})")