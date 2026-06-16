"""
Train a quantile LSTM on 90 days of CoinGecko hourly data.
Saves model.pt to the repo root.
Run: python pipeline/train.py
"""

import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from pipeline.fetch import fetch_coin_list, fetch_hourly
from pipeline.features import N_FEATURES, SEQ_LEN, HORIZON, build_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
TOP_N_COINS   = 50
TRAIN_DAYS    = 90
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
BATCH_SIZE    = 256
MAX_EPOCHS    = 150
LR            = 1e-3
ES_PATIENCE   = 15      # early stopping patience (epochs)
CV_FOLDS      = 5
CV_TEST_DAYS  = 14
MODEL_PATH    = Path("model.pt")
BEST_WEIGHTS  = Path("best_weights.pt")


# ── Model ──────────────────────────────────────────────────────────────────────
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_q10 = nn.Linear(64, horizon)
        self.head_med = nn.Linear(64, horizon)
        self.head_q90 = nn.Linear(64, horizon)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]           # last layer hidden state: (batch, hidden)
        h = self.fc(h)
        return torch.stack([
            self.head_q10(h),
            self.head_med(h),
            self.head_q90(h),
        ], dim=1)            # (batch, 3, horizon)


# ── Loss ───────────────────────────────────────────────────────────────────────
def quantile_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    err = target - pred
    return torch.mean(torch.max(alpha * err, (alpha - 1) * err))


def total_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    q10, med, q90 = out[:, 0, :], out[:, 1, :], out[:, 2, :]
    loss = (
        quantile_loss(q10, y, 0.1)
        + quantile_loss(med, y, 0.5)
        + quantile_loss(q90, y, 0.9)
    )
    # Soft monotonicity penalty: q10 ≤ med ≤ q90
    mono = (
        torch.mean(torch.relu(q10 - med))
        + torch.mean(torch.relu(med - q90))
    )
    return loss + 0.1 * mono


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    logger.info(f"Using device: {d}")
    return d


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = total_loss(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += total_loss(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)


def median_mae(model, loader, device) -> float:
    """MAE of the median head's first-step prediction vs actual."""
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device))[:, 1, 0].cpu().numpy()
            actual = yb[:, 0].numpy()
            errs.extend(np.abs(pred - actual).tolist())
    return float(np.mean(errs))


def fit(model, X, y, device, val_split=0.1) -> float:
    """Train model with early stopping. Returns best val loss."""
    n_val = max(1, int(len(X) * val_split))
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE,
    )

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    best_val = float("inf")
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss  = train_epoch(model, tr_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_WEIGHTS)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= ES_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    BEST_WEIGHTS.unlink(missing_ok=True)
    return best_val


# ── Walk-forward CV ────────────────────────────────────────────────────────────
def walk_forward_cv(X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    """5-fold walk-forward CV. Returns average median MAE."""
    n = len(X)
    test_size  = CV_TEST_DAYS * 24
    fold_step  = test_size // 2
    fold_start = n - CV_FOLDS * fold_step - test_size

    if fold_start <= 0:
        logger.warning("Not enough data for walk-forward CV. Skipping.")
        return float("nan")

    maes = []
    for fold in range(CV_FOLDS):
        val_end   = fold_start + (fold + 1) * fold_step + test_size
        val_start = val_end - test_size
        X_tr, y_tr   = X[:val_start], y[:val_start]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        model = QuantileLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON).to(device)
        fit(model, X_tr, y_tr, device)

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=BATCH_SIZE,
        )
        mae = median_mae(model, val_loader, device)
        maes.append(mae)
        logger.info(f"CV Fold {fold+1}/{CV_FOLDS} | median MAE: {mae:.4f} index pts")

    cv_mae = float(np.mean(maes))
    logger.info(f"Walk-forward CV MAE: {cv_mae:.4f} index pts (target ≤ 2.5)")
    return cv_mae


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()

    # 1. Fetch data
    logger.info("Fetching coin list...")
    coin_ids = fetch_coin_list(TOP_N_COINS)

    logger.info(f"Fetching {TRAIN_DAYS}d hourly data for {len(coin_ids)} coins...")
    btc_df = fetch_hourly("bitcoin",  TRAIN_DAYS)
    eth_df = fetch_hourly("ethereum", TRAIN_DAYS)

    all_X, all_y = [], []
    for coin_id in coin_ids:
        try:
            df = fetch_hourly(coin_id, TRAIN_DAYS)
            X, y = build_sequences(df, btc_df=btc_df, eth_df=eth_df, for_training=True)
            if len(X) == 0:
                continue
            all_X.append(X)
            all_y.append(y)
            logger.info(f"[{coin_id}] {len(X)} sequences")
        except Exception as e:
            logger.error(f"[{coin_id}] skipped: {e}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    logger.info(f"Total sequences: {len(X_all)}")

    # 2. Normalise features (fit on training data)
    scaler = StandardScaler()
    shape = X_all.shape
    X_scaled = scaler.fit_transform(X_all.reshape(-1, N_FEATURES)).reshape(shape)
    X_scaled = X_scaled.astype(np.float32)

    # 3. Walk-forward CV
    logger.info("Running walk-forward CV...")
    cv_mae = walk_forward_cv(X_scaled, y_all, device)

    # 4. Final model on all data
    logger.info("Training final model on all data...")
    model = QuantileLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON).to(device)
    fit(model, X_scaled, y_all, device)

    # 5. Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size":  N_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "seq_len":     SEQ_LEN,
            "horizon":     HORIZON,
        },
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_scale":  scaler.scale_.tolist(),
        "feature_names": list(X_all.shape),
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "cv_mae":        cv_mae,
        "version":       "lstm-1.0.0",
    }, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")

    # Commit updated model
    import subprocess
    subprocess.run(["git", "add", "model.pt"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Update model (cv_mae={cv_mae:.3f})"],
        check=False,  # no-op if nothing changed
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    logger.info("Pushed model.pt to GitHub")


if __name__ == "__main__":
    main()
