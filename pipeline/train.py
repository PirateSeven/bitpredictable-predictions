"""
Train a quantile LSTM on 90 days of CoinGecko hourly data.
Saves model.pt to the repo root.
Run: python pipeline/train.py

Fixes vs v1:
  - QuantileLSTM uses attention pooling over all hidden states (not just h[-1])
  - Walk-forward CV splits each coin's sequences by time independently
    → prevents cross-coin temporal leakage
  - Scaler is fit inside each CV fold on training data only
    → eliminates data leakage from validation period
  - CV metric: 24h cumulative return MAE (matches what the site displays)
"""

import logging
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
ES_PATIENCE   = 15
CV_FOLDS      = 5
CV_TEST_DAYS  = 14
MODEL_PATH    = Path("model.pt")
BEST_WEIGHTS  = Path("best_weights.pt")


# ── Model ──────────────────────────────────────────────────────────────────────
class QuantileLSTM(nn.Module):
    """
    2-layer LSTM with additive attention pooling over all hidden states.
    Attention prevents throwing away temporal information from early timesteps
    (the flaw in using only h[-1]).
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        # Single learned attention query (no bias — avoids position bias)
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_q10 = nn.Linear(64, horizon)
        self.head_med = nn.Linear(64, horizon)
        self.head_q90 = nn.Linear(64, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)                           # (batch, seq_len, hidden)
        w = torch.softmax(self.attn(out), dim=1)        # (batch, seq_len, 1)
        h = (w * out).sum(dim=1)                        # (batch, hidden)
        h = self.fc(h)
        return torch.stack([
            self.head_q10(h),
            self.head_med(h),
            self.head_q90(h),
        ], dim=1)                                        # (batch, 3, horizon)


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
    # Soft monotonicity: q10 ≤ med ≤ q90
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


def cumulative_24h_mae(model, loader, device) -> float:
    """
    MAE of the 24h compounded predicted return vs actual.
    This matches what the site displays as the accuracy score.
    """
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            pred_med = model(xb.to(device))[:, 1, :].cpu().numpy()  # (batch, 24)
            actual   = yb.cpu().numpy()                              # (batch, 24)
            pred_cum   = (np.cumprod(1 + pred_med / 100, axis=1)[:, -1] - 1) * 100
            actual_cum = (np.cumprod(1 + actual   / 100, axis=1)[:, -1] - 1) * 100
            errs.extend(np.abs(pred_cum - actual_cum).tolist())
    return float(np.mean(errs))


def fit(model, X: np.ndarray, y: np.ndarray, device, val_split=0.1) -> float:
    """Train with early stopping. Returns best val loss."""
    n_val  = max(1, int(len(X) * val_split))
    X_tr, y_tr   = X[:-n_val], y[:-n_val]
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

    best_val       = float("inf")
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
            logger.info(f"  epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    BEST_WEIGHTS.unlink(missing_ok=True)
    return best_val


# ── Walk-forward CV ────────────────────────────────────────────────────────────
def walk_forward_cv(
    all_X: list[np.ndarray],
    all_y: list[np.ndarray],
    device: torch.device,
) -> float:
    """
    True temporal walk-forward CV.

    Each coin's sequences are split independently in time so that the
    test window is always strictly newer than the training window.
    The scaler is re-fit on each fold's training data to prevent
    data leakage from the validation period into normalisation.
    """
    test_size = CV_TEST_DAYS * 24   # 14 days × 24h = 336 sequences per fold per coin
    fold_step = test_size // 2      # 168 — step between consecutive fold boundaries

    maes = []
    for fold in range(CV_FOLDS):
        X_tr_parts,  y_tr_parts  = [], []
        X_val_parts, y_val_parts = [], []

        for X_coin, y_coin in zip(all_X, all_y):
            n = len(X_coin)
            # fold CV_FOLDS-1 = most recent window; fold 0 = oldest
            val_end   = n - (CV_FOLDS - 1 - fold) * fold_step
            val_start = val_end - test_size

            if val_start < SEQ_LEN:
                continue  # not enough training history for this coin/fold

            X_tr_parts.append(X_coin[:val_start])
            y_tr_parts.append(y_coin[:val_start])
            X_val_parts.append(X_coin[val_start:val_end])
            y_val_parts.append(y_coin[val_start:val_end])

        if not X_tr_parts:
            logger.warning(f"Fold {fold+1}/{CV_FOLDS}: no valid coins, skipping")
            continue

        X_tr  = np.concatenate(X_tr_parts,  axis=0)
        y_tr  = np.concatenate(y_tr_parts,  axis=0)
        X_val = np.concatenate(X_val_parts, axis=0)
        y_val = np.concatenate(y_val_parts, axis=0)

        # Fit scaler on TRAINING data only — no leakage from validation period
        fold_scaler = StandardScaler()
        shape_tr  = X_tr.shape
        shape_val = X_val.shape
        X_tr_sc  = fold_scaler.fit_transform(
            X_tr.reshape(-1, N_FEATURES)
        ).reshape(shape_tr).astype(np.float32)
        X_val_sc = fold_scaler.transform(
            X_val.reshape(-1, N_FEATURES)
        ).reshape(shape_val).astype(np.float32)

        model = QuantileLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON).to(device)
        fit(model, X_tr_sc, y_tr.astype(np.float32), device)

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_sc), torch.tensor(y_val.astype(np.float32))),
            batch_size=BATCH_SIZE,
        )
        mae = cumulative_24h_mae(model, val_loader, device)
        maes.append(mae)
        logger.info(
            f"CV Fold {fold+1}/{CV_FOLDS} | "
            f"train={len(X_tr):,}  val={len(X_val):,} | "
            f"24h MAE: {mae:.2f}%"
        )

    if not maes:
        logger.warning("Walk-forward CV produced no results.")
        return float("nan")

    cv_mae = float(np.mean(maes))
    logger.info(f"Walk-forward CV 24h MAE: {cv_mae:.2f}% (target ≤ 5%)")
    return cv_mae


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = get_device()

    # 1. Fetch data — build per-coin sequence arrays (order preserved for CV)
    logger.info("Fetching coin list...")
    coin_ids = fetch_coin_list(TOP_N_COINS)

    logger.info(f"Fetching {TRAIN_DAYS}d hourly data for {len(coin_ids)} coins...")
    btc_df = fetch_hourly("bitcoin",  TRAIN_DAYS)
    eth_df = fetch_hourly("ethereum", TRAIN_DAYS)

    all_X_per_coin: list[np.ndarray] = []
    all_y_per_coin: list[np.ndarray] = []

    for coin_id in coin_ids:
        try:
            df = fetch_hourly(coin_id, TRAIN_DAYS)
            X, y = build_sequences(df, btc_df=btc_df, eth_df=eth_df, for_training=True)
            if len(X) == 0:
                continue
            all_X_per_coin.append(X)
            all_y_per_coin.append(y)
            logger.info(f"  [{coin_id}] {len(X)} sequences")
        except Exception as e:
            logger.error(f"  [{coin_id}] skipped: {e}")

    if not all_X_per_coin:
        raise RuntimeError("No training data fetched. Check API key and network.")

    # 2. Walk-forward CV (scaler fit per-fold inside, no global leakage)
    logger.info("Running walk-forward CV...")
    cv_mae = walk_forward_cv(all_X_per_coin, all_y_per_coin, device)

    # 3. Final model: concatenate all coins, fit global scaler on full dataset
    X_all = np.concatenate(all_X_per_coin, axis=0)
    y_all = np.concatenate(all_y_per_coin, axis=0)
    logger.info(f"Total sequences: {len(X_all):,}")

    scaler = StandardScaler()
    shape  = X_all.shape
    X_scaled = scaler.fit_transform(
        X_all.reshape(-1, N_FEATURES)
    ).reshape(shape).astype(np.float32)

    logger.info("Training final model on all data...")
    model = QuantileLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, HORIZON).to(device)
    fit(model, X_scaled, y_all.astype(np.float32), device)

    # 4. Save
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
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "cv_mae_24h":   cv_mae,
        "version":      "lstm-2.0.0",
    }, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")

    import subprocess
    subprocess.run(["git", "add", "model.pt"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Update model v2 (cv_24h_mae={cv_mae:.2f}%)"],
        check=False,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    logger.info("Pushed model.pt to GitHub")


if __name__ == "__main__":
    main()
