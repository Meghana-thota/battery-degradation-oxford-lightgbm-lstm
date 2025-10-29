# src/models/train_lstm.py
"""
LSTM trainer for next-cycle capacity from sequence features.

Usage:
  python src/models/train_lstm.py \
    --npz data/processed/sequences_next_capacity.npz \
    --model_out artifacts/lstm_model.pt \
    --metrics_out artifacts/lstm_metrics.json
"""

import argparse, json, os, math, random
from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to sequences .npz")
    p.add_argument("--valid_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model_out", default="artifacts/lstm_model.pt")
    p.add_argument("--scaler_out", default="artifacts/lstm_scaler.npz")
    p.add_argument("--metrics_out", default="artifacts/lstm_metrics.json")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    # Try common keys
    Xk = next((k for k in ["X","x","features","inputs"] if k in d), None)
    yk = next((k for k in ["y","Y","target","labels"] if k in d), None)
    if Xk is None or yk is None:
        # Fallback: pick 3D array as X, 1D/2D as y
        Xk = next((k for k in keys if d[k].ndim==3), None)
        yk = next((k for k in keys if d[k].ndim in (1,2) and k!=Xk), None)
    if Xk is None or yk is None:
        raise ValueError(f"Could not infer keys from {keys}. Expected X(3D) and y(1D/2D).")
    X, y = d[Xk], d[yk]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"y must be 1D after squeeze; got shape {y.shape}")
    return X.astype(np.float32), y.astype(np.float32)

def train_val_split(X, y, valid_frac, seed):
    n = len(X)
    n_valid = max(1, int(round(n*valid_frac)))
    idx = np.arange(n); rng = np.random.RandomState(seed); rng.shuffle(idx)
    valid_idx = idx[:n_valid]; train_idx = idx[n_valid:]
    return (X[train_idx], y[train_idx]), (X[valid_idx], y[valid_idx])

def fit_standardizer(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    # X: (N, T, F) -> standardize feature-wise across N*T
    N, T, F = X_train.shape
    Xr = X_train.reshape(N*T, F)
    mean = Xr.mean(axis=0)
    std = Xr.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

def apply_standardizer(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - scaler["mean"]) / scaler["std"]

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, T, F)
        self.y = torch.from_numpy(y).unsqueeze(-1)  # (N, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]         # (B, H)
        return self.head(last)       # (B, 1)

def rmse(pred, target):
    return math.sqrt(nn.functional.mse_loss(pred, target).item())

def main():
    args = parse_args()
    set_seed(args.seed)
    for p in [args.model_out, args.scaler_out, args.metrics_out]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    X, y = load_npz(args.npz)
    N, T, F = X.shape
    print(f"[INFO] Loaded sequences: N={N}, T={T}, F={F}")

    (Xtr, ytr), (Xva, yva) = train_val_split(X, y, args.valid_frac, args.seed)

    scaler = fit_standardizer(Xtr)
    Xtr = apply_standardizer(Xtr, scaler)
    Xva = apply_standardizer(Xva, scaler)

    train_ds = SeqDataset(Xtr, ytr)
    valid_ds = SeqDataset(Xva, yva)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(F, args.hidden_size, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                valid_loss += loss.item() * xb.size(0)
        valid_loss /= len(valid_ds)

        print(f"[EPOCH {epoch:03d}] train_mse={train_loss:.6f} | valid_mse={valid_loss:.6f}")

        if valid_loss + 1e-8 < best_loss:
            best_loss = valid_loss
            best_state = { "model": model.state_dict(), "epoch": epoch }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best valid_mse={best_loss:.6f})")
                break

    # Save best model
    torch.save({
        "state_dict": best_state["model"] if best_state else model.state_dict(),
        "input_size": F,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epoch": best_state["epoch"] if best_state else args.epochs,
    }, args.model_out)

    # Save scaler
    np.savez(args.scaler_out, mean=scaler["mean"], std=scaler["std"])

    # Metrics
    valid_rmse = float(math.sqrt(best_loss))
    metrics = {
        "n_samples": int(N),
        "seq_len": int(T),
        "n_features": int(F),
        "best_valid_mse": float(best_loss),
        "best_valid_rmse": valid_rmse,
        "best_epoch": int(best_state["epoch"] if best_state else args.epochs),
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Saved model to:", args.model_out)
    print("[INFO] Saved scaler to:", args.scaler_out)
    print("[INFO] Saved metrics to:", args.metrics_out)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
