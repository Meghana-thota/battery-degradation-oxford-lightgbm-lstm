# src/models/predict_lstm.py
import argparse, os
import numpy as np, torch
from torch import nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                  nn.ReLU(), nn.Linear(hidden_size//2, 1))
    def forward(self, x):
        out,_ = self.lstm(x); return self.head(out[:,-1,:])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="NPZ with X (N,T,F)")
    p.add_argument("--model", default="artifacts/lstm_model.pt")
    p.add_argument("--scaler", default="artifacts/lstm_scaler.npz")
    p.add_argument("--out", default="artifacts/lstm_predictions.npy")
    return p.parse_args()

def main():
    args = parse_args()
    d = np.load(args.npz, allow_pickle=True)
    Xk = next((k for k in ["X","x","features","inputs"] if k in d), None)
    if Xk is None: Xk = next(k for k in d.files if d[k].ndim==3)
    X = d[Xk].astype(np.float32)

    sc = np.load(args.scaler)
    mean, std = sc["mean"].astype(np.float32), sc["std"].astype(np.float32)
    X = (X - mean) / std

    ckpt = torch.load(args.model, map_location="cpu")
    model = LSTMRegressor(ckpt["input_size"], ckpt["hidden_size"],
                          ckpt["num_layers"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(X)).squeeze(-1).numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, preds)
    print(f"[INFO] Saved predictions → {args.out} (shape {preds.shape})")

if __name__ == "__main__":
    main()
