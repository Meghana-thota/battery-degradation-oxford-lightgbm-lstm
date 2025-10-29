# scripts/build_supervised.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parents[1]
SRC = ROOT / "data" / "processed" / "oxford_capacity_trends.csv"
TAB_OUT = ROOT / "data" / "processed" / "tabular_next_capacity.csv"
SEQ_OUT = ROOT / "data" / "processed" / "sequences_next_capacity.npz"

RANDOM_STATE = 42
WINDOW = 5  # sequence length for LSTM

def main():
    df = pd.read_csv(SRC)
    # keep clean, sorted
    df = df.sort_values(["cell_id", "cycle"]).reset_index(drop=True)

    # ----- build next-cycle target per cell -----
    df["y_next"] = (
        df.groupby("cell_id")["capacity_mAh"].shift(-1)
    )
    # drop last point of each cell (no next label)
    df_tab = df.dropna(subset=["y_next"]).reset_index(drop=True)

    # choose features for tabular (you can add/remove later)
    drop_cols = {
        "y_next",  # target handled separately later
        "SoH",     # correlated with capacity; keep things simple first
    }
    # basic numeric features except identifiers & explicit target
    base_cols = [c for c in df_tab.columns if c not in {"cell_id"}]
    feature_cols = [c for c in base_cols if c not in drop_cols]

    # save tabular
    df_tab.to_csv(TAB_OUT, index=False)
    print(" wrote:", TAB_OUT, "shape:", df_tab.shape)
    print("features:", feature_cols[:10], "… (#", len(feature_cols), ")")

    # ----- build sequences for LSTM -----
    # pick a compact set of channels to start with
    seq_feats = ["capacity_mAh", "v_dc_mean", "T_dc_mean"]
    seq_X, seq_y, seq_groups = [], [], []

    for cell, g in df.groupby("cell_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        # we need WINDOW past steps to predict the next step
        for i in range(len(g) - WINDOW):
            window = g.loc[i : i + WINDOW - 1, seq_feats].to_numpy(dtype=np.float32)
            target = g.loc[i + WINDOW, "capacity_mAh"]  # next after window
            # ensure we didn’t cross a gap (shouldn’t for clean set, but safe)
            if window.shape[0] == WINDOW and np.isfinite(target):
                seq_X.append(window)
                seq_y.append(np.float32(target))
                seq_groups.append(cell)

    seq_X = np.stack(seq_X, axis=0) if seq_X else np.zeros((0, WINDOW, len(seq_feats)), dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32)
    seq_groups = np.array(seq_groups)

    np.savez(SEQ_OUT, X=seq_X, y=seq_y, groups=seq_groups, feat_names=np.array(seq_feats))
    print(" wrote:", SEQ_OUT, "X:", seq_X.shape, "y:", seq_y.shape)
    print("cells:", np.unique(seq_groups))

if __name__ == "__main__":
    main()
