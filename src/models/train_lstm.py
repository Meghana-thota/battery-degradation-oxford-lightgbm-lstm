# src/models/train_lstm.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
import json

ROOT = Path(__file__).parents[2]
NPZ = ROOT / "data" / "processed" / "sequences_next_capacity.npz"
OUT_DIR = ROOT / "outputs" / "lstm_tf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BATCH = 32
EPOCHS = 200
PATIENCE = 20

def rmse(y, yhat):
    return mean_squared_error(y, yhat, squared=False)

def main():
    npz = np.load(NPZ, allow_pickle=True)
    X = npz["X"]      # (N, T, F)
    y = npz["y"]      # (N,)
    groups = npz["groups"]  # cell IDs per sequence
    feat_names = npz["feat_names"]

    if X.shape[0] == 0:
        print("No sequences found. Did build_supervised run?")
        return

    # group split by cell for test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # train/val split by cell again
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss2.split(X_train, y_train, g_train))
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # scale features per time-step feature dimension using train set only
    # reshape to 2D for scaler fit
    T, F = X.shape[1], X.shape[2]
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, F)
    scaler.fit(X_tr_2d)
    def apply_scale(x):
        return scaler.transform(x.reshape(-1, F)).reshape(-1, T, F)
    X_tr = apply_scale(X_tr)
    X_val = apply_scale(X_val)
    X_test = apply_scale(X_test)

    # build model
    tf.random.set_seed(RANDOM_STATE)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)  # regression
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
        callbacks=cb
    )

    # eval
    pred_val = model.predict(X_val, verbose=0).ravel()
    pred_test = model.predict(X_test, verbose=0).ravel()
    metrics = {
        "val_MAE": float(mean_absolute_error(y_val, pred_val)),
        "val_RMSE": float(rmse(y_val, pred_val)),
        "test_MAE": float(mean_absolute_error(y_test, pred_test)),
        "test_RMSE": float(rmse(y_test, pred_test)),
    }

    # save
    model.save((OUT_DIR / "model.keras").as_posix())
    np.savez(OUT_DIR / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, feat_names=feat_names)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ LSTM metrics:", metrics)
    print("📁 Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
