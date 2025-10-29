# src/models/train_lgbm.py
import argparse, json, os
from typing import List, Tuple
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--target_col", required=True)
    p.add_argument("--valid_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_out", default="artifacts/lgbm_model.txt")
    p.add_argument("--feature_list_out", default="artifacts/lgbm_features.txt")
    p.add_argument("--metrics_out", default="artifacts/metrics.json")
    p.add_argument("--categoricals", nargs="*", default=None)
    p.add_argument("--learning_rate", type=float, default=0.08)
    p.add_argument("--num_boost_round", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    return p.parse_args()

def _drop_bad_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    all_na_cols = df.columns[df.isna().all()].tolist()
    if all_na_cols:
        df = df.drop(columns=all_na_cols); dropped += all_na_cols
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols); dropped += const_cols
    return df, dropped

def _coerce_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def main():
    args = parse_args()
    for pth in [args.model_out, args.feature_list_out, args.metrics_out]:
        os.makedirs(os.path.dirname(pth), exist_ok=True)

    df = pd.read_csv(args.train_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not in CSV columns")

    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])

    if y.nunique() <= 1:
        raise ValueError(f"Target has {y.nunique()} unique values; need >=2")

    X, dropped = _drop_bad_columns(X)
    print("[INFO] No constant/all-NaN columns found." if not dropped else f"[INFO] Dropped columns: {dropped}")

    auto_cat = (X.select_dtypes(include=["object", "category"]).columns.tolist()
                if args.categoricals is None
                else [c for c in args.categoricals if c in X.columns])
    X = _coerce_categoricals(X, auto_cat)
    print(f"[INFO] Categorical columns: {auto_cat}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, random_state=args.seed
    )

    dtrain = lgb.Dataset(X_train, label=y_train,
                         categorical_feature=auto_cat if auto_cat else None,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid,
                         categorical_feature=auto_cat if auto_cat else None,
                         free_raw_data=False)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=args.learning_rate,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=1e-3,
        min_split_gain=0.0,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        min_data_per_group=1,
        max_cat_threshold=64,
        verbosity=-1,
    )

    print("[INFO] Starting training…")
    # ✅ Use callbacks for early stopping (works across older LightGBM versions)
    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    best_iter = getattr(booster, "best_iteration", args.num_boost_round) or args.num_boost_round
    best_score = booster.best_score.get("valid", {}).get("rmse", np.nan)

    booster.save_model(args.model_out, num_iteration=best_iter)
    with open(args.feature_list_out, "w") as f:
        for c in X.columns:
            f.write(f"{c}\n")

    metrics = {
        "best_iteration": int(best_iter),
        "valid_rmse": float(best_score),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_features_used": int(X.shape[1]),
        "dropped_columns": dropped,
        "categorical_columns": auto_cat,
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] Best iteration: {best_iter}")
    print(f"[INFO] Valid RMSE: {best_score}")
    print(f"[INFO] Model saved to: {args.model_out}")
    print(f"[INFO] Feature list saved to: {args.feature_list_out}")
    print(f"[INFO] Metrics saved to: {args.metrics_out}")

if __name__ == "__main__":
    main()
