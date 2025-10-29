# src/eval/eval_compare.py
import json, argparse, os
from math import sqrt
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lgbm_metrics", default="artifacts/metrics.json")
    p.add_argument("--lstm_metrics", default="artifacts/lstm_metrics.json")
    return p.parse_args()

def loadj(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

if __name__ == "__main__":
    args = parse_args()
    m1 = loadj(args.lgbm_metrics)
    m2 = loadj(args.lstm_metrics)
    print("\n Model Scoreboard ")
    if m1:
        print(f"LGBM:  valid_rmse={m1.get('valid_rmse'):.6f}  (best_iter={m1.get('best_iteration')})")
    else:
        print("LGBM:  metrics not found.")
    if m2:
        print(f"LSTM:  valid_rmse={m2.get('best_valid_rmse'):.6f}  (best_epoch={m2.get('best_epoch')})")
    else:
        print("LSTM:  metrics not found.")
    print(" \n")
