# scripts/make_features.py
import sys, traceback
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data" / "raw" / "oxford" / "Oxford_Battery_Degradation_Dataset_1.mat"
OUT_CSV   = Path(__file__).parents[1] / "data" / "processed" / "oxford_capacity_trends.csv"

def log(*a): print("[make_features]", *a, flush=True)

def series_stats(arr: np.ndarray, prefix: str):
    x = np.asarray(arr).ravel()
    if x.size == 0 or not np.isfinite(x).any():
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_min": np.nan,
                f"{prefix}_max": np.nan, f"{prefix}_range": np.nan}
    return {
        f"{prefix}_mean": float(np.nanmean(x)),
        f"{prefix}_std":  float(np.nanstd(x)),
        f"{prefix}_min":  float(np.nanmin(x)),
        f"{prefix}_max":  float(np.nanmax(x)),
        f"{prefix}_range": float(np.nanmax(x) - np.nanmin(x)),
    }

def duration_seconds(t):
    t = np.asarray(t).ravel()
    if t.size == 0 or not np.isfinite(t).any():
        return np.nan
    return float(np.nanmax(t) - np.nanmin(t))

def ocv_slope_features(q, v):
    q = np.asarray(q).ravel()
    v = np.asarray(v).ravel()
    if len(q) < 2 or len(v) < 2:
        return {"ocv_slope_mean": np.nan, "ocv_slope_std": np.nan,
                "ocv_slope_p25": np.nan, "ocv_slope_p50": np.nan, "ocv_slope_p75": np.nan}
    idx = np.argsort(q); q = q[idx]; v = v[idx]
    dq = np.diff(q); dv = np.diff(v)
    mask = np.abs(dq) > 1e-6
    if not mask.any():
        return {"ocv_slope_mean": np.nan, "ocv_slope_std": np.nan,
                "ocv_slope_p25": np.nan, "ocv_slope_p50": np.nan, "ocv_slope_p75": np.nan}
    s = dv[mask] / dq[mask]
    return {
        "ocv_slope_mean": float(np.nanmean(s)),
        "ocv_slope_std":  float(np.nanstd(s)),
        "ocv_slope_p25":  float(np.nanpercentile(s, 25)),
        "ocv_slope_p50":  float(np.nanpercentile(s, 50)),
        "ocv_slope_p75":  float(np.nanpercentile(s, 75)),
    }

def extract_cycle_features(cyc_struct):
    feats = {}
    c1dc = getattr(cyc_struct, "C1dc", None)
    if c1dc is not None:
        q_dc = getattr(c1dc, "q", np.array([]))
        v_dc = getattr(c1dc, "v", np.array([]))
        T_dc = getattr(c1dc, "T", np.array([]))
        t_dc = getattr(c1dc, "t", np.array([]))
        feats["capacity_mAh"] = float(np.nanmax(q_dc) - np.nanmin(q_dc)) if np.size(q_dc)>0 else np.nan
        feats.update(series_stats(v_dc, "v_dc"))
        feats.update(series_stats(T_dc, "T_dc"))
        feats["t_dc_duration_s"] = duration_seconds(t_dc)

    c1ch = getattr(cyc_struct, "C1ch", None)
    if c1ch is not None:
        v_ch = getattr(c1ch, "v", np.array([]))
        T_ch = getattr(c1ch, "T", np.array([]))
        t_ch = getattr(c1ch, "t", np.array([]))
        feats.update(series_stats(v_ch, "v_ch"))
        feats.update(series_stats(T_ch, "T_ch"))
        feats["t_ch_duration_s"] = duration_seconds(t_ch)

    ocv = getattr(cyc_struct, "OCVdc", None)
    if ocv is not None:
        q_ocv = getattr(ocv, "q", np.array([]))
        v_ocv = getattr(ocv, "v", np.array([]))
        feats.update(ocv_slope_features(q_ocv, v_ocv))

    return feats

def main():
    try:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        log("CWD:", Path.cwd())
        log("Data path:", DATA_PATH)
        if not DATA_PATH.exists():
            log("ERROR: .mat file not found.")
            return

        log("Loading .mat (this can take a moment)...")
        data = scipy.io.loadmat(DATA_PATH.as_posix(), squeeze_me=True, struct_as_record=False)
        cell_keys = [k for k in data.keys() if k.lower().startswith("cell")]
        log("Found cell keys:", cell_keys)
        rows = []

        for cell_key in cell_keys:
            cell_obj = data[cell_key]
            if isinstance(cell_obj, (list, tuple, np.ndarray)):
                cell_obj = cell_obj[0]
            fieldnames = getattr(cell_obj, "_fieldnames", []) or []
            cyc_names = [c for c in fieldnames if c and str(c).lower().startswith("cyc")]
            cyc_names = sorted(cyc_names, key=lambda s: int("".join(ch for ch in str(s) if ch.isdigit()) or 0))
            log(f"{cell_key}: {len(cyc_names)} characterization cycles")

            first_cap = None
            for cyc_name in cyc_names:
                cyc_struct = getattr(cell_obj, cyc_name)
                feats = extract_cycle_features(cyc_struct)
                cyc_num = int("".join(ch for ch in str(cyc_name) if ch.isdigit()) or 0)
                row = {"cell_id": cell_key, "cycle": cyc_num, **feats}

                if first_cap is None and np.isfinite(row.get("capacity_mAh", np.nan)):
                    first_cap = row["capacity_mAh"]
                row["SoH"] = float(row["capacity_mAh"] / first_cap) if (first_cap and np.isfinite(row.get("capacity_mAh", np.nan))) else np.nan
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["cell_id", "cycle"]).reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        log("✅ Saved CSV:", OUT_CSV)
        log("Shape:", df.shape)
        if not df.empty:
            log("Cells:", df['cell_id'].unique().tolist())
            log("\nPer-cell cycles:\n", df.groupby("cell_id")["cycle"].agg(["min", "max", "count"]))
            log("\nHead:\n", df.head(10).to_string(index=False))

    except Exception as e:
        log("FATAL:", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
