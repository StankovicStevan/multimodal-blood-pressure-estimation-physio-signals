import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import csv

REQUIRED_KEYS = ['UID', 'age', 'weight', 'height', 'data_PPG', 'data_ECG', 'data_PCG', 'data_FSR', 'data_BP']
DATASET_DIR = Path('..') / 'dataset' / 'Cuff-less Non-invasive Blood Pressure Estimation Data Set'
PLOT = True


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, dict, str, bytes)) and len(value) == 0)


def describe_value(value: Any) -> str:
    """Return human-readable info about value type and size."""
    if value is None:
        return "None"

    # List-like
    if isinstance(value, list):
        if len(value) == 0:
            return "list (EMPTY)"

        # check list of dicts → tabular
        if isinstance(value[0], dict):
            rows = len(value)
            cols = len(value[0])
            return f"list of dicts (table) → rows={rows}, cols={cols}"
        return f"list → length={len(value)}"

    # NumPy array
    if isinstance(value, np.ndarray):
        return f"numpy array → shape={value.shape}"

    # Dict
    if isinstance(value, dict):
        if len(value) == 0:
            return "dict (EMPTY)"
        # check if dict of lists
        if all(isinstance(v, list) for v in value.values()):
            rows = len(next(iter(value.values())))
            cols = len(value)
            return f"dict of lists (table) → rows={rows}, cols={cols}"
        return f"dict → keys={len(value)}"

    # Fallback scalars
    return f"{type(value).__name__} → value={value}"


def check_required(data: Dict[str, Any], filename: str) -> List[str]:
    print(f"\n--- Checking {filename} ---")
    missing = []
    for k in REQUIRED_KEYS:
        if k not in data:
            print(f" ❌ {k}: MISSING")
            missing.append(k)
        elif is_empty(data[k]):
            print(f" ⚠️ {k}: EMPTY ({describe_value(data[k])})")
            missing.append(k)
        else:
            print(f" ✅ {k}: OK ({describe_value(data[k])})")
    return missing


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    return pd.Series(x, dtype=float).interpolate(limit_direction='both').to_numpy()


def clean_fsr_signal(fsr: Sequence[float], max_jump: float = 50.0) -> np.ndarray:
    x = np.asarray(fsr, dtype=float)
    if x.size < 2:
        return x
    jumps = np.abs(x[1:] - x[:-1]) > max_jump
    mask = np.append(jumps, False)
    x[mask] = np.nan
    return interpolate_nans(x)


def smooth_fsr(x: np.ndarray, window: int = 51, polyorder: int = 3) -> np.ndarray:
    n = len(x)
    if n < 5:
        return x
    if window > n:
        window = n - (1 - n % 2)
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2:
        return x
    return signal.savgol_filter(x, window, polyorder)


def slope_series(x: np.ndarray, lag: int) -> np.ndarray:
    if len(x) <= lag:
        return np.array([])
    return x[lag:] - x[:-lag]


def find_local_mins(x: np.ndarray, k: int, exclusion_window: int) -> List[int]:
    if k <= 0 or len(x) == 0:
        return []
    x = x.copy()
    xmax = np.nanmax(x)
    x[np.isnan(x)] = xmax
    out = []
    half = exclusion_window // 2

    for _ in range(k):
        idx = int(np.argmin(x))
        out.append(idx)
        lo = max(0, idx - half)
        hi = min(len(x), idx + half + 1)
        x[lo:hi] = xmax

    return sorted(out)


def trim_initial_flat(fsr, threshold=5, min_run=2000):
    diffs = np.abs(np.diff(fsr))
    flat = diffs < threshold
    for i in range(min_run, len(flat)):
        if not flat[i]:
            return fsr[i:]
    return fsr


def process_subject(filepath: Path) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        data = json.load(f)

    missing = check_required(data, filepath.name)

    if 'data_FSR' not in data or len(data['data_FSR']) == 0:
        print(" ❌ FSR missing → skipping")
        return {"file": filepath.name, "skipped": True}

    # FSR loaded & logged
    print(f" ℹ️ data_FSR size: {len(data['data_FSR'])}")

    fsr_raw = -np.asarray(data['data_FSR'], dtype=float)
    fsr = trim_initial_flat(fsr_raw, threshold=5, min_run=2000)

    if PLOT:
        plt.figure(figsize=(14, 4))
        plt.title(f"FSR (trimmed start) — {filepath.name}")
        plt.plot(fsr)

    fsr_clean = clean_fsr_signal(fsr)
    fsr_smooth = smooth_fsr(fsr_clean, window=51, polyorder=3)

    diff_n = 1000
    fsr_slope = slope_series(fsr_smooth, diff_n)
    fsr_slope_roll = pd.Series(fsr_slope).rolling(21, center=True, min_periods=1).mean().to_numpy()

    k = len(data.get('data_BP', []))
    print(f" ℹ️ data_BP count: {k}")

    mins = find_local_mins(fsr_slope_roll, k, exclusion_window=15000)
    marks = [m + diff_n // 2 for m in mins]

    if PLOT:
        plt.figure(figsize=(14, 4))
        plt.title(f"Minima (BP points) — {filepath.name}")
        plt.plot(fsr_smooth, label="FSR smooth")
        y0, y1 = np.nanmin(fsr_smooth), np.nanmax(fsr_smooth)
        for idx in marks:
            plt.vlines(idx, y0, y1, colors='red')
        plt.legend()

    return {
        "file": filepath.name,
        "UID": data.get("UID"),
        "bp_points": marks,
        "bp_values": data.get("data_BP", []),
        "skipped": False
    }


def save_timestamp_results_csv(results, out_file="bp_timestamps.csv"):
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "UID", "timestamp", "SBP", "DBP"])

        for r in results:
            if not r.get("skipped"):
                for idx, bp in zip(r["bp_points"], r["bp_values"]):
                    writer.writerow([r["file"], r["UID"], idx, bp["SBP"], bp["DBP"]])


def process_all_subjects():
    results = []
    for fp in sorted(DATASET_DIR.glob("*.json")):
        print(fp.name)
        try:
            results.append(process_subject(fp))
        except Exception as e:
            print(f"❌ Error in {fp.name}: {e}")
            results.append({"file": fp.name, "skipped": True})

    print("\n===== SUMMARY =====")
    for r in results:
        if r.get("skipped"):
            print(f"{r['file']}: SKIPPED")
        else:
            print(f"{r['file']}: OK — BP points={len(r['bp_points'])}")

    save_timestamp_results_csv(results, "bp_timestamps.csv")
    print("\n✅ Saved bp_timestamps.csv")

    return results


if __name__ == "__main__":
    results = process_all_subjects()
