import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly, butter, filtfilt
import random
import os

DATASET_DIR = Path('..') / 'dataset' / 'Cuff-less Non-invasive Blood Pressure Estimation Data Set'
TIMESTAMP_CSV = "bp_timestamps.csv"
OUTPUT_DIR = Path("./preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========= Settings =========
ORIG_FS = 1000
TARGET_FS = 250
WINDOW_SEC = 4
SAMPLES = WINDOW_SEC * TARGET_FS
CHANNELS = ["data_PPG", "data_ECG", "data_PCG", "data_FSR"]

AUGMENT = True          # enable augmentations
P_SECOND_AUG = 0.5       # probability of a second augmented copy


# ========= Filters =========
def bandpass(data, fs, low, high, order=3):
    nyq = fs * 0.5
    low_norm = max(low / nyq, 1e-6)
    high_norm = min(high / nyq, 0.999)
    if low_norm >= high_norm:
        return data
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data)


def lowpass(data, fs, cutoff, order=3):
    nyq = fs * 0.5
    cutoff_norm = min(max(cutoff / nyq, 1e-6), 0.999)
    b, a = butter(order, cutoff_norm, btype='low')
    return filtfilt(b, a, data)


# ========= CSV loader =========
def load_timestamps(csv_file):
    return pd.read_csv(csv_file)


# ========= JSON loader =========
def load_subject_json(subject_file):
    with open(subject_file, "r") as f:
        return json.load(f)


# ========= Extract window =========
def extract_window(sig, center_idx):
    start = center_idx - SAMPLES // 2
    end = center_idx + SAMPLES // 2
    if start < 0 or end > len(sig):
        return None
    return sig[start:end]


# ========= Augmentation =========
def add_gaussian_noise(seg, std=0.01):
    return seg + np.random.normal(0, std, seg.shape)

def random_scaling(seg, min_s=0.9, max_s=1.1):
    factor = np.random.uniform(min_s, max_s)
    return seg * factor

def augment_signal(seg):
    if random.random() < 0.5:
        seg = add_gaussian_noise(seg)
    if random.random() < 0.5:
        seg = random_scaling(seg)
    return seg


# ========= Main pipeline =========
def build_dataset():
    df = load_timestamps(TIMESTAMP_CSV)

    X_signal = []
    X_tabular = []
    y = []

    grouped = df.groupby("file")
    total = 0
    samples_per_subject = {}

    for fname, group in grouped:
        print(f"\nProcessing {fname} ...")

        subject_path = DATASET_DIR / fname
        if not subject_path.exists():
            print(f"❌ Missing JSON for {fname} — skipping")
            continue

        data = load_subject_json(subject_path)

        age, weight, height = data["age"], data["weight"], data["height"]

        # Load signals
        sigs = {c: np.asarray(data[c], dtype=float) for c in CHANNELS}

        # Downsample
        for c in CHANNELS:
            sigs[c] = resample_poly(sigs[c], up=1, down=ORIG_FS // TARGET_FS)

        # Filtering
        sigs["data_ECG"] = bandpass(sigs["data_ECG"], TARGET_FS, 0.5, 40)
        sigs["data_PPG"] = bandpass(sigs["data_PPG"], TARGET_FS, 0.5, 8)
        sigs["data_PCG"] = bandpass(sigs["data_PCG"], TARGET_FS, 20, 120)
        sigs["data_FSR"] = lowpass(sigs["data_FSR"], TARGET_FS, 1)

        before = len(X_signal)

        # Extract windows for this subject
        for _, row in group.iterrows():
            ts_raw = int(row["timestamp"])            # original timestamp @ 1000 Hz
            ts = int(ts_raw * (TARGET_FS / ORIG_FS))  # rescaled to 250 Hz

            SBP, DBP = float(row["SBP"]), float(row["DBP"])

            seg_channels = []
            valid = True

            for c in CHANNELS:
                w = extract_window(sigs[c], ts)
                if w is None:
                    valid = False
                    break
                seg_channels.append(w)

            if not valid:
                continue

            seg = np.stack(seg_channels, axis=-1)

            # normalize
            seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)

            # ORIGINAL
            X_signal.append(seg)
            X_tabular.append([age, weight, height])
            y.append([SBP, DBP])

            # FIRST augmentation
            if AUGMENT:
                aug1 = augment_signal(seg.copy())
                X_signal.append(aug1)
                X_tabular.append([age, weight, height])
                y.append([SBP, DBP])

                # SECOND augmentation (probabilistic)
                if random.random() < P_SECOND_AUG:
                    aug2 = augment_signal(seg.copy())
                    X_signal.append(aug2)
                    X_tabular.append([age, weight, height])
                    y.append([SBP, DBP])

        added = len(X_signal) - before
        samples_per_subject[fname] = added
        total += added
        print(f"✅ Extracted {added} samples from {fname}")

    # Convert to np arrays
    X_signal = np.array(X_signal)
    X_tabular = np.array(X_tabular, dtype=float)
    y = np.array(y, dtype=float)

    print("\n===== SUMMARY =====")
    print("Total samples:", total)
    print("X_signal:", X_signal.shape)
    print("X_tabular:", X_tabular.shape)
    print("y:", y.shape)

    print("\nSamples per subject:")
    for fname, cnt in samples_per_subject.items():
        print(f"  {fname}: {cnt}")

    # Filename tag
    tag = f"{WINDOW_SEC}s_augmented" if AUGMENT else f"{WINDOW_SEC}s"

    # Save arrays
    np.save(OUTPUT_DIR / f"X_signal_{tag}.npy", X_signal)
    np.save(OUTPUT_DIR / f"X_tabular_{tag}.npy", X_tabular)
    np.save(OUTPUT_DIR / f"y_{tag}.npy", y)

    # Save summary JSON
    summary = {
        "window_seconds": WINDOW_SEC,
        "augmentation_enabled": bool(AUGMENT),
        "total_samples": int(len(X_signal)),
        "subjects": {fname: int(cnt) for fname, cnt in samples_per_subject.items()}
    }
    with open(OUTPUT_DIR / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Saved: X_signal_{tag}.npy, X_tabular_{tag}.npy, y_{tag}.npy")
    print(f"✅ Saved summary_{tag}.json")
    print(f"✅ Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    build_dataset()
