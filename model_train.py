import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# PATH
DATA_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "IMU_Gesture_Data")

# SETTINGS (good defaults)
SENSOR_COLS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

RESAMPLE_DT_MS = 15

WINDOW_SAMPLES = 40          # 0.6s windows at 15ms
WINDOW_OVERLAP = 0.75
HP_GRAVITY_KERNEL = 15

# Keep windows per file roughly equal so long clips don't dominate
MAX_WINDOWS_PER_FILE = 20

# Trimming: DO NOT trim idle (we keep it as-is)
TRIM_USE = "gyro"
TRIM_STD_MULT = 0.75
TRIM_PAD_SAMPLES = 10

# Balancing behavior:
# - "oversample_to_max": bring every class up to the max class count (with replacement)
# - "downsample_to_min": cut every class down to the min class count (no replacement)
BALANCE_MODE = "oversample_to_max"
RANDOM_SEED = 42


# LOAD

def load_all_csvs(root):
    paths = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    dfs = []

    for p in paths:
        df = pd.read_csv(p)
        need = ["t_ms"] + SENSOR_COLS
        if any(c not in df.columns for c in need):
            continue

        parts = os.path.normpath(p).split(os.sep)
        inferred_label = parts[-3] if len(parts) >= 3 else "unknown"
        inferred_sess = parts[-2] if len(parts) >= 2 else "session1"

        if "label" not in df.columns:
            df["label"] = inferred_label
        if "session_id" not in df.columns:
            df["session_id"] = inferred_sess

        df["source_file"] = p
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid CSVs found under: {root}")

    all_df = pd.concat(dfs, ignore_index=True)

    # Normalize label names (optional safety)
    all_df["label"] = all_df["label"].astype(str).str.strip().str.lower()
    all_df["session_id"] = all_df["session_id"].astype(str).str.strip()

    return all_df


# PREPROCESS

def trim_active_segment(df, use="gyro", std_mult=0.75, pad=10):
    """
    Keep active region using a magnitude threshold.
    We will NOT use this for idle (idle should remain idle).
    """
    df = df.sort_values("t_ms").reset_index(drop=True)
    x = df[SENSOR_COLS].to_numpy(dtype=float)

    if use == "gyro":
        mag = np.linalg.norm(x[:, 3:6], axis=1)
    else:
        mag = np.linalg.norm(x[:, 0:3], axis=1)

    thr = mag.mean() + std_mult * mag.std()
    idx = np.where(mag > thr)[0]
    if len(idx) == 0:
        return df

    start = max(0, idx[0] - pad)
    end = min(len(df) - 1, idx[-1] + pad)
    return df.iloc[start:end + 1].reset_index(drop=True)

def resample_fixed_dt(df, dt_ms=15):
    """
    Resample to fixed dt using interpolation; remove duplicate timestamps
    (BLE burst timestamps).
    """
    df = df.sort_values("t_ms").reset_index(drop=True).copy()

    t = df["t_ms"].to_numpy(dtype=float)
    t0 = t[0]
    t = t - t0

    # remove duplicate timestamps
    _, uniq_idx = np.unique(t, return_index=True)
    df = df.iloc[uniq_idx].reset_index(drop=True)
    t = df["t_ms"].to_numpy(dtype=float) - t0

    if len(t) < 2:
        return df

    new_t = np.arange(0, t[-1] + 1, dt_ms)

    out = pd.DataFrame({"t_ms": new_t + t0})
    for col in SENSOR_COLS:
        out[col] = np.interp(new_t, t, df[col].to_numpy(dtype=float))

    out["label"] = df["label"].iloc[0]
    out["session_id"] = df["session_id"].iloc[0]
    out["source_file"] = df["source_file"].iloc[0]
    return out


# FEATURES (robust)

def high_pass_mag(acc_mag, kernel=15):
    """Remove slow trend (gravity-ish) using moving average."""
    if len(acc_mag) < kernel:
        return acc_mag - acc_mag.mean()
    k = np.ones(kernel) / kernel
    trend = np.convolve(acc_mag, k, mode="same")
    return acc_mag - trend

def window_features_from_recording(df, window_samples=40, overlap=0.75, max_windows=20):
    """
    Features:
      - gyro axes (gx,gy,gz)
      - gyro magnitude
      - accel magnitude
      - high-pass accel magnitude
    Stats per window: mean/std/min/max/range/rms + slope
    """
    step = max(1, int(window_samples * (1 - overlap)))

    x = df[SENSOR_COLS].to_numpy(dtype=float)
    acc = x[:, 0:3]
    gyr = x[:, 3:6]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    acc_hp = high_pass_mag(acc_mag, kernel=HP_GRAVITY_KERNEL)

    sig = np.column_stack([gyr, gyr_mag, acc_mag, acc_hp])  # 6 channels

    feats = []
    for start in range(0, len(sig) - window_samples + 1, step):
        w = sig[start:start + window_samples]

        means = w.mean(axis=0)
        stds  = w.std(axis=0)
        mins  = w.min(axis=0)
        maxs  = w.max(axis=0)
        rng   = maxs - mins
        rms   = np.sqrt(np.mean(w ** 2, axis=0))
        slope = w[-1] - w[0]

        feats.append(np.concatenate([means, stds, mins, maxs, rng, rms, slope]))

    if not feats:
        return None

    # Limit windows per file (prevents domination by long/strong gestures)
    if len(feats) > max_windows:
        idx = np.linspace(0, len(feats) - 1, max_windows, dtype=int)
        feats = [feats[i] for i in idx]

    X = np.vstack(feats)
    y = np.array([df["label"].iloc[0]] * len(X))
    g = np.array([df["session_id"].iloc[0]] * len(X))  # session grouping
    return X, y, g


# DATASET BUILD

def build_dataset(all_df):
    X_list, y_list, g_list = [], [], []
    counts = {}

    for src, df in all_df.groupby("source_file"):
        label = str(df["label"].iloc[0]).strip().lower()

        # ✅ Don't trim idle; keep it as-is
        if label != "idle":
            df = trim_active_segment(df, use=TRIM_USE, std_mult=TRIM_STD_MULT, pad=TRIM_PAD_SAMPLES)

        df = resample_fixed_dt(df, dt_ms=RESAMPLE_DT_MS)

        out = window_features_from_recording(
            df,
            window_samples=WINDOW_SAMPLES,
            overlap=WINDOW_OVERLAP,
            max_windows=MAX_WINDOWS_PER_FILE
        )
        if out is None:
            continue

        X, y, g = out
        X_list.append(X)
        y_list.append(y)
        g_list.append(g)

        counts[label] = counts.get(label, 0) + len(y)

    if not X_list:
        raise RuntimeError("No windows produced. Try lowering WINDOW_SAMPLES or trimming less.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)

    print("\nWindows per label (after per-file cap):")
    for k in sorted(counts.keys()):
        print(f"  {k:<6}: {counts[k]}")

    return X_all, y_all, g_all


# BALANCING
def balance_by_class(X, y, groups, mode="oversample_to_max", seed=42):
    rng = np.random.default_rng(seed)
    labels = np.unique(y)

    idx_by_label = {lab: np.where(y == lab)[0] for lab in labels}
    counts = {lab: len(idx) for lab, idx in idx_by_label.items()}

    print("\nBefore class-balance:")
    for lab in sorted(counts.keys()):
        print(f"  {lab:<6}: {counts[lab]}")

    if mode == "downsample_to_min":
        target = min(counts.values())
        keep = []
        for lab in labels:
            idx = idx_by_label[lab]
            pick = rng.choice(idx, size=target, replace=False)
            keep.append(pick)
        keep = np.concatenate(keep)

    elif mode == "oversample_to_max":
        target = max(counts.values())
        keep = []
        for lab in labels:
            idx = idx_by_label[lab]
            # oversample minority classes with replacement
            pick = rng.choice(idx, size=target, replace=True)
            keep.append(pick)
        keep = np.concatenate(keep)

    else:
        raise ValueError("mode must be 'oversample_to_max' or 'downsample_to_min'")

    # Shuffle final indices
    keep = rng.permutation(keep)

    Xb = X[keep]
    yb = y[keep]
    gb = groups[keep]

    # Show after
    after_counts = {lab: int((yb == lab).sum()) for lab in np.unique(yb)}
    print("\nAfter class-balance (mode=%s, target=%d):" % (mode, target))
    for lab in sorted(after_counts.keys()):
        print(f"  {lab:<6}: {after_counts[lab]}")

    return Xb, yb, gb

# =========================
# TRAIN / TEST
# =========================
from sklearn.model_selection import train_test_split

def main():
    print("Loading:", DATA_ROOT)
    all_df = load_all_csvs(DATA_ROOT)

    print("Total files:", all_df["source_file"].nunique())
    print("Labels found:", sorted(all_df["label"].unique().tolist()))
    print("Sessions:", sorted(all_df["session_id"].unique().tolist()))

    X, y, groups = build_dataset(all_df)
    
    print("\nTotal windows before split:", len(y))

    # 1. SPLIT FIRST (Using standard stratify to ensure all classes are represented)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,     # Hold out 30% for testing
        stratify=y,        # CRITICAL: Ensures equal percentage of all gestures in train/test
        random_state=42
    )

    # 2. BALANCE SECOND (Only balance the training data!)
    print("\nBalancing Training Data...")
    # We pass dummy groups here because we aren't using GroupSplit anymore
    dummy_groups = np.zeros(len(y_train)) 
    X_train, y_train, _ = balance_by_class(X_train, y_train, dummy_groups, mode=BALANCE_MODE, seed=RANDOM_SEED)

    print("\nTraining Feature dim:", X_train.shape[1])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=900,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("\nTesting...")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Labels:", labels)
    print("Confusion matrix:\n", cm)

    try:
        import joblib
        joblib.dump(model, "gesture_rf.joblib")
        print("\nSaved model: gesture_rf.joblib")
    except Exception as e:
        print("\nCould not save model:", e)

if __name__ == "__main__":
    main()
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# PATH
DATA_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "IMU_Gesture_Data")

# SETTINGS (good defaults)
SENSOR_COLS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

RESAMPLE_DT_MS = 15

WINDOW_SAMPLES = 40          # 0.6s windows at 15ms
WINDOW_OVERLAP = 0.75
HP_GRAVITY_KERNEL = 15

# Keep windows per file roughly equal so long clips don't dominate
MAX_WINDOWS_PER_FILE = 20

# Trimming: DO NOT trim idle (we keep it as-is)
TRIM_USE = "gyro"
TRIM_STD_MULT = 0.75
TRIM_PAD_SAMPLES = 10

# Balancing behavior:
# - "oversample_to_max": bring every class up to the max class count (with replacement)
# - "downsample_to_min": cut every class down to the min class count (no replacement)
BALANCE_MODE = "oversample_to_max"
RANDOM_SEED = 42


# LOAD

def load_all_csvs(root):
    paths = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    dfs = []

    for p in paths:
        df = pd.read_csv(p)
        need = ["t_ms"] + SENSOR_COLS
        if any(c not in df.columns for c in need):
            continue

        parts = os.path.normpath(p).split(os.sep)
        inferred_label = parts[-3] if len(parts) >= 3 else "unknown"
        inferred_sess = parts[-2] if len(parts) >= 2 else "session1"

        if "label" not in df.columns:
            df["label"] = inferred_label
        if "session_id" not in df.columns:
            df["session_id"] = inferred_sess

        df["source_file"] = p
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid CSVs found under: {root}")

    all_df = pd.concat(dfs, ignore_index=True)

    # Normalize label names (optional safety)
    all_df["label"] = all_df["label"].astype(str).str.strip().str.lower()
    all_df["session_id"] = all_df["session_id"].astype(str).str.strip()

    return all_df


# PREPROCESS

def trim_active_segment(df, use="gyro", std_mult=0.75, pad=10):
    """
    Keep active region using a magnitude threshold.
    We will NOT use this for idle (idle should remain idle).
    """
    df = df.sort_values("t_ms").reset_index(drop=True)
    x = df[SENSOR_COLS].to_numpy(dtype=float)

    if use == "gyro":
        mag = np.linalg.norm(x[:, 3:6], axis=1)
    else:
        mag = np.linalg.norm(x[:, 0:3], axis=1)

    thr = mag.mean() + std_mult * mag.std()
    idx = np.where(mag > thr)[0]
    if len(idx) == 0:
        return df

    start = max(0, idx[0] - pad)
    end = min(len(df) - 1, idx[-1] + pad)
    return df.iloc[start:end + 1].reset_index(drop=True)

def resample_fixed_dt(df, dt_ms=15):
    """
    Resample to fixed dt using interpolation; remove duplicate timestamps
    (BLE burst timestamps).
    """
    df = df.sort_values("t_ms").reset_index(drop=True).copy()

    t = df["t_ms"].to_numpy(dtype=float)
    t0 = t[0]
    t = t - t0

    # remove duplicate timestamps
    _, uniq_idx = np.unique(t, return_index=True)
    df = df.iloc[uniq_idx].reset_index(drop=True)
    t = df["t_ms"].to_numpy(dtype=float) - t0

    if len(t) < 2:
        return df

    new_t = np.arange(0, t[-1] + 1, dt_ms)

    out = pd.DataFrame({"t_ms": new_t + t0})
    for col in SENSOR_COLS:
        out[col] = np.interp(new_t, t, df[col].to_numpy(dtype=float))

    out["label"] = df["label"].iloc[0]
    out["session_id"] = df["session_id"].iloc[0]
    out["source_file"] = df["source_file"].iloc[0]
    return out


# FEATURES (robust)

def high_pass_mag(acc_mag, kernel=15):
    """Remove slow trend (gravity-ish) using moving average."""
    if len(acc_mag) < kernel:
        return acc_mag - acc_mag.mean()
    k = np.ones(kernel) / kernel
    trend = np.convolve(acc_mag, k, mode="same")
    return acc_mag - trend

def window_features_from_recording(df, window_samples=40, overlap=0.75, max_windows=20):
    """
    Features:
      - gyro axes (gx,gy,gz)
      - gyro magnitude
      - accel magnitude
      - high-pass accel magnitude
    Stats per window: mean/std/min/max/range/rms + slope
    """
    step = max(1, int(window_samples * (1 - overlap)))

    x = df[SENSOR_COLS].to_numpy(dtype=float)
    acc = x[:, 0:3]
    gyr = x[:, 3:6]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    acc_hp = high_pass_mag(acc_mag, kernel=HP_GRAVITY_KERNEL)

    sig = np.column_stack([gyr, gyr_mag, acc_mag, acc_hp])  # 6 channels

    feats = []
    for start in range(0, len(sig) - window_samples + 1, step):
        w = sig[start:start + window_samples]

        means = w.mean(axis=0)
        stds  = w.std(axis=0)
        mins  = w.min(axis=0)
        maxs  = w.max(axis=0)
        rng   = maxs - mins
        rms   = np.sqrt(np.mean(w ** 2, axis=0))
        slope = w[-1] - w[0]

        feats.append(np.concatenate([means, stds, mins, maxs, rng, rms, slope]))

    if not feats:
        return None

    # Limit windows per file (prevents domination by long/strong gestures)
    if len(feats) > max_windows:
        idx = np.linspace(0, len(feats) - 1, max_windows, dtype=int)
        feats = [feats[i] for i in idx]

    X = np.vstack(feats)
    y = np.array([df["label"].iloc[0]] * len(X))
    g = np.array([df["session_id"].iloc[0]] * len(X))  # session grouping
    return X, y, g


# DATASET BUILD

def build_dataset(all_df):
    X_list, y_list, g_list = [], [], []
    counts = {}

    for src, df in all_df.groupby("source_file"):
        label = str(df["label"].iloc[0]).strip().lower()

        # ✅ Don't trim idle; keep it as-is
        if label != "idle":
            df = trim_active_segment(df, use=TRIM_USE, std_mult=TRIM_STD_MULT, pad=TRIM_PAD_SAMPLES)

        df = resample_fixed_dt(df, dt_ms=RESAMPLE_DT_MS)

        out = window_features_from_recording(
            df,
            window_samples=WINDOW_SAMPLES,
            overlap=WINDOW_OVERLAP,
            max_windows=MAX_WINDOWS_PER_FILE
        )
        if out is None:
            continue

        X, y, g = out
        X_list.append(X)
        y_list.append(y)
        g_list.append(g)

        counts[label] = counts.get(label, 0) + len(y)

    if not X_list:
        raise RuntimeError("No windows produced. Try lowering WINDOW_SAMPLES or trimming less.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)

    print("\nWindows per label (after per-file cap):")
    for k in sorted(counts.keys()):
        print(f"  {k:<6}: {counts[k]}")

    return X_all, y_all, g_all


# BALANCING
def balance_by_class(X, y, groups, mode="oversample_to_max", seed=42):
    rng = np.random.default_rng(seed)
    labels = np.unique(y)

    idx_by_label = {lab: np.where(y == lab)[0] for lab in labels}
    counts = {lab: len(idx) for lab, idx in idx_by_label.items()}

    print("\nBefore class-balance:")
    for lab in sorted(counts.keys()):
        print(f"  {lab:<6}: {counts[lab]}")

    if mode == "downsample_to_min":
        target = min(counts.values())
        keep = []
        for lab in labels:
            idx = idx_by_label[lab]
            pick = rng.choice(idx, size=target, replace=False)
            keep.append(pick)
        keep = np.concatenate(keep)

    elif mode == "oversample_to_max":
        target = max(counts.values())
        keep = []
        for lab in labels:
            idx = idx_by_label[lab]
            # oversample minority classes with replacement
            pick = rng.choice(idx, size=target, replace=True)
            keep.append(pick)
        keep = np.concatenate(keep)

    else:
        raise ValueError("mode must be 'oversample_to_max' or 'downsample_to_min'")

    # Shuffle final indices
    keep = rng.permutation(keep)

    Xb = X[keep]
    yb = y[keep]
    gb = groups[keep]

    # Show after
    after_counts = {lab: int((yb == lab).sum()) for lab in np.unique(yb)}
    print("\nAfter class-balance (mode=%s, target=%d):" % (mode, target))
    for lab in sorted(after_counts.keys()):
        print(f"  {lab:<6}: {after_counts[lab]}")

    return Xb, yb, gb

# =========================
# TRAIN / TEST
# =========================
from sklearn.model_selection import train_test_split

def main():
    print("Loading:", DATA_ROOT)
    all_df = load_all_csvs(DATA_ROOT)

    print("Total files:", all_df["source_file"].nunique())
    print("Labels found:", sorted(all_df["label"].unique().tolist()))
    print("Sessions:", sorted(all_df["session_id"].unique().tolist()))

    X, y, groups = build_dataset(all_df)
    
    print("\nTotal windows before split:", len(y))

    # 1. SPLIT FIRST (Using standard stratify to ensure all classes are represented)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,     # Hold out 30% for testing
        stratify=y,        # CRITICAL: Ensures equal percentage of all gestures in train/test
        random_state=42
    )

    # 2. BALANCE SECOND (Only balance the training data!)
    print("\nBalancing Training Data...")
    # We pass dummy groups here because we aren't using GroupSplit anymore
    dummy_groups = np.zeros(len(y_train)) 
    X_train, y_train, _ = balance_by_class(X_train, y_train, dummy_groups, mode=BALANCE_MODE, seed=RANDOM_SEED)

    print("\nTraining Feature dim:", X_train.shape[1])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=900,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("\nTesting...")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Labels:", labels)
    print("Confusion matrix:\n", cm)

    try:
        import joblib
        joblib.dump(model, "gesture_rf.joblib")
        print("\nSaved model: gesture_rf.joblib")
    except Exception as e:
        print("\nCould not save model:", e)

if __name__ == "__main__":
    main()