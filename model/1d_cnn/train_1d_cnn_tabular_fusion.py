from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import Model, Input
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf


# ==============================
# Constants and paths
# ==============================

DEFAULT_WINDOW_SEC = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 50
RESULTS_DIR = Path("results_1d_cnn")
DATA_DIR = Path("../../datapreprocessing/preprocessed")


# ==============================
# Helper functions
# ==============================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_tag(window_sec: int, augmented: bool = True) -> str:
    """Build filename tag, e.g. '8s_augmented' or '8s'."""
    return f"{window_sec}s{'_augmented' if augmented else ''}"


def load_data(tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X_signal, X_tabular and y from .npy files.
    y is expected to be of shape (N, 2) → [SBP, DBP].
    """
    X_sig = np.load(DATA_DIR / f"X_signal_{tag}.npy")
    X_tab = np.load(DATA_DIR / f"X_tabular_{tag}.npy")
    y = np.load(DATA_DIR / f"y_{tag}.npy")

    print("Loaded shapes:")
    print("  Signal :", X_sig.shape)
    print("  Tabular:", X_tab.shape)
    print("  Labels :", y.shape)

    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y must be (N, 2) for [SBP, DBP], got {y.shape}")

    return X_sig, X_tab, y


def split_data(
    X_signal: np.ndarray,
    X_tabular: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    val_share_of_temp: float = 0.50,
    seed: int = 42,
):
    """
    Split data into 70% train, 15% validation, 15% test.
    (two-step split: train+temp → val+test)
    """
    X_sig_train, X_sig_temp, X_tab_train, X_tab_temp, y_train, y_temp = train_test_split(
        X_signal, X_tabular, y, test_size=test_size, random_state=seed
    )

    X_sig_val, X_sig_test, X_tab_val, X_tab_test, y_val, y_test = train_test_split(
        X_sig_temp, X_tab_temp, y_temp, test_size=val_share_of_temp, random_state=seed
    )

    print("\nSplit sizes:")
    print(f"  Train: {X_sig_train.shape[0]}")
    print(f"  Val  : {X_sig_val.shape[0]}")
    print(f"  Test : {X_sig_test.shape[0]}")

    return (X_sig_train, X_sig_val, X_sig_test,
            X_tab_train, X_tab_val, X_tab_test,
            y_train, y_val, y_test)


def build_model(
    signal_shape: Tuple[int, int],
    tabular_dim: int,
    conv_filters: Tuple[int, int, int] = (32, 64, 128),
    kernels: Tuple[int, int, int] = (7, 5, 5),
    dense_tab: int = 16,
    dense_final: int = 64,
    dropout_tab: float = 0.20,
    dropout_final: float = 0.30,
    lr: float = 1e-3,
) -> Model:
    """
    1D-CNN branch for signal + MLP branch for tabular features → combined regression head (SBP, DBP).
    """
    # Signal branch
    sig_in = Input(shape=signal_shape, name="signal_input")
    x = layers.Conv1D(conv_filters[0], kernels[0], activation='relu')(sig_in)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(conv_filters[1], kernels[1], activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(conv_filters[2], kernels[2], activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Tabular branch
    tab_in = Input(shape=(tabular_dim,), name="tabular_input")
    t = layers.Dense(dense_tab, activation='relu')(tab_in)
    t = layers.Dropout(dropout_tab)(t)

    # Combine both
    combined = layers.Concatenate()([x, t])
    z = layers.Dense(dense_final, activation='relu')(combined)
    z = layers.Dropout(dropout_final)(z)
    out = layers.Dense(2, name="bp_output")(z)

    model = Model([sig_in, tab_in], out, name="cnn1d_tabular_bp")
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss='mae',
        metrics=['mae']
    )
    return model


def train_model(
    model: Model,
    X_sig_train: np.ndarray, X_tab_train: np.ndarray, y_train: np.ndarray,
    X_sig_val: np.ndarray,   X_tab_val: np.ndarray,   y_val: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    out_dir: Path = RESULTS_DIR,
):
    """
    Train with common callbacks: early stopping, LR scheduler, checkpoint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.keras"

    callbacks = [
        EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_mae", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        [X_sig_train, X_tab_train], y_train,
        validation_data=([X_sig_val, X_tab_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )
    return history


def evaluate_and_report(model: Model,
                        X_sig_test: np.ndarray, X_tab_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate on test set + compute SBP/DBP MAE & RMSE manually.
    """
    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate([X_sig_test, X_tab_test], y_test, verbose=1)
    print(f"✅ Test MAE (Keras metric): {test_mae:.3f}")

    y_pred = model.predict([X_sig_test, X_tab_test], verbose=0)

    sbp_true, dbp_true = y_test[:, 0], y_test[:, 1]
    sbp_pred, dbp_pred = y_pred[:, 0], y_pred[:, 1]

    sbp_mae = mean_absolute_error(sbp_true, sbp_pred)
    dbp_mae = mean_absolute_error(dbp_true, dbp_pred)
    sbp_rmse = math.sqrt(mean_squared_error(sbp_true, sbp_pred))
    dbp_rmse = math.sqrt(mean_squared_error(dbp_true, dbp_pred))

    print("\n===== Test Metrics =====")
    print(f"SBP MAE  : {sbp_mae:.3f}")
    print(f"DBP MAE  : {dbp_mae:.3f}")
    print(f"SBP RMSE : {sbp_rmse:.3f}")
    print(f"DBP RMSE : {dbp_rmse:.3f}")

    return y_pred, {
        "sbp_mae": sbp_mae,
        "dbp_mae": dbp_mae,
        "sbp_rmse": sbp_rmse,
        "dbp_rmse": dbp_rmse,
        "keras_test_mae": test_mae
    }


def save_outputs(y_pred: np.ndarray, y_test: np.ndarray, tag: str, out_dir: Path = RESULTS_DIR) -> None:
    """Save predicted and true labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"y_pred_{tag}.npy", y_pred)
    np.save(out_dir / f"y_test_{tag}.npy", y_test)
    print(f"✅ Saved predictions to: {out_dir}")


def plot_training(history, tag: str, out_dir: Path = RESULTS_DIR) -> None:
    """Plot train/val MAE curve."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f"Training Curve ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = out_dir / f"loss_curve_{tag}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved loss curve: {path}")


def plot_scatter(y_test: np.ndarray, y_pred: np.ndarray, tag: str, out_dir: Path = RESULTS_DIR) -> None:
    """Scatter plot: True vs Predicted SBP/DBP."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sbp_true, dbp_true = y_test[:, 0], y_test[:, 1]
    sbp_pred, dbp_pred = y_pred[:, 0], y_pred[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(sbp_true, sbp_pred, alpha=0.6, label="SBP")
    plt.scatter(dbp_true, dbp_pred, alpha=0.6, label="DBP")
    lo, hi = 60, 200
    plt.plot([lo, hi], [lo, hi], 'r--', label="Ideal")
    plt.xlabel("True BP")
    plt.ylabel("Predicted BP")
    plt.title(f"Prediction Scatter — True vs Pred ({tag})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = out_dir / f"scatter_{tag}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved scatter plot: {path}")


# ==============================
# Main function
# ==============================

def main():
    parser = argparse.ArgumentParser(description="1D-CNN + Tabular for BP estimation")
    parser.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC, help="Window length in seconds")
    parser.add_argument("--no-aug", action="store_true", help="Use non-augmented tag")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)

    tag = make_tag(args.window_sec, augmented=not args.no_aug)
    print(f"Using tag: {tag}")

    X_signal, X_tabular, y = load_data(tag)
    (X_sig_train, X_sig_val, X_sig_test,
     X_tab_train, X_tab_val, X_tab_test,
     y_train, y_val, y_test) = split_data(X_signal, X_tabular, y)

    model = build_model(
        signal_shape=(X_signal.shape[1], X_signal.shape[2]),
        tabular_dim=X_tabular.shape[1],
        lr=args.lr
    )

    print("\nModel Summary:")
    model.summary()

    history = train_model(
        model,
        X_sig_train, X_tab_train, y_train,
        X_sig_val, X_tab_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=RESULTS_DIR
    )

    y_pred, metrics = evaluate_and_report(model, X_sig_test, X_tab_test, y_test)

    save_outputs(y_pred, y_test, tag, RESULTS_DIR)
    plot_training(history, tag, RESULTS_DIR)
    plot_scatter(y_test, y_pred, tag, RESULTS_DIR)

    print("\n✅ ALL DONE — Model trained, tested, evaluated, and results saved!")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
