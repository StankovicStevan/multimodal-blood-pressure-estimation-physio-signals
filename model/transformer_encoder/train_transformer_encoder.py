import math
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, concatenate, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ================================
# Config and paths
# ================================
WINDOW_SEC = 8
TAG = f"{WINDOW_SEC}s"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


# ================================
# Data loading
# ================================
def load_data(tag: str):
    """Load signal tensor, tabular features and labels."""
    X = np.load(f"../../datapreprocessing/preprocessed/X_signal_{tag}.npy")
    XT = np.load(f"../../datapreprocessing/preprocessed/X_tabular_{tag}.npy")
    y = np.load(f"../../datapreprocessing/preprocessed/y_{tag}.npy")
    return X, XT, y


# ================================
# CNN feature extractor
# ================================
def cnn_encoder(x_in):
    """
    A lightweight CNN encoder that reduces temporal resolution but keeps the sequence.
    Output: (batch, time/4, channels)
    """
    x = Conv1D(32, 7, padding='same', activation='relu')(x_in)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    return x


# ================================
# Build transformer fusion model
# ================================
def build_model(input_len, tab_dim):
    """Build a CNN + Transformer encoder fusion model."""
    # Signal inputs
    inp_ppg = Input(shape=(input_len, 1), name="ppg_in")
    inp_ecg = Input(shape=(input_len, 1), name="ecg_in")
    inp_pcg = Input(shape=(input_len, 1), name="pcg_in")
    inp_fsr = Input(shape=(input_len, 1), name="fsr_in")

    # CNN encoders
    e_ppg = cnn_encoder(inp_ppg)
    e_ecg = cnn_encoder(inp_ecg)
    e_pcg = cnn_encoder(inp_pcg)
    e_fsr = cnn_encoder(inp_fsr)

    # Feature fusion (concat along channels)
    fusion = concatenate([e_ppg, e_ecg, e_pcg, e_fsr], axis=-1, name="fusion_concat")

    # Transformer encoder
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(fusion, fusion)
    attn = LayerNormalization()(attn)

    ff = Dense(128, activation='relu')(attn)
    ff = Dense(128)(ff)
    ff = LayerNormalization()(ff)

    # Sequence → fixed representation
    rep = GlobalAveragePooling1D(name="global_pool")(ff)

    # Tabular branch
    inp_tab = Input(shape=(tab_dim,), name="tab_in")
    t = Dense(16, activation='relu')(inp_tab)
    t = Dropout(0.2)(t)

    # Late fusion
    h = concatenate([rep, t], name="late_fusion")
    h = Dense(64, activation='relu')(h)
    h = Dropout(0.3)(h)

    out = Dense(2, name="bp_out")(h)

    model = Model(
        inputs=[inp_ppg, inp_ecg, inp_pcg, inp_fsr, inp_tab],
        outputs=out,
        name="transformer_encoder_model"
    )
    model.compile(optimizer=Adam(1e-4), loss='mae', metrics=['mae'])

    return model


# ================================
# Train/Validate/Test utilities
# ================================
def split_data(PPG, ECG, PCG, FSR, XT, y):
    """70% train, 15% val, 15% test."""
    PPG_tr, PPG_tmp, ECG_tr, ECG_tmp, PCG_tr, PCG_tmp, FSR_tr, FSR_tmp, XT_tr, XT_tmp, y_tr, y_tmp = \
        train_test_split(PPG, ECG, PCG, FSR, XT, y, test_size=0.30, random_state=42)

    PPG_va, PPG_te, ECG_va, ECG_te, PCG_va, PCG_te, FSR_va, FSR_te, XT_va, XT_te, y_va, y_te = \
        train_test_split(PPG_tmp, ECG_tmp, PCG_tmp, FSR_tmp, XT_tmp, y_tmp, test_size=0.50, random_state=42)

    print(f"Train={PPG_tr.shape[0]}, Val={PPG_va.shape[0]}, Test={PPG_te.shape[0]}")

    return (PPG_tr, PPG_va, PPG_te,
            ECG_tr, ECG_va, ECG_te,
            PCG_tr, PCG_va, PCG_te,
            FSR_tr, FSR_va, FSR_te,
            XT_tr, XT_va, XT_te,
            y_tr, y_va, y_te)


def compute_metrics(y_true, y_pred):
    sbp_true, dbp_true = y_true[:, 0], y_true[:, 1]
    sbp_pred, dbp_pred = y_pred[:, 0], y_pred[:, 1]

    sbp_mae = mean_absolute_error(sbp_true, sbp_pred)
    dbp_mae = mean_absolute_error(dbp_true, dbp_pred)
    sbp_rmse = math.sqrt(mean_squared_error(sbp_true, sbp_pred))
    dbp_rmse = math.sqrt(mean_squared_error(dbp_true, dbp_pred))

    print("\n===== Transformer Fusion Results =====")
    print(f"SBP MAE  : {sbp_mae:.3f}")
    print(f"DBP MAE  : {dbp_mae:.3f}")
    print(f"SBP RMSE : {sbp_rmse:.3f}")
    print(f"DBP RMSE : {dbp_rmse:.3f}")


def plot_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f"Training Curve (Transformer Encoder, {TAG})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    plt.savefig(RESULTS_DIR / f"loss_curve_transformer_{TAG}.png", dpi=150, bbox_inches="tight")
    print("✅ Saved loss curve")


def plot_scatter(y_true, y_pred):
    sbp_true, dbp_true = y_true[:, 0], y_true[:, 1]
    sbp_pred, dbp_pred = y_pred[:, 0], y_pred[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(sbp_true, sbp_pred, alpha=0.6, label="SBP")
    plt.scatter(dbp_true, dbp_pred, alpha=0.6, label="DBP")
    plt.plot([60, 200], [60, 200], 'r--', label="Ideal")
    plt.xlabel("True BP")
    plt.ylabel("Predicted BP")
    plt.title(f"Prediction Scatter — Transformer ({TAG})")
    plt.legend()
    plt.grid()
    plt.savefig(RESULTS_DIR / f"scatter_transformer_{TAG}.png", dpi=150, bbox_inches="tight")
    print("✅ Saved scatter plot")


# ================================
# Main function
# ================================
def main():
    # Load data
    X, XT, y = load_data(TAG)

    # Split modalities
    PPG = X[:, :, 0:1]
    ECG = X[:, :, 1:2]
    PCG = X[:, :, 2:3]
    FSR = X[:, :, 3:4]

    # Split into train/val/test
    (PPG_tr, PPG_va, PPG_te,
     ECG_tr, ECG_va, ECG_te,
     PCG_tr, PCG_va, PCG_te,
     FSR_tr, FSR_va, FSR_te,
     XT_tr, XT_va, XT_te,
     y_tr, y_va, y_te) = split_data(PPG, ECG, PCG, FSR, XT, y)

    # Build model
    model = build_model(input_len=PPG.shape[1], tab_dim=XT.shape[1])
    model.summary()

    # Train
    history = model.fit(
        [PPG_tr, ECG_tr, PCG_tr, FSR_tr, XT_tr], y_tr,
        validation_data=([PPG_va, ECG_va, PCG_va, FSR_va, XT_va], y_va),
        epochs=50,
        batch_size=8,
        verbose=1
    )

    # Test
    loss, mae = model.evaluate([PPG_te, ECG_te, PCG_te, FSR_te, XT_te], y_te, verbose=1)
    print(f"\n✅ Test MAE: {mae:.3f}")

    y_pred = model.predict([PPG_te, ECG_te, PCG_te, FSR_te, XT_te], verbose=0)

    # Save model and predictions
    model.save(MODELS_DIR / f"transformer_encoder_{TAG}.h5")
    np.save(RESULTS_DIR / f"y_pred_transformer_{TAG}.npy", y_pred)
    np.save(RESULTS_DIR / f"y_test_transformer_{TAG}.npy", y_te)

    # Metrics + plots
    compute_metrics(y_te, y_pred)
    plot_curves(history)
    plot_scatter(y_te, y_pred)

    print("\n✅ ALL DONE — Transformer model trained, evaluated, plotted, saved.")


if __name__ == "__main__":
    main()
