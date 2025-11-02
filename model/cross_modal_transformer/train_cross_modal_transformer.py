import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, concatenate, LayerNormalization, Add
)
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Layer


# ================================
# Paths and settings
# ================================
WINDOW_SEC = 8
TAG = f"{WINDOW_SEC}s"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


# ================================
# Positional Encoding Layer
# ================================
class PositionalEncoding(Layer):
    """Standard sine/cosine positional encoding added to temporal CNN features."""
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]

        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) *
                          (-tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))

        pe_cos = tf.cos(position * div_term)
        pe_sin = tf.sin(position * div_term)

        pe = tf.reshape(tf.stack([pe_sin, pe_cos], axis=-1), (seq_len, -1))
        pe = pe[:, :d_model]              # trim if odd channels
        pe = tf.expand_dims(pe, 0)        # batch dimension

        return x + pe


# ================================
# Model building blocks
# ================================
def cnn_encoder(inp):
    """3-layer CNN encoder for 1D biosignal."""
    x = Conv1D(32, 7, padding='same', activation='relu')(inp)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    return x


def cross_attend(query_seq, key_value_seq, name=None):
    """
    Cross-attention: query from one modality, key/value from another.
    Residual + LayerNorm for transformer-style stability.
    """
    attn = MultiHeadAttention(num_heads=4, key_dim=32, name=name)(query_seq, key_value_seq)
    x = Add()([query_seq, attn])
    x = LayerNormalization()(x)
    return x


def transformer_block(x, prefix, mlp_dim=256):
    """
    Standard transformer block: self-attention + MLP
    with residual connections and normalization.
    """
    attn = MultiHeadAttention(num_heads=4, key_dim=32, name=f"{prefix}_mha")(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    ff = Dense(mlp_dim, activation='relu', name=f"{prefix}_ff1")(x)
    ff = Dense(mlp_dim, name=f"{prefix}_ff2")(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    return x


# ================================
# Build Cross-Modal Transformer Model
# ================================
def build_crossmodal_model(input_length, tab_dim):
    # Inputs
    inp_ppg = Input(shape=(input_length, 1), name="ppg")
    inp_ecg = Input(shape=(input_length, 1), name="ecg")
    inp_pcg = Input(shape=(input_length, 1), name="pcg")
    inp_fsr = Input(shape=(input_length, 1), name="fsr")
    inp_tab = Input(shape=(tab_dim,), name="tab")

    # CNN encoders
    ppg_enc = cnn_encoder(inp_ppg)
    ecg_enc = cnn_encoder(inp_ecg)
    pcg_enc = cnn_encoder(inp_pcg)
    fsr_enc = cnn_encoder(inp_fsr)

    # Add positional encoding
    pos = PositionalEncoding()
    ppg_enc = pos(ppg_enc)
    ecg_enc = pos(ecg_enc)
    pcg_enc = pos(pcg_enc)
    fsr_enc = pos(fsr_enc)

    # Cross-modal attention (all → PPG, ECG, PCG)
    ppg_from_ecg = cross_attend(ppg_enc, ecg_enc, "ppg_from_ecg")
    ppg_from_pcg = cross_attend(ppg_enc, pcg_enc, "ppg_from_pcg")
    ppg_from_fsr = cross_attend(ppg_enc, fsr_enc, "ppg_from_fsr")

    ecg_from_ppg = cross_attend(ecg_enc, ppg_enc, "ecg_from_ppg")
    ecg_from_pcg = cross_attend(ecg_enc, pcg_enc, "ecg_from_pcg")
    ecg_from_fsr = cross_attend(ecg_enc, fsr_enc, "ecg_from_fsr")

    pcg_from_ppg = cross_attend(pcg_enc, ppg_enc, "pcg_from_ppg")
    pcg_from_ecg = cross_attend(pcg_enc, ecg_enc, "pcg_from_ecg")
    pcg_from_fsr = cross_attend(pcg_enc, fsr_enc, "pcg_from_fsr")

    # Concatenate all attended representations
    fusion = concatenate(
        [ppg_from_ecg, ppg_from_pcg, ppg_from_fsr,
         ecg_from_ppg, ecg_from_pcg, ecg_from_fsr,
         pcg_from_ppg, pcg_from_ecg, pcg_from_fsr],
        axis=-1,
        name="crossmodal_concat"
    )

    # Compress channels
    fusion = Conv1D(256, 1, activation='relu', padding='same')(fusion)

    # Transformer layers
    x = transformer_block(fusion, "tr1")
    x = transformer_block(x, "tr2")

    # Global representation
    rep = GlobalAveragePooling1D(name="global_pool")(x)

    # Tabular branch
    t = Dense(16, activation='relu')(inp_tab)
    t = Dropout(0.2)(t)

    # Late fusion
    h = concatenate([rep, t], name="late_fusion")
    h = Dense(64, activation='relu')(h)
    h = Dropout(0.3)(h)
    out = Dense(2, name="bp_out")(h)

    model = Model([inp_ppg, inp_ecg, inp_pcg, inp_fsr, inp_tab], out)
    model.compile(optimizer=Adam(1e-4), loss='mae', metrics=['mae'])
    return model


# ================================
# Training utilities
# ================================
def train_model(model, train_data, val_data):
    callbacks = [
        EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(filepath=MODELS_DIR / f"crossmodal_best_{TAG}.keras",
                        monitor="val_mae", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=60,
        batch_size=8,
        verbose=1,
        callbacks=callbacks
    )
    return history


def evaluate_model(model, test_data):
    loss, mae = model.evaluate(test_data[0], test_data[1], verbose=1)
    print(f"✅ Test MAE = {mae:.3f}")
    y_pred = model.predict(test_data[0], verbose=0)
    return y_pred


def compute_metrics(y_true, y_pred):
    sbp_t, dbp_t = y_true[:, 0], y_true[:, 1]
    sbp_p, dbp_p = y_pred[:, 0], y_pred[:, 1]

    sbp_mae = mean_absolute_error(sbp_t, sbp_p)
    dbp_mae = mean_absolute_error(dbp_t, dbp_p)
    sbp_rmse = math.sqrt(mean_squared_error(sbp_t, sbp_p))
    dbp_rmse = math.sqrt(mean_squared_error(dbp_t, dbp_p))

    print("\n===== Final Cross-Modal Transformer Results =====")
    print(f"SBP MAE  : {sbp_mae:.3f}")
    print(f"DBP MAE  : {dbp_mae:.3f}")
    print(f"SBP RMSE : {sbp_rmse:.3f}")
    print(f"DBP RMSE : {dbp_rmse:.3f}")

    return sbp_mae, dbp_mae, sbp_rmse, dbp_rmse


def plot_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label="Train MAE")
    plt.plot(history.history['val_mae'], label="Val MAE")
    plt.title(f"Training Curve — Cross-Modal Transformer ({TAG})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    path = RESULTS_DIR / f"loss_curve_crossmodal_{TAG}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("✅ Saved loss curve:", path)


def plot_scatter(y_true, y_pred):
    sbp_t, dbp_t = y_true[:, 0], y_true[:, 1]
    sbp_p, dbp_p = y_pred[:, 0], y_pred[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(sbp_t, sbp_p, alpha=0.6, label="SBP")
    plt.scatter(dbp_t, dbp_p, alpha=0.6, label="DBP")
    plt.plot([60, 200], [60, 200], 'r--', label="Ideal")
    plt.xlabel("True BP")
    plt.ylabel("Predicted BP")
    plt.title(f"Prediction Scatter — Cross-Modal ({TAG})")
    plt.legend()
    plt.grid()
    path = RESULTS_DIR / f"scatter_crossmodal_{TAG}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("✅ Saved scatter plot:", path)


# ================================
# Main script
# ================================
def main():
    # Load data
    X = np.load(f"../../datapreprocessing/preprocessed/X_signal_{TAG}.npy")  # (N, T, 4)
    XT = np.load(f"../../datapreprocessing/preprocessed/X_tabular_{TAG}.npy")  # (N, 3)
    y = np.load(f"../../datapreprocessing/preprocessed/y_{TAG}.npy")  # (N, 2)

    # Split into individual modalities
    PPG = X[:, :, 0:1]
    ECG = X[:, :, 1:2]
    PCG = X[:, :, 2:3]
    FSR = X[:, :, 3:4]

    # Train/Val/Test
    splits = train_test_split(PPG, ECG, PCG, FSR, XT, y, test_size=0.30, random_state=42)
    PPG_tr, PPG_tmp, ECG_tr, ECG_tmp, PCG_tr, PCG_tmp, FSR_tr, FSR_tmp, XT_tr, XT_tmp, y_tr, y_tmp = splits

    splits2 = train_test_split(PPG_tmp, ECG_tmp, PCG_tmp, FSR_tmp, XT_tmp, y_tmp, test_size=0.50, random_state=42)
    PPG_va, PPG_te, ECG_va, ECG_te, PCG_va, PCG_te, FSR_va, FSR_te, XT_va, XT_te, y_va, y_te = splits2

    print(f"Train={PPG_tr.shape[0]}, Val={PPG_va.shape[0]}, Test={PPG_te.shape[0]}")

    # Build model
    model = build_crossmodal_model(input_length=PPG.shape[1], tab_dim=XT.shape[1])
    model.summary()

    # Train
    history = train_model(
        model,
        train_data=([PPG_tr, ECG_tr, PCG_tr, FSR_tr, XT_tr], y_tr),
        val_data=([PPG_va, ECG_va, PCG_va, FSR_va, XT_va], y_va)
    )

    # Evaluate
    y_pred = evaluate_model(model, ([PPG_te, ECG_te, PCG_te, FSR_te, XT_te], y_te))
    np.save(RESULTS_DIR / f"y_pred_crossmodal_{TAG}.npy", y_pred)
    np.save(RESULTS_DIR / f"y_test_crossmodal_{TAG}.npy", y_te)
    model.save(MODELS_DIR / f"crossmodal_transformer_{TAG}.h5")

    # Metrics + plots
    compute_metrics(y_te, y_pred)
    plot_curves(history)
    plot_scatter(y_te, y_pred)

    print("\n✅ ALL DONE — Cross-Modal Transformer trained, evaluated, plotted, saved.")


if __name__ == "__main__":
    main()
