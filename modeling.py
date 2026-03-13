import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    joblib = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False


SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent
DATA_FULL = PROJECT_ROOT / "data" / "processed" / "merged_features.csv"
DATA_DEV = PROJECT_ROOT / "models" / "df_sample_5k.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def _params_to_json_serializable(obj):
    """Convert numpy types in param dicts for JSON save."""
    if isinstance(obj, dict):
        return {k: _params_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_params_to_json_serializable(x) for x in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj

# Columns that leak target information and should be dropped from
# baseline same-day predictions.
LEAKAGE_COLS = [
    # explicit future targets
    "mood_next_day",
    "stress_next_day",
    "energy_next_day",
    "user_id_te_mood_next_day",
    "user_id_te_stress_next_day",
    "user_id_te_energy_next_day",
    # mood / stress / energy history features
    "mood_lag1",
    "mood_lag2",
    "mood_roll_3",
    "stress_lag1",
    "stress_roll_3",
    "energy_lag1",
    "energy_roll_3",
]


def load_data(use_dev: bool) -> pd.DataFrame:
    data_path = DATA_DEV if use_dev else DATA_FULL
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(
        data_path,
        parse_dates=["date", "intervention_start", "intervention_end", "week_start"],
        low_memory=False,
    )
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


def prepare_xy_mood(df: pd.DataFrame, target: str = "mood_score") -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [
        "date",
        "user_id",
        "intervention_start",
        "intervention_end",
        "intervention_type",
        "intervention_intensity",
        "week_start",
    ]

    assert target in df.columns, f"{target} not found in df columns"

    X = df.drop(
        columns=drop_cols + [target, "health_twin_index", "stress_level"] + LEAKAGE_COLS,
        errors="ignore",
    )
    y = (
        df[target]
        .ffill()
        .bfill()
        .fillna(df[target].mean())
    )

    X_num = X.select_dtypes(include=[np.number]).copy()
    # Simple numeric imputation for safety
    X_num = X_num.fillna(X_num.median())

    print("Feature matrix (numeric) shape:", X_num.shape)
    print("Example features:", list(X_num.columns[:15]))
    return X_num, y


def train_baseline_models(
    X_num: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    rf_params: dict | None = None,
    gb_params: dict | None = None,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"Linear Regression — MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}")

    # Random Forest
    rf_kw = {"n_estimators": n_estimators, "random_state": SEED, "n_jobs": -1}
    if rf_params:
        rf_kw = {**rf_kw, **rf_params}
    rf = RandomForestRegressor(**rf_kw)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest — MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

    # Gradient Boosting (often better than RF on tabular targets like mood)
    gb_kw = {
        "n_estimators": min(150, n_estimators),
        "max_depth": 5,
        "learning_rate": 0.08,
        "random_state": SEED,
    }
    if gb_params:
        gb_kw = {**gb_kw, **gb_params}
    gb = GradientBoostingRegressor(**gb_kw)
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    print(f"Gradient Boosting — MSE: {mse_gb:.4f}, R²: {r2_gb:.4f}")

    # Simple ensemble: mean of LR, RF, GB (often improves mood R²)
    y_pred_ens = (y_pred_lr + y_pred_rf + y_pred_gb) / 3.0
    mse_ens = mean_squared_error(y_test, y_pred_ens)
    r2_ens = r2_score(y_test, y_pred_ens)
    print(f"Ensemble (LR+RF+GB) — MSE: {mse_ens:.4f}, R²: {r2_ens:.4f}")

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=X_num.columns).sort_values(
        ascending=False
    )
    top20 = feat_imp.head(20)
    print("\nTop 20 feature importances (Random Forest):")
    print(top20)

    # Save metrics and feature importances
    metrics = {
        "lr": {"mse": mse_lr, "r2": r2_lr},
        "rf": {"mse": mse_rf, "r2": r2_rf, "n_estimators": n_estimators},
        "gb": {"mse": mse_gb, "r2": r2_gb},
        "ensemble": {"mse": mse_ens, "r2": r2_ens},
    }
    (MODEL_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    top20.to_csv(MODEL_DIR / "feature_importance_mood_rf_top20.csv", index=True)

    return {
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "gb": gb,
        "metrics": metrics,
        "feature_importance": feat_imp,
    }


def tune_baseline_models(
    X_num: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 15,
    cv: int = 3,
) -> dict:
    """Run RandomizedSearchCV for RF and GB, then train final models with best params."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.2, random_state=SEED
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # RF: no scaling
    rf_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        param_distributions=rf_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    best_rf_params = {k: v for k, v in rf_search.best_params_.items()}

    # GB: scaled
    gb_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.08, 0.1],
        "min_samples_leaf": [1, 2],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        param_distributions=gb_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=SEED,
        n_jobs=-1,
    )
    gb_search.fit(X_train_s, y_train)
    best_gb_params = {k: v for k, v in gb_search.best_params_.items()}

    out = {"rf": best_rf_params, "gb": best_gb_params}
    (MODEL_DIR / "baseline_tuned_params.json").write_text(
        json.dumps(_params_to_json_serializable(out), indent=2)
    )
    print("Tuned baseline params saved to baseline_tuned_params.json")
    n_est = best_rf_params.get("n_estimators", 200)
    return train_baseline_models(
        X_num, y,
        n_estimators=n_est,
        rf_params=best_rf_params,
        gb_params=best_gb_params,
    )


def prepare_xy_multi(
    df: pd.DataFrame,
    targets: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    drop_cols = [
        "date",
        "user_id",
        "intervention_start",
        "intervention_end",
        "intervention_type",
        "intervention_intensity",
        "week_start",
    ]

    for t in targets:
        assert t in df.columns, f"{t} not found in df columns"

    X = df.drop(
        columns=drop_cols + targets + LEAKAGE_COLS,
        errors="ignore",
    )

    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(X_num.median())

    Y_cols = []
    for t in targets:
        col = (
            df[t]
            .ffill()
            .bfill()
            .fillna(df[t].mean())
        )
        Y_cols.append(col.to_numpy())

    Y = np.vstack(Y_cols).T  # shape (n_samples, n_targets)

    print("Multi-output feature matrix shape:", X_num.shape)
    print("Targets:", targets)
    return X_num, Y


def train_multioutput_rf(
    X_num: pd.DataFrame,
    Y: np.ndarray,
    target_names: list[str],
    n_estimators: int,
    rf_params: dict | None = None,
) -> dict:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_num, Y, test_size=0.2, random_state=SEED
    )

    rf_kw = {"n_estimators": n_estimators, "random_state": SEED, "n_jobs": -1}
    if rf_params:
        rf_kw = {**rf_kw, **rf_params}
    rf_multi = RandomForestRegressor(**rf_kw)
    rf_multi.fit(X_train, Y_train)
    Y_pred = rf_multi.predict(X_test)

    metrics: dict[str, dict[str, float]] = {}
    print("\nMulti-output Random Forest metrics:")
    for idx, name in enumerate(target_names):
        y_true = Y_test[:, idx]
        y_hat = Y_pred[:, idx]
        mse = mean_squared_error(y_true, y_hat)
        r2 = r2_score(y_true, y_hat)
        metrics[name] = {"mse": mse, "r2": r2}
        print(f"  {name}: MSE={mse:.4f}, R²={r2:.4f}")

    (MODEL_DIR / "multioutput_metrics.json").write_text(json.dumps(metrics, indent=2))

    return {
        "rf_multi": rf_multi,
        "metrics": metrics,
    }


def tune_multioutput_rf(
    X_num: pd.DataFrame,
    Y: np.ndarray,
    target_names: list[str],
    n_iter: int = 12,
    cv: int = 3,
) -> dict:
    """Tune multi-output RF with RandomizedSearchCV (optimize mean R² across targets)."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_num, Y, test_size=0.2, random_state=SEED
    )
    rf_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [1, 2, 4],
    }
    base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=rf_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=SEED,
        n_jobs=-1,
    )
    # R² for multi-output: sklearn averages R² per output by default
    search.fit(X_train, Y_train)
    best = {k: v for k, v in search.best_params_.items()}
    (MODEL_DIR / "multioutput_tuned_params.json").write_text(
        json.dumps(_params_to_json_serializable(best), indent=2)
    )
    print("Tuned multi-output RF params saved to multioutput_tuned_params.json")
    return train_multioutput_rf(
        X_num, Y, target_names=target_names,
        n_estimators=best.get("n_estimators", 200),
        rf_params=best,
    )


def build_sequences_multi(
    df: pd.DataFrame,
    targets: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X_seq, Y) for LSTM: each sample is seq_len consecutive days, target is last day."""
    df_sorted = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    drop_cols = [
        "date", "user_id", "intervention_start", "intervention_end",
        "intervention_type", "intervention_intensity", "week_start",
    ]
    for t in targets:
        assert t in df_sorted.columns, f"{t} not found"
    X = df_sorted.drop(columns=drop_cols + targets + LEAKAGE_COLS, errors="ignore")
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(X_num.median())
    feature_names = list(X_num.columns)
    X_arr = X_num.to_numpy(dtype=np.float32)

    Y_cols = []
    for t in targets:
        col = df_sorted[t].ffill().bfill().fillna(df_sorted[t].mean())
        Y_cols.append(col.to_numpy())
    Y_arr = np.vstack(Y_cols).T.astype(np.float32)

    X_seq_list, Y_list = [], []
    for _, grp in df_sorted.groupby("user_id", sort=False):
        inds = grp.index.to_numpy()
        for start in range(0, len(inds) - seq_len + 1):
            win = inds[start : start + seq_len]
            X_seq_list.append(X_arr[win])
            Y_list.append(Y_arr[win[-1]])
    X_seq = np.array(X_seq_list, dtype=np.float32)
    Y_seq = np.array(Y_list, dtype=np.float32)
    print(f"LSTM sequences: {X_seq.shape}, targets {Y_seq.shape}")
    return X_seq, Y_seq, feature_names


def train_lstm_multi(
    X_seq: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    epochs: int = 30,
    batch_size: int = 32,
    seq_len: int = 7,
    units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> dict:
    if not HAS_TF:
        print("TensorFlow not available; skipping LSTM.")
        return {}
    n_samples, _, n_features = X_seq.shape
    n_targets = Y.shape[1]

    # Scale inputs and targets (target scaling stops one output dominating loss).
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_seq, Y, test_size=0.2, random_state=SEED
    )
    scaler_x = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    X_train_s = scaler_x.fit_transform(X_train_2d).reshape(X_train.shape)
    X_test_s = scaler_x.transform(X_test_2d).reshape(X_test.shape)
    scaler_y = StandardScaler()
    Y_train_s = scaler_y.fit_transform(Y_train)
    Y_test_s = scaler_y.transform(Y_test)

    if n_samples < 2000:
        print(f"  (LSTM has only {n_samples} sequences; run without --dev for more data.)")

    keras.utils.set_random_seed(SEED)

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(24, activation="relu"),
        layers.Dense(n_targets),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    model.fit(
        X_train_s, Y_train_s,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    Y_pred_s = model.predict(X_test_s, verbose=0)
    Y_pred = scaler_y.inverse_transform(Y_pred_s)
    metrics = {}
    print("\nLSTM multi-output metrics:")
    for idx, name in enumerate(target_names):
        mse = mean_squared_error(Y_test[:, idx], Y_pred[:, idx])
        r2 = r2_score(Y_test[:, idx], Y_pred[:, idx])
        metrics[name] = {"mse": float(mse), "r2": float(r2)}
        print(f"  {name}: MSE={mse:.4f}, R²={r2:.4f}")
    out_path = MODEL_DIR / "lstm_multi_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    model.save(MODEL_DIR / "lstm_multi.keras")
    return {"model": model, "metrics": metrics, "scaler_x": scaler_x, "scaler_y": scaler_y}


def train_mlp_multi(
    X_num: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    epochs: int = 40,
    batch_size: int = 32,
    hidden: tuple = (128, 64),
    mood_weight: float = 1.0,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    feature_names: list[str] | None = None,
) -> dict:
    if not HAS_TF:
        print("TensorFlow not available; skipping MLP.")
        return {}
    n_features = X_num.shape[1]
    n_targets = Y.shape[1]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_num, Y, test_size=0.2, random_state=SEED
    )
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    scaler_y = StandardScaler()
    Y_train_s = scaler_y.fit_transform(Y_train)
    Y_test_s = scaler_y.transform(Y_test)
    keras.utils.set_random_seed(SEED)

    # Optional: weight mood MSE higher so the model focuses on mood_score
    if mood_weight != 1.0 and n_targets >= 1:
        w = [float(mood_weight)] + [1.0] * (n_targets - 1)
        weights_tensor = tf.constant(w, dtype=tf.float32)

        def weighted_mse(y_true, y_pred):
            se = tf.square(y_true - y_pred)
            return tf.reduce_mean(se * weights_tensor)
        loss_fn = weighted_mse
    else:
        loss_fn = "mse"

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(hidden[0], activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(hidden[1], activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(n_targets),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn, metrics=["mae"])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    model.fit(
        X_train_s, Y_train_s,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    Y_pred_s = model.predict(X_test_s, verbose=0)
    Y_pred = scaler_y.inverse_transform(Y_pred_s)
    metrics = {}
    print("\nMLP multi-output metrics:")
    for idx, name in enumerate(target_names):
        mse = mean_squared_error(Y_test[:, idx], Y_pred[:, idx])
        r2 = r2_score(Y_test[:, idx], Y_pred[:, idx])
        metrics[name] = {"mse": float(mse), "r2": float(r2)}
        print(f"  {name}: MSE={mse:.4f}, R²={r2:.4f}")
    (MODEL_DIR / "mlp_multi_metrics.json").write_text(json.dumps(metrics, indent=2))
    model_path = MODEL_DIR / "mlp_multi.keras"
    model.save(model_path)
    print("MLP model saved:", model_path.name)
    if joblib is not None:
        joblib.dump(scaler_x, MODEL_DIR / "mlp_scaler_x.joblib")
        joblib.dump(scaler_y, MODEL_DIR / "mlp_scaler_y.joblib")
        print("MLP scalers saved: mlp_scaler_x.joblib, mlp_scaler_y.joblib")
    if feature_names is not None:
        meta = {"feature_names": feature_names, "target_names": target_names}
        (MODEL_DIR / "mlp_metadata.json").write_text(json.dumps(meta, indent=2))
        print("MLP metadata saved: mlp_metadata.json")
    return {"model": model, "metrics": metrics, "scaler_x": scaler_x, "scaler_y": scaler_y}


def tune_mlp_multi(
    X_num: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    n_trials: int = 10,
    epochs: int = 50,
    feature_names: list[str] | None = None,
) -> dict:
    """Random search over MLP hyperparameters; keep best by mean R² across targets."""
    hidden_options = [(64, 32), (128, 64), (256, 128), (128, 64, 32)]
    dropout_options = [0.1, 0.2, 0.3]
    lr_options = [1e-3, 3e-4, 5e-4]
    batch_options = [16, 32]
    rng = random.Random(SEED)
    best_score = -np.inf
    best_params: dict | None = None
    best_result: dict | None = None

    for i in range(n_trials):
        if HAS_TF:
            keras.backend.clear_session()
        hidden = rng.choice(hidden_options)
        dropout = rng.choice(dropout_options)
        lr = rng.choice(lr_options)
        batch_size = rng.choice(batch_options)
        mood_weight = rng.choice([1.0, 1.5, 2.0])
        params = {"hidden": hidden, "dropout": dropout, "learning_rate": lr, "batch_size": batch_size, "mood_weight": mood_weight}
        result = train_mlp_multi(
            X_num, Y, target_names=target_names,
            epochs=epochs, batch_size=batch_size, hidden=hidden,
            mood_weight=mood_weight, dropout=dropout, learning_rate=lr,
            feature_names=feature_names,
        )
        mean_r2 = np.mean([result["metrics"][t]["r2"] for t in target_names])
        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = params
            best_result = result
        print(f"  Trial {i+1}/{n_trials}: mean R²={mean_r2:.4f} (best={best_score:.4f})")

    if best_params is None or best_result is None:
        return {}
    # Save best params and final model is already from last train_mlp_multi; refit best once more to save
    (MODEL_DIR / "mlp_tuned_params.json").write_text(
        json.dumps(_params_to_json_serializable(best_params), indent=2)
    )
    print("MLP tuned params saved to mlp_tuned_params.json. Re-training best config for final save.")
    return train_mlp_multi(
        X_num, Y, target_names=target_names,
        epochs=epochs, batch_size=best_params["batch_size"], hidden=best_params["hidden"],
        mood_weight=best_params["mood_weight"], dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        feature_names=feature_names,
    )


def tune_lstm_multi(
    X_seq: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    seq_len: int,
    n_trials: int = 8,
    epochs: int = 40,
) -> dict:
    """Random search over LSTM hyperparameters; keep best by mean R²."""
    units_options = [24, 32, 48, 64]
    dropout_options = [0.1, 0.2, 0.3]
    lr_options = [1e-3, 3e-4, 5e-4]
    batch_options = [16, 32]
    rng = random.Random(SEED)
    best_score = -np.inf
    best_params: dict | None = None

    for i in range(n_trials):
        if HAS_TF:
            keras.backend.clear_session()
        units = rng.choice(units_options)
        dropout = rng.choice(dropout_options)
        lr = rng.choice(lr_options)
        batch_size = rng.choice(batch_options)
        result = train_lstm_multi(
            X_seq, Y, target_names=target_names,
            epochs=epochs, batch_size=batch_size, seq_len=seq_len,
            units=units, dropout=dropout, learning_rate=lr,
        )
        mean_r2 = np.mean([result["metrics"][t]["r2"] for t in target_names])
        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = {"units": units, "dropout": dropout, "learning_rate": lr, "batch_size": batch_size}
        print(f"  Trial {i+1}/{n_trials}: mean R²={mean_r2:.4f} (best={best_score:.4f})")

    if best_params is None:
        return {}
    (MODEL_DIR / "lstm_tuned_params.json").write_text(
        json.dumps(_params_to_json_serializable(best_params), indent=2)
    )
    print("LSTM tuned params saved. Re-training best config for final save.")
    return train_lstm_multi(
        X_seq, Y, target_names=target_names,
        epochs=epochs, batch_size=best_params["batch_size"], seq_len=seq_len,
        units=best_params["units"], dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
    )


def train_nextday_model(
    df: pd.DataFrame,
    drop_cols: list[str],
    n_estimators: int,
    sample_frac: float | None = None,
) -> dict:
    df_td = df.copy().sort_values(["user_id", "date"])
    df_td["mood_next_day"] = df_td.groupby("user_id")["mood_score"].shift(-1)
    df_td = df_td.dropna(subset=["mood_next_day"])

    if sample_frac is not None and 0 < sample_frac < 1.0:
        df_td = df_td.sample(frac=sample_frac, random_state=SEED)
        print(f"Next-day training on sampled fraction: {sample_frac}, shape={df_td.shape}")
    else:
        print("Next-day training on full next-day dataset, shape:", df_td.shape)

    X_td = df_td.drop(
        columns=drop_cols + ["mood_score", "mood_next_day", "health_twin_index", "stress_level"],
        errors="ignore",
    )
    X_td_num = X_td.select_dtypes(include=[np.number]).copy()
    X_td_num = X_td_num.fillna(X_td_num.median())
    y_td = df_td["mood_next_day"]

    Xtr, Xte, ytr, yte = train_test_split(
        X_td_num, y_td, test_size=0.2, random_state=SEED
    )

    rf_next = RandomForestRegressor(
        n_estimators=n_estimators, random_state=SEED, n_jobs=-1
    )
    rf_next.fit(Xtr, ytr)
    y_pred_next = rf_next.predict(Xte)

    mse_next = mean_squared_error(yte, y_pred_next)
    r2_next = r2_score(yte, y_pred_next)
    print(
        "Next-day Random Forest — MSE:",
        mse_next,
        "R2:",
        r2_next,
    )

    metrics_next = {"mse": mse_next, "r2": r2_next, "n_estimators": n_estimators}
    (MODEL_DIR / "nextday_metrics.json").write_text(json.dumps(metrics_next, indent=2))

    return {"rf_next": rf_next, "metrics": metrics_next}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline and next-day mood models for Everyday Health Twin."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use small dev sample instead of full dataset.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of trees for baseline Random Forest (default: 200).",
    )
    parser.add_argument(
        "--train-nextday",
        action="store_true",
        help="Also train next-day mood Random Forest.",
    )
    parser.add_argument(
        "--nextday-sample-frac",
        type=float,
        default=0.3,
        help=(
            "Fraction of next-day rows to sample to avoid OOM (default: 0.3). "
            "Set to 1.0 to use all rows."
        ),
    )
    parser.add_argument(
        "--nextday-estimators",
        type=int,
        default=150,
        help="Number of trees for next-day Random Forest (default: 150).",
    )
    parser.add_argument(
        "--multi-target",
        action="store_true",
        help="Train multi-output RF for mood_score, stress_level, health_twin_index.",
    )
    parser.add_argument(
        "--multi-estimators",
        type=int,
        default=200,
        help="Number of trees for multi-output Random Forest (default: 200).",
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="Train LSTM multi-output model (mood, stress, health_twin_index).",
    )
    parser.add_argument(
        "--lstm-seq-len",
        type=int,
        default=7,
        help="Sequence length in days for LSTM (default: 7).",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=30,
        help="Epochs for LSTM (default: 30).",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Train MLP multi-output model (mood, stress, health_twin_index).",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=40,
        help="Epochs for MLP (default: 40).",
    )
    parser.add_argument(
        "--mlp-mood-weight",
        type=float,
        default=2.0,
        help="Weight for mood_score in MLP loss (default: 2.0). Use 1.0 for equal weights.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning for baseline RF/GB, multi-output RF, and optionally MLP/LSTM.",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=15,
        help="RandomizedSearchCV n_iter for RF/GB (default: 15). Use fewer with --dev.",
    )
    parser.add_argument(
        "--tune-n-trials",
        type=int,
        default=10,
        help="Random search trials for MLP/LSTM when --tune (default: 10).",
    )
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="Run only the MLP pipeline (no baseline, no multi-output RF, no LSTM). Lighter on memory/CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_dev = args.dev

    print("=== Everyday Health Twin - Modeling ===")
    print("Mode:", "DEV (sample)" if use_dev else "FULL DATA")
    if args.mlp_only:
        print("MLP only (skipping baseline, multi-output RF, LSTM).")

    df = load_data(use_dev=use_dev)

    multi_targets = ["mood_score", "stress_level", "health_twin_index"]

    if not args.mlp_only:
        # Baseline mood_score prediction (with optional tuning)
        X_num, y = prepare_xy_mood(df, target="mood_score")
        n_iter_sklearn = max(5, args.tune_n_iter // 2) if use_dev and args.tune else args.tune_n_iter
        if args.tune:
            print("Tuning baseline (RF + GB)...")
            baseline = tune_baseline_models(X_num, y, n_iter=n_iter_sklearn, cv=3)
        else:
            baseline = train_baseline_models(
                X_num=X_num,
                y=y,
                n_estimators=args.rf_estimators,
            )

        # Optional multi-output prediction for mood, stress, health_twin_index
        if args.multi_target:
            X_multi, Y_multi = prepare_xy_multi(df, targets=multi_targets)
            if args.tune:
                print("Tuning multi-output RF...")
                tune_multioutput_rf(X_multi, Y_multi, target_names=multi_targets, n_iter=max(6, n_iter_sklearn), cv=3)
            else:
                train_multioutput_rf(
                    X_num=X_multi,
                    Y=Y_multi,
                    target_names=multi_targets,
                    n_estimators=args.multi_estimators,
                )

        # LSTM and MLP need TensorFlow
        if (args.lstm or args.mlp) and not HAS_TF:
            print("\nTensorFlow not available. Install with: pip install tensorflow")
            print("Skipping LSTM and MLP.")

        # Optional LSTM multi-output (sequences of consecutive days per user)
        if args.lstm and HAS_TF:
            X_seq, Y_seq, _ = build_sequences_multi(
                df, targets=multi_targets, seq_len=args.lstm_seq_len
            )
            if len(X_seq) > 0:
                if args.tune:
                    print("Tuning LSTM...")
                    tune_lstm_multi(
                        X_seq, Y_seq, target_names=multi_targets,
                        seq_len=args.lstm_seq_len,
                        n_trials=min(6, args.tune_n_trials),
                        epochs=args.lstm_epochs,
                    )
                else:
                    train_lstm_multi(
                        X_seq,
                        Y_seq,
                        target_names=multi_targets,
                        epochs=args.lstm_epochs,
                        seq_len=args.lstm_seq_len,
                    )
            else:
                print("LSTM skipped: no sequences (need more days per user).")

    # MLP: run when --mlp or --mlp-only
    if args.mlp_only and not HAS_TF:
        print("\nTensorFlow required for --mlp-only. Install with: pip install tensorflow")
    elif (args.mlp or args.mlp_only) and HAS_TF:
        X_multi, Y_multi = prepare_xy_multi(df, targets=multi_targets)
        feature_names = X_multi.columns.tolist()
        if args.tune:
            print("Tuning MLP...")
            tune_mlp_multi(
                X_multi.to_numpy(),
                Y_multi,
                target_names=multi_targets,
                n_trials=args.tune_n_trials,
                epochs=args.mlp_epochs,
                feature_names=feature_names,
            )
        else:
            train_mlp_multi(
                X_multi.to_numpy(),
                Y_multi,
                target_names=multi_targets,
                epochs=args.mlp_epochs,
                mood_weight=args.mlp_mood_weight,
                feature_names=feature_names,
            )

    # Optional next-day mood prediction (skipped when --mlp-only)
    if args.train_nextday and not args.mlp_only:
        drop_cols = [
            "date",
            "user_id",
            "intervention_start",
            "intervention_end",
            "intervention_type",
            "intervention_intensity",
            "week_start",
        ]
        train_nextday_model(
            df=df,
            drop_cols=drop_cols,
            n_estimators=args.nextday_estimators,
            sample_frac=args.nextday_sample_frac,
        )

    print("\nDone. Metrics written under:", MODEL_DIR)


if __name__ == "__main__":
    main()

