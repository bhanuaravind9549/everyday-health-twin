"""
Predict mood_score, stress_level, and health_twin_index using the trained MLP.
Loads model and scalers from models/ (run modeling.py --multi-target --mlp first on full data).

Usage:
  python predict.py --input path/to/features.csv --output predictions.csv
  python predict.py --input path/to/features.csv   # prints predictions to stdout
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_mlp_artifacts():
    """Load MLP model, scalers, and metadata. Raises if any file is missing."""
    model_path = MODEL_DIR / "mlp_multi.keras"
    scaler_x_path = MODEL_DIR / "mlp_scaler_x.joblib"
    scaler_y_path = MODEL_DIR / "mlp_scaler_y.joblib"
    meta_path = MODEL_DIR / "mlp_metadata.json"

    for p in (model_path, scaler_x_path, scaler_y_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}. Run modeling.py with --multi-target --mlp (and optionally --tune) on full data first."
            )

    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required for prediction. Install with: pip install tensorflow")

    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required to load scalers. Install with: pip install joblib")

    model = keras.models.load_model(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    target_names = meta["target_names"]
    return model, scaler_x, scaler_y, feature_names, target_names


def prepare_features(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Select and align features, fill missing with column median."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Required: {feature_names[:5]}... ({len(feature_names)} total)."
        )
    X = df[feature_names].copy()
    X = X.fillna(X.median())
    return X.to_numpy(dtype=np.float32)


def predict(
    model,
    scaler_x,
    scaler_y,
    feature_names: list[str],
    target_names: list[str],
    X: np.ndarray,
) -> np.ndarray:
    """Return predicted values (n_samples, n_targets) in original scale."""
    X_s = scaler_x.transform(X)
    Y_s = model.predict(X_s, verbose=0)
    return scaler_y.inverse_transform(Y_s)


def run(
    input_path: Path | None = None,
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load artifacts, run predictions on input CSV or DataFrame, return DataFrame with predictions.
    If output_path is set, also write to CSV.
    """
    model, scaler_x, scaler_y, feature_names, target_names = load_mlp_artifacts()

    if df is None:
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)

    X = prepare_features(df, feature_names)
    Y_pred = predict(model, scaler_x, scaler_y, feature_names, target_names, X)

    out = df.copy()
    for i, name in enumerate(target_names):
        out[f"{name}_pred"] = Y_pred[:, i]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Predictions written to {output_path}", file=sys.stderr)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Predict mood_score, stress_level, health_twin_index with the trained MLP."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input CSV with same feature columns as training (e.g. merged_features.csv or a subset).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, predictions are printed to stdout.",
    )
    args = parser.parse_args()

    if args.input is None:
        parser.error("--input is required (path to features CSV).")

    result = run(input_path=args.input, output_path=args.output)

    if args.output is None:
        pred_cols = [c for c in result.columns if c.endswith("_pred")]
        result[pred_cols].to_csv(sys.stdout, index=False)
    else:
        print(f"Done. Output has {len(result)} rows with columns: mood_score_pred, stress_level_pred, health_twin_index_pred.")


if __name__ == "__main__":
    main()
