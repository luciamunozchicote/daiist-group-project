"""
Component 2 (tabular ML): estimated % power loss of a soiled panel, from
component 1's description of the photo plus irradiance and time of day.

Trained by tools/train_component2.py on DeepSolarEye photos that went through
component 1 (data/deepsolareye_gemini.csv). The trained model is saved in
models/component2.joblib; this module only loads it and predicts.
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "component2.joblib"

GEMINI_FEATURES = ["covered_fraction", "n_spots", "spread_one_area", "spread_several_areas", "label_soiled"]
CONTEXT_FEATURES = ["irradiance", "hour"]
ALL_FEATURES = GEMINI_FEATURES + CONTEXT_FEATURES


def to_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Component 1's output (+ context) as the numeric columns the model reads."""
    return pd.DataFrame({
        "covered_fraction": rows["covered_fraction"].astype(float),
        "n_spots": rows["n_spots"].astype(float),
        "spread_one_area": (rows["spread"] == "one_area").astype(float),
        "spread_several_areas": (rows["spread"] == "several_areas").astype(float),
        "label_soiled": (rows["label"] == "soiled").astype(float),
        "irradiance": rows["irradiance"].astype(float),
        "hour": rows["hour"].astype(float),
    })


_bundle = None


def _load() -> dict | None:
    global _bundle
    if _bundle is None and MODEL_FILE.is_file():
        _bundle = joblib.load(MODEL_FILE)
    return _bundle


def predict(c1: dict, irradiance: float, hour: float) -> dict:
    """% power loss with a range, from component 1's output and the context."""
    bundle = _load()
    if bundle is None:
        return {"loss_pct": None, "note": "component 2 is not trained yet: run tools/train_component2.py"}
    row = pd.DataFrame([c1 | {"irradiance": irradiance, "hour": hour}])
    x = to_features(row)[bundle["features"]]
    loss = float(min(1.0, max(0.0, bundle["model"].predict(x)[0])))
    margin = bundle["error_p80"]  # 80% of test errors were smaller than this
    return {
        "loss_pct": round(100 * loss, 1),
        "loss_range_pct": [round(100 * max(0.0, loss - margin), 1), round(100 * min(1.0, loss + margin), 1)],
        "model": bundle["name"],
    }
