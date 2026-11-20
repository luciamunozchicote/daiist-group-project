"""
Train and evaluate component 2 (team decisions 3, 5 and 6).

    uv run python tools/train_component2.py

- Data: data/deepsolareye_gemini.csv (component 1's output on DeepSolarEye photos
  + true power loss, irradiance and hour).
- Split BY DAY, never at random: photos are consecutive video frames, so a random
  split would put near-identical frames in train and test and inflate the scores.
  Test = the last TEST_DAYS days (the "future"); also leave-one-day-out CV.
- Several models, each WITH and WITHOUT component 1's features: if Gemini's
  features do not lower the error, the chain adds nothing.
- Metrics: MAE in points of % loss, and how often the clean / don't-clean call is
  right at a loss threshold (PLACEHOLDER: the team sets it from the € formula).
- Learning curve: test MAE as the training set grows (do more photos help?).

Writes models/component2.joblib (the model the app uses) and
models/component2_report.json (all the numbers, for REPORT.md).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from component2 import ALL_FEATURES, CONTEXT_FEATURES, GEMINI_FEATURES, to_features  # noqa: E402

DATA = ROOT / "data" / "deepsolareye_gemini.csv"
OUT_MODEL = ROOT / "models" / "component2.joblib"
OUT_REPORT = ROOT / "models" / "component2_report.json"
TEST_DAYS = 4
# Decision threshold from the € formula: clean when the energy lost by the dirt,
# until the next scheduled cleaning, is worth more than cleaning.
#   threshold = cleaning cost / (energy per m2 until next cleaning x price)
CLEANING_EUR_PER_M2 = 0.09        # Micheli et al., Energy 2021 (1 MW plant, Granada), doi:10.1016/j.energy.2020.119018
YIELD_KWH_PER_KWP_YEAR = 1718     # same plant, 2019 AC yield
MODULE_KWP_PER_M2 = 0.21          # ~21% efficient modern module (team: check against your module datasheet)
DAYS_TO_NEXT_CLEANING = 30        # TEAM DECISION: how often the plant is cleaned anyway
PRICE_EUR_PER_KWH = 0.0325        # Spain solar PPA, LevelTen Q4 2025 (main case)
PRICE_SENSITIVITY = {"OMIE solar captured 2025 (UNEF)": 0.03639,
                     "OMIE solar captured H1 2026 (UNEF)": 0.02968}


def loss_threshold(price: float, days: int = DAYS_TO_NEXT_CLEANING) -> float:
    kwh_per_m2 = YIELD_KWH_PER_KWP_YEAR * MODULE_KWP_PER_M2 / 365 * days
    return CLEANING_EUR_PER_M2 / (kwh_per_m2 * price)


CLEAN_IF_LOSS_ABOVE = round(loss_threshold(PRICE_EUR_PER_KWH), 3)
SEED = 42
# How the app's model is picked (team decision). Always on the training days only
# (leave-one-day-out CV), never on the test days, so the test score stays honest.
SELECT_BY = "cv_by_day_decision_accuracy"  # or "cv_by_day_mae_pts"

MODELS = {
    "mean (no model)": lambda: DummyRegressor(strategy="mean"),
    "linear (ridge)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
    "gradient boosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                              max_iter=300, random_state=SEED),
}
FEATURE_SETS = {
    "gemini + context": ALL_FEATURES,
    "context only (no gemini)": CONTEXT_FEATURES,
    "gemini only": GEMINI_FEATURES,
}


def decision_accuracy(y_true, y_pred) -> float:
    return float(np.mean((y_true > CLEAN_IF_LOSS_ABOVE) == (y_pred > CLEAN_IF_LOSS_ABOVE)))


def main() -> None:
    data = pd.read_csv(DATA)
    data["date"] = pd.to_datetime(data["day"], format="%Y-%b-%d")
    days = sorted(data["date"].unique())
    test_days = days[-TEST_DAYS:]
    train, test = data[~data["date"].isin(test_days)], data[data["date"].isin(test_days)]
    x_train, x_test = to_features(train), to_features(test)
    y_train, y_test = train["loss"].to_numpy(), test["loss"].to_numpy()
    groups = train["day"].to_numpy()
    print(f"{len(data)} photos, {len(days)} days. Train: {len(train)} photos ({len(days) - TEST_DAYS} days). "
          f"Test: {len(test)} photos (last {TEST_DAYS} days: "
          f"{', '.join(pd.Timestamp(d).strftime('%d %b') for d in test_days)}).\n")

    results = []
    for model_name, make in MODELS.items():
        for set_name, features in FEATURE_SETS.items():
            if model_name == "mean (no model)" and set_name != "gemini + context":
                continue
            model = make().fit(x_train[features], y_train)
            pred = np.clip(model.predict(x_test[features]), 0, 1)
            cv_pred = np.clip(cross_val_predict(make(), x_train[features], y_train,
                                                groups=groups, cv=LeaveOneGroupOut()), 0, 1)
            results.append({
                "model": model_name,
                "features": set_name,
                "test_mae_pts": round(100 * mean_absolute_error(y_test, pred), 2),
                "cv_by_day_mae_pts": round(100 * mean_absolute_error(y_train, cv_pred), 2),
                "cv_by_day_decision_accuracy": round(decision_accuracy(y_train, cv_pred), 3),
                "test_decision_accuracy": round(decision_accuracy(y_test, pred), 3),
            })
    table = pd.DataFrame(results)
    print("MAE = average error in points of % loss (lower is better).")
    print(f"Decision accuracy = clean/don't-clean right at a {CLEAN_IF_LOSS_ABOVE:.0%} loss threshold (from the € formula).")
    for label, price in PRICE_SENSITIVITY.items():
        print(f"  sensitivity, {label} at {price} €/kWh: threshold {loss_threshold(price):.1%}")
    print()
    print(table.to_string(index=False), "\n")

    # Pick among models using gemini + context, by SELECT_BY on the CV (ties: lower CV MAE).
    candidates = table[(table["features"] == "gemini + context") & (table["model"] != "mean (no model)")]
    higher_is_better = SELECT_BY.endswith("accuracy")
    best = candidates.sort_values([SELECT_BY, "cv_by_day_mae_pts"],
                                  ascending=[not higher_is_better, True]).iloc[0]["model"]
    model = MODELS[best]().fit(x_train[ALL_FEATURES], y_train)
    errors = np.abs(np.clip(model.predict(x_test[ALL_FEATURES]), 0, 1) - y_test)
    error_p80 = float(np.quantile(errors, 0.8))
    print(f"Model for the app, picked by {SELECT_BY}: {best} (gemini + context). "
          f"80% of test errors are below {100 * error_p80:.1f} points -> shown as the range.\n")

    # Learning curve: does more data help?
    curve = []
    rng = np.random.default_rng(SEED)
    for size in [n for n in (50, 100, 200, 300, 400, 500, 600, 800) if n < len(train)] + [len(train)]:
        maes = []
        for _ in range(5 if size < len(train) else 1):
            idx = rng.choice(len(train), size=size, replace=False)
            m = MODELS[best]().fit(x_train.iloc[idx][ALL_FEATURES], y_train[idx])
            maes.append(mean_absolute_error(y_test, np.clip(m.predict(x_test[ALL_FEATURES]), 0, 1)))
        curve.append({"train_photos": int(size), "test_mae_pts": round(100 * float(np.mean(maes)), 2)})
    print("Learning curve (" + best + "):")
    print(pd.DataFrame(curve).to_string(index=False), "\n")

    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(ALL_FEATURES, np.round(model.feature_importances_, 3).tolist()))
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, x_test[ALL_FEATURES], y_test, n_repeats=10,
                                      random_state=SEED, scoring="neg_mean_absolute_error")
        importance = dict(zip(ALL_FEATURES, np.round(perm.importances_mean, 4).tolist()))
    print("Feature importance:", json.dumps(importance), "\n")

    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "name": f"{best} v1", "features": ALL_FEATURES,
                 "error_p80": error_p80}, OUT_MODEL)
    OUT_REPORT.write_text(json.dumps({
        "n_photos": len(data), "n_train": len(train), "n_test": len(test),
        "test_days": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_days],
        "decision_threshold": CLEAN_IF_LOSS_ABOVE,
        "results": results, "selected_by": SELECT_BY, "provisional_model": best, "error_p80_pts": round(100 * error_p80, 2),
        "learning_curve": curve, "feature_importance": importance,
    }, indent=1))
    print(f"Saved {OUT_MODEL.relative_to(ROOT)} and {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
