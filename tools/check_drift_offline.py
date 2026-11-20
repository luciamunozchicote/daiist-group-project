"""
Try the drift alert on the photo features only, without calling Gemini (no quota):

    uv run python tools/check_drift_offline.py --images ~/Desktop/daiist-data/deepsolareye/Solar_Panel_Soiling_Image_dataset/PanelImages

- False alarms: draws many random batches of BATCH normal DeepSolarEye photos
  (labelled ones, not in the reference set) and counts how often the alert fires.
- Detection: runs eval/shifted.csv, all together and per kind of shift.

Use it to choose monitor.PSI_ALERT / NOISE_MULTIPLE (team decision) and to
justify them in REPORT.md. `main.py drift-demo` is the official check (it also
uses Gemini's agreement).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402
import pipeline  # noqa: E402

BATCH = 20      # like monitor.MIN_WINDOW: the smallest window the app alerts on
N_BATCHES = 200


def inputs(name: str) -> list[str]:
    with (ROOT / "eval" / name).open(newline="") as f:
        return [r["input"] for r in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder with the DeepSolarEye .jpg files")
    args = parser.parse_args()
    images = Path(args.images).expanduser()

    reference = pd.DataFrame([pipeline.image_features(str(ROOT / p)) for p in inputs("reference.csv")])
    in_reference = {Path(p).name for p in inputs("reference.csv")}
    with (ROOT / "data" / "deepsolareye_gemini.csv").open(newline="") as f:
        pool = sorted({r["file"] for r in csv.DictReader(f)} - in_reference)
    features = {name: pipeline.image_features(str(images / name)) for name in pool}

    rng = random.Random(0)
    fired = pd.Series(0, index=reference.columns)
    any_fired = 0
    for _ in range(N_BATCHES):
        report = monitor.drift(reference, pd.DataFrame([features[n] for n in rng.sample(pool, BATCH)]))
        fired += report.set_index("feature")["alert"].astype(int)
        any_fired += bool(report["alert"].any())
    print(f"Normal photos, {N_BATCHES} random batches of {BATCH}: the alert fired in {any_fired} "
          f"({any_fired / N_BATCHES:.0%}). Per feature: {fired.to_dict()}\n")

    shifted = inputs("shifted.csv")
    table = pd.DataFrame([pipeline.image_features(str(ROOT / p)) for p in shifted])
    report = monitor.drift(reference, table)
    print(f"Shifted photos, all {len(shifted)} together:\n{report.to_string(index=False)}\n")
    for kind in sorted({Path(p).name.split('_')[0] for p in shifted}):
        rows = table[[Path(p).name.startswith(kind + "_") for p in shifted]]
        alerts = monitor.drift(reference, rows).query("alert")["feature"].tolist()
        print(f"  only {kind} ({len(rows)} photos): {'fires on ' + ', '.join(alerts) if alerts else 'quiet'}")


if __name__ == "__main__":
    main()
