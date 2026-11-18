"""
Pass a sample of DeepSolarEye photos through component 1 (Gemini) and save its
output next to the true power loss, irradiance and time. That table is the
training data for component 2 (team decisions 3 and 5).

    uv run python tools/label_deepsolareye.py --images ~/Desktop/daiist-data/deepsolareye/Solar_Panel_Soiling_Image_dataset/PanelImages

- Sample: the same number of photos per (day, loss band), so every day and every
  level of loss is represented. Days matter: train/test must be split by day.
- One Gemini call per photo at temperature 0 (no votes: they would triple the
  cost and hit the free daily limit).
- Resumable: photos already in the output file are skipped, so you can stop it
  and run it again (or tomorrow, when the daily limit resets).

DeepSolarEye (Mehta et al., WACV 2018, CC BY-NC-SA): https://deep-solar-eye.github.io
Download: one zip from the project page; see SETUP.md.
"""

import argparse
import csv
import random
import re
import sys
import time
import urllib.error
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import env  # noqa: E402
import vision  # noqa: E402

NAME = re.compile(
    r"solar_[A-Za-z]+_([A-Za-z]+)_(\d+)_(\d+)__(\d+)__(\d+)_(\d+)_L_(\d+(?:\.\d+)?(?:e-?\d+)?)_I_(\d+(?:\.\d+)?(?:e-?\d+)?)\.jpg$"
)
LOSS_BANDS = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 1.01)]
FIELDS = ["file", "day", "hour", "irradiance", "loss", "loss_band",
          "label", "n_spots", "covered_fraction", "spread", "is_solar_panel",
          "tokens", "model", "prompt_version"]


def parse(name: str) -> dict | None:
    match = NAME.match(name)
    if not match:
        return None
    month, day, hour, minute, second, year, loss, irradiance = match.groups()
    loss = float(loss)
    band = next(i for i, (lo, hi) in enumerate(LOSS_BANDS) if lo <= loss < hi)
    return {
        "file": name,
        "day": f"{year}-{month}-{int(day):02d}",
        "hour": round(int(hour) + int(minute) / 60 + int(second) / 3600, 4),
        "irradiance": float(irradiance),
        "loss": loss,
        "loss_band": band,
    }


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """About the same number of photos from every (day, loss band) group."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["day"], row["loss_band"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    sample, keys = [], sorted(groups)
    while len(sample) < n and any(groups.values()):
        for key in keys:  # round robin, so small groups are not crowded out
            if groups[key] and len(sample) < n:
                sample.append(groups[key].pop())
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder with the DeepSolarEye .jpg files")
    parser.add_argument("--n", type=int, default=1000, help="how many photos in total (default 1000)")
    parser.add_argument("--min-irradiance", type=float, default=0.0,
                        help="skip photos darker than this irradiance (0-1); default keeps all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "data" / "deepsolareye_gemini.csv"))
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between calls (free-tier rate limit)")
    args = parser.parse_args()

    env.load_env()
    rows = [r for r in (parse(p.name) for p in Path(args.images).expanduser().glob("*.jpg")) if r]
    rows = [r for r in rows if r["irradiance"] >= args.min_irradiance]
    sample = stratified_sample(rows, args.n, args.seed)
    print(f"{len(rows)} photos found, {len(sample)} sampled "
          f"from {len({r['day'] for r in sample})} days and {len(LOSS_BANDS)} loss bands.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with out.open(newline="") as f:
            done = {row["file"] for row in csv.DictReader(f)}
    todo = [r for r in sample if r["file"] not in done]
    print(f"{len(done)} already labelled, {len(todo)} to go.")

    new_file = not out.is_file()
    failures_in_a_row = 0
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(todo, 1):
            try:
                answer = vision.describe(str(Path(args.images).expanduser() / row["file"]), n_votes=1)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                failures_in_a_row += 1
                print(f"  {row['file']}: {getattr(error, 'code', None) or type(error).__name__}")
                if failures_in_a_row >= 5:
                    print("5 failures in a row: probably the daily limit. Run again later; done photos are kept.")
                    break
                time.sleep(30)
                continue
            failures_in_a_row = 0
            writer.writerow(row | {k: answer[k] for k in
                                   ("label", "n_spots", "covered_fraction", "spread", "is_solar_panel", "tokens")}
                            | {"model": vision.MODEL, "prompt_version": vision.PROMPT_VERSION})
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(args.pause)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
