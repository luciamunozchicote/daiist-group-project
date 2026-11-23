"""
Suggest DeepSolarEye photos you could use as test cases in eval/cases.csv.

    uv run python tools/case_candidates.py --images ~/Desktop/daiist-data/deepsolareye/Solar_Panel_Soiling_Image_dataset/PanelImages

Only photos that are NOT in the training data (data/deepsolareye_gemini.csv) nor in
eval/reference.csv, from the test days (27-30 Jun), so the evaluation is fair.
The true power loss is in the file name (L_...). Prints a few per loss band; you
choose which ones become cases, copy them to eval/inputs/cases/ and write the row.
"""

import argparse
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(0.0, 0.05), (0.05, 0.08), (0.08, 0.11), (0.11, 0.30), (0.30, 1.01)]
LOSS = re.compile(r"_L_([\d.e-]+?)_I_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--per-band", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--copy", action="store_true",
                        help="also copy them to eval/inputs/candidates/ (so teammates can pick without the dataset)")
    args = parser.parse_args()
    images = Path(args.images).expanduser()
    with (ROOT / "data" / "deepsolareye_gemini.csv").open(newline="") as f:
        used = {r["file"] for r in csv.DictReader(f)}
    with (ROOT / "eval" / "reference.csv").open(newline="") as f:
        used |= {Path(r["input"]).name for r in csv.DictReader(f)}
    pool = [p.name for day in ("27", "28", "29", "30") for p in images.glob(f"solar_*_Jun_{day}_*.jpg")
            if p.name not in used]
    rng = random.Random(args.seed)
    for lo, hi in BANDS:
        band = [n for n in pool if lo <= float(LOSS.search(n).group(1)) < hi]
        print(f"\nTrue loss {lo:.0%}-{hi:.0%} ({len(band)} photos):")
        for name in rng.sample(band, min(args.per_band, len(band))):
            print(f"  {float(LOSS.search(name).group(1)):6.1%}  {name}")
            if args.copy:
                out = ROOT / "eval" / "inputs" / "candidates"
                out.mkdir(parents=True, exist_ok=True)
                (out / name).write_bytes((images / name).read_bytes())


if __name__ == "__main__":
    main()
