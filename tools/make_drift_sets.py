"""
Build the inputs of the drift check (README step 5, eval/README.md):

    uv run python tools/make_drift_sets.py --images ~/Desktop/daiist-data/deepsolareye/Solar_Panel_Soiling_Image_dataset/PanelImages

- eval/reference.csv: N_REFERENCE normal DeepSolarEye photos from the photos
  component 2 was trained on (data/deepsolareye_gemini.csv), spread over all days.
  Copied to eval/inputs/reference/.
- eval/shifted.csv: other DeepSolarEye photos changed on purpose, the shifts the
  team expects in the field (decision 9), copied to eval/inputs/shifted/:
    dark_*     darker: dusk, overcast, a cheaper camera (brightness x DARK_FACTOR)
    calima_*   Saharan dust haze: blended with a dusty orange (CALIMA_MIX)
    flat_*     low contrast: fog, a dirty lens, overexposure (contrast x FLAT_FACTOR)

Photos are 192x192 jpgs (~10 KB each), so they can live in the repo.
DeepSolarEye (Mehta et al., WACV 2018) is CC BY-NC-SA: cite it in REPORT.md.
Team cases in eval/cases.csv must not reuse the reference photos.
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "deepsolareye_gemini.csv"
N_REFERENCE = 40
N_PER_SHIFT = 5
DARK_FACTOR = 0.45
CALIMA_COLOUR = (205, 150, 80)
CALIMA_MIX = 0.40
FLAT_FACTOR = 0.35
SEED = 7


def shift(image: Image.Image, kind: str) -> Image.Image:
    image = image.convert("RGB")
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance(DARK_FACTOR)
    if kind == "calima":
        return Image.blend(image, Image.new("RGB", image.size, CALIMA_COLOUR), CALIMA_MIX)
    if kind == "flat":
        return ImageEnhance.Contrast(image).enhance(FLAT_FACTOR)
    raise ValueError(kind)


def write_inputs(path: Path, inputs: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input"])
        writer.writerows([i] for i in inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="folder with the DeepSolarEye .jpg files")
    args = parser.parse_args()
    images = Path(args.images).expanduser()
    rng = random.Random(SEED)

    with LABELLED.open(newline="") as f:
        labelled = list(csv.DictReader(f))
    by_day = {}
    for row in labelled:
        by_day.setdefault(row["day"], []).append(row["file"])
    for files in by_day.values():
        rng.shuffle(files)
    reference, days = [], sorted(by_day)
    while len(reference) < N_REFERENCE:  # round robin over days
        for day in days:
            if by_day[day] and len(reference) < N_REFERENCE:
                reference.append(by_day[day].pop())

    out = ROOT / "eval" / "inputs" / "reference"
    out.mkdir(parents=True, exist_ok=True)
    for name in reference:
        (out / name).write_bytes((images / name).read_bytes())
    write_inputs(ROOT / "eval" / "reference.csv", [f"eval/inputs/reference/{n}" for n in reference])

    used = {row["file"] for row in labelled}
    pool = sorted(p.name for p in images.glob("solar_*.jpg") if p.name not in used)
    picked = rng.sample(pool, 3 * N_PER_SHIFT)
    out = ROOT / "eval" / "inputs" / "shifted"
    out.mkdir(parents=True, exist_ok=True)
    shifted = []
    for i, kind in enumerate(("dark", "calima", "flat")):
        for name in picked[i * N_PER_SHIFT:(i + 1) * N_PER_SHIFT]:
            with Image.open(images / name) as image:
                shift(image, kind).save(out / f"{kind}_{name}", quality=95)
            shifted.append(f"eval/inputs/shifted/{kind}_{name}")
    write_inputs(ROOT / "eval" / "shifted.csv", shifted)
    print(f"{len(reference)} reference photos and {len(shifted)} shifted photos written under eval/.")


if __name__ == "__main__":
    main()
