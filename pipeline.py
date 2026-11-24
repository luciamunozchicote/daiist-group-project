"""
Solar Triage: the two components and how one feeds the other.

    photo ──► component 1 (vision, Gemini) ──► component 2 (tabular, % power loss) ──► € / decision

Component 1: vision.py (Gemini). Component 2: component2.py, a regression
trained by tools/train_component2.py on DeepSolarEye photos that went through
component 1 (tools/label_deepsolareye.py).

`run()` must keep returning the dict app.py, evaluate.py, main.py and check.py read:

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # cost of this request in euros
    }

`raw_input` is the path of an image (the App tab uses gr.Image(type="filepath");
eval/cases.csv uses paths like eval/inputs/c01.jpg).
"""

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import component2
import economics
import vision

VERSIONS = {
    "component_1": f"{vision.MODEL}/{vision.PROMPT_VERSION}",
    "component_2": "regression-v1",
}

N_VOTES = 3  # main answer + 2 votes; agreement = confidence (team decision 1)

# Team decision (2026-11-23): a person reviews when the 3 Gemini answers have no
# majority (agreement 1/3), when Gemini answers "unknown", and always for damage.
REVIEW_BELOW_AGREEMENT = 0.67

# DeepSolarEye file names carry irradiance and time:
# solar_Wed_Jun_28_7__5__6_2017_L_0.0123_I_0.0566.jpg
_DSE_NAME = re.compile(r"solar_[A-Za-z]+_[A-Za-z]+_\d+_(\d+)__(\d+)__\d+_\d+_L_[\d.e-]+?_I_(\d+(?:\.\d+)?(?:e-?\d+)?)")


def context_from_name(path: str) -> dict:
    """Irradiance (0-1) and hour of day, if the file name has them (DeepSolarEye)."""
    match = _DSE_NAME.search(Path(path).name)
    if not match:
        return {"irradiance": None, "hour": None}
    hour, minute, irradiance = match.groups()
    return {"irradiance": float(irradiance), "hour": round(int(hour) + int(minute) / 60, 2)}


def image_features(path: str) -> dict[str, float]:
    """Numbers that describe the photo, for the drift check (decision 9):
    brightness for darker photos, colour balance for haze (calima)."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((96, 96)), dtype=float) / 255
    r, g, b = pixels[..., 0].mean(), pixels[..., 1].mean(), pixels[..., 2].mean()
    return {
        "brightness": round(float(pixels.mean()), 4),
        "mean_r": round(float(r), 4),
        "mean_g": round(float(g), 4),
        "mean_b": round(float(b), 4),
        "yellowness": round(float((r + g) / 2 - b), 4),  # haze/calima pushes this up
        "contrast": round(float(pixels.std()), 4),
    }


def component_1(path: str) -> dict:
    """Vision: what is wrong with the panel (vision.py)."""
    return vision.describe(path, n_votes=N_VOTES)


# Irradiance and hour: the user can type them in the app (team decision, 2026-11-23).
# If they don't, and the photo is not from DeepSolarEye (whose file names carry
# them), the app uses these typical values and says so in its answer.
DEFAULT_CONTEXT = {"irradiance": 0.5, "hour": 12.0}


def component_2(c1: dict, context: dict) -> dict:
    """Tabular ML: estimated % power loss from component 1's output + irradiance and hour.
    Trained only on soiling (DeepSolarEye), so it only answers for soiled panels."""
    if c1["label"] != "soiled":
        return {"loss_pct": None, "note": "no estimate: the model only learned soiling losses"}
    assumed = [k for k in ("irradiance", "hour") if context.get(k) is None]
    filled = {k: DEFAULT_CONTEXT[k] if context.get(k) is None else context[k] for k in ("irradiance", "hour")}
    result = component2.predict(c1, **filled)
    if assumed:
        result["assumed"] = {k: filled[k] for k in assumed}
    return result


# What the product can tell the maintenance manager to do. evaluate.py compares
# `action` with the `expected` column of eval/cases.csv, so cases use these words.
ACTIONS = ["no_action", "clean", "wait", "check_shade", "technician", "out_of_scope", "unavailable"]


def decide(c1: dict, c2: dict) -> tuple[str, str, bool]:
    """(action, sentence for the user, review flag). Team decisions 4 and 7;
    the cleaning threshold comes from the € formula (economics.py)."""
    if not c1["is_solar_panel"]:
        return "out_of_scope", "The photo does not show a solar panel.", False
    if c1["label"] == "visible_damage":
        return "technician", "Send a technician: visible damage (damage always goes to a person).", True
    review = c1["agreement"] < REVIEW_BELOW_AGREEMENT or c1["label"] == "unknown"
    if c1["label"] == "shaded":
        return "check_shade", "Check vegetation / obstacles: shade is not fixed by cleaning.", review
    if c1["label"] in ("clean", "unknown"):
        return "no_action", "No action: the panel looks clean.", review
    if c2.get("loss_pct") is None:
        return "clean", "Cleaning candidate (no loss estimate).", True
    low, high = c2["loss_range_pct"]
    threshold = 100 * economics.CLEAN_IF_LOSS_ABOVE
    gain = economics.net_gain_per_m2(c2["loss_pct"] / 100)
    if c2["loss_pct"] > threshold:
        return ("clean", f"Clean it: estimated loss {c2['loss_pct']}% (likely {low}-{high}%), above the "
                f"{threshold:.1f}% at which cleaning pays off; about €{gain:.2f} per m² gained "
                f"before the next scheduled cleaning.", review)
    return ("wait", f"Wait for the scheduled cleaning: estimated loss {c2['loss_pct']}% (likely {low}-{high}%), "
            f"below the {threshold:.1f}% at which cleaning pays off.", review)


def _summary(path: str) -> str:
    """What the log keeps about the input: size and type only. Never the photo,
    and not the file name either (a name could contain personal data)."""
    with Image.open(path) as image:
        return f"image {image.width}x{image.height} px, {Path(path).suffix.lower().lstrip('.') or 'unknown type'}"


def run(raw_input: str, irradiance: float | None = None, hour: float | None = None) -> dict:
    """One request. `irradiance` (0-1) and `hour` (0-24) are optional: values the user
    types win over the ones in a DeepSolarEye file name."""
    path = str(raw_input).strip()
    try:
        features = image_features(path)
    except (FileNotFoundError, IsADirectoryError, UnidentifiedImageError, OSError, ValueError):
        return {
            "input_summary": "not an image",
            "features": {},
            "outputs": {"component_1": None, "component_2": None,
                        "action": "out_of_scope", "decision": "Send a photo of a solar panel."},
            "versions": VERSIONS,
            "review": False,
            "tokens": 0,
            "cost_eur": 0.0,
        }

    timing = {}
    context = context_from_name(path)
    if irradiance is not None:
        context["irradiance"] = min(max(float(irradiance), 0.0), 1.0)
    if hour is not None:
        context["hour"] = min(max(float(hour), 0.0), 24.0)
    try:
        started = time.perf_counter()
        c1 = component_1(path)  # 3 separate Gemini calls, one after the other
        timing["component_1"] = round(time.perf_counter() - started, 3)
    except Exception as error:  # Gemini down, over the daily limit, or key missing
        reason = getattr(error, "code", None) or type(error).__name__
        return {
            "input_summary": _summary(path),
            "features": features,  # no agreement: NaN would break the Postgres log (jsonb)
            "outputs": {"component_1": {"error": f"Gemini unavailable ({reason})"}, "component_2": None,
                        "context": context, "action": "unavailable",
                        "decision": "Component 1 unavailable: sent to a person."},
            "versions": VERSIONS,
            "review": True,
            "tokens": 0,
            "cost_eur": 0.0,
        }
    started = time.perf_counter()
    c2 = component_2(c1, context)
    timing["component_2"] = round(time.perf_counter() - started, 4)
    action, decision, review = decide(c1, c2)
    if c2.get("assumed"):
        shown = ", ".join(f"{k} {v}" for k, v in c2["assumed"].items())
        decision += f" (Assumed {shown}: type the real values for a better estimate.)"
    return {
        "input_summary": _summary(path),
        "features": features | {"agreement": c1["agreement"]},
        "outputs": {"component_1": c1, "component_2": c2, "context": context,
                    "action": action, "decision": decision, "timing_s": timing},
        "versions": VERSIONS,
        "review": review,
        "tokens": c1["tokens"],
        "cost_eur": c1["cost_eur"],
    }
