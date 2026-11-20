"""
Component 1 (vision): Gemini looks at a photo of a solar panel and describes it.

Team decisions (see REPORT.md):
- Pretrained model through an API, version pinned, JSON answer.
- Main answer at temperature 0. Confidence = how many of N answers agree on the
  label; the extra answers use a higher temperature so they can disagree.
  Gemini's self-reported confidence is NOT used.
- 4 labels: clean / soiled / shaded / visible_damage.

Uses the REST API with the standard library only, so the app stays small enough
for Render's free plan.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "panel-v1"
LABELS = ["clean", "soiled", "shaded", "visible_damage"]
VOTE_TEMPERATURE = 0.8

# Price per million tokens in EUR, for the cost column. The free tier costs 0 €,
# but we report what the product would cost on the paid plan:
# $0.30 input / $2.50 output (thinking included) per 1M tokens, Google DeepMind
# model page https://deepmind.google/models/gemini/flash-lite/ (checked 2026-11-20),
# at 0.86 €/$. TODO (team): confirm on https://ai.google.dev/pricing before submitting.
USD_TO_EUR = 0.86
EUR_PER_M_INPUT_TOKENS = 0.30 * USD_TO_EUR
EUR_PER_M_OUTPUT_TOKENS = 2.50 * USD_TO_EUR

PROMPT = f"""You inspect solar panels for a maintenance company.
Look at the large solar panel in the foreground (the one closest to the camera).
Answer only about that panel.

Return JSON with exactly these keys:
- "label": one of {LABELS}
    clean = no visible dirt, shade or damage
    soiled = dust, sand, bird droppings or other material on the glass
    shaded = a shadow falls on part of the panel
    visible_damage = cracks, broken glass, burn marks or other physical damage
- "n_spots": integer, number of separate dirty, shaded or damaged spots (0 if clean)
- "covered_fraction": number 0-1, share of the panel surface that is covered or affected
- "spread": "none", "one_area" or "several_areas" (are the spots together or spread over the panel?)
- "is_solar_panel": true or false (false if the photo does not show a solar panel)
"""

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class MissingKey(RuntimeError):
    pass


def _call(image_bytes: bytes, mime: str, temperature: float, retries: int = 4) -> tuple[dict, int, int]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingKey("GEMINI_API_KEY is not set: see SETUP.md and .env.example.")
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}},
        ]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": temperature},
    }
    request = urllib.request.Request(
        _URL.format(model=MODEL),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            # output is billed with the model's thinking tokens
            tokens_out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            return json.loads(text), usage.get("promptTokenCount", 0), tokens_out
        except urllib.error.HTTPError as error:
            # 429 = rate limit, 5xx = Gemini overloaded: wait and retry
            if error.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            # network hiccup or Gemini too slow: wait and retry
            if attempt < retries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
        except (json.JSONDecodeError, KeyError, IndexError):
            if attempt < retries - 1:
                continue
            raise
    raise RuntimeError("unreachable")


def _clean(answer: dict) -> dict:
    """Make the answer safe to use: known label, numbers in range."""
    label = answer.get("label") if answer.get("label") in LABELS else "unknown"
    spread = answer.get("spread") if answer.get("spread") in ("none", "one_area", "several_areas") else "unknown"
    try:
        n_spots = max(0, int(answer.get("n_spots", 0)))
    except (TypeError, ValueError):
        n_spots = 0
    try:
        covered = min(1.0, max(0.0, float(answer.get("covered_fraction", 0))))
    except (TypeError, ValueError):
        covered = 0.0
    return {
        "label": label,
        "n_spots": n_spots,
        "covered_fraction": round(covered, 3),
        "spread": spread,
        "is_solar_panel": bool(answer.get("is_solar_panel", True)),
    }


def describe(image_path: str, n_votes: int = 1) -> dict:
    """Gemini's description of the panel in the photo.

    n_votes=1: one call at temperature 0 (used to label training data).
    n_votes=3: the main call plus 2 at VOTE_TEMPERATURE; `agreement` is the share
    of answers with the main label (used in the app as the confidence).
    """
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    image_bytes = path.read_bytes()

    answer, tokens_in, tokens_out = _call(image_bytes, mime, temperature=0)
    main = _clean(answer)
    labels = [main["label"]]
    for _ in range(n_votes - 1):
        vote, t_in, t_out = _call(image_bytes, mime, temperature=VOTE_TEMPERATURE)
        labels.append(_clean(vote)["label"])
        tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out

    agreement = Counter(labels)[main["label"]] / len(labels)
    cost = (tokens_in * EUR_PER_M_INPUT_TOKENS + tokens_out * EUR_PER_M_OUTPUT_TOKENS) / 1e6
    return main | {
        "agreement": round(agreement, 3),
        "votes": labels,
        "tokens": tokens_in + tokens_out,
        "cost_eur": round(cost, 6),
    }
