"""
Evaluation. `uv run python main.py evaluate` (this file, or evaluate.ipynb)

The template only runs every case in eval/cases.csv through the pipeline and
reports what does not depend on your product: review rate per kind, latency and
cost. The TODOs are yours:

  1. each component on its own, with the metric that fits it;
  2. `is_correct()` for the end-to-end result, and the comparison with the baseline;
  3. if outputs are free text, an LLM judge, checked against eval/human_labels.csv
     (not needed here: our outputs are labels and numbers).

Write what you report into eval/ (results.csv is written for you) and commit it,
so the numbers in REPORT.md can be traced back.
"""

import csv
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import economics
import env
import pipeline

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
RESULTS = ROOT / "eval" / "results.csv"


def is_correct(case: dict, result: dict) -> bool | None:
    """Is the end-to-end result right for this case?

    `expected` in eval/cases.csv is one of pipeline.ACTIONS (no_action, clean, wait,
    check_shade, technician, out_of_scope). For `review` cases, the product must
    also send the case to a person. None if `expected` is not an action (fix the case).
    """
    expected = case["expected"].strip().lower()
    if expected not in pipeline.ACTIONS:
        return None
    if case["kind"].strip().lower() == "review" and not result["review"]:
        return False
    return _same_decision(result["outputs"]["action"], expected)


# `no_action` (looks clean) and `wait` (dirty, but not worth a visit yet) lead to the
# same thing for the manager: nobody goes to clean. Counted as the same decision.
_DONT_CLEAN = {"no_action", "wait"}


def _same_decision(action: str | None, expected: str) -> bool:
    return action == expected or (action in _DONT_CLEAN and expected in _DONT_CLEAN)


def baseline_correct(case: dict) -> bool | None:
    """Would today's way of working (the `baseline` column, also an action) get it right?"""
    baseline, expected = case["baseline"].strip().lower(), case["expected"].strip().lower()
    if baseline not in pipeline.ACTIONS or expected not in pipeline.ACTIONS:
        return None
    return _same_decision(baseline, expected)


def component_1_metrics() -> None:
    """Component 1 on its own: does Gemini call a panel `soiled` when it really loses
    enough power to be worth cleaning? On the 1,000 labelled DeepSolarEye photos
    (data/deepsolareye_gemini.csv: Gemini's label next to the measured loss)."""
    data = pd.read_csv(ROOT / "data" / "deepsolareye_gemini.csv")
    truth = data["loss"] > economics.CLEAN_IF_LOSS_ABOVE
    said_soiled = data["label"] == "soiled"
    print(f"\n## Component 1 (Gemini), {len(data)} DeepSolarEye photos")
    print(f"True class: loss above the cleaning threshold ({economics.CLEAN_IF_LOSS_ABOVE:.1%}).")
    print(classification_report(truth, said_soiled, target_names=["below threshold", "above threshold"],
                                digits=3, zero_division=0))
    print("Confusion matrix (rows = truth below/above, columns = Gemini not soiled/soiled):")
    print(confusion_matrix(truth, said_soiled))
    print("Labels Gemini gave, by measured loss band:")
    bands = pd.cut(data["loss"], [0, 0.05, 0.2, 0.5, 1.01], right=False,
                   labels=["0-5%", "5-20%", "20-50%", "50-100%"])
    print(pd.crosstab(bands, data["label"], normalize="index").round(2).to_string())
    report = {
        "what": "Gemini label 'soiled' vs measured loss above the cleaning threshold",
        "data": "data/deepsolareye_gemini.csv (1 Gemini call per photo, temperature 0)",
        "n_photos": int(len(data)),
        "threshold_loss": round(float(economics.CLEAN_IF_LOSS_ABOVE), 4),
        "accuracy": round(float((truth == said_soiled).mean()), 3),
        "classification_report": classification_report(
            truth, said_soiled, target_names=["below threshold", "above threshold"],
            output_dict=True, zero_division=0),
        "confusion_matrix": {"rows": "truth below/above", "columns": "Gemini not soiled/soiled",
                             "values": confusion_matrix(truth, said_soiled).tolist()},
        "label_share_by_loss_band": pd.crosstab(bands, data["label"], normalize="index").round(3).to_dict("index"),
    }
    (ROOT / "models" / "component1_report.json").write_text(json.dumps(report, indent=1, default=str))
    print("Saved models/component1_report.json")


def component_2_metrics() -> None:
    """Component 2 on its own: written by tools/train_component2.py (split by day)."""
    report = json.loads((ROOT / "models" / "component2_report.json").read_text())
    table = pd.DataFrame(report["results"])
    print(f"\n## Component 2 (regression), {report['n_train']} train / {report['n_test']} test photos, "
          f"test days {', '.join(report['test_days'])}")
    print(f"Model in the app: {report['provisional_model']}, picked by {report.get('selected_by')}.")
    print(table.to_string(index=False))


def main() -> None:
    env.load_env()
    with CASES.open(newline="", encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if (r.get("input") or "").strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  {i}/{len(cases)}", end="\r", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        rows.append({
            "id": case["id"],
            "kind": case["kind"].strip().lower(),
            "review": result["review"],
            "correct": is_correct(case, result),
            "baseline_correct": baseline_correct(case),
            "action": result["outputs"].get("action"),
            "expected": case["expected"].strip().lower(),
            "latency_s": round(time.perf_counter() - started, 3),
            "tokens": result["tokens"],
            "cost_eur": result["cost_eur"],
            "outputs": result["outputs"],
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    print(f"Wrote {RESULTS.relative_to(ROOT)} ({len(results)} cases).\n")

    by_kind = results.groupby("kind").agg(
        cases=("id", "count"),
        sent_to_review=("review", "mean"),
        correct=("correct", "mean"),
        baseline_correct=("baseline_correct", "mean"),
        median_latency_s=("latency_s", "median"),
        mean_cost_eur=("cost_eur", "mean"),
    )
    print("## End to end, eval/cases.csv")
    print(by_kind.round(3).to_string())
    scored = results.dropna(subset=["correct"])
    if len(scored) < len(results):
        print(f"\n{len(results) - len(scored)} cases have an `expected` that is not one of {pipeline.ACTIONS}: not scored.")
    if len(scored):
        both = scored.dropna(subset=["baseline_correct"])
        print(f"\nProduct correct: {scored['correct'].mean():.0%} of {len(scored)} cases.")
        if len(both):
            print(f"Baseline correct: {both['baseline_correct'].mean():.0%} of {len(both)} cases "
                  f"(product on the same cases: {both['correct'].mean():.0%}).")
        # Second comparison (team decision 2026-11-23): the "always clean" rule.
        always_clean = scored["expected"].map(lambda e: _same_decision("clean", e.strip().lower()))
        print(f"Rule 'always clean' correct: {always_clean.mean():.0%} of {len(scored)} cases.")
        print("\nWhat the product did vs what was expected (rows = expected):")
        print(pd.crosstab(scored["expected"], scored["action"]).to_string())

    print("\n## Cost per request")
    print(f"Tokens: mean {results['tokens'].mean():.0f}. Euros (paid Gemini price): mean €{results['cost_eur'].mean():.5f}, "
          f"€{1000 * results['cost_eur'].mean():.2f} per 1,000 photos. "
          f"Seconds: median {results['latency_s'].median():.1f}, max {results['latency_s'].max():.1f}.")
    timing = results["outputs"].apply(lambda o: o.get("timing_s") or {}).apply(pd.Series)
    if not timing.empty:
        parts = ", ".join(f"{c} median {timing[c].median():.2f} s" for c in timing.columns)
        print(f"Seconds by component: {parts} (component 1 = 3 Gemini calls in a row).")
    print(f"Total cost of this run: €{results['cost_eur'].sum():.4f}, {results['tokens'].sum()} tokens.")

    component_1_metrics()
    component_2_metrics()
    # Outputs are labels and numbers, not free text, so there is no LLM judge.


if __name__ == "__main__":
    main()
