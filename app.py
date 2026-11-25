"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import os
import time

import gradio as gr
import pandas as pd

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Solar Triage"
DESCRIPTION = "For maintenance managers of solar plants: upload a photo of a panel and get what is wrong (dirt, shade, damage), the estimated power loss and what to do."


def serve(raw_input: str, irradiance: float | None = None, hour: float | None = None) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input, irradiance=irradiance, hour=hour)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def on_run(raw_input, irradiance=None, hour=None):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Upload a photo of a solar panel first.")
    record = serve(raw_input, irradiance=irradiance, hour=hour)
    outputs = record["outputs"]
    status = "⚠️ **Sent to a person for review.**" if record["review"] else "✅ Handled automatically."
    return (
        outputs["component_1"],
        outputs["component_2"],
        f"### {outputs['decision']}",
        f"{status}  Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}.",
        record["id"],
    )


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        # With fewer requests the chance-level PSI is huge (threshold ~8 with 1 request),
        # so the check stays off instead of showing a meaningless threshold.
        drift_table = pd.DataFrame(columns=["feature", "psi", "threshold", "alert"])
        alert_text += (f"\n\nDrift check starts after {monitor.MIN_WINDOW} requests "
                       f"(now {len(logs)}).")
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


# Gradio keeps a temporary copy of each uploaded photo on the server's disk:
# delete copies older than 1 hour, every hour. The logs never store the photo.
with gr.Blocks(title=TITLE, delete_cache=(3600, 3600)) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        raw_input = gr.Image(type="filepath", label="Photo of the solar panel")
        with gr.Row():
            irradiance = gr.Number(value=None, minimum=0, maximum=1, label="Irradiance (optional, 0-1 as in DeepSolarEye; 1 = strongest sun)")
            hour = gr.Number(value=None, minimum=0, maximum=24, label="Hour the photo was taken (optional, 0-24)")
        run_button = gr.Button("Run", variant="primary")
        api_button = gr.Button(visible=False)  # keeps /run photo-only for check.py
        decision = gr.Markdown()
        with gr.Row():
            out_a = gr.JSON(label="Component 1 · vision (Gemini)")
            out_b = gr.JSON(label="Component 2 · estimated power loss")
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        run_outputs = [out_a, out_b, decision, status, request_id]
        run_button.click(on_run, [raw_input, irradiance, hour], run_outputs, api_name=False)
        api_button.click(on_run, raw_input, run_outputs, api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
