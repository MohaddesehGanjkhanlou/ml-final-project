"""Gradio dashboard for the Jet Engine Hospital exported inference artifacts."""
from pathlib import Path
import json

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = APP_ROOT / "artifacts"
if not ARTIFACT_ROOT.exists():
    ARTIFACT_ROOT = APP_ROOT.parent / "artifacts"

SUBSETS = [p.name.replace("_predictions.csv.gz", "")
           for p in sorted((ARTIFACT_ROOT / "demo").glob("*_predictions.csv.gz"))]
if not SUBSETS:
    raise RuntimeError("No dashboard artifacts found. Run src/production_pipeline.py first.")

_frames = {}
_summaries = {}
def frame(subset):
    if subset not in _frames:
        _frames[subset] = pd.read_csv(ARTIFACT_ROOT / "demo" / f"{subset}_predictions.csv.gz")
    return _frames[subset]

def summary(subset):
    if subset not in _summaries:
        path = ARTIFACT_ROOT / "results" / f"{subset}_summary.json"
        _summaries[subset] = json.loads(path.read_text(encoding="utf-8"))
    return _summaries[subset]

def engines_for_subset(subset):
    values = frame(subset).engine_id.drop_duplicates().astype(int).tolist()
    return gr.Dropdown(choices=values, value=values[0], label="Engine")

def cycles_for_engine(subset, engine):
    values = frame(subset).query("engine_id == @engine").cycle.astype(int).tolist()
    return gr.Slider(minimum=min(values), maximum=max(values), value=max(values),
                     step=1, label="Cycle")

def render(subset, engine, cycle, sensor):
    data = frame(subset)
    history = data[(data.engine_id == int(engine)) & (data.cycle <= int(cycle))].copy()
    row = history.iloc[-1]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(history.cycle, history[sensor], color="#3465a4", linewidth=2)
    axes[0].axvline(row.cycle, color="#c0392b", linestyle="--")
    axes[0].set_ylabel(sensor); axes[0].set_title(f"Engine {int(engine)} timeline")
    axes[1].plot(history.cycle, history.rul_prediction, label="RUL estimate", color="#2e8b57")
    axes[1].fill_between(history.cycle, history.rul_lower, history.rul_upper,
                         color="#2e8b57", alpha=.18, label="90% interval")
    axes[1].axvline(row.cycle, color="#c0392b", linestyle="--")
    axes[1].set(xlabel="cycle", ylabel="cycles remaining"); axes[1].legend()
    fig.tight_layout()

    rul = (f"### RUL\n**{row.rul_prediction:.1f} cycles**  \n"
           f"90% interval: **[{row.rul_lower:.1f}, {row.rul_upper:.1f}]**")
    risk_lines = [f"- {h} cycles: **{row[f'risk_h{h}']:.1%}** "
                  f"(threshold {row[f'threshold_h{h}']:.1%})" for h in (10, 20, 30)]
    risk = "### Calibrated failure risk\n" + "\n".join(risk_lines)
    anomaly = (f"### Anomaly\nNormalized score: **{row.anomaly_normalized_score:.1%}**  \n"
               f"Healthy-reference percentile: **{row.anomaly_percentile:.1%}**  \n"
               f"Threshold: {row.anomaly_threshold:.1%}  \n"
               f"Threshold margin: **{row.anomaly_threshold_margin:+.1%}**  \n"
               f"Persistent: **{'yes' if row.anomaly_persistent else 'no'}**  \n"
               "_The normalized anomaly score is not a probability._")
    colors = {"CONTINUE":"🟢", "INSPECT":"🟠", "STOP":"🔴"}
    action = (f"## {colors[row.recommendation]} {row.recommendation}\n"
              f"Confidence: **{row.decision_confidence}** "
              f"({int(row.signal_agreement_count)}/3 evidence families agree)  \n"
              f"RUL: **{row.rul_evidence_action}** · "
              f"Failure risk: **{row.classification_evidence_action}** · "
              f"Anomaly: **{row.anomaly_evidence_action}**  \n"
              f"Trigger: {row.explanation}  \nNext review: cycle {int(row.cycle)+5}")
    meta = summary(subset)["model_metadata"]
    metadata = (f"Dataset: **{subset}** · Model: **{meta['version']}** · "
                f"Window: **{meta['feature_window']} cycles** · Trained: {meta['trained_utc']}")
    return fig, rul, risk, anomaly, action, metadata

initial = SUBSETS[0]
initial_engine = int(frame(initial).engine_id.iloc[0])
initial_max = int(frame(initial).query("engine_id == @initial_engine").cycle.max())

with gr.Blocks(title="Jet Engine Hospital", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✈️ Jet Engine Hospital\nCalibrated, uncertainty-aware turbofan maintenance support")
    with gr.Row():
        subset = gr.Dropdown(SUBSETS, value=initial, label="Dataset")
        engine = gr.Dropdown(frame(initial).engine_id.drop_duplicates().astype(int).tolist(),
                             value=initial_engine, label="Engine")
        cycle = gr.Slider(1, initial_max, value=initial_max, step=1, label="Cycle")
        sensor = gr.Dropdown([f"sensor_{i}" for i in range(1,22)], value="sensor_2", label="Sensor")
    timeline = gr.Plot(label="Evidence timeline")
    with gr.Row():
        rul_card = gr.Markdown(); risk_card = gr.Markdown(); anomaly_card = gr.Markdown()
    action_card = gr.Markdown(); metadata_card = gr.Markdown()
    inputs=[subset,engine,cycle,sensor]; outputs=[timeline,rul_card,risk_card,anomaly_card,action_card,metadata_card]
    subset.change(engines_for_subset, subset, engine).then(cycles_for_engine,[subset,engine],cycle).then(render,inputs,outputs)
    engine.change(cycles_for_engine,[subset,engine],cycle).then(render,inputs,outputs)
    cycle.change(render,inputs,outputs); sensor.change(render,inputs,outputs)
    demo.load(render,inputs,outputs)

if __name__ == "__main__":
    demo.launch()
