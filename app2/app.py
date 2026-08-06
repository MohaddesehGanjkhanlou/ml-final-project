"""Streamlit dashboard for the Jet Engine Hospital inference artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = APP_ROOT / "artifacts"
DEMO_ROOT = ARTIFACT_ROOT / "demo"
RESULTS_ROOT = ARTIFACT_ROOT / "results"

st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(max_entries=3, show_spinner="Loading engine evidence…")
def load_predictions(subset: str) -> pd.DataFrame:
    """Load one compressed prediction table and normalize selector columns."""
    path = DEMO_ROOT / f"{subset}_predictions.csv.gz"
    data = pd.read_csv(path)
    data["engine_id"] = data["engine_id"].astype(int)
    data["cycle"] = data["cycle"].astype(int)
    return data


@st.cache_data(max_entries=3)
def load_summary(subset: str) -> dict:
    """Load model metadata for a dataset subset."""
    path = RESULTS_ROOT / f"{subset}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def available_subsets() -> list[str]:
    """Return datasets that have both predictions and summary metadata."""
    if not DEMO_ROOT.exists() or not RESULTS_ROOT.exists():
        return []
    subsets = []
    for path in sorted(DEMO_ROOT.glob("*_predictions.csv.gz")):
        subset = path.name.removesuffix("_predictions.csv.gz")
        if (RESULTS_ROOT / f"{subset}_summary.json").is_file():
            subsets.append(subset)
    return subsets


def make_timeline(history: pd.DataFrame, sensor: str, selected_cycle: int):
    """Build the sensor and remaining-useful-life evidence chart."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    cycle = history["cycle"].to_numpy()

    axes[0].plot(cycle, history[sensor].to_numpy(), color="#2563eb", linewidth=2)
    axes[0].axvline(selected_cycle, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel(sensor.replace("_", " ").title())
    axes[0].set_title("Observed sensor history")

    prediction = history["rul_prediction"].to_numpy()
    lower = history["rul_lower"].to_numpy()
    upper = history["rul_upper"].to_numpy()
    axes[1].plot(cycle, prediction, label="RUL estimate", color="#15803d", linewidth=2)
    axes[1].fill_between(
        cycle,
        lower,
        upper,
        color="#22c55e",
        alpha=0.18,
        label="90% interval",
    )
    axes[1].axvline(selected_cycle, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Cycles remaining")
    axes[1].set_title("Remaining useful life")
    axes[1].legend(loc="best")
    figure.tight_layout()
    return figure


def percent(value: float) -> str:
    return f"{float(value):.1%}"


subsets = available_subsets()
if not subsets:
    st.error(
        "No dashboard artifacts were found. Expected compressed predictions in "
        "`app2/artifacts/demo` and matching summaries in `app2/artifacts/results`."
    )
    st.stop()

st.title("✈️ Jet Engine Hospital")
st.caption("Calibrated, uncertainty-aware turbofan maintenance support")

with st.sidebar:
    st.header("Engine selection")
    subset = st.selectbox("Dataset", subsets, help="NASA C-MAPSS dataset subset")
    data = load_predictions(subset)

    engines = data["engine_id"].drop_duplicates().sort_values().tolist()
    engine_id = st.selectbox("Engine", engines)
    engine_data = data.loc[data["engine_id"] == engine_id].sort_values("cycle")

    minimum_cycle = int(engine_data["cycle"].min())
    maximum_cycle = int(engine_data["cycle"].max())
    selected_cycle = st.slider(
        "Cycle",
        min_value=minimum_cycle,
        max_value=maximum_cycle,
        value=maximum_cycle,
        step=1,
    )

    sensors = sorted(
        (column for column in data.columns if column.startswith("sensor_")),
        key=lambda name: int(name.split("_")[1]),
    )
    default_sensor = sensors.index("sensor_2") if "sensor_2" in sensors else 0
    sensor = st.selectbox("Sensor", sensors, index=default_sensor)

    st.divider()
    st.caption(
        "This dashboard reads frozen, validation-selected outputs bundled with the "
        "repository. It does not upload data or run model training."
    )

history = engine_data.loc[engine_data["cycle"] <= selected_cycle].copy()
if history.empty:
    st.error("No evidence is available for this engine and cycle.")
    st.stop()

row = history.iloc[-1]
recommendation = str(row["recommendation"])
state = {
    "CONTINUE": ("🟢", st.success),
    "INSPECT": ("🟠", st.warning),
    "STOP": ("🔴", st.error),
}
state_icon, state_message = state.get(recommendation, ("⚪", st.info))

st.subheader(f"{subset} · Engine {engine_id} · Cycle {selected_cycle}")
state_message(
    f"{state_icon} **{recommendation}** — {row['explanation']}  "
    f"Next scheduled review: cycle {selected_cycle + 5}."
)

metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("Predicted RUL", f"{row['rul_prediction']:.1f} cycles")
    st.caption(f"90% interval: {row['rul_lower']:.1f}–{row['rul_upper']:.1f} cycles")
with metric_columns[1]:
    st.metric("Decision confidence", str(row["decision_confidence"]).title())
    st.caption(f"{int(row['signal_agreement_count'])}/3 evidence families agree")
with metric_columns[2]:
    st.metric("Anomaly percentile", percent(row["anomaly_percentile"]))
    st.caption("Relative to healthy-reference evidence")
with metric_columns[3]:
    st.metric("Current action", recommendation.title())
    st.caption(f"Review again at cycle {selected_cycle + 5}")

overview_tab, evidence_tab, details_tab = st.tabs(
    ["Overview", "Evidence", "Model details"]
)

with overview_tab:
    figure = make_timeline(history, sensor, selected_cycle)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.subheader("Calibrated failure risk")
    risk_columns = st.columns(3)
    for column, horizon in zip(risk_columns, (10, 20, 30)):
        risk = float(row[f"risk_h{horizon}"])
        threshold = float(row[f"threshold_h{horizon}"])
        with column:
            st.metric(f"Within {horizon} cycles", percent(risk))
            st.progress(min(max(risk, 0.0), 1.0))
            st.caption(f"Decision threshold: {percent(threshold)}")

with evidence_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Anomaly evidence")
        st.metric("Normalized score", percent(row["anomaly_normalized_score"]))
        st.metric("Healthy-reference percentile", percent(row["anomaly_percentile"]))
        st.write(f"Threshold: **{percent(row['anomaly_threshold'])}**")
        st.write(f"Threshold margin: **{float(row['anomaly_threshold_margin']):+.1%}**")
        st.write(
            "Persistent anomaly: "
            f"**{'Yes' if bool(row['anomaly_persistent']) else 'No'}**"
        )
        st.caption("The normalized anomaly score is not a probability.")

    with right:
        st.subheader("Evidence-family decisions")
        evidence = pd.DataFrame(
            {
                "Evidence family": ["RUL interval", "Failure risk", "Anomaly"],
                "Action": [
                    row["rul_evidence_action"],
                    row["classification_evidence_action"],
                    row["anomaly_evidence_action"],
                ],
            }
        )
        st.dataframe(evidence, hide_index=True, width="stretch")
        st.write(f"**Trigger:** {row['explanation']}")

    with st.expander("Selected-cycle values"):
        selected_columns = [
            "engine_id",
            "cycle",
            sensor,
            "rul_prediction",
            "rul_lower",
            "rul_upper",
            "risk_h10",
            "risk_h20",
            "risk_h30",
            "anomaly_normalized_score",
            "recommendation",
        ]
        st.dataframe(
            row[selected_columns].astype(str).rename("Value").to_frame(),
            width="stretch",
        )

with details_tab:
    metadata = load_summary(subset).get("model_metadata", {})
    st.subheader("Frozen artifact metadata")
    model_columns = st.columns(3)
    model_columns[0].metric("Model version", metadata.get("version", "Unknown"))
    model_columns[1].metric("Feature window", f"{metadata.get('feature_window', '—')} cycles")
    model_columns[2].metric("Dataset", subset)
    st.write(f"**Trained UTC:** {metadata.get('trained_utc', 'Unknown')}")
    st.info(
        "The displayed values are precomputed inference artifacts. This dashboard is "
        "for decision support and demonstration, not an autonomous maintenance command."
    )
