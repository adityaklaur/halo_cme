from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from src.dashboard_utils import fingerprint_figure, overview_figure


ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
SCI = ROOT / "data" / "scientific"
PHASE3 = ROOT / "outputs" / "phase3"
PHASE4 = ROOT / "outputs" / "phase4"
PHASE5 = ROOT / "outputs" / "phase5"
PHASE6 = ROOT / "outputs" / "phase6"
ENABLE_DATA_MANAGER = False


REPLAY_DATASETS = {
    "August 2024": {
        "source_id": "aug2024",
        "feature_file": PROC / "aug2024_features_1min.csv",
        "spectra_file": PROC / "aug2024_spectra_1min.npz",
        "report_file": PROC / "pipeline_report.json",
        "detector_available": True,
        "phase6_interval_id": "AUG2024_DEMO_INTERVAL",
    },
    "September 2024": {
        "source_id": "sep2024",
        "feature_file": PROC / "events" / "sep2024_features_1min.csv",
        "spectra_file": PROC / "events" / "sep2024_spectra_1min.npz",
        "report_file": PROC / "events" / "sep2024_report.json",
        "detector_available": False,
        "phase6_interval_id": "SEP2024_QUIET_INTERVAL",
    },
    "October 2024": {
        "source_id": "oct2024",
        "feature_file": PROC / "events" / "oct2024_features_1min.csv",
        "spectra_file": PROC / "events" / "oct2024_spectra_1min.npz",
        "report_file": PROC / "events" / "oct2024_report.json",
        "detector_available": False,
        "phase6_interval_id": "OCT2024_ICME_INTERVAL",
    },
    "November 2024": {
        "source_id": "nov2024",
        "feature_file": PROC / "events" / "nov2024_features_1min.csv",
        "spectra_file": PROC / "events" / "nov2024_spectra_1min.npz",
        "report_file": PROC / "events" / "nov2024_report.json",
        "detector_available": False,
        "phase6_interval_id": None,
    },
    "March 2025": {
        "source_id": "mar2025",
        "feature_file": PROC / "events" / "mar2025_features_1min.csv",
        "spectra_file": PROC / "events" / "mar2025_spectra_1min.npz",
        "report_file": PROC / "events" / "mar2025_report.json",
        "detector_available": False,
        "phase6_interval_id": "MAR2025_CIR_HSS_INTERVAL",
    },
}


st.set_page_config(page_title="TopoCross-SWIS - Phases 2 to 6", layout="wide")


@st.cache_data(show_spinner=False)
def load_all():
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    npz = np.load(PROC / "aug2024_spectra_1min.npz")
    spec = {k: npz[k] for k in npz.files}
    with open(PROC / "pipeline_report.json", "r", encoding="utf-8") as file:
        report = json.load(file)
    candidates = pd.read_csv(PROC / "cme_candidate_ranking.csv", parse_dates=["cme_time"])
    tests = pd.read_csv(PROC / "state_statistical_tests.csv")
    labels = pd.read_csv(ROOT / "data" / "labels" / "event_boundaries.csv", parse_dates=["boundary_time_utc"])
    phase2_catalog = pd.read_csv(SCI / "phase2_event_catalog.csv", parse_dates=["start_utc", "end_utc"])
    phase2_coverage = pd.read_csv(SCI / "phase2_modality_coverage.csv")
    phase2_queue = pd.read_csv(SCI / "phase2_acquisition_queue.csv")
    with open(SCI / "phase2_manifest.json", "r", encoding="utf-8") as file:
        phase2_manifest = json.load(file)
    phase3_counts = pd.read_csv(PHASE3 / "phase3_label_counts.csv")
    phase3_events = pd.read_csv(PHASE3 / "phase3_event_register.csv", parse_dates=["event_start", "event_end"])
    phase3_boundaries = pd.read_csv(PHASE3 / "phase3_boundary_register.csv", parse_dates=["boundary_time_utc"])
    with open(PHASE3 / "phase3_report.json", "r", encoding="utf-8") as file:
        phase3_report = json.load(file)
    phase4_summary = pd.read_csv(PHASE4 / "phase4_event_summary.csv")
    phase4_dictionary = pd.read_csv(PHASE4 / "phase4_feature_dictionary.csv")
    with open(PHASE4 / "phase4_report.json", "r", encoding="utf-8") as file:
        phase4_report = json.load(file)
    phase5_summary = pd.read_csv(PHASE5 / "phase5_summary_metrics.csv")
    phase5_folds = pd.read_csv(PHASE5 / "phase5_fold_metrics.csv")
    phase5_delays = pd.read_csv(PHASE5 / "phase5_detection_delays.csv", parse_dates=["reference_positive_onset", "detected_at"])
    with open(PHASE5 / "phase5_report.json", "r", encoding="utf-8") as file:
        phase5_report = json.load(file)
    phase6_summary = pd.read_csv(PHASE6 / "phase6_summary_metrics.csv")
    phase6_folds = pd.read_csv(PHASE6 / "phase6_fold_metrics.csv")
    phase6_importance = pd.read_csv(PHASE6 / "phase6_feature_importance.csv")
    phase6_delays = pd.read_csv(PHASE6 / "phase6_detection_delays.csv", parse_dates=["reference_positive_onset", "detected_at"])
    with open(PHASE6 / "phase6_report.json", "r", encoding="utf-8") as file:
        phase6_report = json.load(file)
    return (
        df,
        spec,
        report,
        candidates,
        tests,
        labels,
        phase2_catalog,
        phase2_coverage,
        phase2_queue,
        phase2_manifest,
        phase3_counts,
        phase3_events,
        phase3_boundaries,
        phase3_report,
        phase4_summary,
        phase4_dictionary,
        phase4_report,
        phase5_summary,
        phase5_folds,
        phase5_delays,
        phase5_report,
        phase6_summary,
        phase6_folds,
        phase6_importance,
        phase6_delays,
        phase6_report,
    )


DETECTOR_FLOAT_COLUMNS = [
    "z_js_opdi",
    "z_hellinger_opdi",
    "z_wasserstein_opdi",
    "z_proton_bulk_speed",
    "z_proton_density",
    "z_proton_thermal",
    "z_alpha_proton_ratio",
    "z_bmag_gse",
    "z_Bx_gse",
    "z_By_gse",
    "z_Bz_gse",
    "opdi_anomaly_score",
    "plasma_anomaly_score",
    "mag_anomaly_score",
    "conventional_anomaly_score",
    "combined_anomaly_score",
    "transition_component_bmag_gse",
    "transition_component_proton_bulk_speed",
    "transition_component_proton_thermal",
    "transition_component_proton_density",
    "transition_component_js_opdi",
    "transition_score",
]


def ensure_replay_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize every processed source to the columns used by the replay UI.

    August contains the prototype detector outputs. The independent September,
    October, November and March products intentionally contain measured and
    processed observables only. Missing detector columns therefore remain NaN;
    they are never inferred from labels or substituted with OMNI values.
    """
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    for column in DETECTOR_FLOAT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    if "transition_threshold_exceeded" not in out.columns:
        out["transition_threshold_exceeded"] = False
    if "is_change_point" not in out.columns:
        out["is_change_point"] = False
    else:
        out["is_change_point"] = out["is_change_point"].fillna(False).astype(bool)
    if "state" not in out.columns:
        out["state"] = "NOT AVAILABLE"
    else:
        out["state"] = out["state"].fillna("NOT AVAILABLE").astype(str)
    if "ground_truth_state" not in out.columns:
        out["ground_truth_state"] = np.nan
    return out.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_replay_dataset(label: str):
    meta = REPLAY_DATASETS[label]
    frame = ensure_replay_columns(pd.read_csv(meta["feature_file"], parse_dates=["timestamp"]))
    with np.load(meta["spectra_file"]) as npz:
        spectra = {key: npz[key] for key in npz.files}
    with open(meta["report_file"], "r", encoding="utf-8") as file:
        source_report = json.load(file)
    return frame, spectra, source_report


@st.cache_data(show_spinner=False)
def load_phase6_predictions() -> pd.DataFrame:
    """Load minute-level Phase 6 held-out predictions for dashboard replay."""
    path = PHASE6 / "phase6_predictions.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "independent_interval_id",
                "timestamp",
                "model",
                "probability",
                "raw_predicted_binary",
                "predicted_binary",
            ]
        )
    predictions = pd.read_csv(path, parse_dates=["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    return predictions


def phase6_prediction_at(
    predictions: pd.DataFrame,
    interval_id: str | None,
    timestamp: pd.Timestamp,
    model: str | None = None,
) -> dict | None:
    """Return an exact held-out Phase 6 dashboard prediction for one minute.

    The dashboard intentionally uses the mean probability from all available
    Phase 6 baseline models at the exact timestamp.  Logistic Regression can
    become numerically saturated near 0/1 on held-out intervals, which made a
    rounded single-model confidence appear as 100% for long stretches.
    Averaging the independent baseline probabilities produces a more useful
    dashboard confidence without changing the formal Phase 6 model outputs,
    thresholds, persistence rule, or evaluation metrics.

    No nearest-neighbour or interpolation fallback is used.
    """
    if not interval_id or predictions.empty:
        return None

    matches = predictions.loc[
        (predictions["independent_interval_id"] == interval_id)
        & (predictions["timestamp"] == pd.Timestamp(timestamp))
    ].copy()
    if matches.empty:
        return None

    probabilities = pd.to_numeric(matches["probability"], errors="coerce").dropna()
    if probabilities.empty:
        return None

    event_probability = float(np.clip(probabilities.mean(), 0.0, 1.0))
    positive = event_probability >= 0.5
    confidence = event_probability if positive else 1.0 - event_probability
    model_names = sorted(matches.loc[probabilities.index, "model"].astype(str).unique())

    return {
        "prediction": "EVENT DETECTED" if positive else "NORMAL",
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "event_probability": event_probability,
        "model": "Ensemble (" + ", ".join(model_names) + ")",
        "model_count": len(model_names),
    }


def replay_event_rows(catalog: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    return catalog.loc[(catalog["end_utc"] >= start) & (catalog["start_utc"] <= end)].copy()


def replay_boundaries(
    replay_label: str,
    catalog: pd.DataFrame,
    phase3_boundaries: pd.DataFrame,
    august_labels: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if replay_label == "August 2024":
        return august_labels.copy()
    events = replay_event_rows(catalog, frame)
    return phase3_boundaries.loc[phase3_boundaries["event_id"].isin(events["event_id"])].copy()


def fmt_metric(value, pattern: str) -> str:
    return format(float(value), pattern) if pd.notna(value) else "NA"


def run_rebuild():
    return run_script_with_terminal_logs(ROOT / "scripts" / "rebuild_processed.py")


def run_uploaded_swis_only_build():
    selected = st.session_state.get("selected_swis_only_pairs", [])
    extra_args = ["--pairs", ",".join(selected)] if selected else []
    return run_script_with_terminal_logs(ROOT / "scripts" / "build_uploaded_swis_only.py", extra_args)


def run_phase2_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "build_phases2_and3.py")


def run_phase3_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "phase3_ground_truth.py")


def run_phase4_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "build_phase4_features.py")


def run_phase5_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "run_phase5_experiment.py")


def run_phase6_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "run_phase6_ml.py")


def run_phases2_to6_build():
    return run_script_with_terminal_logs(ROOT / "scripts" / "build_phases2_to6.py")


def run_script_with_terminal_logs(script_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(script_path), *(extra_args or [])]
    print(f"[TopoCross] Starting: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    print(f"[TopoCross] Finished with exit code {return_code}", flush=True)
    return subprocess.CompletedProcess(command, return_code, stdout="".join(lines), stderr="")


def process_output(result: subprocess.CompletedProcess) -> str:
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    return output or "Full logs were streamed to the terminal running Streamlit."


def save_uploaded_raw_files(files, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in files or []:
        safe_name = Path(uploaded.name).name
        target = destination / safe_name
        target.write_bytes(uploaded.getbuffer())
        saved.append(target)
    return saved


def available_swis_only_pairs() -> list[str]:
    pattern = re.compile(r"_(20\d{6})_.*_(V\d+)\.cdf$")
    pairs_by_kind = []
    for kind in ["th1", "th2"]:
        folder = ROOT / "data" / "raw" / kind
        pairs = set()
        if folder.exists():
            for path in folder.glob("*.cdf"):
                match = pattern.search(path.name)
                if match:
                    pairs.add(f"{match.group(1)}_{match.group(2)}")
        pairs_by_kind.append(pairs)
    return sorted(set.intersection(*pairs_by_kind)) if pairs_by_kind else []


def clear_uploaded_raw_files() -> None:
    raw = ROOT / "data" / "raw"
    for child in ["th1", "th2", "blk", "mag"]:
        folder = raw / child
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def full_rebuild_coverage() -> tuple[pd.DataFrame, bool]:
    cfg = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text())
    version = cfg["swis"]["version"]
    start = pd.Timestamp(cfg["processing"]["start_date"])
    end = pd.Timestamp(cfg["processing"]["end_date"])
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="D")]
    specs = {
        "TH1": [(ROOT / "data" / "raw" / "th1", lambda d: f"*{d}*{version}.cdf"), (ROOT / "tha1", lambda d: f"*{d}*{version}.cdf")],
        "TH2": [(ROOT / "data" / "raw" / "th2", lambda d: f"*{d}*{version}.cdf"), (ROOT / "tha2", lambda d: f"*{d}*{version}.cdf")],
        "BLK": [(ROOT / "data" / "raw" / "blk", lambda d: f"*{d}*{version}.cdf"), (ROOT / "swis_BLK", lambda d: f"*{d}*{version}.cdf")],
        "MAG L2": [(ROOT / "data" / "raw" / "mag", lambda d: f"L2_AL1_MAG_{d}_V00.nc"), (ROOT / "mag_2026Aug23T210145602", lambda d: f"L2_AL1_MAG_{d}_V00.nc")],
    }
    rows = []
    all_present = True
    for date in dates:
        row = {"date": date}
        for label, locations in specs.items():
            found = False
            for folder, pattern in locations:
                if folder.exists() and list(folder.glob(pattern(date))):
                    found = True
                    break
            row[label] = "OK" if found else "MISSING"
            all_present = all_present and found
        rows.append(row)
    return pd.DataFrame(rows), all_present


def event_lines(fig, boundaries, rows):
    for _, boundary in boundaries.iterrows():
        time = pd.Timestamp(boundary["boundary_time_utc"])
        for row in rows:
            fig.add_vline(x=time, line_dash="dash", row=row, col=1, opacity=0.55)
    return fig


def spectrogram_pair(spec, lo, hi):
    time = pd.to_datetime(spec["time"].astype("datetime64[ms]"))
    mask = (time >= lo) & (time <= hi)
    energy = spec["energy"]
    f1 = np.asarray(spec["th1_flux"], float)[mask]
    f2 = np.asarray(spec["th2_flux"], float)[mask]
    z1 = np.log10(np.where(f1 > 0, f1, np.nan)).T
    z2 = np.log10(np.where(f2 > 0, f2, np.nan)).T
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["TH1 energy-time spectrum", "TH2 energy-time spectrum"],
    )
    fig.add_trace(go.Heatmap(x=time[mask], y=energy, z=z1, coloraxis="coloraxis"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=time[mask], y=energy, z=z2, coloraxis="coloraxis"), row=2, col=1)
    fig.update_yaxes(type="log", title="Energy (eV)", row=1, col=1)
    fig.update_yaxes(type="log", title="Energy (eV)", row=2, col=1)
    fig.update_xaxes(title="UTC", row=2, col=1)
    fig.update_layout(
        height=650,
        coloraxis_colorbar=dict(title="log10 flux"),
        margin=dict(l=40, r=20, t=55, b=40),
    )
    return fig


def event_context(df, lo, hi, labels):
    d = df[(df.timestamp >= lo) & (df.timestamp <= hi)]
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[
            "Cross-plane divergence (OPDI)",
            "SWIS proton plasma moments",
            "Aditya-L1 MAG (GSE)",
            "Automatic transition score",
            "Detector evidence",
        ],
    )
    for column, name in [("js_opdi", "JS"), ("hellinger_opdi", "Hellinger"), ("wasserstein_opdi", "Wasserstein")]:
        fig.add_trace(go.Scatter(x=d.timestamp, y=d[column], mode="lines", name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.timestamp, y=d.proton_bulk_speed, mode="lines", name="Proton bulk speed"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.timestamp, y=d.proton_thermal, mode="lines", name="Proton thermal speed"), row=2, col=1)
    for column in ["Bx_gse", "By_gse", "Bz_gse", "bmag_gse"]:
        fig.add_trace(go.Scatter(x=d.timestamp, y=d[column], mode="lines", name=column), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.timestamp, y=d.transition_score, mode="lines", name="Transition score"), row=4, col=1)
    cps = d[d.is_change_point.astype(bool)]
    fig.add_trace(go.Scatter(x=cps.timestamp, y=cps.transition_score, mode="markers", name="Detected transition"), row=4, col=1)
    for column, name in [
        ("opdi_anomaly_score", "OPDI-only"),
        ("conventional_anomaly_score", "Conventional"),
        ("combined_anomaly_score", "Combined"),
    ]:
        fig.add_trace(go.Scatter(x=d.timestamp, y=d[column], mode="lines", name=name), row=5, col=1)
    event_lines(fig, labels, range(1, 6))
    fig.update_yaxes(title="distance", row=1, col=1)
    fig.update_yaxes(title="km/s", row=2, col=1)
    fig.update_yaxes(title="nT", row=3, col=1)
    fig.update_yaxes(title="score", row=4, col=1)
    fig.update_yaxes(title="score", row=5, col=1)
    fig.update_xaxes(title="UTC", row=5, col=1)
    fig.update_layout(height=1050, hovermode="x unified", legend=dict(orientation="h"), margin=dict(l=40, r=20, t=55, b=40))
    return fig


def explain_figure(row):
    pairs = [
        ("OPDI level divergence", row.get("opdi_anomaly_score", np.nan)),
        ("Plasma moments", row.get("plasma_anomaly_score", np.nan)),
        ("Magnetic field", row.get("mag_anomaly_score", np.nan)),
        ("Transition", row.get("transition_score", np.nan)),
    ]
    labels = [pair[0] for pair in pairs]
    values = [max(0, float(pair[1])) if pd.notna(pair[1]) else 0 for pair in pairs]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    fig.update_layout(
        height=300,
        title="Why did the detector react?",
        xaxis_title="evidence score",
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


st.title("TopoCross-SWIS - Phases 2 to 6")
st.caption("Aditya-L1 multi-event registry, ground truth, feature engineering, OPDI ablation, and baseline machine learning")

if ENABLE_DATA_MANAGER:
    with st.sidebar:
        st.header("Data controls")
        if st.button("Refresh loaded data"):
            load_all.clear()
            st.rerun()
        if st.button("Rebuild processed dataset"):
            with st.spinner("Rebuilding from local raw CDF/NetCDF files..."):
                result = run_rebuild()
            load_all.clear()
            if result.returncode == 0:
                st.success("Rebuild complete. Refreshing dashboard...")
                st.code(process_output(result), language="text")
                st.rerun()
            else:
                st.error("Rebuild failed.")
                st.code(process_output(result), language="text")

try:
    (
        aug_df,
        aug_spec,
        aug_report,
        aug_candidates,
        tests,
        aug_labels,
        phase2_catalog,
        phase2_coverage,
        phase2_queue,
        phase2_manifest,
        phase3_counts,
        phase3_events,
        phase3_boundaries,
        phase3_report,
        phase4_summary,
        phase4_dictionary,
        phase4_report,
        phase5_summary,
        phase5_folds,
        phase5_delays,
        phase5_report,
        phase6_summary,
        phase6_folds,
        phase6_importance,
        phase6_delays,
        phase6_report,
    ) = load_all()
except FileNotFoundError:
    st.warning("Processed outputs are not built yet.")
    st.code("python3 scripts/rebuild_processed.py\nstreamlit run app.py", language="bash")
    if st.button("Build processed data now"):
        with st.spinner("Building from local raw CDF/NetCDF files..."):
            result = run_rebuild()
        if result.returncode == 0:
            load_all.clear()
            st.success("Build complete. Reloading...")
            st.rerun()
        st.error("Build failed.")
        st.code(process_output(result), language="text")
    st.stop()

with st.sidebar:
    st.header("Replay dataset")
    selected_replay_label = st.selectbox(
        "Processed source",
        list(REPLAY_DATASETS.keys()),
        index=0,
        key="replay_dataset_selector",
    )

selected_replay_meta = REPLAY_DATASETS[selected_replay_label]
if selected_replay_label == "August 2024":
    df = ensure_replay_columns(aug_df)
    spec = aug_spec
    replay_report = aug_report
else:
    df, spec, replay_report = load_replay_dataset(selected_replay_label)

labels = replay_boundaries(selected_replay_label, phase2_catalog, phase3_boundaries, aug_labels, df)
selected_events = replay_event_rows(phase2_catalog, df)

if selected_replay_meta["detector_available"]:
    st.warning(
        "Scientific-status note: the configured August shock reference is approximate. "
        "The detector does not use event labels as inputs; timing offsets are exploratory until the boundary is reconciled."
    )
else:
    source_gaps = []
    missing_swis = replay_report.get("missing_swis_dates", [])
    partial_swis = replay_report.get("partial_swis_dates", [])
    missing_mag = replay_report.get("missing_mag_dates", [])
    partial_mag = replay_report.get("partial_mag_dates", [])
    if missing_swis:
        source_gaps.append("missing SWIS: " + ", ".join(missing_swis))
    if partial_swis:
        source_gaps.append("partial SWIS: " + ", ".join(partial_swis))
    if missing_mag:
        source_gaps.append("missing MAG: " + ", ".join(missing_mag))
    if partial_mag:
        source_gaps.append("partial MAG: " + ", ".join(partial_mag))
    st.info(
        f"{selected_replay_label} is loaded from the independent multi-event processed source. "
        "Measured/processed SWIS, MAG, spectra, and OMNI reference columns are shown. "
        "Prototype detector state/anomaly outputs were not generated for this source. "
        "The dashboard therefore reports the prototype detector as Not Available and shows the Phase 6 held-out ML prediction separately when that exact minute was evaluated."
    )
    if source_gaps:
        st.warning("Source-quality note: " + "; ".join(source_gaps))

if selected_replay_label == "August 2024":
    primary = aug_report.get("primary_transition_nearest_configured_reference") or {}
    shock_ref = pd.Timestamp(aug_report["configured_ground_truth"]["shock_reference"])
    default_time = pd.Timestamp(primary.get("detected_at", shock_ref))
    reset_time = max(df.timestamp.min(), default_time - pd.Timedelta(minutes=3))
else:
    default_time = selected_events["start_utc"].min() if not selected_events.empty else df.timestamp.min()
    reset_time = max(df.timestamp.min(), pd.Timestamp(default_time))
default_idx = int(np.argmin(np.abs((df.timestamp - reset_time).dt.total_seconds().to_numpy())))
if st.session_state.get("active_replay_dataset") != selected_replay_label:
    st.session_state.active_replay_dataset = selected_replay_label
    st.session_state.replay_idx = default_idx
elif "replay_idx" not in st.session_state:
    st.session_state.replay_idx = default_idx
else:
    st.session_state.replay_idx = min(max(0, int(st.session_state.replay_idx)), len(df) - 1)

with st.sidebar:
    st.header("Event replay")
    st.write("**Processed coverage**")
    st.write(f"{df.timestamp.min()} to {df.timestamp.max()}")
    auto_replay = st.toggle("Play replay", value=False)
    replay_speed = st.select_slider("Replay speed", options=[1, 5, 10], value=1, format_func=lambda x: f"{x} min/step")
    prev_col, reset_col, next_col = st.columns(3)
    if prev_col.button("Back"):
        st.session_state.replay_idx = max(0, int(st.session_state.replay_idx) - replay_speed)
    if reset_col.button("Reset"):
        st.session_state.replay_idx = default_idx
    if next_col.button("Next"):
        st.session_state.replay_idx = min(len(df) - 1, int(st.session_state.replay_idx) + replay_speed)
    idx = st.slider("Minute index", 0, len(df) - 1, int(st.session_state.replay_idx), 1)
    st.session_state.replay_idx = idx
    current = df.iloc[idx]
    st.write("**Spacecraft time (UTC)**")
    st.code(str(current.timestamp))
    range_mode = st.radio("Time selection mode", ["Replay window", "Custom time range"], horizontal=True)
    if range_mode == "Replay window":
        window_h = st.select_slider("Display window", options=[1, 3, 6, 12, 24, 48], value=12, format_func=lambda x: f"+/- {x} h")
        custom_range = None
    else:
        default_range = (
            max(df.timestamp.min().to_pydatetime(), (current.timestamp - pd.Timedelta(hours=6)).to_pydatetime()),
            min(df.timestamp.max().to_pydatetime(), (current.timestamp + pd.Timedelta(hours=6)).to_pydatetime()),
        )
        custom_range = st.slider(
            "Custom UTC range",
            min_value=df.timestamp.min().to_pydatetime(),
            max_value=df.timestamp.max().to_pydatetime(),
            value=default_range,
            format="YYYY-MM-DD HH:mm",
        )
        window_h = None
    st.divider()
    st.write("**Analysis grid**")
    prototype_cfg = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text(encoding="utf-8"))
    st.write(f"{prototype_cfg['swis']['common_energy_min_ev']:.0f}-{prototype_cfg['swis']['common_energy_max_ev']:.0f} eV")
    st.write(f"{prototype_cfg['swis']['common_grid_points']} common log-energy points")
    st.write("SWIS revision:", prototype_cfg["swis"]["version"])
    st.divider()
    st.write("**Configured event boundaries**")
    for _, row in labels.iterrows():
        st.write(f"{row['boundary_type']}: {row['boundary_time_utc']}")
    reference_markers = aug_report.get("configured_ground_truth", {}).get("reference_markers", {}) if selected_replay_label == "August 2024" else {}
    if reference_markers:
        st.write("**Reference markers**")
        for name, marker in reference_markers.items():
            st.write(f"{name.replace('_', ' ')}: {marker}")

row = df.iloc[idx]
if custom_range is None:
    lo = row.timestamp - pd.Timedelta(hours=window_h)
    hi = row.timestamp + pd.Timedelta(hours=window_h)
else:
    lo = pd.Timestamp(custom_range[0])
    hi = pd.Timestamp(custom_range[1])

phase6_predictions = load_phase6_predictions()
phase6_ranking = phase6_report.get("model_ranking", [])
phase6_dashboard_model = phase6_ranking[0] if phase6_ranking else "Logistic Regression"
phase6_status = phase6_prediction_at(
    phase6_predictions,
    selected_replay_meta.get("phase6_interval_id"),
    row.timestamp,
    phase6_dashboard_model,
)

if selected_replay_meta["detector_available"]:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("SYSTEM STATE", str(row.state))
    c2.metric("JS OPDI", fmt_metric(row.js_opdi, ".4f"))
    c3.metric("Transition score", fmt_metric(row.transition_score, ".2f"))
    c4.metric("Proton speed", f"{fmt_metric(row.proton_bulk_speed, '.1f')} km/s" if pd.notna(row.proton_bulk_speed) else "NA")
    c5.metric("|B| GSE", f"{fmt_metric(row.bmag_gse, '.1f')} nT" if pd.notna(row.bmag_gse) else "NA")
    c6.metric("alpha/p density", fmt_metric(row.alpha_proton_ratio, ".3f"))
else:
    status1, status2, status3 = st.columns(3)
    status1.metric("PROTOTYPE DETECTOR", "Not Available")
    if phase6_status is not None:
        status2.metric("PHASE 6 PREDICTION", phase6_status["prediction"])
        status3.metric("CONFIDENCE", f"{phase6_status['confidence']:.2%}")
        st.caption(
            f"Phase 6 dashboard model: {phase6_status['model']} · "
            f"ensemble event probability: {phase6_status['event_probability']:.1%} · "
            "confidence is derived from the mean held-out probability across the available Phase 6 baseline models."
        )
    else:
        status2.metric("PHASE 6 PREDICTION", "NOT EVALUATED")
        status3.metric("CONFIDENCE", "NA")
        if selected_replay_meta.get("phase6_interval_id") is None:
            st.caption(
                "This source was not included in Phase 6 held-out modeling, so no ML prediction or confidence is available."
            )
        else:
            st.caption(
                "No exact Phase 6 prediction exists for this replay minute. "
                "Predictions are shown only for timestamps actually included in held-out evaluation."
            )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("JS OPDI", fmt_metric(row.js_opdi, ".4f"))
    d2.metric("Proton speed", f"{fmt_metric(row.proton_bulk_speed, '.1f')} km/s" if pd.notna(row.proton_bulk_speed) else "NA")
    d3.metric("|B| GSE", f"{fmt_metric(row.bmag_gse, '.1f')} nT" if pd.notna(row.bmag_gse) else "NA")
    d4.metric("alpha/p density", fmt_metric(row.alpha_proton_ratio, ".3f"))

tab_names = [
    "Event Replay",
    "Whole Event",
    "Scientific Dataset",
    "Ground Truth",
    "Feature Engineering",
    "OPDI Ablation",
    "Baseline ML",
    "Validation",
    "CME Source Candidates",
]
if ENABLE_DATA_MANAGER:
    tab_names.append("Data Manager")
tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs[:9]
tab10 = tabs[9] if ENABLE_DATA_MANAGER else None

with tab1:
    st.subheader("Synchronized TH1 / TH2 spectra")
    st.plotly_chart(spectrogram_pair(spec, lo, hi), width="stretch")
    st.subheader("Cross-plane + plasma + magnetic context")
    st.plotly_chart(event_context(df, lo, hi, labels), width="stretch")

    left, right = st.columns(2)
    t_spec = pd.to_datetime(spec["time"].astype("datetime64[ms]"))
    sidx = int(np.argmin(np.abs((t_spec - row.timestamp).total_seconds())))
    with left:
        st.plotly_chart(
            fingerprint_figure(
                spec["th1_probability"][sidx],
                spec["th2_probability"][sidx],
                spec["energy"],
                title=f"SWIS Cross-Plane Fingerprint - {row.timestamp} UTC",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(explain_figure(row), width="stretch")
        st.markdown("**Current evidence**")
        if pd.notna(row.opdi_anomaly_score):
            st.write(f"OPDI level anomaly score: **{row.opdi_anomaly_score:.2f}**")
            if pd.notna(row.transition_component_js_opdi):
                st.write(f"OPDI transition contribution: **{row.transition_component_js_opdi:.2f}**")
            st.write(f"Conventional plasma + MAG score: **{row.conventional_anomaly_score:.2f}**")
            st.write(f"Combined score: **{row.combined_anomaly_score:.2f}**")
            if bool(row.is_change_point):
                st.error("Automatic persistent transition detected at this minute.")
            elif row.state == "ICME CANDIDATE":
                st.warning("Recent transition + sustained conventional disturbance: ICME CANDIDATE state.")
            elif row.state == "WATCH":
                st.info("Unusual cross-plane / environmental behavior: WATCH state.")
        else:
            st.info(
                "Detector evidence is not computed for this independent source. "
                "The replay is showing the processed scientific measurements without fabricating detector scores."
            )

with tab2:
    st.subheader(f"Full {selected_replay_label} processed-source overview")
    bounds = [{"time": str(r.boundary_time_utc), "label": r.boundary_type} for _, r in labels.iterrows()]
    st.plotly_chart(overview_figure(df, spec, bounds), width="stretch")
    selected_feature_path = selected_replay_meta["feature_file"]
    st.download_button(
        "Download selected processed 1-minute feature table",
        data=selected_feature_path.read_bytes(),
        file_name=selected_feature_path.name,
        mime="text/csv",
    )

with tab3:
    st.subheader("Phase 2 event dataset")
    st.caption(
        "The registry separates event windows from independent source intervals. "
        "The five source intervals are counted independently even when one source contains several scientific windows."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Registered windows", phase2_manifest["event_windows"])
    m2.metric("Independent intervals", phase2_manifest["independent_intervals"])
    m3.metric("Positive events", phase2_manifest["positive_events"])
    m4.metric("Control windows", phase2_manifest["control_windows"])
    m5.metric("One-minute records", f"{phase2_manifest['one_minute_records']:,}")

    if phase2_manifest.get("research_ready"):
        st.success("The configured Phase 2 dataset satisfies the current research-readiness contract.")
    else:
        st.warning(
            "Five independent source intervals are assembled, but the November orientation control is not research-ready: "
            "its uploaded SWIS ZIP is corrupt and MAG coverage for 25 November is absent."
        )

    st.markdown("#### Registered event windows")
    catalog_columns = [
        "event_id",
        "event_class",
        "sample_role",
        "negative_control",
        "start_utc",
        "end_utc",
        "records_observed",
        "time_coverage_fraction",
        "spectra_available",
        "cme_candidates",
        "label_status",
        "data_status",
        "phase2_ready",
    ]
    st.dataframe(phase2_catalog[catalog_columns], width="stretch", hide_index=True)

    st.markdown("#### Modality completeness by event")
    coverage_pivot = phase2_coverage.pivot(index="event_id", columns="modality", values="coverage_fraction")
    heatmap = go.Figure(
        data=go.Heatmap(
            z=coverage_pivot.to_numpy() * 100,
            x=coverage_pivot.columns,
            y=coverage_pivot.index,
            zmin=0,
            zmax=100,
            colorscale=[[0.0, "#7f1d1d"], [0.9, "#f59e0b"], [1.0, "#16a34a"]],
            colorbar={"title": "Coverage %"},
            text=np.round(coverage_pivot.to_numpy() * 100, 1),
            texttemplate="%{text}%",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        )
    )
    heatmap.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(heatmap, width="stretch")
    st.caption(
        "OMNI_REFERENCE is optional NASA near-Earth context. It is not included in the Aditya-L1 modality contract "
        "and never fills missing TH1, TH2, BLK or MAG values."
    )

    st.markdown("#### Source gaps and blocked inputs")
    st.dataframe(phase2_queue, width="stretch", hide_index=True)
    st.caption(
        "October 12 and partial September dates remain explicit source gaps. The November item can be completed "
        "after valid SWIS and MAG files covering 25 November are supplied."
    )

    st.markdown("#### Scientific guardrails")
    for item in phase2_manifest.get("scientific_guardrails", []):
        st.write("-", item)

    download_columns = st.columns(4)
    download_columns[0].download_button(
        "Event catalog CSV",
        data=(SCI / "phase2_event_catalog.csv").read_bytes(),
        file_name="phase2_event_catalog.csv",
        mime="text/csv",
    )
    download_columns[1].download_button(
        "Feature table CSV",
        data=(SCI / "phase2_feature_table.csv").read_bytes(),
        file_name="phase2_feature_table.csv",
        mime="text/csv",
    )
    download_columns[2].download_button(
        "Coverage CSV",
        data=(SCI / "phase2_modality_coverage.csv").read_bytes(),
        file_name="phase2_modality_coverage.csv",
        mime="text/csv",
    )
    download_columns[3].download_button(
        "Manifest JSON",
        data=(SCI / "phase2_manifest.json").read_bytes(),
        file_name="phase2_manifest.json",
        mime="application/json",
    )
    if st.button("Rebuild Phases 2 to 6"):
        with st.spinner("Rebuilding the event registry, labels, features, ablation, and baseline ML outputs..."):
            result = run_phases2_to6_build()
        load_all.clear()
        if result.returncode == 0:
            st.success("Phases 2 to 6 rebuilt successfully.")
            st.rerun()
        else:
            st.error("Linked Phase 2-6 build failed.")
            st.code(process_output(result), language="text")

with tab4:
    st.subheader("Phase 3 multi-event ground truth")
    st.caption(
        "Phase 3 consumes the Phase 2 registry. Exact substructure labels are used only where configured boundaries exist; "
        "other intervals retain their sourced event-window class."
    )
    validation = phase3_report["validation"]
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Labeled records", f"{validation['records']:,}")
    g2.metric("Event windows", validation["event_windows"])
    g3.metric("Independent intervals", validation["independent_intervals"])
    g4.metric("Ready events", f"{validation['phase3_ready_events']}/{validation['event_windows']}")
    g5.metric("Unknown labels", validation["unknown_labels"])

    if phase3_report.get("research_ready"):
        st.success("All Phase 3 labels and their Phase 2 source modalities are research-ready.")
    else:
        blocked = ", ".join(validation.get("blocked_events", [])) or "unspecified event"
        st.warning(
            "Ground-truth construction is valid, but full research readiness remains blocked by Phase 2 source coverage: "
            f"{blocked}."
        )

    st.markdown("#### Minute-level label distribution")
    label_chart = go.Figure(
        go.Bar(
            x=phase3_counts["research_label"],
            y=phase3_counts["n"],
            text=phase3_counts["n"],
            marker_color=[
                "#64748b",
                "#dc2626",
                "#f97316",
                "#7c3aed",
                "#4f46e5",
                "#0891b2",
                "#16a34a",
                "#ca8a04",
            ][: len(phase3_counts)],
        )
    )
    label_chart.update_layout(
        height=390,
        xaxis_title="Research label",
        yaxis_title="One-minute records",
        margin=dict(l=35, r=20, t=20, b=80),
    )
    st.plotly_chart(label_chart, width="stretch")

    st.markdown("#### Event label register")
    event_columns = [
        "event_id",
        "event_class",
        "sample_role",
        "research_labels",
        "phase3_policy",
        "label_confidence",
        "boundary_status",
        "records",
        "eligible_fraction",
        "phase3_ready",
    ]
    st.dataframe(phase3_events[event_columns], width="stretch", hide_index=True)

    st.markdown("#### Boundary register")
    st.dataframe(
        phase3_boundaries[
            [
                "event_id",
                "boundary_type",
                "boundary_time_utc",
                "scientific_status",
                "used_for_minute_labeling",
                "label_source",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "October reference boundaries are displayed but explicitly excluded from exact minute-level substructure labeling."
    )

    st.markdown("#### Phase 3 scientific guardrails")
    for item in phase3_report.get("scientific_guardrails", []):
        st.write("-", item)

    phase3_downloads = st.columns(5)
    for column, filename, label, mime in [
        (phase3_downloads[0], "phase3_ground_truth_dataset.csv", "Ground truth CSV", "text/csv"),
        (phase3_downloads[1], "phase3_label_counts.csv", "Label counts CSV", "text/csv"),
        (phase3_downloads[2], "phase3_event_register.csv", "Event register CSV", "text/csv"),
        (phase3_downloads[3], "phase3_boundary_register.csv", "Boundary register CSV", "text/csv"),
        (phase3_downloads[4], "phase3_report.json", "Phase 3 report", "application/json"),
    ]:
        column.download_button(
            label,
            data=(PHASE3 / filename).read_bytes(),
            file_name=filename,
            mime=mime,
        )
    if st.button("Rebuild Phase 3 ground truth"):
        with st.spinner("Applying Phase 3 policies to the Phase 2 multi-event registry..."):
            result = run_phase3_build()
        load_all.clear()
        if result.returncode == 0:
            st.success("Phase 3 ground truth rebuilt successfully.")
            st.rerun()
        else:
            st.error("Phase 3 build failed.")
            st.code(process_output(result), language="text")

with tab5:
    st.subheader("Phase 4 complete feature engineering")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Records", f"{phase4_report['records']:,}")
    f2.metric("Derived feature columns", phase4_report["derived_feature_columns"])
    f3.metric("Ablation-ready rows", f"{phase4_report['exploratory_ablation_rows']:,}")
    f4.metric("Spectral-shape rows", f"{phase4_report['spectral_shape_complete_rows']:,}")
    if phase4_report.get("blocked_events_from_phase3"):
        st.warning("Phase 4 is complete, but the existing Phase 2/3 data block remains for: " + ", ".join(phase4_report["blocked_events_from_phase3"]))
    else:
        st.success("All registered event rows have the required Phase 4 source modalities.")

    st.markdown("#### Per-event feature availability")
    st.dataframe(phase4_summary, width="stretch", hide_index=True)
    st.markdown("#### Feature dictionary")
    selected_group = st.selectbox("Feature group", ["All"] + sorted(phase4_dictionary["feature_group"].unique().tolist()), key="phase4_feature_group")
    dictionary_view = phase4_dictionary if selected_group == "All" else phase4_dictionary.loc[phase4_dictionary["feature_group"] == selected_group]
    st.dataframe(dictionary_view, width="stretch", hide_index=True)
    st.markdown("#### Phase 4 guardrails")
    for item in phase4_report.get("scientific_guardrails", []):
        st.write("-", item)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Feature dataset CSV", data=(PHASE4 / "phase4_feature_dataset.csv").read_bytes(), file_name="phase4_feature_dataset.csv", mime="text/csv")
    d2.download_button("Feature dictionary CSV", data=(PHASE4 / "phase4_feature_dictionary.csv").read_bytes(), file_name="phase4_feature_dictionary.csv", mime="text/csv")
    d3.download_button("Event summary CSV", data=(PHASE4 / "phase4_event_summary.csv").read_bytes(), file_name="phase4_event_summary.csv", mime="text/csv")
    d4.download_button("Phase 4 report", data=(PHASE4 / "phase4_report.json").read_bytes(), file_name="phase4_report.json", mime="application/json")
    if st.button("Rebuild Phase 4 features"):
        with st.spinner("Computing conventional, OPDI, rolling, compression, and spectral-shape features..."):
            result = run_phase4_build()
        load_all.clear()
        if result.returncode == 0:
            st.success("Phase 4 feature engineering rebuilt successfully.")
            st.rerun()
        st.error("Phase 4 build failed.")
        st.code(process_output(result), language="text")

with tab6:
    st.subheader("Phase 5 central scientific question: does OPDI add information?")
    st.caption("Event-wise exploratory ablation: Conventional vs OPDI only vs Combined. No individual minute is shared between train and held-out source intervals.")
    evidence = phase5_report.get("evidence_status", "UNKNOWN")
    if evidence == "EXPLORATORY_SUPPORT_FOR_ADDED_OPDI_INFORMATION":
        st.success(evidence.replace("_", " ").title())
    elif evidence == "EXPLORATORY_MIXED_EVIDENCE":
        st.warning("Exploratory mixed evidence: Combined improves ranking PR-AUC and detection timing, but current thresholded F1/false alarms do not improve over Conventional.")
    else:
        st.info(evidence.replace("_", " ").title())

    metric_view = phase5_summary[["mode", "detection_rate", "false_alarms_per_day", "precision", "recall", "f1", "pr_auc", "median_detection_delay_minutes"]].copy()
    st.dataframe(metric_view, width="stretch", hide_index=True)
    metric_chart = go.Figure()
    for metric in ["f1", "pr_auc", "recall", "precision"]:
        metric_chart.add_trace(go.Bar(name=metric, x=phase5_summary["mode"], y=phase5_summary[metric]))
    metric_chart.update_layout(barmode="group", height=420, yaxis_title="Score", yaxis_range=[0, 1], margin=dict(l=30, r=20, t=30, b=30))
    st.plotly_chart(metric_chart, width="stretch")

    st.markdown("#### Event-wise fold metrics")
    st.dataframe(phase5_folds, width="stretch", hide_index=True)
    st.markdown("#### Detection-delay register")
    st.dataframe(phase5_delays, width="stretch", hide_index=True)
    st.caption("Delay is only calculated where a labeled positive onset occurs inside the event window. Constant positive windows without an exact onset are not assigned a delay.")
    st.markdown("#### Phase 5 guardrails")
    for item in phase5_report.get("scientific_guardrails", []):
        st.write("-", item)
    p1, p2, p3, p4 = st.columns(4)
    p1.download_button("Summary metrics CSV", data=(PHASE5 / "phase5_summary_metrics.csv").read_bytes(), file_name="phase5_summary_metrics.csv", mime="text/csv")
    p2.download_button("Fold metrics CSV", data=(PHASE5 / "phase5_fold_metrics.csv").read_bytes(), file_name="phase5_fold_metrics.csv", mime="text/csv")
    p3.download_button("Predictions CSV", data=(PHASE5 / "phase5_predictions.csv").read_bytes(), file_name="phase5_predictions.csv", mime="text/csv")
    p4.download_button("Phase 5 report", data=(PHASE5 / "phase5_report.json").read_bytes(), file_name="phase5_report.json", mime="application/json")
    if st.button("Re-run Phase 5 ablation"):
        with st.spinner("Running event-wise Conventional vs OPDI vs Combined ablation..."):
            result = run_phase5_build()
        load_all.clear()
        if result.returncode == 0:
            st.success("Phase 5 experiment completed.")
            st.rerun()
        st.error("Phase 5 experiment failed.")
        st.code(process_output(result), language="text")

with tab7:
    st.subheader("Phase 6 baseline machine-learning models")
    st.caption("Leakage-controlled leave-one-independent-interval-out comparison of Logistic Regression, Random Forest, and HistGradientBoosting using label-free Phase 4 features.")
    ranking = phase6_report.get("model_ranking", [])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eligible rows", f"{phase6_report.get('eligible_rows', 0):,}")
    m2.metric("Candidate features", phase6_report.get("candidate_feature_count", 0))
    m3.metric("Held-out folds", phase6_report.get("folds", 0))
    m4.metric("Top baseline", ranking[0] if ranking else "NA")
    st.warning("Phase 6 is exploratory: only four independent source intervals are currently research-usable, so model ranking is not yet a confirmatory generalization result.")

    metric_cols = ["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_alarms_per_day", "detection_rate", "median_detection_delay_minutes"]
    st.dataframe(phase6_summary[metric_cols], width="stretch", hide_index=True)
    ml_chart = go.Figure()
    for metric in ["f1", "pr_auc", "roc_auc", "recall", "precision"]:
        ml_chart.add_trace(go.Bar(name=metric, x=phase6_summary["model"], y=phase6_summary[metric]))
    ml_chart.update_layout(barmode="group", height=430, yaxis_title="Score", yaxis_range=[0, 1], margin=dict(l=30, r=20, t=30, b=30))
    st.plotly_chart(ml_chart, width="stretch")

    st.markdown("#### Event-wise fold metrics")
    st.dataframe(phase6_folds, width="stretch", hide_index=True)
    st.markdown("#### Baseline feature importance")
    if not phase6_importance.empty:
        importance_summary = (
            phase6_importance.groupby(["model", "feature"], as_index=False)["normalized_importance"].mean()
            .sort_values(["model", "normalized_importance"], ascending=[True, False])
        )
        selected_model = st.selectbox("Model for feature ranking", sorted(importance_summary["model"].unique()), key="phase6_importance_model")
        st.dataframe(importance_summary.loc[importance_summary["model"] == selected_model].head(20), width="stretch", hide_index=True)
        st.caption("Logistic Regression importance is based on absolute standardized coefficients; Random Forest uses native impurity importance. HistGradientBoosting does not expose native feature importance here.")
    st.markdown("#### Detection-delay register")
    st.dataframe(phase6_delays, width="stretch", hide_index=True)

    st.markdown("#### Phase 6 guardrails")
    for item in phase6_report.get("scientific_guardrails", []):
        st.write("-", item)
    q1, q2, q3, q4 = st.columns(4)
    q1.download_button("Summary metrics CSV", data=(PHASE6 / "phase6_summary_metrics.csv").read_bytes(), file_name="phase6_summary_metrics.csv", mime="text/csv")
    q2.download_button("Fold metrics CSV", data=(PHASE6 / "phase6_fold_metrics.csv").read_bytes(), file_name="phase6_fold_metrics.csv", mime="text/csv")
    q3.download_button("Predictions CSV", data=(PHASE6 / "phase6_predictions.csv").read_bytes(), file_name="phase6_predictions.csv", mime="text/csv")
    q4.download_button("Phase 6 report", data=(PHASE6 / "phase6_report.json").read_bytes(), file_name="phase6_report.json", mime="application/json")
    if st.button("Re-run Phase 6 baseline ML"):
        with st.spinner("Training and evaluating the three baseline models event-wise..."):
            result = run_phase6_build()
        load_all.clear()
        if result.returncode == 0:
            st.success("Phase 6 baseline ML completed.")
            st.rerun()
        else:
            st.error("Phase 6 baseline ML failed.")
            st.code(process_output(result), language="text")

with tab8:
    st.subheader("What has actually been validated?")
    st.caption(
        "This validation panel is the August 2024 detector-prototype validation. "
        "Changing the replay source does not relabel another month as detector-validated."
    )
    shock_ref = pd.Timestamp(aug_report["configured_ground_truth"]["shock_reference"])
    nearest = aug_report["evaluation_against_configured_reference"]["nearest_change_to_reference_shock"]
    offset = aug_report["evaluation_against_configured_reference"]["nearest_change_offset_minutes"]
    a, b, c = st.columns(3)
    a.metric("Legacy approx. benchmark", str(shock_ref))
    b.metric("Nearest automatic transition", str(nearest))
    c.metric("Offset", f"{offset:+.0f} min" if offset is not None else "NA")
    st.caption("This offset is relative to the legacy approximate internal benchmark. It must not be reported as validated early-warning lead time.")
    markers = aug_report.get("configured_ground_truth", {}).get("reference_markers", {})
    if markers:
        st.markdown("#### Reference markers")
        marker_rows = [{"reference": name.replace("_", " "), "time_utc": value} for name, value in markers.items()]
        marker_rows.append({"reference": "TopoCross detected transition", "time_utc": str(nearest)})
        st.dataframe(pd.DataFrame(marker_rows), width="stretch", hide_index=True)
    st.markdown("#### Exploratory OPDI separation by event phase")
    js = tests[tests.metric == "js_opdi"].copy()
    show = js[
        [
            "comparison",
            "quiet_median",
            "event_median",
            "cliffs_delta_event_vs_quiet",
            "mannwhitney_p_two_sided",
            "n_quiet",
            "n_event",
        ]
    ]
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption("These are within-event exploratory statistics, not proof of generalization to unseen ICMEs.")
    st.markdown("#### Detector state counts")
    st.dataframe(pd.DataFrame({"state": list(aug_report["state_counts"]), "minutes": list(aug_report["state_counts"].values())}), width="stretch", hide_index=True)
    st.markdown("#### Scientific guardrails")
    for item in aug_report.get("scientific_status", []):
        st.write("-", item)

with tab9:
    st.subheader("Prototype CME compatibility ranking")
    if selected_replay_label != "August 2024":
        st.info(
            f"A CME compatibility ranking has not been built for {selected_replay_label}. "
            "The existing heuristic source-matching table is tied to the August 2024 prototype transition, so it is not reused for another month."
        )
    else:
        st.caption(
            f"Ranking reference time: {aug_report.get('cme_source_match_reference_time', 'NA')} "
            f"({aug_report.get('cme_source_match_reference', 'unknown')})."
        )
        cols = [
            "cme_time",
            "linear_speed_km_s",
            "space_speed_km_s",
            "source_location",
            "flare",
            "observed_transit_hours",
            "ballistic_transit_hours",
            "compatibility_score",
            "catalog",
        ]
        st.dataframe(aug_candidates[cols], width="stretch", hide_index=True)
        st.caption(
            "Compatibility combines simple transit-time, direction, halo status and speed terms. "
            "It is a heuristic ranking for source-association exploration, not a causal probability."
        )

if ENABLE_DATA_MANAGER and tab10 is not None:
    with tab10:
        st.subheader("Data manager")
        st.write(
            "Use this tab when your team adds more Aditya-L1 files locally. Official mission data should be downloaded "
            "from ISRO/ISSDC PRADAN after login, then placed in the matching local folders before rebuilding."
        )
        st.link_button("Open PRADAN Aditya-L1 Portal", "https://pradan.issdc.gov.in/al1")
        st.link_button("Open public Zenodo SWIS sample", "https://zenodo.org/records/15861770")

        st.markdown("#### Local raw-file coverage")
        coverage_rows = []
        raw_checks = {
            "TH1 SWIS": [(ROOT / "data" / "raw" / "th1", "*.cdf"), (ROOT / "tha1", "*.cdf")],
            "TH2 SWIS": [(ROOT / "data" / "raw" / "th2", "*.cdf"), (ROOT / "tha2", "*.cdf")],
            "BLK plasma": [(ROOT / "data" / "raw" / "blk", "*.cdf"), (ROOT / "swis_BLK", "*.cdf")],
            "MAG L2": [(ROOT / "data" / "raw" / "mag", "L2_AL1_MAG_*_V00.nc"), (ROOT / "mag_2026Aug23T210145602", "L2_AL1_MAG_*_V00.nc")],
        }
        for label, locations in raw_checks.items():
            files = []
            folders = []
            for folder, pattern in locations:
                files.extend(folder.glob(pattern) if folder.exists() else [])
                if folder.exists():
                    folders.append(str(folder.relative_to(ROOT)))
            coverage_rows.append(
                {
                    "dataset": label,
                    "folder": ", ".join(folders) if folders else "not present",
                    "files": len(files),
                    "status": "available" if files else "missing",
                }
            )
        st.dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True)

        st.markdown("#### Upload more raw Aditya-L1 files")
        st.caption(
            "Upload matching TH1, TH2, BLK `.cdf` files and MAG `L2_AL1_MAG_*_V00.nc` files. "
            "Files are saved into `data/raw/`, which is ignored by Git. After uploading, update the date range in "
            "`config/prototype.yaml` if needed, then click rebuild."
        )
        clear_confirmed = st.checkbox("I want to clear previously uploaded raw files before a new upload")
        if st.button("Clear uploaded raw files", disabled=not clear_confirmed):
            clear_uploaded_raw_files()
            st.success("Cleared `data/raw/th1`, `data/raw/th2`, `data/raw/blk`, and `data/raw/mag`.")
            st.rerun()
        th1_uploads = st.file_uploader("Upload TH1 SWIS CDF files", type=["cdf"], accept_multiple_files=True, key="raw_th1")
        th2_uploads = st.file_uploader("Upload TH2 SWIS CDF files", type=["cdf"], accept_multiple_files=True, key="raw_th2")
        blk_uploads = st.file_uploader("Upload BLK plasma CDF files", type=["cdf"], accept_multiple_files=True, key="raw_blk")
        mag_uploads = st.file_uploader("Upload MAG L2 NetCDF files", type=["nc"], accept_multiple_files=True, key="raw_mag")
        if st.button("Save uploaded raw files"):
            saved = []
            saved.extend(save_uploaded_raw_files(th1_uploads, ROOT / "data" / "raw" / "th1"))
            saved.extend(save_uploaded_raw_files(th2_uploads, ROOT / "data" / "raw" / "th2"))
            saved.extend(save_uploaded_raw_files(blk_uploads, ROOT / "data" / "raw" / "blk"))
            valid_mag_uploads = []
            invalid_mag_names = []
            for uploaded in mag_uploads or []:
                name = Path(uploaded.name).name
                if name.startswith("L2_AL1_MAG_") and name.endswith("_V00.nc"):
                    valid_mag_uploads.append(uploaded)
                else:
                    invalid_mag_names.append(name)
            if invalid_mag_names:
                st.error(
                    "These MAG files were not saved because full detector mode requires L2 MAG files named "
                    "`L2_AL1_MAG_YYYYMMDD_V00.nc`."
                )
                st.code("\n".join(invalid_mag_names), language="text")
            saved.extend(save_uploaded_raw_files(valid_mag_uploads, ROOT / "data" / "raw" / "mag"))
            if saved:
                st.success(f"Saved {len(saved)} raw files. Click rebuild after confirming `config/prototype.yaml` date range.")
                st.code("\n".join(str(path.relative_to(ROOT)) for path in saved[-30:]), language="text")
            else:
                st.warning("No raw files selected.")

        st.markdown("#### Build uploaded TH1/TH2 as SWIS-only")
        st.caption(
            "Use this when you uploaded only TH1 and TH2 files. It calculates OPDI and spectra, "
            "but it does not run BLK/MAG detector states or CME validation."
        )
        swis_pairs = available_swis_only_pairs()
        selected_pairs = st.multiselect(
            "Choose uploaded TH1/TH2 date-version pairs",
            options=swis_pairs,
            default=swis_pairs,
            help="Clear old uploads first if you only want to process the newest pair.",
        )
        st.session_state.selected_swis_only_pairs = selected_pairs
        if st.button("Build uploaded SWIS-only dataset"):
            if not selected_pairs:
                st.warning("No matching TH1/TH2 date-version pairs selected.")
                st.stop()
            with st.spinner("Building SWIS-only OPDI dataset from uploaded TH1/TH2 files..."):
                result = run_uploaded_swis_only_build()
            if result.returncode == 0:
                st.success("SWIS-only build complete.")
                st.code(process_output(result), language="text")
            else:
                st.error("SWIS-only build failed.")
                st.code(process_output(result), language="text")

        external_report = ROOT / "data" / "external" / "zenodo_swis_20231106_12" / "processed" / "zenodo_swis_only_report.json"
        if external_report.exists():
            with open(external_report, "r", encoding="utf-8") as file:
                ext = json.load(file)
            st.success(
                f"External Zenodo SWIS-only dataset available: {ext['records']:,} one-minute records "
                f"from {ext['start']} to {ext['end']}."
            )
            st.caption(ext["note"])
        else:
            st.info("Optional Zenodo SWIS-only sample is not processed yet. Run `python3 scripts/download_zenodo_swis.py` and `python3 scripts/build_zenodo_swis_only.py`.")

        st.markdown("#### Add or evaluate another processed dataset")
        st.caption(
            "For raw CDF/NetCDF data, use the upload section above or manually add files into `data/raw/th1`, "
            "`data/raw/th2`, `data/raw/blk`, and `data/raw/mag`; edit `config/prototype.yaml` date range if needed, then click rebuild. "
            "For a quick check, upload a processed feature CSV with columns like `timestamp`, `js_opdi`, and `state`."
        )
        upload = st.file_uploader("Upload processed feature CSV for quick evaluation", type=["csv"])
        if upload is not None:
            uploaded_df = pd.read_csv(upload)
            st.write(f"Uploaded rows: {len(uploaded_df):,}")
            st.dataframe(uploaded_df.head(100), width="stretch")
            required = {"timestamp", "js_opdi"}
            missing = sorted(required - set(uploaded_df.columns))
            if missing:
                st.warning(f"Missing required columns for OPDI evaluation: {', '.join(missing)}")
            else:
                uploaded_df["timestamp"] = pd.to_datetime(uploaded_df["timestamp"], errors="coerce")
                st.metric("Uploaded JS OPDI median", f"{uploaded_df['js_opdi'].median():.4f}")
                if "state" in uploaded_df:
                    st.dataframe(uploaded_df["state"].value_counts().rename_axis("state").reset_index(name="minutes"), width="stretch", hide_index=True)

        st.markdown("#### Rebuild full detector dataset from local raw files")
        st.caption(
            "Full rebuild requires matching TH1, TH2, BLK, and MAG files for the configured dates/version in "
            "`config/prototype.yaml`. TH1/TH2-only uploads should use the SWIS-only build above."
        )
        coverage, can_full_rebuild = full_rebuild_coverage()
        st.dataframe(coverage, width="stretch", hide_index=True)
        if not can_full_rebuild:
            st.warning(
                "Full rebuild is disabled because one or more required files are missing. "
                "MAG files must be `L2_AL1_MAG_YYYYMMDD_V00.nc`; `L1_MAG...` files do not satisfy the full detector input."
            )
        if st.button("Rebuild now from Data Manager", disabled=not can_full_rebuild):
            with st.spinner("Rebuilding processed files..."):
                result = run_rebuild()
            load_all.clear()
            if result.returncode == 0:
                st.success("Rebuild complete.")
                st.code(process_output(result), language="text")
                st.rerun()
            else:
                st.error("Rebuild failed.")
                st.code(process_output(result), language="text")

st.divider()
st.caption(
    "TopoCross-SWIS v4.1 - Phases 2-6 implemented: multi-event registry/replay, ground truth, "
    "feature engineering, OPDI ablation, and baseline machine learning."
)

if auto_replay:
    time.sleep(max(0.2, 1.0 / replay_speed))
    st.session_state.replay_idx = min(len(df) - 1, int(st.session_state.replay_idx) + replay_speed)
    st.rerun()
