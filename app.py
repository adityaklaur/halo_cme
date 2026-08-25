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
ENABLE_DATA_MANAGER = False


st.set_page_config(page_title="TopoCross-SWIS - Aug 2024", layout="wide")


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
    return df, spec, report, candidates, tests, labels


def run_rebuild():
    return run_script_with_terminal_logs(ROOT / "scripts" / "rebuild_processed.py")


def run_uploaded_swis_only_build():
    selected = st.session_state.get("selected_swis_only_pairs", [])
    extra_args = ["--pairs", ",".join(selected)] if selected else []
    return run_script_with_terminal_logs(ROOT / "scripts" / "build_uploaded_swis_only.py", extra_args)


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


st.title("TopoCross-SWIS - August 2024")
st.caption("Aditya-L1 ASPEX/SWIS TH1 + TH2 + BLK + MAG - real spacecraft data - research prototype")

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
    df, spec, report, candidates, tests, labels = load_all()
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

st.warning(
    "Scientific-status note: the configured shock reference is approximate. "
    "The detector does not use event labels as inputs; timing offsets are exploratory until the boundary is reconciled."
)

primary = report.get("primary_transition_nearest_configured_reference") or {}
shock_ref = pd.Timestamp(report["configured_ground_truth"]["shock_reference"])
default_time = pd.Timestamp(primary.get("detected_at", shock_ref))
default_idx = int(np.argmin(np.abs((df.timestamp - default_time).dt.total_seconds().to_numpy())))
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = default_idx

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
    st.write(f"{report['common_energy_grid_ev'][0]:.0f}-{report['common_energy_grid_ev'][1]:.0f} eV")
    st.write(f"{report['common_energy_grid_points']} common log-energy points")
    st.write("SWIS revision:", report["swis_version"])
    st.divider()
    st.write("**Configured event boundaries**")
    for _, row in labels.iterrows():
        st.write(f"{row['boundary_type']}: {row['boundary_time_utc']}")
    reference_markers = report.get("configured_ground_truth", {}).get("reference_markers", {})
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

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("SYSTEM STATE", row.state)
c2.metric("JS OPDI", f"{row.js_opdi:.4f}" if pd.notna(row.js_opdi) else "NA")
c3.metric("Transition score", f"{row.transition_score:.2f}" if pd.notna(row.transition_score) else "NA")
c4.metric("Proton speed", f"{row.proton_bulk_speed:.1f} km/s" if pd.notna(row.proton_bulk_speed) else "NA")
c5.metric("|B| GSE", f"{row.bmag_gse:.1f} nT" if pd.notna(row.bmag_gse) else "NA")
c6.metric("alpha/p density", f"{row.alpha_proton_ratio:.3f}" if pd.notna(row.alpha_proton_ratio) else "NA")

tab_names = ["Event Replay", "Whole Event", "Validation", "CME Source Candidates"]
if ENABLE_DATA_MANAGER:
    tab_names.append("Data Manager")
tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4 = tabs[:4]
tab5 = tabs[4] if ENABLE_DATA_MANAGER else None

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
        st.write(f"OPDI level anomaly score: **{row.opdi_anomaly_score:.2f}**")
        if "transition_component_js_opdi" in row:
            st.write(f"OPDI transition contribution: **{row.transition_component_js_opdi:.2f}**")
        st.write(f"Conventional plasma + MAG score: **{row.conventional_anomaly_score:.2f}**")
        st.write(f"Combined score: **{row.combined_anomaly_score:.2f}**")
        if bool(row.is_change_point):
            st.error("Automatic persistent transition detected at this minute.")
        elif row.state == "ICME CANDIDATE":
            st.warning("Recent transition + sustained conventional disturbance: ICME CANDIDATE state.")
        elif row.state == "WATCH":
            st.info("Unusual cross-plane / environmental behavior: WATCH state.")

with tab2:
    st.subheader("Full 9-15 August 2024 overview")
    bounds = [{"time": str(r.boundary_time_utc), "label": r.boundary_type} for _, r in labels.iterrows()]
    st.plotly_chart(overview_figure(df, spec, bounds), width="stretch")
    st.download_button(
        "Download processed 1-minute feature table",
        data=(PROC / "aug2024_features_1min.csv").read_bytes(),
        file_name="aug2024_features_1min.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("What has actually been validated?")
    nearest = report["evaluation_against_configured_reference"]["nearest_change_to_reference_shock"]
    offset = report["evaluation_against_configured_reference"]["nearest_change_offset_minutes"]
    a, b, c = st.columns(3)
    a.metric("Legacy approx. benchmark", str(shock_ref))
    b.metric("Nearest automatic transition", str(nearest))
    c.metric("Offset", f"{offset:+.0f} min" if offset is not None else "NA")
    st.caption("This offset is relative to the legacy approximate internal benchmark. It must not be reported as validated early-warning lead time.")
    markers = report.get("configured_ground_truth", {}).get("reference_markers", {})
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
    st.dataframe(pd.DataFrame({"state": list(report["state_counts"]), "minutes": list(report["state_counts"].values())}), width="stretch", hide_index=True)
    st.markdown("#### Scientific guardrails")
    for item in report.get("scientific_status", []):
        st.write("-", item)

with tab4:
    st.subheader("Prototype CME compatibility ranking")
    st.caption(
        f"Ranking reference time: {report.get('cme_source_match_reference_time', 'NA')} "
        f"({report.get('cme_source_match_reference', 'unknown')})."
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
    st.dataframe(candidates[cols], width="stretch", hide_index=True)
    st.caption(
        "Compatibility combines simple transit-time, direction, halo status and speed terms. "
        "It is a heuristic ranking for source-association exploration, not a causal probability."
    )

if ENABLE_DATA_MANAGER and tab5 is not None:
    with tab5:
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
st.caption("TopoCross-SWIS v1.0 - thresholds and ground truth remain configurable.")

if auto_replay:
    time.sleep(max(0.2, 1.0 / replay_speed))
    st.session_state.replay_idx = min(len(df) - 1, int(st.session_state.replay_idx) + replay_speed)
    st.rerun()
