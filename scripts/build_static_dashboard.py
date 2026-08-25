#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "TopoCross_dashboard.html"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    report = json.loads((PROC / "pipeline_report.json").read_text())
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=["OPDI", "SWIS plasma", "MAG GSE", "Transition detector", "Anomaly evidence"],
    )
    for column, name in [("js_opdi", "JS"), ("hellinger_opdi", "Hellinger"), ("wasserstein_opdi", "Wasserstein")]:
        fig.add_trace(go.Scatter(x=df.timestamp, y=df[column], mode="lines", name=name), row=1, col=1)
    for column, name in [("proton_bulk_speed", "Speed"), ("proton_density", "Density"), ("proton_thermal", "Thermal")]:
        fig.add_trace(go.Scatter(x=df.timestamp, y=df[column], mode="lines", name=name), row=2, col=1)
    for column in ["Bx_gse", "By_gse", "Bz_gse", "bmag_gse"]:
        fig.add_trace(go.Scatter(x=df.timestamp, y=df[column], mode="lines", name=column), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.transition_score, mode="lines", name="Transition score"), row=4, col=1)
    change_points = df[df.is_change_point.astype(bool)]
    fig.add_trace(
        go.Scatter(x=change_points.timestamp, y=change_points.transition_score, mode="markers", name="Persistent transition"),
        row=4,
        col=1,
    )
    for column, name in [
        ("opdi_anomaly_score", "OPDI"),
        ("conventional_anomaly_score", "Conventional"),
        ("combined_anomaly_score", "Combined"),
    ]:
        fig.add_trace(go.Scatter(x=df.timestamp, y=df[column], mode="lines", name=name), row=5, col=1)
    for key in ["shock_reference", "icme_ejecta_start", "icme_ejecta_end"]:
        time = pd.Timestamp(report["configured_ground_truth"][key])
        for row in range(1, 6):
            fig.add_vline(x=time, row=row, col=1, line_dash="dash")
    fig.update_layout(
        title="TopoCross-SWIS - August 2024 static dashboard",
        height=1200,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=45, r=25, t=80, b=40),
    )
    fig.write_html(OUT, include_plotlyjs=True, full_html=True)
    print(OUT)


if __name__ == "__main__":
    main()
