#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
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
        title="TopoCross-SWIS - August 2024 detector overview",
        height=1200,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=45, r=25, t=80, b=40),
    )
    phase3_counts = pd.read_csv(ROOT / "outputs" / "phase3" / "phase3_label_counts.csv")
    phase3_events = pd.read_csv(ROOT / "outputs" / "phase3" / "phase3_event_register.csv")
    phase3 = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.20,
        subplot_titles=["Phase 3 minute-level label distribution", "Event-level modeling eligibility"],
    )
    phase3.add_trace(
        go.Bar(
            x=phase3_counts["research_label"],
            y=phase3_counts["n"],
            text=phase3_counts["n"],
            name="Labeled minutes",
        ),
        row=1,
        col=1,
    )
    phase3.add_trace(
        go.Bar(
            x=phase3_events["eligible_fraction"] * 100,
            y=phase3_events["event_id"],
            orientation="h",
            text=(phase3_events["eligible_fraction"] * 100).round(1).astype(str) + "%",
            name="Eligible records",
            marker_color=phase3_events["phase3_ready"].map({True: "#16a34a", False: "#dc2626"}),
        ),
        row=2,
        col=1,
    )
    phase3.update_yaxes(title="One-minute records", row=1, col=1)
    phase3.update_xaxes(title="Eligible records (%)", range=[0, 105], row=2, col=1)
    phase3.update_layout(
        title="Phase 2 multi-event registry + Phase 3 ground truth",
        height=900,
        margin=dict(l=50, r=25, t=80, b=50),
        showlegend=False,
    )
    phase5_summary = pd.read_csv(ROOT / "outputs" / "phase5" / "phase5_summary_metrics.csv")
    phase6_summary = pd.read_csv(ROOT / "outputs" / "phase6" / "phase6_summary_metrics.csv")
    science = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.20,
        subplot_titles=["Phase 5 OPDI ablation", "Phase 6 baseline ML"],
    )
    for metric in ["f1", "pr_auc", "recall", "precision"]:
        science.add_trace(go.Bar(name=f"P5 {metric}", x=phase5_summary["mode"], y=phase5_summary[metric]), row=1, col=1)
    for metric in ["f1", "pr_auc", "roc_auc", "recall", "precision"]:
        science.add_trace(go.Bar(name=f"P6 {metric}", x=phase6_summary["model"], y=phase6_summary[metric]), row=2, col=1)
    science.update_yaxes(range=[0, 1], title="Score")
    science.update_layout(title="Scientific evaluation through Phase 6", height=950, barmode="group", margin=dict(l=50, r=25, t=80, b=50))

    html = """<!doctype html><html><head><meta charset="utf-8"><title>TopoCross-SWIS Phases 2 to 6</title></head><body>"""
    html += pio.to_html(fig, include_plotlyjs=True, full_html=False)
    html += pio.to_html(phase3, include_plotlyjs=False, full_html=False)
    html += pio.to_html(science, include_plotlyjs=False, full_html=False)
    html += "</body></html>"
    OUT.write_text(html, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
