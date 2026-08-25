from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STATE_COLOR = {
    "NORMAL": "#2ca02c",
    "WATCH": "#ffbf00",
    "ALERT": "#d62728",
    "ICME CANDIDATE": "#9467bd",
}


def fingerprint_figure(th1_probability, th2_probability, energy, title: str = "SWIS Cross-Plane Fingerprint"):
    energy = np.asarray(energy, dtype=float)
    p1 = np.asarray(th1_probability, dtype=float)
    p2 = np.asarray(th2_probability, dtype=float)
    valid = np.isfinite(energy) & np.isfinite(p1) & np.isfinite(p2)
    fig = go.Figure()
    if valid.sum() >= 2:
        fig.add_trace(go.Scatterpolar(r=p1[valid], theta=np.linspace(0, 360, valid.sum(), endpoint=False), fill="toself", name="TH1"))
        fig.add_trace(go.Scatterpolar(r=p2[valid], theta=np.linspace(0, 360, valid.sum(), endpoint=False), fill="toself", name="TH2"))
    fig.update_layout(
        title=title,
        height=360,
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        margin=dict(l=30, r=30, t=55, b=30),
    )
    return fig


def overview_figure(df: pd.DataFrame, spec: dict, boundaries: list[dict] | None = None):
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[
            "TH1 / TH2 cross-plane divergence",
            "Detector state",
            "SWIS proton moments",
            "MAG GSE",
            "Transition and evidence scores",
        ],
    )
    for col, name in [("js_opdi", "JS"), ("hellinger_opdi", "Hellinger"), ("wasserstein_opdi", "Wasserstein")]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.timestamp, y=d[col], name=name, mode="lines"), row=1, col=1)

    y_state = {state: i for i, state in enumerate(["NORMAL", "WATCH", "ALERT", "ICME CANDIDATE"])}
    colors = [STATE_COLOR.get(str(x), "#777") for x in d["state"]]
    fig.add_trace(
        go.Scatter(
            x=d.timestamp,
            y=d["state"].map(y_state),
            mode="markers",
            marker=dict(size=4, color=colors),
            name="state",
            text=d["state"],
        ),
        row=2,
        col=1,
    )
    for col in ["proton_bulk_speed", "proton_density", "proton_thermal"]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.timestamp, y=d[col], name=col, mode="lines"), row=3, col=1)
    for col in ["Bx_gse", "By_gse", "Bz_gse", "bmag_gse"]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.timestamp, y=d[col], name=col, mode="lines"), row=4, col=1)
    for col in ["transition_score", "opdi_anomaly_score", "combined_anomaly_score"]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.timestamp, y=d[col], name=col, mode="lines"), row=5, col=1)

    cps = d[d.get("is_change_point", False).astype(bool)] if "is_change_point" in d else d.iloc[0:0]
    if len(cps):
        fig.add_trace(
            go.Scatter(x=cps.timestamp, y=cps.transition_score, mode="markers", marker=dict(size=9), name="change point"),
            row=5,
            col=1,
        )

    for boundary in boundaries or []:
        time = pd.Timestamp(boundary["time"])
        label = str(boundary.get("label", "boundary"))
        for row in range(1, 6):
            fig.add_vline(x=time, line_dash="dash", opacity=0.5, row=row, col=1)
        fig.add_annotation(x=time, y=1, yref="paper", text=label, showarrow=False, textangle=-90)

    fig.update_yaxes(title="distance", row=1, col=1)
    fig.update_yaxes(title="state", tickmode="array", tickvals=list(y_state.values()), ticktext=list(y_state), row=2, col=1)
    fig.update_yaxes(title="SWIS", row=3, col=1)
    fig.update_yaxes(title="nT", row=4, col=1)
    fig.update_yaxes(title="score", row=5, col=1)
    fig.update_xaxes(title="UTC", row=5, col=1)
    fig.update_layout(height=1050, hovermode="x unified", legend=dict(orientation="h"), margin=dict(l=45, r=20, t=60, b=40))
    return fig
