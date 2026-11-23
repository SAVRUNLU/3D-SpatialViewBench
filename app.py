#!/usr/bin/env python3
"""Streamlit dashboard for 3D-SpatialViewBench results.

Run with:  streamlit run app.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODELS = ["deepseek-v4.1-flash", "glm-5.3", "gemma-4-31b"]
N_ITEMS = 60
RANDOM_BASELINE = 0.25

# Distill palette.
C_MODELS = dict(zip(MODELS, ["#2B5C8F", "#C85A32", "#2E7D5B"]))  # steel blue, sienna, sage
C_PAPER = "#FCFCFA"      # ivory page
C_CARD = "#FFFFFF"       # figure containers
C_BORDER = "#E6E4DD"     # card hairline
C_RULE = "#D8D6CF"       # metadata bar rules
C_INK = "#1A1A1A"        # charcoal
C_MUTED = "#5C5B57"      # warm gray captions
C_REF = "#A8A69E"        # baseline dash
OBJECT_COLORS = ["#2B5C8F", "#C85A32", "#2E7D5B", "#8A6D3B"]

st.set_page_config(page_title="3D-SpatialViewBench", layout="wide")

# Font overrides are scoped to html/body and to code/pre/metric nodes only.
# Wildcard or span selectors here clobber Streamlit's Material Symbols icon
# font, which makes expander arrows render as literal "_arrow_down" text.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

header, footer {{visibility: hidden !important;}}
#MainMenu {{visibility: hidden !important;}}
div[data-testid="stDecoration"] {{display: none !important;}}
div[data-testid="collapsedControl"] {{display: none !important;}}

html, body {{
    color: {C_INK};
    font-family: 'Inter', -apple-system, "Segoe UI", sans-serif;
}}
.stApp {{background: {C_PAPER};}}
div[data-testid="stHeader"] {{background: transparent;}}

h1, h2, h3, h4 {{
    font-family: 'Newsreader', Georgia, serif !important;
    color: {C_INK} !important;
    font-weight: 500 !important;
}}

[data-testid="stMetricValue"], code, pre {{
    font-family: 'JetBrains Mono', Consolas, monospace !important;
}}

[data-testid="stMetric"] {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
}}
[data-testid="stMetricLabel"] p {{
    color: {C_MUTED} !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em;
}}

div[data-testid="stTabs"] button {{
    font-family: 'Inter', sans-serif;
    color: {C_MUTED};
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_INK};
}}

hr, [data-testid="stHorizontalBlock"] [data-testid="stWidgetLabel"] {{
    border-color: {C_RULE};
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    metrics = pd.read_csv(BASE / "results" / "summary_metrics.csv")
    predictions = json.loads((BASE / "results" / "raw_predictions.json").read_text(encoding="utf-8"))
    dataset = json.loads((BASE / "dataset.json").read_text(encoding="utf-8"))
    return metrics, predictions, dataset


metrics_df, predictions, dataset = load_data()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_positions(item):
    positions = {}
    for part in item["scene_description"].split("; "):
        m = re.search(r"^(Object [A-Z]) is .*? at world coordinates \((-?\d+), (-?\d+), (-?\d+)\)", part)
        if m:
            positions[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return positions


def col(basis, k):
    """k-th column of a row-major 3x3 basis = camera axis k in world coords."""
    return [basis[0][k], basis[1][k], basis[2][k]]


def camera_maneuver(item):
    gt = item["ground_truth_math"]
    return gt.get("camera_rotation_y_deg", gt.get("camera_yaw_deg")), gt.get("camera_pitch_deg")


def question_label(item):
    q = item["question"]
    return f"{item['id']} · {item['difficulty']} — {q[:80]}{'…' if len(q) > 80 else ''}"


def figure(fig):
    """Wrap a Plotly figure in a white figure card with a hairline border."""
    st.markdown(
        f"<div style='background:{C_CARD};border:1px solid {C_BORDER};"
        f"border-radius:4px;padding:0.4rem;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)


def bar_metric(value_col, title, y_fmt, extra_line=None):
    fig = go.Figure()
    for model in MODELS:
        row = metrics_df.loc[metrics_df["model"] == model].iloc[0]
        fig.add_trace(go.Bar(
            x=[model],
            y=[float(row[value_col])],
            name=model,
            marker_color=C_MODELS[model],
            width=0.55,
            text=[y_fmt(row[value_col])],
            textposition="outside",
            textfont=dict(color=C_INK, family="JetBrains Mono, monospace", size=12),
            hovertemplate=f"<b>{model}</b><br>{title}: %{{y:.3f}}<extra></extra>",
        ))
    if extra_line is not None:
        fig.add_hline(
            y=extra_line,
            line_color=C_REF, line_width=1, line_dash="dash",
            annotation_text="25% random baseline",
            annotation_position="top left",
            annotation_font=dict(color=C_MUTED, size=11),
        )
    y_max = max(float(metrics_df[value_col].max()), extra_line or 0) * 1.22
    fig.update_layout(
        plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, showlegend=False,
        title=dict(text=title, font=dict(size=14, color=C_INK, family="Newsreader, serif")),
        height=290, margin=dict(l=8, r=8, t=42, b=8),
        yaxis=dict(range=[0, y_max], gridcolor=C_BORDER, zeroline=False,
                   title=None, tickfont=dict(color=C_MUTED, size=11)),
        xaxis=dict(showgrid=False, title=None, tickfont=dict(color=C_MUTED, size=11)),
        bargap=0.45,
        font=dict(color=C_INK),
    )
    return fig


# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
errors_total = int(metrics_df["Errors"].sum())
compliance_min, compliance_max = metrics_df["JSON Compliance %"].min(), metrics_df["JSON Compliance %"].max()

st.markdown(f"""
<div style="max-width:1060px;margin:0.5rem auto 0;">
  <div style="font-family:'Newsreader',Georgia,serif;font-size:2.35rem;font-weight:600;color:{C_INK};line-height:1.15;">
    3D-SpatialViewBench
  </div>
  <p style="color:{C_MUTED};font-size:1.02rem;line-height:1.6;margin:0.7rem 0 1.1rem;max-width:52rem;">
    Can a text-only LLM compose rotation matrices and re-express a world vector in a rotated
    camera frame, given exact coordinates and no perception? Three models, 60 items, greedy
    decoding. Only <code style="font-family:'JetBrains Mono',monospace;">gemma-4-31b</code>
    clears the 25% chance floor; the other two score as if picking letters at random.
  </p>
</div>
""", unsafe_allow_html=True)

mb1, mb2, mb3, mb4, mb5 = st.columns([1, 2, 2.5, 2, 1.5])
mb1.markdown(
    f"<div style='border-top:1px solid {C_RULE};border-bottom:1px solid {C_RULE};"
    f"padding:0.55rem 0;'><span style='color:{C_MUTED};font-size:0.72rem;"
    f"letter-spacing:0.08em;text-transform:uppercase;'>Models</span>&nbsp;&nbsp;"
    f"<span style='font-family:JetBrains Mono,monospace;color:{C_INK};'>3</span></div>",
    unsafe_allow_html=True,
)
mb2.markdown(
    f"<div style='border-top:1px solid {C_RULE};border-bottom:1px solid {C_RULE};"
    f"padding:0.55rem 0;'><span style='color:{C_MUTED};font-size:0.72rem;"
    f"letter-spacing:0.08em;text-transform:uppercase;'>Items</span>&nbsp;&nbsp;"
    f"<span style='font-family:JetBrains Mono,monospace;color:{C_INK};'>60 &times; 3 = 180 calls</span></div>",
    unsafe_allow_html=True,
)
mb3.markdown(
    f"<div style='border-top:1px solid {C_RULE};border-bottom:1px solid {C_RULE};"
    f"padding:0.55rem 0;'><span style='color:{C_MUTED};font-size:0.72rem;"
    f"letter-spacing:0.08em;text-transform:uppercase;'>Decoding</span>&nbsp;&nbsp;"
    f"<span style='font-family:JetBrains Mono,monospace;color:{C_INK};'>greedy (T=0)</span></div>",
    unsafe_allow_html=True,
)
mb4.markdown(
    f"<div style='border-top:1px solid {C_RULE};border-bottom:1px solid {C_RULE};"
    f"padding:0.55rem 0;'><span style='color:{C_MUTED};font-size:0.72rem;"
    f"letter-spacing:0.08em;text-transform:uppercase;'>JSON compliance</span>&nbsp;&nbsp;"
    f"<span style='font-family:JetBrains Mono,monospace;color:{C_INK};'>{compliance_min:.1f}–{compliance_max:.1f}%</span></div>",
    unsafe_allow_html=True,
)
mb5.markdown(
    f"<div style='border-top:1px solid {C_RULE};border-bottom:1px solid {C_RULE};"
    f"padding:0.55rem 0;'><span style='color:{C_MUTED};font-size:0.72rem;"
    f"letter-spacing:0.08em;text-transform:uppercase;'>API errors</span>&nbsp;&nbsp;"
    f"<span style='font-family:JetBrains Mono,monospace;color:{C_INK};'>{errors_total}</span></div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

tab_overview, tab_explorer = st.tabs(["Overview", "3D Scene Inspector"])

# --------------------------------------------------------------------------- #
# Tab 1 — Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    best = metrics_df.loc[metrics_df["Overall Accuracy"].idxmax()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best overall", f"{best['Overall Accuracy']:.1%}", best["model"])
    k2.metric("Random baseline", "25.0%")
    k3.metric("Best JSON compliance", f"{compliance_max:.1f}%")
    k4.metric("API errors", errors_total)

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    g1c1, g1c2 = st.columns(2)
    g2c1, g2c2 = st.columns(2)
    with g1c1:
        figure(bar_metric("Overall Accuracy", "Overall accuracy", lambda v: f"{v:.1%}", RANDOM_BASELINE))
    with g1c2:
        figure(bar_metric("Level 1 Accuracy", "Level 1 — single yaw", lambda v: f"{v:.1%}", RANDOM_BASELINE))
    with g2c1:
        figure(bar_metric("Level 2 Accuracy", "Level 2 — translate + yaw", lambda v: f"{v:.1%}", RANDOM_BASELINE))
    with g2c2:
        figure(bar_metric("Level 3 Accuracy", "Level 3 — yaw then pitch", lambda v: f"{v:.1%}", RANDOM_BASELINE))

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        figure(bar_metric("JSON Compliance %", "JSON compliance", lambda v: f"{v:.1f}%"))
    with s2:
        figure(bar_metric("Avg Latency (s)", "Mean latency (s)", lambda v: f"{v:.1f}s"))

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Summary matrix")
    st.markdown(
        "<p style='color:" + C_MUTED + ";font-size:0.9rem;margin-top:-0.4rem;'>"
        "Per-model aggregates over the 60-item set. Level columns cover 20 items each.</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        metrics_df.set_index("model").style.format({
            "Overall Accuracy": "{:.2%}", "Level 1 Accuracy": "{:.2%}",
            "Level 2 Accuracy": "{:.2%}", "Level 3 Accuracy": "{:.2%}",
            "JSON Compliance %": "{:.2f}%", "Avg Latency (s)": "{:.1f}",
        }),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Tab 2 — 3D Scene Inspector
# --------------------------------------------------------------------------- #
with tab_explorer:
    sel_col, view_col = st.columns([3, 1])
    with sel_col:
        idx = st.selectbox(
            "Select a question",
            options=range(len(dataset)),
            format_func=lambda i: question_label(dataset[i]),
            label_visibility="collapsed",
        )
    with view_col:
        show_trail = st.toggle("Rotation trail", value=False)

    item = dataset[idx]
    gt = item["ground_truth_math"]
    positions = parse_positions(item)
    yaw, pitch = camera_maneuver(item)
    shift = gt.get("camera_translation_world")

    st.markdown(
        f"<p style='color:{C_MUTED};font-size:0.92rem;line-height:1.55;margin-bottom:0.2rem;'>"
        f"<b style='color:{C_INK};'>{item['id']} · {item['difficulty']}</b><br>"
        f"{item['scene_description']}<br>"
        f"<i>{item['question']}</i></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family:JetBrains Mono,monospace;font-size:0.85rem;color:"
        + C_INK + ";'>"
        + "&nbsp;&nbsp;&nbsp;".join(f"{k}: {v}" for k, v in item["options"].items())
        + "</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("### Scene")
        fig = go.Figure()
        all_pts = list(positions.values()) + [(0, 0, 0)] + ([shift] if shift else [])

        for i, (label, pos) in enumerate(positions.items()):
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers+text",
                marker=dict(size=8, color=OBJECT_COLORS[i % len(OBJECT_COLORS)]),
                text=[label], textposition="top center",
                textfont=dict(color=C_INK, size=11),
                name=label,
                hovertemplate=f"<b>{label}</b><br>({pos[0]}, {pos[1]}, {pos[2]})<extra></extra>",
            ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers",
            marker=dict(size=5, color=C_INK, symbol="diamond"),
            name="Camera (start)", hovertemplate="Camera start (0,0,0)<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, -2.5], mode="lines",
            line=dict(color=C_INK, width=3, dash="dot"),
            name="Initial facing (−Z)", hoverinfo="skip",
        ))

        origin_after = list(shift) if shift else [0, 0, 0]
        if shift:
            fig.add_trace(go.Scatter3d(
                x=[shift[0]], y=[shift[1]], z=[shift[2]], mode="markers",
                marker=dict(size=6, color=C_INK, symbol="x"),
                name="Camera (translated)", hovertemplate="Camera after translation<extra></extra>",
            ))

        if "camera_basis_after" in gt:
            basis = gt["camera_basis_after"]
            fwd, right_ax, up = col(basis, 2), col(basis, 0), col(basis, 1)
            L = 2.5

            def seg(vec, color, name):
                return go.Scatter3d(
                    x=[origin_after[0], origin_after[0] + vec[0] * L],
                    y=[origin_after[1], origin_after[1] + vec[1] * L],
                    z=[origin_after[2], origin_after[2] + vec[2] * L],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=name, hoverinfo="skip",
                )

            fig.add_trace(seg(fwd, "#B03A2E", "fwd"))
            fig.add_trace(seg(right_ax, "#2B5C8F", "right"))
            fig.add_trace(seg(up, "#2E7D5B", "up"))

            if show_trail:
                fig.add_trace(go.Scatter3d(
                    x=[0, origin_after[0] + fwd[0] * L],
                    y=[0, origin_after[1] + fwd[1] * L],
                    z=[-2.5, origin_after[2] + fwd[2] * L],
                    mode="lines", line=dict(color="#B03A2E", width=2, dash="dash"),
                    name="Rotation trail", hoverinfo="skip", showlegend=False,
                ))

        pad = 2.5
        fig.update_layout(
            height=560,
            scene=dict(
                xaxis=dict(title="X (right)", range=[min(p[0] for p in all_pts) - pad, max(p[0] for p in all_pts) + pad], gridcolor=C_BORDER, zerolinecolor=C_RULE),
                yaxis=dict(title="Y (up)", range=[min(p[1] for p in all_pts) - pad, max(p[1] for p in all_pts) + pad], gridcolor=C_BORDER, zerolinecolor=C_RULE),
                zaxis=dict(title="Z (back)", range=[min(p[2] for p in all_pts) - pad, max(p[2] for p in all_pts) + pad], gridcolor=C_BORDER, zerolinecolor=C_RULE),
                aspectmode="cube",
                bgcolor=C_CARD,
                camera=dict(eye=dict(x=1.6, y=1.1, z=1.4)),
            ),
            margin=dict(l=0, r=0, b=0, t=8),
            paper_bgcolor=C_CARD,
            font=dict(color=C_MUTED),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title="",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        )
        figure(fig)
        st.markdown(
            f"<p style='color:{C_MUTED};font-size:0.82rem;margin-top:-0.6rem;'>"
            f"Diamond = camera start. Solid axes = camera basis after rotation "
            f"(red = fwd, blue = right, green = up). Drag to orbit, scroll to zoom.</p>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Ground truth")
        st.markdown(
            f"<p style='font-family:JetBrains Mono,monospace;font-size:1.05rem;color:{C_INK};'>"
            f"{item['correct_answer']} — {item['options'][item['correct_answer']]}</p>",
            unsafe_allow_html=True,
        )

        cv = gt.get("camera_frame_relative_vector")
        if cv:
            st.markdown(
                f"<p style='color:{C_MUTED};font-size:0.85rem;margin-bottom:0.1rem;'>"
                f"Relative vector (camera frame: x right, y up, z back)</p>"
                f"<p style='font-family:JetBrains Mono,monospace;color:{C_INK};'>{tuple(cv)}</p>",
                unsafe_allow_html=True,
            )

        rotated = gt.get("positions_in_new_camera_frame")
        if rotated:
            st.dataframe(
                pd.DataFrame(
                    [{"Object": k, "x": v[0], "y": v[1], "z": v[2]} for k, v in rotated.items()]
                ).set_index("Object"),
                use_container_width=True,
            )

        st.markdown("### Transformation")
        steps = []
        if shift:
            steps.append(f"translate {tuple(shift)} (world)")
        steps.append(f"yaw {yaw}°")
        if pitch:
            steps.append(f"pitch {pitch}° (local right)")
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:0.88rem;color:"
            + C_INK + ";'> &rarr; ".join(steps) + "</p>",
            unsafe_allow_html=True,
        )
        if "camera_basis_after" in gt:
            st.dataframe(
                pd.DataFrame(gt["camera_basis_after"], index=["r_x", "r_y", "r_z"],
                             columns=["right", "up", "back"]),
                use_container_width=True,
            )
            st.markdown(
                f"<p style='color:{C_MUTED};font-size:0.8rem;margin-top:-0.5rem;'>"
                f"Columns = camera axes in world coordinates. v<sub>cam</sub> = Mᵀ(p − t).</p>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.0rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Model outputs")
    cols = st.columns(len(MODELS))
    for col_, model in zip(cols, MODELS):
        rec = next((p for p in predictions.get(model, []) if p["id"] == item["id"]), None)
        with col_:
            st.markdown(
                f"<div style='background:{C_CARD};border:1px solid {C_BORDER};"
                f"border-top:3px solid {C_MODELS[model]};border-radius:4px;"
                f"padding:0.8rem 0.9rem;min-height:7.5rem;'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                f"color:{C_INK};font-weight:500;'>{model}</div>",
                unsafe_allow_html=True,
            )
            if not rec:
                st.markdown(
                    f"<div style='color:{C_MUTED};font-size:0.85rem;margin-top:0.4rem;'>"
                    f"no record</div></div>", unsafe_allow_html=True)
                continue
            if rec["correct"]:
                verdict = (f"<span style='color:{C_MODELS[model]};font-weight:600;'>correct</span>"
                           f" · option {rec['selected_option']}")
            else:
                chosen = rec["selected_option"]
                detail = item["options"].get(chosen, chosen) if chosen else "no parseable answer"
                verdict = (f"<span style='color:{C_MUTED};'>wrong</span>"
                           f" · chose {chosen or '—'} ({detail})")
            lat = rec.get("latency_seconds")
            lat_txt = f"{lat:.1f}s" if lat else "—"
            if not rec.get("json_compliant"):
                lat_txt += " · non-JSON"
            st.markdown(
                f"<div style='font-size:0.85rem;color:{C_INK};margin-top:0.45rem;'>"
                f"{verdict}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;"
                f"color:{C_MUTED};margin-top:0.3rem;'>{lat_txt}</div>",
                unsafe_allow_html=True,
            )
            trace = (rec.get("reasoning") or rec.get("raw_output")
                     or rec.get("error") or "_empty_")
            st.markdown(
                f"<div style='border-top:1px solid {C_BORDER};margin-top:0.6rem;"
                f"padding-top:0.55rem;font-size:0.78rem;color:{C_MUTED};"
                f"line-height:1.5;max-height:14rem;overflow-y:auto;white-space:pre-wrap;'>"
                f"{trace}</div></div>",
                unsafe_allow_html=True,
            )
