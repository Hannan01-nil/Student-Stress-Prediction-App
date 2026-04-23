from datetime import datetime, timedelta
from pathlib import Path
import random
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import load_or_train_model, predict_from_ui_inputs

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "style.css"
DATASET_PATH = BASE_DIR / "student_lifestyle_dataset_Final.csv"
MODEL_PATH = BASE_DIR / "saved_model.pkl"


st.set_page_config(
    page_title="Student Stress Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open(STYLE_PATH, encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<script>
(function () {
  const killFallbackToggleText = () => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const badNodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const txt = (node.nodeValue || "").trim();
      if (txt.startsWith("keyboard_double_arrow")) badNodes.push(node);
    }
    badNodes.forEach((n) => {
      if (n.parentElement) n.parentElement.style.display = "none";
      n.nodeValue = "";
    });

    document.querySelectorAll('button[kind="header"], [data-testid*="collapsedControl"]').forEach((el) => {
      el.style.display = "none";
      el.style.visibility = "hidden";
      el.style.pointerEvents = "none";
    });
  };

  killFallbackToggleText();
  const observer = new MutationObserver(killFallbackToggleText);
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
</script>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_artifact():
    return load_or_train_model(DATASET_PATH, MODEL_PATH)


def render_html(html):
    st.markdown(dedent(html), unsafe_allow_html=True)


def premium_logo(size=60):
    return f"""
<div class="brand-logo" style="width:{size}px;height:{size}px;">
  <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
      <linearGradient id="logoGrad" x1="10" y1="10" x2="54" y2="54" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#7c6af7"/>
        <stop offset="0.55" stop-color="#3dd6f5"/>
        <stop offset="1" stop-color="#5afac3"/>
      </linearGradient>
    </defs>
    <circle cx="32" cy="32" r="24" fill="rgba(255,255,255,0.04)" stroke="url(#logoGrad)" stroke-width="2"/>
    <path d="M24 24C27 20.5 33 19.8 37.5 22C41.5 24 44 28.5 43 33.2C42 38.4 37.8 42.5 32.5 43.5" fill="none" stroke="url(#logoGrad)" stroke-width="2.2" stroke-linecap="round"/>
    <path d="M22 33.5C24.3 33.5 24.6 29.3 27 29.3C29.4 29.3 29.8 36.4 32.3 36.4C34.8 36.4 34.9 27.3 37.8 27.3C40.1 27.3 40.8 31.4 43 31.4" fill="none" stroke="url(#logoGrad)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    <rect x="23" y="37" width="4" height="7" rx="1.2" fill="#7c6af7"/>
    <rect x="30" y="33" width="4" height="11" rx="1.2" fill="#3dd6f5"/>
    <rect x="37" y="35" width="4" height="9" rx="1.2" fill="#5afac3"/>
    <circle cx="24" cy="24" r="1.9" fill="#7c6af7"/>
    <circle cx="40" cy="25" r="1.9" fill="#3dd6f5"/>
    <circle cx="32" cy="43.5" r="1.9" fill="#5afac3"/>
  </svg>
</div>
"""

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-shell">
          <div class="sidebar-brand-mark">{premium_logo(48)}</div>
          <div class="sidebar-brand-copy">
            <div class="brand-sidebar-title">StressPredict</div>
            <div class="brand-sidebar-sub">v1.0 • Wellness AI</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    about_html = """
    <div class="glass-card sidebar-section-card">
      <div class="sidebar-section-head">
        <span class="sidebar-section-icon">📘</span>
        <div class="sidebar-copy">
          <div class="card-title">About Project</div>
          <p style="font-size:0.85rem; color:#b0b8c6; line-height:1.6; margin:0;">
            This system estimates student stress using academic and lifestyle indicators.
            Enter your daily habits to receive a personalized wellness score and actionable suggestions.
          </p>
        </div>
      </div>
    </div>
    """
    render_html(about_html)

    created_by_html = """
    <div class="glass-card sidebar-section-card">
      <div class="sidebar-section-head">
        <span class="sidebar-section-icon">👨‍💻</span>
        <div class="sidebar-copy">
          <div class="card-title">Created By</div>
          <div class="creator-wrap">
            <div class="creator-mark">H</div>
            <div>
              <div class="creator-name">Mohamed Hannan</div>
              <div class="creator-sub">BCA Student · VIT Vellore</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    render_html(created_by_html)

    tech_html = """
    <div class="glass-card sidebar-section-card">
      <div class="sidebar-section-head">
        <span class="sidebar-section-icon">🧰</span>
        <div class="sidebar-copy">
          <div class="card-title">Technology</div>
          <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
            <span style="background:rgba(124,106,247,0.2);color:#a89af9;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">Python</span>
            <span style="background:rgba(61,214,245,0.15);color:#6ee7f7;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">Streamlit</span>
            <span style="background:rgba(245,166,35,0.15);color:#fcd07a;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">Plotly</span>
            <span style="background:rgba(34,211,160,0.15);color:#5ef5c6;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">Pandas</span>
            <span style="background:rgba(248,113,113,0.15);color:#fca5a5;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">NumPy</span>
          </div>
        </div>
      </div>
    </div>
    """
    render_html(tech_html)

    footer_html = """
    <div class="sidebar-copy" style="margin-top:1rem;font-size:0.75rem;color:#7e8a9a;text-align:center;line-height:1.6;">
      © 2026 Mohamed Hannan<br>Built with Python & Streamlit
    </div>
    """
    render_html(footer_html)


if "history" not in st.session_state:
    st.session_state.history = []
if "predicted" not in st.session_state:
    st.session_state.predicted = False


def contributors_dict(
    sleep, study, pressure, screen, support, exercise, attendance, assign_load, fin_pressure, personal
):
    return {
        "Academic Pressure": round((pressure / 10) * 18, 1),
        "Assignment Load": round((assign_load / 10) * 14, 1),
        "Financial Pressure": round((fin_pressure / 10) * 10, 1),
        "Personal Stress": round((personal / 10) * 12, 1),
        "Low Sleep": round(max(0, (7 - sleep) / 7 * 20), 1),
        "Screen Time": round(max(0, (screen - 4) / 8 * 12), 1),
        "Low Attendance": round(max(0, (75 - attendance) / 75 * 8), 1),
        "Low Exercise": round(max(0, (4 - exercise) / 4 * 8), 1),
        "Low Social Support": round(max(0, (5 - support) / 5 * 8), 1),
    }


def lifestyle_balance(sleep, exercise, support, attendance, screen):
    healthy = (
        min(sleep / 8, 1) * 25
        + (exercise / 7) * 20
        + (support / 10) * 20
        + (attendance / 100) * 20
        + max(0, (8 - screen) / 8) * 15
    )
    healthy = min(100, round(healthy, 1))
    stress_factor = round(100 - healthy, 1)
    return healthy, stress_factor


def stress_trend(base_score):
    noise = np.random.normal(0, 3, 7)
    return [max(0, min(100, base_score + n)) for n in noise]


def generate_insights(
    sleep, study, pressure, screen, support, exercise, attendance, assign_load, fin_pressure, personal
):
    insights = []
    if sleep < 6:
        insights.append(("😴", "Sleep hours are below healthy range — aim for 7–9 hours nightly."))
    elif sleep >= 7:
        insights.append(("✅", "Your sleep schedule is healthy — keep it consistent."))
    if pressure >= 8:
        insights.append(("🔥", "Academic pressure is your highest stress contributor."))
    elif pressure <= 4:
        insights.append(("✅", "Academic pressure is well-managed."))
    if screen > 7:
        insights.append(("📱", "Screen time is high — it may impact focus and sleep quality."))
    if support >= 7:
        insights.append(("🤝", "Strong social support is actively helping reduce your stress."))
    elif support <= 3:
        insights.append(("⚠️", "Low social support can amplify stress — reach out to peers or family."))
    if attendance < 65:
        insights.append(("📋", "Low attendance may increase academic anxiety and backlog pressure."))
    if exercise >= 4:
        insights.append(("🏃", "Regular exercise is a strong natural stress reliever — great habit!"))
    elif exercise <= 1:
        insights.append(("💪", "Minimal exercise detected — even 20-min walks can reduce stress significantly."))
    if fin_pressure >= 8:
        insights.append(("💸", "Financial pressure is a major stressor — consider counselling or assistance schemes."))
    if personal >= 8:
        insights.append(("❤️", "High personal/relationship stress — prioritise self-care and communication."))
    if study > 10:
        insights.append(("📚", "Studying >10 hours daily without breaks leads to burnout."))
    return insights


def generate_suggestions(category):
    if category == "Low":
        return [
            ("🌟", "Maintain your current routine — you're doing great!"),
            ("📈", "Continue your healthy habits and document what works."),
            ("🎯", "Stay consistent — small daily actions compound over time."),
            ("🧘", "Consider mindfulness practices to sustain mental clarity."),
        ]
    if category == "Moderate":
        return [
            ("😴", "Improve your sleep schedule — set a fixed bedtime and wake time."),
            ("⏸️", "Take structured study breaks (Pomodoro: 25 min study / 5 min rest)."),
            ("🏃", "Exercise at least 3–4 times a week — even brisk walks count."),
            ("📵", "Reduce unnecessary screen time, especially 1 hour before sleep."),
            ("📝", "Break large tasks into smaller milestones to reduce overwhelm."),
        ]
    return [
        ("🧠", "Prioritise mental wellness above academic performance right now."),
        ("💬", "Speak with a mentor, counsellor, parents, or trusted friends."),
        ("📉", "Reduce study overload immediately — quality beats quantity."),
        ("🔄", "Restructure your daily routine with dedicated rest periods."),
        ("🆘", "Seek professional guidance if stress feels unmanageable."),
        ("🍎", "Focus on nutrition and hydration — basics are often neglected."),
    ]


def generate_warnings(sleep, screen, attendance, pressure, score):
    warnings = []
    if score >= 65:
        warnings.append("⚠️ High stress detected — this may affect focus, sleep, health, and academic performance.")
    if sleep < 5:
        warnings.append("⚠️ Severe sleep deprivation detected — long-term health risks are significant.")
    if screen > 8:
        warnings.append("⚠️ Excessive screen exposure (>8 hrs) may impair mental clarity and disrupt sleep cycles.")
    if attendance < 60:
        warnings.append("⚠️ Poor attendance (<60%) may create serious academic backlog and exam pressure.")
    if pressure > 8:
        warnings.append("⚠️ Very high academic pressure — academic burnout risk is elevated.")
    return warnings


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#b0b8c6", size=12),
    margin=dict(t=40, b=20, l=20, r=20),
)


def random_demo():
    return dict(
        sleep=round(random.uniform(4, 9), 1),
        study=round(random.uniform(2, 13), 1),
        pressure=random.randint(3, 10),
        screen=round(random.uniform(2, 11), 1),
        support=random.randint(2, 10),
        exercise=random.randint(0, 6),
        attendance=random.randint(45, 100),
        assign_load=random.randint(2, 10),
        fin_pressure=random.randint(1, 10),
        personal=random.randint(1, 10),
    )


def generate_report(inputs, prediction, insights, suggestions, warnings):
    metrics = prediction["metrics"]
    lines = [
        "=" * 60,
        "   STUDENT STRESS PREDICTION REPORT",
        "   Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "   By: Mohamed Hannan | VIT Vellore",
        "=" * 60,
        "",
        f"STRESS SCORE   : {prediction['score']}%",
        f"STRESS CATEGORY: {prediction['category'].upper()} STRESS",
        f"ML MODEL       : {prediction['model_name']}",
        f"VALIDATION ACC : {metrics.get('validation_score', 0.0):.3f}",
        "",
        "—" * 60,
        "INPUT SUMMARY",
        "—" * 60,
        f"Sleep Hours / Day    : {inputs['sleep']} hrs",
        f"Study Hours / Day    : {inputs['study']} hrs",
        f"Academic Pressure    : {inputs['pressure']} / 10",
        f"Screen Time          : {inputs['screen']} hrs",
        f"Social Support       : {inputs['support']} / 10",
        f"Exercise Days / Week : {inputs['exercise']}",
        f"Attendance %         : {inputs['attendance']}%",
        f"Assignment Load      : {inputs['assign_load']} / 10",
        f"Financial Pressure   : {inputs['fin_pressure']} / 10",
        f"Personal Stress      : {inputs['personal']} / 10",
        "",
        "—" * 60,
        "SMART INSIGHTS",
        "—" * 60,
    ]
    for icon, text in insights:
        lines.append(f"  {icon} {text}")
    lines += ["", "—" * 60, "IMPROVEMENT SUGGESTIONS", "—" * 60]
    for icon, text in suggestions:
        lines.append(f"  {icon} {text}")
    if warnings:
        lines += ["", "—" * 60, "WARNINGS", "—" * 60]
        for warning in warnings:
            lines.append(f"  {warning}")
    lines += [
        "",
        "—" * 60,
        "DISCLAIMER",
        "—" * 60,
        "This tool provides estimated wellness insights based on entered",
        "data and a trained machine learning model. It is NOT a medical",
        "diagnosis. For serious stress or mental health concerns, consult",
        "a qualified professional.",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


st.markdown("""
<h1 style='
font-size:48px;
font-weight:900;
color:white;
margin-bottom:5px;
'>
Student Stress Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='
font-size:18px;
color:#9ca3af;
margin-top:0;
margin-bottom:25px;
'>
AI-powered wellness dashboard to estimate student stress and provide improvement guidance.
</p>
""", unsafe_allow_html=True)


ctrl_col1, ctrl_col2 = st.columns(2)

with ctrl_col1:
    demo_clicked = st.button("🎲 Demo Data", use_container_width=True)

with ctrl_col2:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if "demo" not in st.session_state:
    st.session_state.demo = {}

if demo_clicked:
    st.session_state.demo = random_demo()
    st.session_state.predicted = False

if reset_clicked:
    st.session_state.demo = {}
    st.session_state.predicted = False

demo = st.session_state.demo

artifact_error = None
artifact = None
try:
    artifact = get_model_artifact()
except Exception as exc:
    artifact_error = str(exc)

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-heading">🎛️ Lifestyle Inputs</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-title">Sleep & Study</div>', unsafe_allow_html=True)
    sleep_val = st.slider("😴 Sleep Hours / Day", 0.0, 12.0, float(demo.get("sleep", 6.5)), 0.5)
    study_val = st.slider("📚 Study Hours / Day", 0.0, 14.0, float(demo.get("study", 7.0)), 0.5)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown('<div class="card-title">Academic Factors</div>', unsafe_allow_html=True)
    pressure_val = st.slider("🔥 Academic Pressure (1–10)", 1, 10, demo.get("pressure", 5))
    assign_val = st.slider("📝 Assignment Load (1–10)", 1, 10, demo.get("assign_load", 5))
    attendance_val = st.slider("📋 Attendance %", 0, 100, demo.get("attendance", 80))
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown('<div class="card-title">Lifestyle & Wellbeing</div>', unsafe_allow_html=True)
    screen_val = st.slider("📱 Screen Time (hrs/day)", 0.0, 12.0, float(demo.get("screen", 5.0)), 0.5)
    exercise_val = st.slider("🏃 Exercise Days / Week", 0, 7, demo.get("exercise", 2))
    support_val = st.slider("🤝 Social Support (1–10)", 1, 10, demo.get("support", 6))
    fin_val = st.slider("💸 Financial Pressure (1–10)", 1, 10, demo.get("fin_pressure", 4))
    personal_val = st.slider(
        "❤️ Personal / Relationship Stress (1–10)", 1, 10, demo.get("personal", 4)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔮 Predict My Stress", use_container_width=True)


ui_inputs = dict(
    sleep=sleep_val,
    study=study_val,
    pressure=pressure_val,
    screen=screen_val,
    support=support_val,
    exercise=exercise_val,
    attendance=attendance_val,
    assign_load=assign_val,
    fin_pressure=fin_val,
    personal=personal_val,
)

prediction = None
if artifact is not None:
    prediction = predict_from_ui_inputs(ui_inputs, artifact)


with col_result:
    st.markdown('<div class="section-heading">📈 Prediction Results</div>', unsafe_allow_html=True)

    if artifact_error:
        st.markdown(
            f"""
        <div class="glass-card" style="border-color:rgba(248,113,113,0.35);">
          <div class="card-title" style="color:#f87171;">Model Error</div>
          <div style="font-size:0.92rem; color:#fca5a5; line-height:1.7;">
            Unable to load or train the machine learning model.<br>{artifact_error}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif predict_btn or st.session_state.predicted:
        if predict_btn:
            st.session_state.predicted = True

        score = prediction["score"]
        category = prediction["category"]

        entry = {"time": datetime.now().strftime("%H:%M"), "score": score, "category": category}
        if predict_btn:
            st.session_state.history.append(entry)

        if category == "Low":
            score_color = "#22d3a0"
            pill_class = "cat-low"
            bar_gradient = "linear-gradient(90deg,#22d3a0,#6ee7b7)"
        elif category == "Moderate":
            score_color = "#fbbf24"
            pill_class = "cat-mod"
            bar_gradient = "linear-gradient(90deg,#f59e0b,#fcd34d)"
        else:
            score_color = "#f87171"
            pill_class = "cat-high"
            bar_gradient = "linear-gradient(90deg,#ef4444,#f87171)"

        st.markdown(
            f"""
        <div class="glass-card" style="border-color:rgba({','.join(str(int(score_color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},0.3);">
          <div class="score-badge">
            <div class="score-number" style="color:{score_color};">{score}</div>
            <div class="score-label">STRESS SCORE OUT OF 100</div>
            <div><span class="category-pill {pill_class}">{category} Stress</span></div>
          </div>
          <div class="prog-track">
            <div class="prog-fill" style="width:{score}%; background:{bar_gradient};"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#7e8a9a;margin-top:0.5rem;">
            <span>0 — Low</span><span>35 — Moderate</span><span>65 — High — 100</span>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"size": 36, "color": score_color, "family": "Syne"}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#7e8a9a",
                        "tickfont": {"color": "#7e8a9a", "size": 10},
                    },
                    "bar": {"color": score_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(255,255,255,0.08)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(34,211,160,0.15)"},
                        {"range": [35, 65], "color": "rgba(251,191,36,0.12)"},
                        {"range": [65, 100], "color": "rgba(248,113,113,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": score_color, "width": 3},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
                title={
                    "text": "Stress Level Gauge",
                    "font": {"color": "#7e8a9a", "size": 13, "family": "DM Sans"},
                },
            )
        )
        gauge.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        healthy_pct, stress_pct = lifestyle_balance(
            sleep_val, exercise_val, support_val, attendance_val, screen_val
        )
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(
            f"""<div class="metric-mini">
          <div class="metric-mini-val" style="color:#22d3a0;">{healthy_pct}%</div>
          <div class="metric-mini-label">Healthy Balance</div></div>""",
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f"""<div class="metric-mini">
          <div class="metric-mini-val" style="color:#f87171;">{stress_pct}%</div>
          <div class="metric-mini-label">Stress Factors</div></div>""",
            unsafe_allow_html=True,
        )
        mc3.markdown(
            f"""<div class="metric-mini">
          <div class="metric-mini-val" style="color:#fbbf24;">{len(st.session_state.history)}</div>
          <div class="metric-mini-label">Predictions Run</div></div>""",
            unsafe_allow_html=True,
        )

        metric_label = "Accuracy" if prediction["problem_type"] == "classification" else "R²"
        metric_text = f"Model: {prediction['model_name']} | {metric_label}: {prediction['metrics'].get('validation_score', 0.0):.3f}"
        if prediction["problem_type"] == "regression":
            metric_text += f" | RMSE: {prediction['metrics'].get('rmse', 0.0):.3f}"
        st.markdown(
            f"""
        <div style="margin-top:0.85rem;font-size:0.76rem;color:#7e8a9a;text-align:center;">
          {metric_text}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="glass-card" style="text-align:center; padding:3rem 2rem;">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🔮</div>
          <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#b0b8c6;">
            Ready to Predict
          </div>
          <div style="font-size:0.88rem; color:#7e8a9a; margin-top:0.5rem; line-height:1.6;">
            Fill in your lifestyle inputs on the left<br>and click <strong style="color:#a89af9;">Predict My Stress</strong>.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


if st.session_state.predicted and prediction is not None:
    score = prediction["score"]
    category = prediction["category"]
    contrib = contributors_dict(
        sleep_val,
        study_val,
        pressure_val,
        screen_val,
        support_val,
        exercise_val,
        attendance_val,
        assign_val,
        fin_val,
        personal_val,
    )
    healthy_pct, stress_pct = lifestyle_balance(
        sleep_val, exercise_val, support_val, attendance_val, screen_val
    )

    st.markdown('<div class="section-heading">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="medium")

    with ch1:
        sorted_contrib = dict(sorted(contrib.items(), key=lambda item: item[1], reverse=True))
        bar_fig = go.Figure(
            go.Bar(
                x=list(sorted_contrib.values()),
                y=list(sorted_contrib.keys()),
                orientation="h",
                marker=dict(
                    color=list(sorted_contrib.values()),
                    colorscale=[[0, "#22d3a0"], [0.5, "#fbbf24"], [1, "#f87171"]],
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{value}" for value in sorted_contrib.values()],
                textposition="outside",
                textfont=dict(color="#b0b8c6", size=11),
            )
        )
        bar_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            title=dict(text="Top Stress Contributors", font=dict(family="Syne", color="#e8eaf0", size=14)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Contribution Score"),
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        pie_fig = go.Figure(
            go.Pie(
                labels=["Healthy Habits", "Stress Factors"],
                values=[healthy_pct, stress_pct],
                hole=0.55,
                marker=dict(colors=["#22d3a0", "#f87171"], line=dict(color="rgba(0,0,0,0)", width=0)),
                textfont=dict(family="DM Sans", color="white", size=12),
            )
        )
        pie_fig.add_annotation(
            text=f"{healthy_pct}%",
            x=0.5,
            y=0.5,
            font=dict(size=22, color="#22d3a0", family="Syne"),
            showarrow=False,
        )
        pie_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            title=dict(text="Lifestyle Balance", font=dict(family="Syne", color="#e8eaf0", size=14)),
            legend=dict(font=dict(color="#b0b8c6")),
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

    ch3, ch4 = st.columns(2, gap="medium")

    with ch3:
        np.random.seed(int(score))
        trend = stress_trend(score)
        days = [(datetime.today() + timedelta(days=index)).strftime("%a %d") for index in range(7)]
        line_fig = go.Figure()
        line_fig.add_trace(
            go.Scatter(
                x=days,
                y=trend,
                mode="lines+markers",
                line=dict(color="#7c6af7", width=2.5, shape="spline"),
                marker=dict(size=7, color="#7c6af7", line=dict(color="#3dd6f5", width=1.5)),
                fill="tozeroy",
                fillcolor="rgba(124,106,247,0.08)",
                name="Stress Trend",
            )
        )
        line_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            title=dict(text="7-Day Stress Projection", font=dict(family="Syne", color="#e8eaf0", size=14)),
            yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)", title="Stress %"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(line_fig, use_container_width=True, config={"displayModeBar": False})

    with ch4:
        categories = [
            "Sleep Quality",
            "Exercise",
            "Social Support",
            "Low Screen Time",
            "Attendance",
            "Low Pressure",
        ]
        values = [
            (sleep_val / 9) * 10,
            (exercise_val / 7) * 10,
            support_val,
            max(0, (8 - screen_val) / 8 * 10),
            (attendance_val / 100) * 10,
            max(0, 10 - pressure_val),
        ]
        values_rounded = [round(value, 1) for value in values]
        radar_fig = go.Figure(
            go.Scatterpolar(
                r=values_rounded + [values_rounded[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(61,214,245,0.1)",
                line=dict(color="#3dd6f5", width=2),
                marker=dict(color="#3dd6f5", size=6),
            )
        )
        radar_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            title=dict(text="Lifestyle Balance Radar", font=dict(family="Syne", color="#e8eaf0", size=14)),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                angularaxis=dict(tickcolor="#7e8a9a", gridcolor="rgba(255,255,255,0.07)"),
                radialaxis=dict(
                    range=[0, 10],
                    visible=True,
                    gridcolor="rgba(255,255,255,0.07)",
                    tickcolor="#7e8a9a",
                ),
            ),
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    ins_col, sug_col = st.columns(2, gap="large")
    insights = generate_insights(
        sleep_val,
        study_val,
        pressure_val,
        screen_val,
        support_val,
        exercise_val,
        attendance_val,
        assign_val,
        fin_val,
        personal_val,
    )
    suggestions = generate_suggestions(category)
    warnings = generate_warnings(sleep_val, screen_val, attendance_val, pressure_val, score)

    with ins_col:
        st.markdown('<div class="section-heading">🧠 Smart Analysis</div>', unsafe_allow_html=True)
        for icon, text in insights:
            st.markdown(
                f"""
            <div class="insight-chip">
              <span class="insight-icon">{icon}</span>
              <span>{text}</span>
            </div>""",
                unsafe_allow_html=True,
            )

    with sug_col:
        st.markdown('<div class="section-heading">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
        for icon, text in suggestions:
            st.markdown(
                f"""
            <div class="sugg-card">
              <span class="sugg-icon">{icon}</span>
              <span class="sugg-text">{text}</span>
            </div>""",
                unsafe_allow_html=True,
            )

    if warnings:
        st.markdown('<div class="section-heading">⚠️ Cautions & Warnings</div>', unsafe_allow_html=True)
        for warning in warnings:
            st.markdown(f'<div class="warn-banner">{warning}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📥 Download Report</div>', unsafe_allow_html=True)
    report_text = generate_report(ui_inputs, prediction, insights, suggestions, warnings)
    st.download_button(
        label="📄 Download Stress Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"stress_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=False,
    )

    if len(st.session_state.history) > 1:
        with st.expander(f"📋 Session History  ({len(st.session_state.history)} predictions)"):
            history_df = pd.DataFrame(st.session_state.history)
            hist_fig = go.Figure(
                go.Scatter(
                    x=history_df["time"],
                    y=history_df["score"],
                    mode="lines+markers+text",
                    text=[str(score_value) for score_value in history_df["score"]],
                    textposition="top center",
                    line=dict(color="#7c6af7", width=2),
                    marker=dict(color="#3dd6f5", size=8),
                )
            )
            hist_fig.update_layout(
                **PLOTLY_LAYOUT,
                height=220,
                yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.06)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                title=dict(text="Score History This Session", font=dict(family="Syne", color="#e8eaf0", size=13)),
            )
            st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})


st.markdown(
    """
<div class="disclaimer-box">
  <strong style="color:#b0b8c6;">📋 Disclaimer</strong><br>
  This tool provides <em>estimated</em> wellness insights based on self-reported data.
  It is <strong>not a medical diagnosis</strong> and should not replace professional advice.
  For serious stress, anxiety, or mental health concerns, please consult a qualified healthcare
  professional, counsellor, or trusted adult.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; padding: 1.5rem 0 0.5rem; font-size:0.8rem; color:#4a5568;">
  © 2026 <strong style="color:#7c6af7;">Mohamed Hannan</strong> &nbsp;|&nbsp;
  Built with Python & Streamlit
</div>
""",
    unsafe_allow_html=True,
)
