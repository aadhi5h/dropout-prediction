import streamlit as st
import pandas as pd
import random
from xgboost import XGBClassifier

st.set_page_config(page_title="Dropout Risk Console", page_icon="🛰️", layout="centered")

GROUP_NUMBER = 8
PRIORITY_MEMBER = "P S Aadhish"
PRIORITY_ID = "VML24AD092"
OTHER_MEMBERS = [
    ("Prajwal Pratheev", "VML24AD091"),
    ("Nived Sunil", "VML24AD086"),
    ("Navalrag V P", "VML24AD084"),
    ("Sanha Sadik", "VML24AD099"),
    ("Muzeen K", "VML24AD083"),
]

if "member_order" not in st.session_state:
    shuffled = OTHER_MEMBERS.copy()
    random.shuffle(shuffled)
    st.session_state.member_order = [(PRIORITY_MEMBER, PRIORITY_ID)] + shuffled

# ==============================================================================
# THEME
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #070c14;
    --panel: #0e151f;
    --panel-border: #1c2733;
    --text: #dce6f0;
    --text-dim: #64758a;
    --cyan: #35e0c2;
    --amber: #ffb454;
    --coral: #ff5c72;
    --mono: 'JetBrains Mono', monospace;
    --display: 'Space Grotesk', sans-serif;
}

html, body, [class*="css"] { font-family: var(--display); color: var(--text); }
.stApp { background: radial-gradient(circle at 15% 0%, #0b1420 0%, var(--bg) 45%); }

#MainMenu, footer, header { visibility: hidden; }

/* ---- boot header ---- */
.console-header { margin: 0.5rem 0 0.25rem 0; }
.console-header .kicker {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.12em;
    color: var(--cyan); display: flex; align-items: center; gap: 0.5rem;
}
.console-header .kicker .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--cyan);
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
.console-header h1 {
    font-family: var(--display); font-weight: 700; font-size: 2.1rem;
    margin: 0.3rem 0 0 0; color: var(--text); line-height: 1.15;
}
.console-header h1 .cursor {
    display: inline-block; width: 3px; height: 1.6rem; background: var(--cyan);
    margin-left: 6px; vertical-align: -3px; animation: blink 1.1s step-end infinite;
}
@keyframes blink { 0%,49% { opacity: 1; } 50%,100% { opacity: 0; } }

.stat-row { display: flex; gap: 0.5rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap; }
.stat-chip {
    font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim);
    border: 1px solid var(--panel-border); background: var(--panel);
    padding: 0.3rem 0.6rem; border-radius: 4px;
}
.stat-chip b { color: var(--text); font-weight: 500; }

/* ---- section labels ---- */
.section-label {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.08em;
    color: var(--text-dim); border-bottom: 1px solid var(--panel-border);
    padding-bottom: 0.4rem; margin: 1.6rem 0 0.8rem 0;
}

/* ---- bordered panels (native st.container(border=True)) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 6px !important;
}

/* ---- sliders ---- */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--cyan) !important;
    box-shadow: 0 0 0 4px rgba(53, 224, 194, 0.15) !important;
}
div[data-testid="stSlider"] label p { font-family: var(--mono); font-size: 0.82rem; color: var(--text-dim); }
div[data-testid="stTickBar"] { display: none; }

/* ---- buttons ---- */
.stButton button {
    font-family: var(--mono); font-weight: 500; letter-spacing: 0.04em;
    background: var(--cyan); color: #06110d; border: none; border-radius: 4px;
    padding: 0.6rem 1.2rem; transition: filter 0.15s ease;
}
.stButton button:hover { filter: brightness(1.1); }
.stButton button:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }

/* ---- sidebar / manifest ---- */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--panel-border); }
.manifest-title {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; color: var(--text-dim);
    display: flex; justify-content: space-between; margin-bottom: 0.8rem;
}
.manifest-title span.id { color: var(--cyan); }
.manifest-item {
    font-family: var(--mono); font-size: 0.82rem; color: var(--text);
    display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0;
    border-bottom: 1px solid var(--panel-border);
}
.manifest-item .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); flex-shrink: 0; }
.manifest-item .id { color: var(--text-dim); margin-left: auto; font-size: 0.72rem; }
.manifest-item.lead .dot { background: var(--amber); }
.manifest-link a { font-family: var(--mono); font-size: 0.75rem; color: var(--text-dim) !important; }

/* ---- risk readout ---- */
.readout { display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; margin-top: 1rem; }
.risk-ring {
    width: 180px; height: 180px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; position: relative;
}
.risk-ring::before {
    content: ''; position: absolute; inset: 12px; border-radius: 50%; background: var(--panel);
}
.risk-ring-value { position: relative; z-index: 2; text-align: center; font-family: var(--mono); }
.risk-ring-value .pct { font-size: 2.1rem; font-weight: 600; }
.risk-ring-value .lbl { font-size: 0.65rem; color: var(--text-dim); letter-spacing: 0.08em; margin-top: 2px; }

.log-line {
    font-family: var(--mono); font-size: 0.9rem; line-height: 1.6; flex: 1; min-width: 240px;
    border-left: 2px solid var(--line-color, var(--cyan)); padding-left: 0.9rem;
}
.log-line .tag {
    font-size: 0.7rem; letter-spacing: 0.08em; color: var(--line-color, var(--cyan));
    display: block; margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR — CREW MANIFEST
# ==============================================================================

with st.sidebar:
    st.markdown(f"""
    <div class="manifest-title">CREW MANIFEST <span class="id">GROUP {GROUP_NUMBER:02d}</span></div>
    """, unsafe_allow_html=True)

    for name, member_id in st.session_state.member_order:
        lead_class = "lead" if member_id == PRIORITY_ID else ""
        st.markdown(f"""
        <div class="manifest-item {lead_class}">
            <span class="dot"></span>{name}<span class="id">{member_id}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="manifest-link">$ git clone <a href="https://github.com/aadhi5h/dropout-prediction">dropout-prediction</a></div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# MODEL
# ==============================================================================

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model_xgb.json")
    return model

model = load_model()

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
<div class="console-header">
    <div class="kicker"><span class="dot"></span>SYSTEM ONLINE</div>
    <h1>Dropout Risk Console<span class="cursor"></span></h1>
</div>
<div class="stat-row">
    <div class="stat-chip">DATASET <b>KDD Cup 2015</b></div>
    <div class="stat-chip">ENROLLMENTS <b>120,542</b></div>
    <div class="stat-chip">MODEL <b>XGBoost</b></div>
    <div class="stat-chip">AUC <b>0.879</b></div>
</div>
""", unsafe_allow_html=True)

st.markdown("Enter a student's engagement stats from the current course to get a live dropout-risk read.")

# ==============================================================================
# INPUT TELEMETRY
# ==============================================================================

st.markdown('<div class="section-label">INPUT TELEMETRY — ACTIVITY</div>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        active_days = st.slider("Active days in course", 0, 30, 5)
        total_events = st.slider("Total events logged", 0, 500, 50)
        server_events = st.slider("Server-logged actions", 0, 300, 25)
    with col2:
        browser_events = st.slider("Browser-logged actions", 0, 300, 25)
        distinct_objects_touched = st.slider("Distinct course objects touched", 0, 100, 10)
        session_span_days = st.slider("Days between first and last activity", 0, 90, 5)

st.markdown('<div class="section-label">INPUT TELEMETRY — RECENCY & CONTENT</div>', unsafe_allow_html=True)
with st.container(border=True):
    col3, col4 = st.columns(2)
    with col3:
        days_before_course_end = st.slider("Days before course end at last activity", 0, 60, 10)
        access_count = st.slider("Access events", 0, 300, 20)
    with col4:
        problem_count = st.slider("Problem attempts", 0, 200, 10)
        video_count = st.slider("Video views", 0, 200, 10)

    with st.expander("More activity detail (optional)"):
        page_close_count = st.slider("Page close events", 0, 200, 10)
        navigate_count = st.slider("Navigation events", 0, 200, 10)
        discussion_count = st.slider("Discussion posts", 0, 100, 0)
        wiki_count = st.slider("Wiki views", 0, 50, 0)

events_per_active_day = round(total_events / max(active_days, 1), 2)

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("Run risk analysis", type="primary", use_container_width=False)

# ==============================================================================
# RISK READOUT
# ==============================================================================

if predict:
    row = pd.DataFrame([{
        "total_events": total_events,
        "active_days": active_days,
        "distinct_objects_touched": distinct_objects_touched,
        "access_count": access_count,
        "problem_count": problem_count,
        "page_close_count": page_close_count,
        "navigate_count": navigate_count,
        "video_count": video_count,
        "discussion_count": discussion_count,
        "wiki_count": wiki_count,
        "server_events": server_events,
        "browser_events": browser_events,
        "days_before_course_end": days_before_course_end,
        "session_span_days": session_span_days,
        "events_per_active_day": events_per_active_day,
    }])

    proba = model.predict_proba(row)[0][1]
    pct = round(proba * 100, 1)

    if pct >= 60:
        color, band, action = "var(--coral)", "HIGH RISK", (
            "Recommend outreach this week — a personal check-in or reminder now has "
            "the best chance of re-engaging this student."
        )
    elif pct >= 30:
        color, band, action = "var(--amber)", "MEDIUM RISK", (
            "Worth monitoring — an automated nudge (reminder email, suggested next "
            "module) is usually enough at this stage."
        )
    else:
        color, band, action = "var(--cyan)", "LOW RISK", (
            "Activity pattern matches students who complete the course — no "
            "intervention needed."
        )

    st.markdown('<div class="section-label">RISK READOUT</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"""
        <div class="readout">
            <div class="risk-ring" style="background: conic-gradient({color} {pct * 3.6}deg, #1b2330 0deg);">
                <div class="risk-ring-value">
                    <div class="pct" style="color:{color}">{pct}%</div>
                    <div class="lbl">DROPOUT PROBABILITY</div>
                </div>
            </div>
            <div class="log-line" style="--line-color: {color};">
                <span class="tag">STATUS: {band}</span>
                {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div class="stat-chip" style="display:inline-block;">TOP FEATURES '
    '<b>active_days 63%</b> · <b>server_events 13%</b> · <b>days_before_course_end 10%</b></div>',
    unsafe_allow_html=True,
)