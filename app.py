import streamlit as st
import pandas as pd
import random
from xgboost import XGBClassifier

st.set_page_config(page_title="Dropout Risk Console", layout="wide", initial_sidebar_state="expanded")

GROUP_NUMBER = 8
PRIORITY_MEMBER = "P S Aadhish (VML24AD092)"
OTHER_MEMBERS = [
    "Prajwal Pratheev (VML24AD091)",
    "Nived Sunil (VML24AD086)",
    "Navalrag V P (VML24AD084)",
    "Sanha Sadik (VML24AD099)",
    "Muzeen K (VML24AD083)",
]

if "member_order" not in st.session_state:
    shuffled = OTHER_MEMBERS.copy()
    random.shuffle(shuffled)
    st.session_state.member_order = [PRIORITY_MEMBER] + shuffled

# ----------------------------------------------------------------------------
# CUSTOM PALETTE
# A deep ink-teal and aged brass pairing on a cool bone-white ground.
# Chosen to read as an instrument panel, not a generic SaaS dashboard.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink:        #142B2E;
    --ink-soft:   #3E5457;
    --ground:     #F4F6F5;
    --panel:      #FFFFFF;
    --line:       #DDE3E1;
    --teal:       #0E5C56;
    --teal-deep:  #0A3F3B;
    --brass:      #A6763A;
    --rust:       #A3402F;
    --ochre:      #B4842A;
    --sage:       #4C7A5E;
}

html, body {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--ground);
    color: var(--ink);
}

h1, h2, h3, .serif {
    font-family: 'Source Serif 4', serif;
}

#MainMenu, footer {visibility: hidden;}

header[data-testid="stHeader"] {
    background: transparent;
    height: 2.2rem;
}

.block-container {
    padding-top: 1.2rem !important;
}

.console-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 2px solid var(--teal-deep);
    padding-bottom: 0.6rem;
    margin-bottom: 1.4rem;
}
.console-header h1 {
    font-size: 2.6rem;
    margin: 0;
    font-weight: 600;
}
.console-header span {
    color: var(--ink-soft);
    font-size: 0.85rem;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
}

.panel-label {
    font-size: 0.78rem;
    color: var(--ink-soft);
    letter-spacing: 0.02em;
    margin-bottom: 0.6rem;
}

.risk-number {
    font-family: 'Source Serif 4', serif;
    font-size: 3.4rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.2rem 0 0.3rem 0;
}

.risk-bar-track {
    width: 100%;
    height: 10px;
    background: var(--line);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.9rem;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 6px;
}

.band-tag {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 3px;
    margin-bottom: 0.7rem;
}

.action-text {
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--ink);
}

div.stButton > button {
    background: var(--teal-deep);
    border: none;
    border-radius: 3px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
}
div.stButton > button, div.stButton > button * {
    color: #FFFFFF !important;
}
div.stButton > button:hover {
    background: var(--teal);
}

section[data-testid="stSidebar"] {
    background: var(--teal-deep);
}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
    color: #F4F6F5 !important;
}
section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] a * {
    color: #D8C79A !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model_xgb.json")
    return model


model = load_model()

with st.sidebar:
    st.markdown("### Group 8")
    for name in st.session_state.member_order:
        st.write(name)
    st.divider()
    st.markdown("[View source on GitHub](https://github.com/aadhi5h/dropout-prediction)")

st.markdown("""
<div class="console-header">
    <h1>Dropout Risk Console</h1>
    <span>XGBoost model &middot; trained on KDD Cup 2015 &middot; AUC 0.879</span>
</div>
""", unsafe_allow_html=True)

col_input, col_result = st.columns([1.15, 1], gap="large")

with col_input:
    st.markdown('<div class="panel-label">STUDENT ENGAGEMENT INPUT</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        active_days = st.slider("Active days in course", 0, 30, 5)
        total_events = st.slider("Total events logged", 0, 500, 50)
        server_events = st.slider("Server-logged actions", 0, 300, 25)
        browser_events = st.slider("Browser-logged actions", 0, 300, 25)
        distinct_objects_touched = st.slider("Distinct objects touched", 0, 100, 10)
    with s2:
        days_before_course_end = st.slider("Days before course end at last activity", 0, 60, 10)
        session_span_days = st.slider("Days between first and last activity", 0, 90, 5)
        access_count = st.slider("Access events", 0, 300, 20)
        problem_count = st.slider("Problem attempts", 0, 200, 10)
        video_count = st.slider("Video views", 0, 200, 10)

    with st.expander("More activity detail"):
        page_close_count = st.slider("Page close events", 0, 200, 10)
        navigate_count = st.slider("Navigation events", 0, 200, 10)
        discussion_count = st.slider("Discussion posts", 0, 100, 0)
        wiki_count = st.slider("Wiki views", 0, 50, 0)

events_per_active_day = round(total_events / max(active_days, 1), 2)

with col_result:
    st.markdown('<div class="panel-label">PREDICTED RISK</div>', unsafe_allow_html=True)
    predict_clicked = st.button("Predict dropout risk")

    if predict_clicked:
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
        pct = float(f"{proba * 100:.2f}")

        if pct >= 60:
            color, band, action = "var(--rust)", "HIGH RISK", \
                "Flag for outreach this week. Low active-day count is the strongest single dropout signal in this dataset -- a personal check-in now gives the best chance of re-engagement."
        elif pct >= 30:
            color, band, action = "var(--ochre)", "MEDIUM RISK", \
                "Worth monitoring. Engagement is inconsistent -- an automated nudge or a suggested next module is usually enough at this stage."
        else:
            color, band, action = "var(--sage)", "LOW RISK", \
                "On track. This activity pattern matches students who complete the course -- no intervention needed."

        st.markdown(f"""
        <div class="panel">
            <div class="risk-number" style="color:{color} !important;">{pct:.2f}%</div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <div class="band-tag" style="background:{color}22; color:{color} !important;">{band}</div>
            <div class="action-text">{action}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel">
            <div class="action-text" style="color: var(--ink-soft);">
                Set the sliders on the left to describe a student's engagement, then press
                <b>Predict dropout risk</b> to see the model's live assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div style="margin-top:1.2rem; font-size:0.78rem; color:var(--ink-soft);">'
    'Top features: active_days (63% importance) &middot; server_events (13%) &middot; days_before_course_end (10%)'
    '</div>',
    unsafe_allow_html=True
)