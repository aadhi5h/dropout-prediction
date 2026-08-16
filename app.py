import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

st.set_page_config(page_title="Student Dropout Risk Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("model_xgb.json")
    return model

model = load_model()

st.title("Student Dropout Risk Predictor")
st.caption("Trained on the KDD Cup 2015 MOOC dataset (120,542 enrollments, AUC 0.879)")

st.markdown("Enter a student's engagement stats from the current course, and the model estimates their dropout risk.")

col1, col2 = st.columns(2)

with col1:
    active_days = st.slider("Active days in course", 0, 30, 5)
    total_events = st.slider("Total events logged", 0, 500, 50)
    server_events = st.slider("Server-logged actions", 0, 300, 25)
    browser_events = st.slider("Browser-logged actions", 0, 300, 25)
    distinct_objects_touched = st.slider("Distinct course objects touched", 0, 100, 10)

with col2:
    days_before_course_end = st.slider("Days before course end at last activity", 0, 60, 10)
    session_span_days = st.slider("Days between first and last activity", 0, 90, 5)
    access_count = st.slider("Access events", 0, 300, 20)
    problem_count = st.slider("Problem attempts", 0, 200, 10)
    video_count = st.slider("Video views", 0, 200, 10)

with st.expander("More activity detail (optional)"):
    page_close_count = st.slider("Page close events", 0, 200, 10)
    navigate_count = st.slider("Navigation events", 0, 200, 10)
    discussion_count = st.slider("Discussion posts", 0, 100, 0)
    wiki_count = st.slider("Wiki views", 0, 50, 0)

events_per_active_day = round(total_events / max(active_days, 1), 2)

if st.button("Predict dropout risk", type="primary"):
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

    st.metric("Predicted dropout probability", f"{pct}%")

    if pct >= 60:
        st.error("High risk. Recommend outreach this week - a personal check-in or reminder now has the best chance of re-engaging this student.")
    elif pct >= 30:
        st.warning("Medium risk. Worth monitoring - an automated nudge (reminder email, suggested next module) is usually enough at this stage.")
    else:
        st.success("Low risk. Activity pattern matches students who complete the course - no intervention needed.")

st.divider()
st.caption("Model: XGBoost classifier. Top features: active_days (63% importance), server_events (13%), days_before_course_end (10%).")