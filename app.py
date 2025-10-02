import streamlit as st
from typing import List, Dict, Tuple

st.set_page_config(page_title="MakerFest Career Suggestor", page_icon="🎓", layout="centered")

st.title("🎓 MakerFest — Career Suggestor")
st.caption("Answer a few quick questions to get a fun career suggestion. (Rules-based MVP)")

CAREERS = ["Engineer", "Doctor", "Artist"]

def predict_rules(interest: str, style: str, enjoy: List[str], values: str):
    scores = {c: 0 for c in CAREERS}

    # Interest
    if interest in ["Math/Physics", "Computers/Data"]:
        scores["Engineer"] += 2
    if interest == "Biology/Health":
        scores["Doctor"] += 2
    if interest == "Art/Design":
        scores["Artist"] += 2

    # Style
    if style in ["Analytical", "Hands-on"]:
        scores["Engineer"] += 1
    if style == "Helping People":
        scores["Doctor"] += 1
    if style == "Creative":
        scores["Artist"] += 1

    # Enjoy
    for e in enjoy:
        if e in ["Building things", "Fixing machines", "Solving puzzles"]:
            scores["Engineer"] += 1
        if e in ["Experiments", "Talking to people"]:
            scores["Doctor"] += 1
        if e == "Drawing/creating":
            scores["Artist"] += 1

    # Values
    if values == "Innovation":
        scores["Engineer"] += 1
    if values == "Helping others":
        scores["Doctor"] += 1
    if values == "Creativity":
        scores["Artist"] += 1
    if values == "Precision/Accuracy":
        scores["Engineer"] += 1
        scores["Doctor"] += 1

    best = max(scores, key=scores.get)
    return best, scores

with st.form("career_form"):
    col1, col2 = st.columns(2)
    with col1:
        interest = st.selectbox("Favorite subject area", ["Math/Physics", "Biology/Health", "Art/Design", "Computers/Data", "Languages/History"], index=3)
        style = st.selectbox("How do you like to work?", ["Hands-on", "Creative", "Analytical", "Helping People"], index=2)
    with col2:
        enjoy = st.multiselect("Pick up to 3 things you enjoy", ["Building things", "Solving puzzles", "Drawing/creating", "Talking to people", "Experiments", "Fixing machines"], default=["Solving puzzles"])
        values = st.selectbox("What matters most to you?", ["Innovation", "Helping others", "Creativity", "Precision/Accuracy"], index=0)

    submitted = st.form_submit_button("🎯 Get my suggestion")

if submitted:
    # trim enjoy to 3 max
    enjoy = enjoy[:3]
    career, scores = predict_rules(interest, style, enjoy, values)
    st.success(f"Suggested career: **{career}**")
    st.caption(f"Scores → Engineer: {scores['Engineer']} • Doctor: {scores['Doctor']} • Artist: {scores['Artist']}")
else:
    st.info("Fill the form and press **Get my suggestion**")

st.divider()
st.caption("MVP uses transparent rules. Swap in a trained model later if desired.")
