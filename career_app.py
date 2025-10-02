import streamlit as st

CAREERS = ["Engineer", "Doctor", "Artist"]

def predict_rules(interest, style, enjoy, values):
    scores = {c: 0 for c in CAREERS}
    if interest in ["Math/Physics", "Computers/Data"]:
        scores["Engineer"] += 2
    if interest == "Biology/Health":
        scores["Doctor"] += 2
    if interest == "Art/Design":
        scores["Artist"] += 2

    if style in ["Analytical", "Hands-on"]:
        scores["Engineer"] += 1
    if style == "Helping People":
        scores["Doctor"] += 1
    if style == "Creative":
        scores["Artist"] += 1

    for e in enjoy:
        if e in ["Building things", "Fixing machines", "Solving puzzles"]:
            scores["Engineer"] += 1
        if e in ["Experiments", "Talking to people"]:
            scores["Doctor"] += 1
        if e == "Drawing/creating":
            scores["Artist"] += 1

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

def run():
    st.subheader("🎓 Career Suggestor")
    with st.form("career_form"):
        col1, col2 = st.columns(2)
        with col1:
            interest = st.selectbox("Favorite subject area",
                ["Math/Physics", "Biology/Health", "Art/Design", "Computers/Data", "Languages/History"])
            style = st.selectbox("How do you like to work?",
                ["Hands-on", "Creative", "Analytical", "Helping People"])
        with col2:
            enjoy = st.multiselect("Pick up to 3 things you enjoy",
                ["Building things", "Solving puzzles", "Drawing/creating", "Talking to people", "Experiments", "Fixing machines"])
            values = st.selectbox("What matters most to you?",
                ["Innovation", "Helping others", "Creativity", "Precision/Accuracy"])
        submitted = st.form_submit_button("🎯 Get my suggestion")

    if submitted:
        enjoy = enjoy[:3]
        career, scores = predict_rules(interest, style, enjoy, values)
        st.success(f"Suggested career: **{career}**")
        st.caption(f"Scores → Engineer: {scores['Engineer']} • Doctor: {scores['Doctor']} • Artist: {scores['Artist']}")
