import streamlit as st
import career_app, translator_app

st.set_page_config(page_title="MakerFest Station", page_icon="🎉", layout="centered")
st.title("🎉 MakerFest Station")

apps = {
    "Career Suggestor": career_app,
    "Translator": translator_app,
}

choice = st.sidebar.radio("Choose a station:", list(apps.keys()))
apps[choice].run()
