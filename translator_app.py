import streamlit as st
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import os
from io import BytesIO
import base64

# App title and instructions
st.title("Translator App with Text-to-Speech")
st.write("You can translate into many languages using Google Translate.")
st.write("Enter text and choose your target language:")

# Input text
text = st.text_input("Enter input:", "")

# Language setup
name_to_code = {name.title(): code for code, name in LANGUAGES.items()}
sorted_language_names = sorted(name_to_code.keys())
classification_space = st.sidebar.selectbox("Language to be translated into:", sorted_language_names)
option = name_to_code[classification_space]

# Translation
if st.button('Translate'):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=option)
        st.success("Translated Text:")
        st.write(translated.text)

        # Text-to-Speech
        tts = gTTS(text=translated.text, lang=option)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

        # Play button
        st.audio(audio_bytes, format="audio/mp3")

    except Exception as e:
        st.error(f"Translation failed: {e}")
