import streamlit as st
from googletrans import Translator, LANGUAGES
from gtts import gTTS
from io import BytesIO

def run():
    st.subheader("🌐 Translator with Text-to-Speech")

    text = st.text_input("Enter input:", "")

    # Language setup
    name_to_code = {name.title(): code for code, name in LANGUAGES.items()}
    sorted_language_names = sorted(name_to_code.keys())
    classification_space = st.sidebar.selectbox("Language to be translated into:", sorted_language_names)
    option = name_to_code[classification_space]

    if st.button("Translate"):
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

            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("⬇️ Download MP3", data=audio_bytes, file_name="translation.mp3", mime="audio/mpeg")

        except Exception as e:
            st.error(f"Translation failed: {e}")
