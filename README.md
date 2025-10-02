# MakerFest Station 🎉

Streamlit app with multiple mini-apps:
- **Career Suggestor** (rules-based)
- **Translator + Text-to-Speech** (googletrans + gTTS)

## Repo Structure
```
app.py               # Main launcher with sidebar menu
career_app.py        # Career Suggestor app
translator_app.py    # Translator with TTS app (lazy imports)
requirements.txt     # Python packages
runtime.txt          # Pin Python to 3.11 for Streamlit Cloud
```

## Why this layout?
- Each app is isolated in its own file.
- `translator_app.py` uses *lazy imports* so import errors don't crash the entire app.
- `runtime.txt` pins Python to **3.11** which avoids `ModuleNotFoundError` / `cgi` removal issues on Python 3.13 with some transitive deps.

## Local Run
```bash
conda create -n makerfest python=3.11 -y
conda activate makerfest
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. Ensure **runtime.txt** contains `3.11` so the cloud uses Python 3.11.
3. Deploy, selecting `app.py` as the entrypoint.

## Notes
- The Translator requires internet access for both translation and TTS.
- If you still see dependency import errors, try clearing Streamlit Cloud build cache and redeploy.
