# MakerFest Career Suggestor (MVP)

Rules-based Streamlit app that mirrors Notebook 01 (no camera/overlays yet).

## Run locally
```bash
conda activate makerfest
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push these files to a public GitHub repo.
2. Go to share.streamlit.io, connect your repo, and choose `app.py` as the entry point.
3. Add a `secrets.toml` only if you need secrets (not required for this MVP).
