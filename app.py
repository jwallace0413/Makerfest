import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt

st.set_page_config(page_title="TF2 SavedModel Webcam Classifier", page_icon="📷", layout="centered")
st.title("📷 TF2 SavedModel: Image Classifier")
st.caption("Capture from camera → run TF2 SavedModel → show predictions")

# ---------------- Sidebar: Model, Labels & Settings ----------------
st.sidebar.header("Model & Labels")

# Where your SavedModel lives (folder containing saved_model.pb & variables/)
default_model_dir = "models/my_saved_model"
model_dir = st.sidebar.text_input("SavedModel directory", value=default_model_dir)

labels_path = st.sidebar.text_input("Labels file (one per line)", value="models/assets/labels.txt")

st.sidebar.subheader("Signature & Tensor Keys")
sig_name = st.sidebar.text_input("Signature name", value="serving_default")
# If your signature expects a named input (common), set it here.
# If left blank, the app will try to auto-detect the first input tensor.
input_key_override = st.sidebar.text_input("Input tensor key (optional)", value="")

st.sidebar.subheader("Preprocessing")
target_w = st.sidebar.number_input("Target width", value=224, step=1)
target_h = st.sidebar.number_input("Target height", value=224, step=1)
scale_01 = st.sidebar.checkbox("Scale to [0,1]", value=True)
mean_val = st.sidebar.number_input("Subtract mean", value=0.0)
std_val = st.sidebar.number_input("Divide by std (if >0)", value=0.0)
bgr = st.sidebar.checkbox("Swap RGB→BGR", value=False)
center_crop = st.sidebar.checkbox("Center-crop to square before resize", value=True)

st.sidebar.subheader("Display")
top_k = st.sidebar.slider("Top-K predictions", 1, 10, 3)
show_probs_chart = st.sidebar.checkbox("Show probability bar chart", value=True)

# ---------------- Utilities ----------------
@st.cache_data(show_spinner=False)
def load_labels(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        return labels
    except Exception as e:
        st.warning(f"Could not load labels: {e}")
        return None

@st.cache_resource(show_spinner=True)
def load_saved_model(model_dir: str, signature_name: str):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"SavedModel directory not found: {model_dir}")
    loaded = tf.saved_model.load(model_dir)
    if signature_name not in loaded.signatures:
        available = list(loaded.signatures.keys())
        raise RuntimeError(
            f"Signature '{signature_name}' not found. Available signatures: {available}"
        )
    infer = loaded.signatures[signature_name]
    return loaded, infer

def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))

def preprocess_image(
    pil_img: Image.Image,
    want_uint8: bool,
    target_hw=(224, 224),
    bgr=False,
    scale_01=True,
    mean_val=0.0,
    std_val=0.0,
    do_center_crop=True,
) -> np.ndarray:
    img = pil_img.convert("RGB")
    if do_center_crop:
        img = center_crop_to_square(img)
    img = img.resize(target_hw, Image.Resampling.LANCZOS)
    arr = np.asarray(img)

    if bgr:
        arr = arr[..., ::-1]  # RGB->BGR

    if want_uint8:
        # Model expects uint8 in [0..255]; don't normalize
        return np.expand_dims(arr.astype(np.uint8), axis=0)

    # Model expects float inputs
    x = arr.astype(np.float32)
    if scale_01:
        x = x / 255.0
    if mean_val != 0.0:
        x = x - mean_val
    if std_val and std_val > 0:
        x = x / std_val
    return np.expand_dims(x, axis=0)

def run_inference_and_debug(infer_fn, batch: np.ndarray, input_key_override: str = ""):
    input_sig = infer_fn.structured_input_signature[1]
    input_keys = list(input_sig.keys())
    key = input_key_override.strip() if input_key_override.strip() else input_keys[0]

    # Detect expected dtype from signature
    expected_dtype = list(input_sig.values())[0].dtype  # TF dtype

    # Enforce dtype to reduce silent coercions
    if expected_dtype.name.startswith("uint8"):
        if batch.dtype != np.uint8:
            st.warning("Model expects uint8 input; converting batch to uint8.")
            batch = batch.astype(np.uint8)
    else:
        if batch.dtype != np.float32:
            st.warning("Model expects float input; converting batch to float32.")
            batch = batch.astype(np.float32)

    out_dict = infer_fn(**{key: tf.convert_to_tensor(batch)})
    outputs = list(out_dict.values())
    if not outputs:
        raise RuntimeError("Model returned no outputs.")
    y = outputs[0].numpy()
    return y, expected_dtype, key, list(out_dict.keys())

# ---------------- Main UI: Camera / Upload ----------------
st.subheader("Step 1 — Take or upload a picture")
img_data = st.camera_input("Use your webcam")
tab1, tab2 = st.tabs(["📸 Classify", "⬆️ Upload Image"])
uploaded_img = tab2.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

pil_img = None
if img_data is not None:
    pil_img = Image.open(img_data)
elif uploaded_img is not None:
    pil_img = Image.open(uploaded_img)

# ---------------- Load model & labels ----------------
labels = load_labels(labels_path) if labels_path else None

loaded = infer = None
load_error = None
if model_dir:
    try:
        loaded, infer = load_saved_model(model_dir, sig_name)
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(load_error)

# ---------------- Inference ----------------
if pil_img:
    st.image(pil_img, caption="Captured image", use_container_width=True)

    if infer is not None and loaded is not None:
        with st.spinner("Running inference..."):
            try:
                # Detect desired input dtype from signature for preprocessing
                expected_dtype = list(infer.structured_input_signature[1].values())[0].dtype
                want_uint8 = expected_dtype.name.startswith("uint8")

                batch = preprocess_image(
                    pil_img,
                    want_uint8=want_uint8,
                    target_hw=(int(target_w), int(target_h)),
                    bgr=bgr,
                    scale_01=scale_01,
                    mean_val=mean_val,
                    std_val=std_val,
                    do_center_crop=center_crop,
                )

                # --- Debug: show input stats ---
                with st.expander("🔍 Debug: Preprocessed batch stats"):
                    x = batch.astype(np.float32)
                    st.write({
                        "shape": batch.shape,
                        "dtype": str(batch.dtype),
                        "min": float(x.min()),
                        "max": float(x.max()),
                        "mean": float(x.mean()),
                        "std": float(x.std()),
                    })

                logits_or_probs, exp_dtype, used_key, out_keys = run_inference_and_debug(
                    infer, batch, input_key_override
                )

                # Ensure 1D vector of class *probabilities* (we trust the model)
                probs = logits_or_probs
                if probs.ndim > 1:
                    probs = probs.reshape(-1)

                # Safety: if not valid probs, normalize by sum (no softmax)
                probs = probs.astype(np.float32)
                if not np.all(np.isfinite(probs)) or probs.min() < 0 or probs.max() > 1.0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
                    s = probs.sum()
                    if s == 0 or not np.isfinite(s):
                        st.warning("Model output didn't look like probabilities; applying safe normalization.")
                        s = np.sum(np.clip(probs, 0, None)) + 1e-8
                    probs = np.clip(probs, 0, None) / s

                # Align to labels if present
                if labels is not None:
                    num = min(len(labels), probs.shape[0])
                    probs = probs[:num]
                    use_labels = labels[:num]
                else:
                    use_labels = [f"class_{i}" for i in range(probs.shape[0])]

                # --- Show raw diagnostics ---
                with st.expander("🔎 Debug: Raw output & signature"):
                    st.write("Input key used:", used_key)
                    st.write("Model expects dtype:", str(exp_dtype))
                    st.write("Output keys:", out_keys)
                    st.write("First 10 raw values:", logits_or_probs.reshape(-1)[:10])
                    st.write("Sum of probs (after normalization):", float(probs.sum()))

                # ---- Results: list top-k + horizontal full chart ----
                st.subheader("Results")

                # Quick Top-K list
                idxs = np.argsort(-probs)[:top_k]
                for r, i in enumerate(idxs, start=1):
                    st.write(f"**{r}. {use_labels[i]}** — {probs[i]*100:.2f}%")

                # Full table & horizontal bar chart for ALL classes
                df = pd.DataFrame({
                    "Label": use_labels,
                    "Confidence (%)": (probs * 100).round(2),
                }).sort_values("Confidence (%)", ascending=True)

                if show_probs_chart:
                    height = max(300, int(18 * len(df)))
                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Confidence (%):Q", title="Confidence (%)"),
                            y=alt.Y("Label:N", sort=None, title="Class"),
                            tooltip=[
                                alt.Tooltip("Label:N", title="Label"),
                                alt.Tooltip("Confidence (%):Q", title="Confidence (%)", format=".2f"),
                            ],
                        )
                        .properties(height=height)
                    )
                    text = chart.mark_text(align="left", dx=3).encode(
                        text=alt.Text("Confidence (%):Q", format=".2f")
                    )
                    st.altair_chart(chart + text, use_container_width=True)

                with st.expander("See all scores as a table"):
                    st.dataframe(df[::-1], use_container_width=True)  # show high→low

            except Exception as e:
                st.error(f"Inference failed: {e}")
    else:
        st.info("Point the sidebar to your SavedModel directory and labels file to run classification.")
else:
    st.warning("Take a picture or upload an image to begin.")

# ---------------- Footer help ----------------
st.caption(
    "This app trusts the model's probabilities and never applies softmax. "
    "If outputs aren't valid probabilities, it only normalizes by sum as a safety net."
)
