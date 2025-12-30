import os
import json
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import subprocess
import sys
import re

# ------------------ SESSION STATE ------------------
st.session_state.setdefault("open_camera", False)

# ------------------ PATHS ------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
model_path = os.path.join(model_dir, "corn-or-maize-leaf-disease-dataset.h5")
link_file = os.path.join(model_dir, "trained_model_link.txt")

# ------------------ HELPERS ------------------
def _is_hdf5_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False

def try_download_model(link, dest_path):
    if not link or not link.startswith("http"):
        return False, "Invalid model download link."

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        import gdown
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    gdown.download(link, dest_path, quiet=False)

    if _is_hdf5_file(dest_path):
        return True, "Model downloaded successfully."
    else:
        return False, "Downloaded file is not a valid HDF5 model."

# ------------------ READ MODEL LINK ------------------
download_link = None
if os.path.exists(link_file):
    with open(link_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        m = re.search(r'(https?://\S+)', content)
        if m:
            download_link = m.group(1).rstrip(').,')

# ------------------ FIX CORRUPTED FILE ------------------
if os.path.exists(model_path) and not _is_hdf5_file(model_path):
    os.remove(model_path)

# ------------------ LOAD MODEL (ROBUST + MOBILE SAFE) ------------------
model = None
model_load_error = None

# Auto-download if missing
if not os.path.exists(model_path) and download_link:
    with st.spinner("Downloading AI model (first time only)…"):
        success, msg = try_download_model(download_link, model_path)
        if not success:
            st.error(msg)

# Retry loading (mobile networks)
if os.path.exists(model_path):
    for _ in range(3):
        try:
            model = tf.keras.models.load_model(model_path)
            break
        except Exception as e:
            model_load_error = str(e)
            time.sleep(2)

# ------------------ LOAD CLASS INDICES ------------------
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# ------------------ IMAGE PREPROCESS ------------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------ PREDICTION ------------------
def predict_with_confidence(model, image_source, class_indices, top_k=3):
    img = load_and_preprocess_image(image_source)
    preds = model.predict(img)[0]
    top_idx = preds.argsort()[-top_k:][::-1]
    return [(class_indices[str(i)], float(preds[i])) for i in top_idx]

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="Corn Disease Classifier", page_icon="🌽", layout="centered")

# ------------------ STYLES ------------------
st.markdown("""
<style>
.stApp { background-color:#008000; }
.app-header, .quote-card, .uploader-card {
    background-color:#008000;
    padding:16px;
    border-radius:14px;
}
button {
    background-color:#3C6301 !important;
    color:white !important;
    border-radius:10px !important;
    font-weight:700 !important;
}
h2,p { color:#ffffff; }
.footer { text-align:center; font-size:12px; color:#ffffff; }
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="app-header">
  <div style="font-size:40px">🌽</div>
  <div>
    <h2>Corn Disease Classifier</h2>
    <p>Detect corn leaf diseases using AI</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ QUOTES ------------------
QUOTES = [
    "“Agriculture is our wisest pursuit.” — Thomas Jefferson",
    "“To forget how to dig the earth is to forget ourselves.” — Mahatma Gandhi",
    "“The nation that destroys its soil destroys itself.” — Franklin D. Roosevelt",
    "“Farming looks easy when your plow is a pencil.” — Dwight D. Eisenhower",
]

if "quote_idx" not in st.session_state:
    st.session_state.quote_idx = 0

left, right = st.columns([3, 1])

with right:
    st.markdown('<div class="quote-card">', unsafe_allow_html=True)
    st.markdown(QUOTES[st.session_state.quote_idx])
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Next Quote"):
        st.session_state.quote_idx = (st.session_state.quote_idx + 1) % len(QUOTES)
        st.experimental_rerun()

# ------------------ INPUT ------------------
with left:
    st.markdown('<div class="uploader-card">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload a corn leaf image", ["jpg", "jpeg", "png"])

    if st.button("📷 Open Camera"):
        st.session_state.open_camera = True

    if st.session_state.open_camera:
        if st.button("❌ Close Camera"):
            st.session_state.open_camera = False

    camera_image = None
    if st.session_state.open_camera:
        camera_image = st.camera_input("Capture image")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ IMAGE SOURCE ------------------
image_source = uploaded_image if uploaded_image else camera_image

# ------------------ DISPLAY & PREDICT ------------------
if image_source is not None:
    image = Image.open(image_source)
    c1, c2 = st.columns(2)

    with c1:
        st.image(image.resize((180, 180)))
        st.markdown("<p style='color:#fff;text-align:center'>Input Image</p>", unsafe_allow_html=True)

    with c2:
        if model is None:
            st.error("Model not loaded yet. Please wait a few seconds.")
            if model_load_error:
                st.warning(model_load_error)
        else:
            if st.button("Predict"):
                results = predict_with_confidence(model, image_source, class_indices)
                top_label, top_conf = results[0]
                st.success(f"{top_label} — {int(top_conf*100)}%")
                for label, prob in results:
                    st.progress(prob, text=f"{label} ({int(prob*100)}%)")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<div class='footer'>Works on desktop & mobile 🌱</div>", unsafe_allow_html=True)
