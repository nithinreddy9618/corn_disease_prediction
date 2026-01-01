import os
import json
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import subprocess
import sys
import 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(
    page_title="Corn Disease Classifier",
    page_icon="🌽",
    layout="centered"
)

# ------------------ BACKGROUND IMAGE (SINGLE CORN LEAF) ------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: url("https://w0.peakpx.com/wallpaper/876/1014/HD-wallpaper-mint-green-green-leaves-zoran-blooming-bush-closeup-dark-dark-green-deep-deep-green-forrest-glow-green-leafed-plant-hedgerow-in-blossoms-landscapes-light-lovely-landscape-macro-plant.jpg");
    background-size: cover;
    background-position: center;
}

/* Center main container */
.block-container {
    max-width: 850px;
    padding-top: 40px;
}

/* Glass card */
.main-card {
    background:  rgba(15, 77, 15, 0.35);
;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* Buttons */
button {
    background-color: #2ecc71 !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Headings & text */
h1, h2, h3, p, label {
    color: #ffffff !important;
}

/* Progress bar */
.stProgress > div > div {
    background-color: rgba(15, 77, 15, 0.35);
}
</style>
""", unsafe_allow_html=True)


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
        m = re.search(r'(https?://\S+)', f.read())
        if m:
            download_link = m.group(1)

# ------------------ FIX CORRUPTED FILE ------------------
if os.path.exists(model_path) and not _is_hdf5_file(model_path):
    os.remove(model_path)

# ------------------ LOAD MODEL ------------------
model = None
model_load_error = None

if not os.path.exists(model_path) and download_link:
    with st.spinner("Downloading AI model (first time only)…"):
        success, msg = try_download_model(download_link, model_path)
        if not success:
            st.error(msg)

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

# ------------------ FERTILIZER SUGGESTIONS ------------------
FERTILIZER_SUGGESTIONS = {
        "Gray_leaf_spot": {
        "fertilizer": "Zinc Sulphate 33%",
        "advice": "Apply zinc sulphate to correct micronutrient deficiency.",
        "buy_link": "https://www.kisanshop.in/product/multi-zinc-sulphate-33-fertilizer"
    },

    # Common Rust
    "Common_rust": {
        "fertilizer": "Potassium Fertilizer (NPK 10-5-20)",
        "advice": "Potassium improves disease resistance in corn.",
        "buy_link": "https://www.kisanshop.in/product/biowall-pluse-npk-134013-water-soluble-fertilizer"
    },

    # Northern Leaf Blight
    "Northern_Leaf_Blight": {
        "fertilizer": "Zinc Sulphate",
        "advice": "Apply zinc sulphate if deficiency symptoms are visible.",
        "buy_link": "https://www.kisanshop.in/product/katyayani-zinc-sulphate-fertilizer"
    },

    # Healthy Crop
    "healthy": {
        "fertilizer": "Urea / DAP",
        "advice": "Maintain balanced nutrition for healthy crop growth.",
        "buy_link": "https://www.kisanshop.in/product/iffco-nano-dap-fertilizer"
    }
}

# ------------------ IMAGE PREPROCESS ------------------
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(img_array, axis=0)
def get_fertilizer_recommendation(label):
    for key in FERTILIZER_SUGGESTIONS:
        if key.lower() in label.lower():
            return FERTILIZER_SUGGESTIONS[key]
    return None



# ------------------ PREDICTION ------------------
def predict_with_confidence(model, image_source, class_indices, top_k=3):
    img = load_and_preprocess_image(image_source)
    preds = model.predict(img)[0]
    top_idx = preds.argsort()[-top_k:][::-1]
    return [(class_indices[str(i)], float(preds[i])) for i in top_idx]

# ------------------ HEADER ------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;">🌽 Corn Disease Detection</h1>

<p style="text-align:center;">Upload or capture a corn image to predict disease</p>
<hr>
""", unsafe_allow_html=True)


# ------------------ INPUT ------------------
uploaded_image = st.file_uploader("Upload a corn leaf image", ["jpg", "jpeg", "png"])

if st.button("📷 Open Camera"):
    st.session_state.open_camera = True

if st.session_state.open_camera:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False
    camera_image = st.camera_input("Capture image")
else:
    camera_image = None

image_source = uploaded_image if uploaded_image else camera_image

# ------------------ DISPLAY & PREDICT ------------------
if image_source is not None:
    image = Image.open(image_source)
    st.image(image.resize((180, 180)), caption="Input Image")

    if model is None:
        st.error("Model not loaded yet.")
        if model_load_error:
            st.warning(model_load_error)
    else:
        if st.button("Predict"):
            results = predict_with_confidence(model, image_source, class_indices)
            top_label, top_conf = results[0]

            # ✅ Show ONLY ONE disease
            st.success(f"🦠 Detected Disease: **{top_label}**")
            st.write(f"🔍 Confidence: **{int(top_conf*100)}%**")

            # 🌱 Fertilizer Recommendation
            rec = get_fertilizer_recommendation(top_label)

            if rec:
                st.markdown("### 🌱 Fertilizer Recommendation")
                st.success(f"**Recommended Fertilizer:** {rec['fertilizer']}")
                st.write(f"**Advice:** {rec['advice']}")

                st.markdown(
                    f"""
                    <a href="{rec['buy_link']}" target="_blank">
                        <button style="
                            background-color:#0F4D0F;
                            color:white;
                            padding:12px 20px;
                            border:none;
                            border-radius:10px;
                            font-size:15px;
                            cursor:pointer;">
                            🛒 Buy Fertilizer Now
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("No fertilizer recommendation available.")





# ------------------ FOOTER ------------------
st.markdown("<div class='footer'>Works on desktop & mobile 🌱</div>", unsafe_allow_html=True)

