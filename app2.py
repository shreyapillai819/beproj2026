# ==========================================================
# VLM Framework for Healthcare (FINAL FIXED VERSION)
# GradCAM + Audio Safe + Analytics + Stable Deployment
# ==========================================================

import os
import json
import streamlit as st
from datetime import datetime
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import snapshot_download
from supabase import create_client

# GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- CONFIG ----------------
HF_REPO_ID = "shreyapillai1312/skin_vlm"
LOCAL_MODEL_DIR = "hf_model"

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISEASE_JSON = "disease.json"

# ---------------- SUPABASE ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- GEMINI ----------------
import google.generativeai as genai

def call_gemini(prompt):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        return response.text if response.text else "No response generated"
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_dir = snapshot_download(repo_id=HF_REPO_ID, local_dir=LOCAL_MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return processor, model, model.config.id2label

# ---------------- GRADCAM ----------------
def generate_gradcam(img, pixel_values, model):
    try:
        # ViT-compatible layer
        target_layers = [model.vit.encoder.layer[-1].output]

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=pixel_values)[0]

        img_np = np.array(img.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        return cam_image

    except Exception as e:
        st.warning(f"GradCAM failed: {e}")
        return None

# ---------------- PREDICT ----------------
def preprocess_image(img, processor):
    return processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)

def predict(pixel_values, model, id2label, topk):
    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        vals, idx = torch.topk(probs, k=topk)

    labels = [id2label.get(i.item(), str(i.item())) for i in idx[0]]
    return labels, vals[0].cpu().numpy()

# ---------------- UI ----------------
st.set_page_config(page_title="VLM Healthcare", layout="wide")

st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(to right, #2196F3, #21CBF3);
padding:15px;
border-radius:15px;
color:white;'>
🧠 VLM SkinCare Enhanced
</h1>
""", unsafe_allow_html=True)

processor, model, id2label = load_model()

# Sidebar
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K Predictions", 1, 5, 3)
show_cam = st.sidebar.checkbox("Show Grad-CAM")

# Upload Section
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload image", type=["jpg", "png"])
    age = st.number_input("Age", 0, 120, 30)

with col2:
    st.subheader("🎤 Voice Input")
    audio = st.audio_input("Ask your question via voice")

# ---------------- RUN ----------------
if st.button("Predict"):

    if uploaded is None:
        st.warning("Please upload an image")
        st.stop()

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prediction
    pixel_values = preprocess_image(img, processor)
    labels, probs = predict(pixel_values, model, id2label, top_k)

    st.subheader("🔍 Predictions")
    for l, p in zip(labels, probs):
        st.write(f"**{l}**: {p*100:.2f}%")

    # GradCAM
    if show_cam:
        cam_img = generate_gradcam(img, pixel_values, model)
        if cam_img is not None:
            st.image(cam_img, caption="Grad-CAM Visualization")

    # ---------------- ANALYTICS ----------------
    st.subheader("📊 Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(labels, probs)
    ax.set_xlabel("Disease")
    ax.set_ylabel("Confidence")
    ax.set_title("Top-K Predictions")

    st.pyplot(fig, clear_figure=True)

    # ---------------- Q&A ----------------
    st.subheader("💬 Ask Questions")

    question = st.text_input("Type your question")

    # Audio safe handling
    if audio is not None:
        st.info("Audio received (transcription not enabled yet)")

    if st.button("Ask") and question:
        prompt = f"""
Patient age: {age}
Predictions: {list(zip(labels, [float(p) for p in probs]))}

User Question: {question}

Rules:
- Do NOT give final diagnosis
- Keep answer simple
"""
        answer = call_gemini(prompt)
        st.success(answer)

# ---------------- HISTORY ANALYTICS ----------------
st.subheader("📈 Usage Analytics")

if st.button("Load Analytics"):
    try:
        logs = supabase.table("logs").select("*").execute().data

        if not logs:
            st.info("No logs found")
        else:
            import pandas as pd
            df = pd.DataFrame(logs)

            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')

            fig2, ax2 = plt.subplots()
            ax2.plot(df['time'], range(len(df)))

            ax2.set_title("App Usage Over Time")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Sessions")

            st.pyplot(fig2, clear_figure=True)

    except Exception as e:
        st.error(f"Analytics error: {e}")
