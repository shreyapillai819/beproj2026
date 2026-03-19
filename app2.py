# ==========================================================
# VLM Framework for Healthcare (PREMIUM UI VERSION)
# ==========================================================

import os
import json
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import snapshot_download

# ---------------- CONFIG ----------------
HF_REPO_ID = "shreyapillai1312/skin_vlm"
LOCAL_MODEL_DIR = "hf_model"

GEMINI_API_KEY ="AIzaSyCAMWuKGm4aWYiafHSFmHt-ZSw-4CDjFrk"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "users.db"
CSV_FILE = "all_users_logs.csv"
DISEASE_JSON = "disease.json"

# ---------------- Gemini ----------------
try:
    import google.generativeai as genai
except:
    genai = None

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    conn.commit()
    conn.close()

init_db()

def signup(u,p):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO users VALUES (?,?)",(u,p))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def login(u,p):
    conn = sqlite3.connect(DB_PATH)
    res = conn.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p)).fetchone()
    conn.close()
    return res is not None

# ---------------- SAVE ----------------
def save_log():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "username": st.session_state.user,
        "image": st.session_state.image_name,
        "patient": st.session_state.patient_data,
        "predictions": list(zip(st.session_state.topk_labels, st.session_state.topk_probs.tolist())),
        "selected": st.session_state.topk_labels[st.session_state.selected_idx],
        "chat": st.session_state.chat_history,
        "time": timestamp
    }

    df = pd.DataFrame([data])

    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_dir = snapshot_download(repo_id=HF_REPO_ID, local_dir=LOCAL_MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    return processor, model, model.config.id2label

def preprocess_image(img, processor):
    return processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)

def predict(pixel_values, model, id2label, topk):
    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        vals, idx = torch.topk(probs, k=topk)

    labels = [id2label.get(str(i.item()), id2label.get(i.item(), str(i.item()))) for i in idx[0]]
    return labels, vals[0].cpu().numpy()

# ---------------- GEMINI ----------------
def call_gemini(prompt):
    if genai is None:
        return "Gemini SDK missing"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
    return model.generate_content(prompt).text

# ---------------- UI STYLE ----------------
st.set_page_config(page_title="VLM Healthcare", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#e3f2fd,#ffffff);
}
.card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}
.pred-btn button {
    border-radius: 10px;
    transition: 0.2s;
}
.pred-btn button:hover {
    transform: scale(1.05);
    background-color:#bbdefb;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(to right, #2196F3, #21CBF3);
padding:15px;
border-radius:15px;
color:white;'>
🧠 VLM Framework for Healthcare
</h1>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    tab1, tab2 = st.tabs(["Login","Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(u,p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Signup"):
            if signup(u,p):
                st.success("Account created")
            else:
                st.error("User exists")

    st.stop()

# ---------------- MAIN ----------------
st.markdown(f"### 👋 Welcome, {st.session_state.user}")

processor, model, id2label = load_model()

st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top-K predictions", 1, 7, 3)
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

col1, col2 = st.columns([1.2,1])

with col1:
    uploaded = st.file_uploader("📤 Upload skin image", type=["jpg","png"])

    age = st.number_input("Patient age",0,120,40)
    gender = st.selectbox("Gender",["Female","Male","Other"])
    skin = st.selectbox("Skin type",["Type I","Type II","Type III","Type IV","Type V","Type VI"])

    run_button = st.button("🚀 Predict")

with col2:
    out_area = st.empty()

# SESSION
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# PREDICT
if run_button and uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.session_state.uploaded_image = img
    st.session_state.image_name = uploaded.name
    st.session_state.patient_data = {"age":age,"gender":gender,"skin":skin}

    pixel_values = preprocess_image(img, processor)
    labels, probs = predict(pixel_values, model, id2label, top_k)

    st.session_state.topk_labels = labels
    st.session_state.topk_probs = probs

# DISPLAY
if "uploaded_image" in st.session_state:
    img = st.session_state.uploaded_image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,*img.size], outline="#0D47A1", width=40)
    out_area.image(img)

    st.subheader("🔍 Predictions")

    cols = st.columns(len(st.session_state.topk_labels))
    for i, label in enumerate(st.session_state.topk_labels):
        with cols[i]:
            if st.button(f"{label}\n{st.session_state.topk_probs[i]*100:.1f}%", key=i):
                st.session_state.selected_idx = i

    selected = st.session_state.topk_labels[st.session_state.selected_idx]

    # DISEASE CARDS
    if os.path.exists(DISEASE_JSON):
        data = json.load(open(DISEASE_JSON))
        disease = data.get(selected)
        if disease:
            st.markdown(f"## 🩺 {disease.get('full_name','Unknown')}")
            for k,v in disease.items():
                if k!="full_name":
                    st.markdown(
                        f"<div class='card'><b>{k.replace('_',' ').title()}</b><br>{v}</div>",
                        unsafe_allow_html=True
                    )

    # CHAT
    st.subheader("💬 Ask Questions")
    q = st.text_input("Type your question")

    if st.button("Ask"):
        ans = call_gemini(f"Disease: {selected}\nQuestion: {q}")
        st.session_state.chat_history.append(("You", q))
        st.session_state.chat_history.append(("AI", ans))

    for role,text in st.session_state.chat_history:
        if role=="You":
            st.markdown(f"<div class='card'>🧑 {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card'>🤖 {text}</div>", unsafe_allow_html=True)

    # SAVE
    if st.button("💾 Save Session"):
        save_log()
        st.success("Saved successfully ✅")

# FOOTER
st.markdown("---")
st.markdown("<center><b>Group 4 BE Project 2026</b></center>", unsafe_allow_html=True)
