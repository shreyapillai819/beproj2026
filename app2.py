# ==========================================================
# VLM Framework for Healthcare (FINAL DEPLOYMENT VERSION)
# ==========================================================

import os
import json
import streamlit as st
from datetime import datetime
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import snapshot_download
from supabase import create_client

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
try:
    import google.generativeai as genai
except:
    genai = None

# ---------------- AUTH ----------------
def signup(u, p):
    try:
        supabase.table("users").insert({
            "username": u,
            "password": p
        }).execute()
        return True
    except:
        return False

def login(u, p):
    res = supabase.table("users") \
        .select("*") \
        .eq("username", u) \
        .eq("password", p) \
        .execute()

    return len(res.data) > 0

# ---------------- SAVE LOG ----------------
def save_log():
    data = {
        "username": st.session_state.user,
        "image": st.session_state.image_name,
        "patient": st.session_state.patient_data,
        "predictions": [
            {"label": l, "prob": float(p)}
            for l, p in zip(
                st.session_state.topk_labels,
                st.session_state.topk_probs
            )
        ],
        "selected": st.session_state.topk_labels[st.session_state.selected_idx],
        "chat": st.session_state.chat_history,
        "time": datetime.now().isoformat()
    }

    supabase.table("logs").insert(data).execute()

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
        return "Gemini not installed"

    if GEMINI_API_KEY is None:
        return "API key missing"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        response = model.generate_content(prompt)
        return response.text if response.text else "No response generated"

    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ---------------- UI ----------------
st.set_page_config(page_title="VLM Healthcare", layout="wide")

st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(to right, #2196F3, #21CBF3);
padding:15px;
border-radius:15px;
color:white;'>
🧠 VLM SkinCare
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
            if login(u, p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Signup"):
            if signup(u, p):
                st.success("Account created")
            else:
                st.error("User exists")

    st.stop()

# ---------------- MAIN ----------------
st.markdown(f"### 👋 Welcome, {st.session_state.user}")

processor, model, id2label = load_model()

top_k = st.sidebar.slider("Top-K predictions", 1, 7, 3)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded = st.file_uploader("Upload image", type=["jpg", "png"])
    age = st.number_input("Age", 0, 120, 40)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    skin = st.selectbox("Skin type", ["Type I", "II", "III", "IV", "V", "VI"])
    run = st.button("Predict")

with col2:
    out = st.empty()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# ---------------- PREDICT ----------------
if run and uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.session_state.uploaded_image = img
    st.session_state.image_name = uploaded.name
    st.session_state.patient_data = {"age": age, "gender": gender, "skin": skin}

    pixel_values = preprocess_image(img, processor)
    labels, probs = predict(pixel_values, model, id2label, top_k)

    st.session_state.topk_labels = labels
    st.session_state.topk_probs = probs

# ---------------- DISPLAY ----------------
if "uploaded_image" in st.session_state:
    img = st.session_state.uploaded_image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, *img.size], outline="#0D47A1", width=20)
    out.image(img)

    st.subheader("Predictions")
    cols = st.columns(len(st.session_state.topk_labels))

    for i, label in enumerate(st.session_state.topk_labels):
        with cols[i]:
            if st.button(f"{label}\n{st.session_state.topk_probs[i]*100:.1f}%", key=i):
                st.session_state.selected_idx = i

    selected = st.session_state.topk_labels[st.session_state.selected_idx]

    # ---------------- DISEASE INFO ----------------
    disease = None
    if os.path.exists(DISEASE_JSON):
        with open(DISEASE_JSON) as f:
            data = json.load(f)

        disease = data.get(selected)

        if disease:
            st.markdown(f"## 🩺 {disease.get('full_name','Unknown')}")
            for k, v in disease.items():
                if k != "full_name":
                    st.markdown(f"**{k}**: {v}")

    # ---------------- CHAT ----------------
    st.subheader("💬 Ask Questions")
    question = st.text_input("Enter your question here:")

    if st.button("Ask"):
        disease_context = ""

        if os.path.exists(DISEASE_JSON):
            with open(DISEASE_JSON) as f:
                data = json.load(f)

            for i, label in enumerate(st.session_state.topk_labels):
                disease_data = data.get(label)

                if disease_data:
                    disease_context += f"\n--- {label} ({st.session_state.topk_probs[i]*100:.1f}%) ---\n"
                    for k, v in disease_data.items():
                        disease_context += f"{k}: {v}\n"

        prompt = f"""
You are a helpful medical assistant.

Patient Info:
{st.session_state.patient_data}

Predicted Diseases:
{list(zip(st.session_state.topk_labels, [float(p) for p in st.session_state.topk_probs]))}

Primary Focus Disease:
{selected}

Disease Context:
{disease_context}

User Question:
{question}

Instructions:
- Understand user intent first

If question is:
• Disease-specific → use predictions
• Comparison → compare diseases
• General → answer normally

Rules:
- Do NOT give a definitive diagnosis
- Keep answers clear, concise, and structured
"""

        answer = call_gemini(prompt)

        st.session_state.chat_history.append(("User", question))
        st.session_state.chat_history.append(("AI", answer))

    for role, text in st.session_state.chat_history:
        st.write(f"{role}: {text}")

    # ---------------- SAVE ----------------
    if st.button("Save Session"):
        save_log()
        st.success("Saved!")
