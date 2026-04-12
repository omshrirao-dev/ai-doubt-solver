import streamlit as st
import json
import difflib
import numpy as np
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import traceback

st.set_page_config(layout="wide")

# ------------------------------
# 🎨 Theme changer
# ------------------------------

themes = [
"linear-gradient(-45deg,#0f172a,#1e1b4b,#312e81,#0f766e)",
"linear-gradient(-45deg,#020617,#6d28d9,#ec4899,#0ea5e9)",
"linear-gradient(-45deg,#022c22,#0f766e,#06b6d4,#1e3a8a)",
"linear-gradient(-45deg,#0f172a,#1e40af,#7c3aed,#0ea5e9)",
"linear-gradient(-45deg,#022c22,#065f46,#1e293b,#0f172a)"
]

if "theme_index" not in st.session_state:
    st.session_state.theme_index = 0

def next_theme():
    st.session_state.theme_index = (st.session_state.theme_index + 1) % len(themes)

bg = themes[st.session_state.theme_index]

st.markdown(f"""
<style>
.stApp {{
background:{bg};
background-size:400% 400%;
animation:gradient 15s ease infinite;
}}
@keyframes gradient {{
0% {{background-position:0% 50%;}}
50% {{background-position:100% 50%;}}
100% {{background-position:0% 50%;}}
}}
.hero {{
text-align:center;
font-size:48px;
font-weight:700;
color:white;
padding:20px;
}}
.card {{
background:rgba(255,255,255,0.07);
padding:20px;
border-radius:16px;
margin-bottom:20px;
backdrop-filter:blur(10px);
box-shadow:0 8px 25px rgba(0,0,0,0.4);
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------

col1,col2 = st.columns([9,1])

with col1:
    st.markdown("<div class='hero'>🚀 AI Doubt Solver</div>", unsafe_allow_html=True)

with col2:
    st.button("🎨", on_click=next_theme)

# ------------------------------
# Load dataset (cached)
# ------------------------------

@st.cache_data
def load_data():
    with open("chemistry_sample.json") as f:
        return json.load(f)

dataset = load_data()

questions = [d.get("question","") for d in dataset]

# ------------------------------
# Load AI model (cached)
# ------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# Load embeddings (cached)
# ------------------------------

@st.cache_data
def get_embeddings(questions):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# ------------------------------
# OCR (cached)
# ------------------------------

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ------------------------------
# Matching functions
# ------------------------------

def exact_match(query):
    scores = [
        difflib.SequenceMatcher(None, query.lower(), q.lower()).ratio()
        for q in questions
    ]
    best = max(scores)

    if best > 0.75:
        return dataset[scores.index(best)]

    return None


def semantic_search(query):
    model = load_model()
    embeddings = get_embeddings(questions)

    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]
    top = scores.argsort(descending=True)[:3]

    return [dataset[int(i)] for i in top]

# ------------------------------
# Display function
# ------------------------------

def show_question(q):
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Question")
    st.write(q.get("question",""))

    options = q.get("options", {})
    if options:
        st.subheader("Options")
        for k,v in options.items():
            if v.strip():
                st.write(f"{k}) {v}")

    st.success(f"Correct Answer: {q.get('correct_answer','')}")

    st.subheader("Solution")
    st.write(q.get("solution",""))

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Inputs
# ------------------------------

col1,col2 = st.columns(2)

with col1:
    text_query = st.text_input("💬 Ask your doubt")

with col2:
    img = st.file_uploader("📷 Upload question image")

# ------------------------------
# MAIN APP (SAFE EXECUTION)
# ------------------------------

try:

    # TEXT SEARCH
    if text_query:
        result = exact_match(text_query)

        if result:
            show_question(result)
        else:
            similar = semantic_search(text_query)
            for q in similar:
                show_question(q)

    # IMAGE SEARCH
    if img:
        reader = load_ocr()

        image = Image.open(img)
        st.image(image, width=300)

        extracted = " ".join(reader.readtext(np.array(image), detail=0))
        st.info(extracted)

        result = exact_match(extracted)

        if result:
            show_question(result)
        else:
            similar = semantic_search(extracted)
            for q in similar:
                show_question(q)

except Exception as e:
    st.error("⚠️ Something went wrong")
    st.text(str(e))
    st.text(traceback.format_exc())        st.subheader("Options")
        for k,v in options.items():
            if v.strip():
                st.write(f"{k}) {v}")

    st.success(f"Correct Answer: {q.get('correct_answer','')}")

    st.subheader("Solution")

    solution = q.get("solution","")

    for line in solution.split("\n"):
        st.write(line)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Inputs
# ------------------------------

col1,col2 = st.columns(2)

with col1:
    text_query = st.text_input("💬 Ask your doubt")

with col2:
    img = st.file_uploader("📷 Upload question image")

# ------------------------------
# Text Search
# ------------------------------

if text_query:

    result = exact_match(text_query)

    if result:
        show_question(result)

    else:
        similar = semantic_search(text_query)

        for q in similar:
            show_question(q)

# ------------------------------
# Image Search
# ------------------------------

if img:

    image = Image.open(img)

    st.image(image, width=300)

    extracted = " ".join(reader.readtext(np.array(image), detail=0))

    st.info(extracted)

    result = exact_match(extracted)

    if result:
        show_question(result)

    else:
        similar = semantic_search(extracted)

        for q in similar:
            show_question(q)
