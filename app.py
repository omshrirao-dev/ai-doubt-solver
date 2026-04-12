import streamlit as st
import json
import difflib
from PIL import Image
import numpy as np
import easyocr

st.set_page_config(layout="wide")

# ------------------------------
# Load Data
# ------------------------------

@st.cache_data
def load_data():
    with open("chemistry_sample.json") as f:
        return json.load(f)

dataset = load_data()
questions = [d.get("question","") for d in dataset]

# ------------------------------
# LIGHT SEARCH (no AI model)
# ------------------------------

def search(query):
    scores = [
        difflib.SequenceMatcher(None, query.lower(), q.lower()).ratio()
        for q in questions
    ]
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return [dataset[i] for i in top]

# ------------------------------
# OCR (optional safe load)
# ------------------------------

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ------------------------------
# UI
# ------------------------------

st.title("🚀 AI Doubt Solver")

query = st.text_input("Ask your doubt")
img = st.file_uploader("Upload question image")

# ------------------------------
# TEXT SEARCH
# ------------------------------

if query:
    results = search(query)
    for q in results:
        st.subheader(q["question"])
        for k,v in q.get("options",{}).items():
            st.write(f"{k}) {v}")
        st.success(f"Answer: {q.get('correct_answer','')}")
        st.write(q.get("solution",""))
        st.divider()

# ------------------------------
# IMAGE SEARCH
# ------------------------------

if img:
    reader = load_ocr()
    image = Image.open(img)
    st.image(image)

    text = " ".join(reader.readtext(np.array(image), detail=0))
    st.info(text)

    results = search(text)
    for q in results:
        st.subheader(q["question"])
        st.write(q.get("solution",""))
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
