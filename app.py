import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np

# ======================
# 🎨 SMART CSS DESIGN
# ======================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white;}
.block-container {background: rgba(255,255,255,0.08); padding: 2rem; border-radius: 18px; backdrop-filter: blur(14px); box-shadow: 0 10px 30px rgba(0,0,0,0.3);}
h1, h2, h3 {color: #e0f2fe;}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA
# ======================
with open("chemistry_sample.json", "r") as f:
    dataset = json.load(f)

# ======================
# LOAD MODELS
# ======================
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'], gpu=False)

# ======================
# EMBEDDINGS
# ======================
questions = [item["question"] for item in dataset]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# ======================
# SEMANTIC SOLVER
# ======================
def semantic_solver(user_question, top_k=3):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [dataset[int(idx)] for idx in top_results]

# ======================
# SOLUTION FORMATTER
# ======================
def format_solution(text):
    if not text:
        return ["Solution not available"]
    parts = text.replace("\n","\n\n")
    return parts

# ======================
# DISPLAY QUESTION
# ======================
def display_question_block(r, index):

    st.subheader(f"Similar Question {index+1}")

    # Question
    st.write("### Question")
    st.write(r["question"])

    # Options (only if exist)
    if any(r["options"].values()):
        st.write("### Options")
        for key,val in r["options"].items():
            if val.strip():
                st.write(f"{key}) {val}")

    # Answer
    st.success(f"Correct Answer: {r['correct_answer']}")

    # Solution
    st.write("### Detailed Solution")
    st.write(format_solution(r["solution"]))

    st.markdown("---")

# ======================
# HERO
# ======================
st.markdown("""
<h1 style='text-align:center;'>🚀 AI Doubt Solver</h1>
<p style='text-align:center;'>Instant doubt solving + AI practice generator</p>
""", unsafe_allow_html=True)

# ======================
# TEXT INPUT
# ======================
user_input = st.text_input("Ask your doubt")

if user_input:
    results = semantic_solver(user_input)
    for i,r in enumerate(results):
        display_question_block(r,i)

# ======================
# IMAGE INPUT
# ======================
uploaded_file = st.file_uploader("Upload question image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img_array = np.array(image)
    text = " ".join(reader.readtext(img_array, detail=0))

    st.write("### Extracted text")
    st.info(text)

    results = semantic_solver(text)
    for i,r in enumerate(results):
        display_question_block(r,i)
