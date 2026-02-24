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

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}

/* Glass container */
.block-container {
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Input styling */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 10px;
}

/* File uploader */
.stFileUploader {
    background: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 12px;
}

/* Headers */
h1, h2, h3 {
    color: #e0f2fe;
}

/* Answer box */
.stSuccess {
    border-radius: 12px;
}

/* Solution box */
.stInfo {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# 📂 LOAD DATA
# ======================
with open("chemistry_sample.json", "r") as f:
    dataset = json.load(f)

# ======================
# 🤖 LOAD MODELS
# ======================
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'], gpu=False)

# ======================
# 🔍 CREATE EMBEDDINGS
# ======================
questions = [item["question"] for item in dataset]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# ======================
# 🔎 SEMANTIC SOLVER
# ======================
def semantic_solver(user_question, top_k=3):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [dataset[int(idx)] for idx in top_results]

# ======================
# 📝 FORMAT SOLUTION
# ======================
def format_solution(solution_text):
    steps = solution_text.replace("\n", ".").split(".")
    return [s.strip() for s in steps if len(s.strip()) > 5]

# ======================
# 🚀 HERO SECTION
# ======================
st.markdown("""
<h1 style='text-align:center; font-size:48px;'>🚀 AI Doubt Solver</h1>
<p style='text-align:center; font-size:18px; color:#cbd5f5;'>
Instant doubt solving + AI practice generator for students
</p>
""", unsafe_allow_html=True)

# ======================
# ✍️ TEXT DOUBT
# ======================
st.subheader("✍️ Type your doubt")

user_input = st.text_input("Ask your doubt")

if user_input:
    results = semantic_solver(user_input)

    for i, r in enumerate(results):
        st.subheader(f"Similar Question {i+1}")

        # Question
        st.write("### Question")
        st.write(r["question"])

        # Options
        st.write("### Options")
        st.write(f"A) {r['options']['A']}")
        st.write(f"B) {r['options']['B']}")
        st.write(f"C) {r['options']['C']}")
        st.write(f"D) {r['options']['D']}")

        # Answer
        st.success(f"Correct Answer: {r['correct_answer']}")

        # Solution
        steps = format_solution(r["solution"])
        st.write("### Stepwise Solution")
        for j, s in enumerate(steps):
            st.write(f"{j+1}. {s}")

# ======================
# 📸 IMAGE DOUBT
# ======================
st.subheader("📸 Upload question image")

uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image")

    img_array = np.array(image)
    ocr_result = reader.readtext(img_array, detail=0)
    extracted_text = " ".join(ocr_result)

    st.subheader("Extracted text")
    st.info(extracted_text)

    results = semantic_solver(extracted_text)

    for i, r in enumerate(results):
        st.subheader(f"Similar Question {i+1}")

        # Question
        st.write("### Question")
        st.write(r["question"])

        # Options
        st.write("### Options")
        st.write(f"A) {r['options']['A']}")
        st.write(f"B) {r['options']['B']}")
        st.write(f"C) {r['options']['C']}")
        st.write(f"D) {r['options']['D']}")

        # Answer
        st.success(f"Correct Answer: {r['correct_answer']}")

        # Solution
        steps = format_solution(r["solution"])
        st.write("### Stepwise Solution")
        for j, s in enumerate(steps):
            st.write(f"{j+1}. {s}")
