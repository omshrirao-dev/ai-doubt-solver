import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np

# -----------------------
# Load dataset
# -----------------------
with open("chemistry_sample.json", "r") as f:
    dataset = json.load(f)

# -----------------------
# Load embedding model
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# Load OCR model (cloud-safe)
# -----------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------
# Create embeddings
# -----------------------
questions = [item["question"] for item in dataset]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# -----------------------
# Semantic solver
# -----------------------
def semantic_solver(user_question, top_k=3):
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [dataset[int(idx)] for idx in top_results]

# -----------------------
# Format solution into steps
# -----------------------
def format_solution(solution_text):
    steps = solution_text.replace("\n", ".").split(".")
    return [s.strip() for s in steps if len(s.strip()) > 5]

# -----------------------
# UI
# -----------------------
st.title("📚 AI Doubt Solver + Image Solver")

# ======================
# TEXT DOUBT
# ======================
st.subheader("✍️ Type your doubt")

user_input = st.text_input("Ask your doubt")

if user_input:
    results = semantic_solver(user_input)

    for i, r in enumerate(results):
        st.subheader(f"Similar Question {i+1}")
        st.write(r["question"])

        st.success(f"Answer: {r['correct_answer']}")

        steps = format_solution(r["solution"])
        st.write("### Stepwise Solution")
        for j, s in enumerate(steps):
            st.write(f"{j+1}. {s}")

# ======================
# IMAGE DOUBT
# ======================
st.subheader("📸 Upload question image")

uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image")

    # Convert to numpy
    img_array = np.array(image)

    # OCR extraction
    ocr_result = reader.readtext(img_array, detail=0)
    extracted_text = " ".join(ocr_result)

    st.subheader("Extracted text")
    st.info(extracted_text)

    results = semantic_solver(extracted_text)

    for i, r in enumerate(results):
        st.subheader(f"Similar Question {i+1}")
        st.write(r["question"])

        st.success(f"Answer: {r['correct_answer']}")

        steps = format_solution(r["solution"])
        st.write("### Stepwise Solution")
        for j, s in enumerate(steps):
            st.write(f"{j+1}. {s}")
