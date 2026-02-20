import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image

# 🔴 IMPORTANT: Update path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

    results = []
    for idx in top_results:
        results.append(dataset[int(idx)])

    return results

# -----------------------
# Format solution into steps
# -----------------------
def format_solution(solution_text):
    steps = solution_text.replace("\n", ".").split(".")
    formatted_steps = []
    for step in steps:
        step = step.strip()
        if len(step) > 5:
            formatted_steps.append(step)
    return formatted_steps

# -----------------------
# UI START
# -----------------------
st.title("📚 AI Doubt Solver + Practice Generator")

# =======================
# TEXT DOUBT
# =======================
st.subheader("✍️ Type your doubt")

user_input = st.text_input("Ask your doubt:")

if user_input:
    results = semantic_solver(user_input, top_k=3)

    for i, result in enumerate(results):
        st.subheader(f"Similar Question {i+1}")
        st.write(result["question"])

        st.write("Options:")
        st.write(f"A) {result['options']['A']}")
        st.write(f"B) {result['options']['B']}")
        st.write(f"C) {result['options']['C']}")
        st.write(f"D) {result['options']['D']}")

        st.success(f"Answer: {result['correct_answer']}")

        # Stepwise solution
        steps = format_solution(result["solution"])
        st.write("### Stepwise Solution")
        for j, step in enumerate(steps):
            st.write(f"{j+1}. {step}")

# =======================
# IMAGE DOUBT
# =======================
st.subheader("📸 Upload question image")

uploaded_file = st.file_uploader("Upload question image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded question")

    extracted_text = pytesseract.image_to_string(image)

    st.subheader("Extracted text")
    st.info(extracted_text)

    results = semantic_solver(extracted_text, top_k=3)

    for i, result in enumerate(results):
        st.subheader(f"Similar Question {i+1}")
        st.write(result["question"])

        st.write("Options:")
        st.write(f"A) {result['options']['A']}")
        st.write(f"B) {result['options']['B']}")
        st.write(f"C) {result['options']['C']}")
        st.write(f"D) {result['options']['D']}")

        st.success(f"Answer: {result['correct_answer']}")

        # Stepwise solution
        steps = format_solution(result["solution"])
        st.write("### Stepwise Solution")
        for j, step in enumerate(steps):
            st.write(f"{j+1}. {step}")