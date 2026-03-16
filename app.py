import streamlit as st
import openai
import base64
from PIL import Image
import io

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Doubt Solver",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Doubt Solver")
st.write("Ask your academic doubts and get instant AI explanations.")

# -----------------------------
# OpenAI API Key
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Session State for History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# User Input
# -----------------------------
st.subheader("Ask your doubt")

question = st.text_area(
    "Enter your question",
    height=120
)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_image = st.file_uploader(
    "Upload image of the question (optional)",
    type=["png", "jpg", "jpeg"]
)

# Show uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Question Image", use_column_width=True)

# -----------------------------
# Solve Button
# -----------------------------
if st.button("Solve Doubt"):

    if question == "" and uploaded_image is None:
        st.warning("Please enter a question or upload an image.")
    
    else:

        with st.spinner("AI is thinking..."):

            prompt = f"""
You are an expert teacher.

Explain the following question clearly step-by-step so that a student can easily understand.

Question:
{question}
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful teacher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            answer = response["choices"][0]["message"]["content"]

            # Save history
            st.session_state.history.append({
                "question": question,
                "answer": answer
            })

            st.success("Solution generated!")

            st.markdown("### 📘 Solution")
            st.write(answer)

# -----------------------------
# Show Previous Doubts
# -----------------------------
st.markdown("---")
st.subheader("Previous Doubts")

for item in reversed(st.session_state.history):

    with st.expander(item["question"][:80] + "..."):

        st.markdown("**Question:**")
        st.write(item["question"])

        st.markdown("**Answer:**")
        st.write(item["answer"])
