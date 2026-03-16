import streamlit as st
import openai
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Doubt Solver",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Doubt Solver")
st.write("Ask your academic doubts and get AI explanations.")

# -------------------------
# OpenAI Key
# -------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------------------------
# Session History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Question Input
# -------------------------
st.subheader("Enter your doubt")

question = st.text_area(
    "Type your question",
    height=150
)

# -------------------------
# Image Upload
# -------------------------
uploaded_image = st.file_uploader(
    "Upload question image (optional)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Question")

# -------------------------
# Solve Button
# -------------------------
if st.button("Solve Doubt"):

    if question == "" and uploaded_image is None:
        st.warning("Please enter a question or upload an image")

    else:

        with st.spinner("AI solving your doubt..."):

            prompt = f"""
You are an expert teacher.

Explain this question step-by-step so that a student understands clearly.

Question:
{question}
"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":"You are a helpful teacher"},
                    {"role":"user","content":prompt}
                ]
            )

            answer = response.choices[0].message.content

            st.session_state.history.append({
                "question": question,
                "answer": answer
            })

            st.success("Solution generated!")

            st.markdown("### 📘 Answer")
            st.write(answer)

# -------------------------
# Previous Doubts
# -------------------------
st.markdown("---")
st.subheader("Previous Doubts")

for item in reversed(st.session_state.history):

    with st.expander(item["question"][:80]):

        st.write("**Question:**")
        st.write(item["question"])

        st.write("**Answer:**")
        st.write(item["answer"])
