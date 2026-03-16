import streamlit as st
from openai import OpenAI
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Doubt Solver",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Doubt Solver")
st.write("Ask your academic doubts and get instant AI explanations.")

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Session State
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Question Input
# -----------------------------
st.subheader("Enter your doubt")

question = st.text_area(
    "Type your question here",
    height=150
)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_image = st.file_uploader(
    "Upload image of the question (optional)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Question", use_column_width=True)

# -----------------------------
# Solve Button
# -----------------------------
if st.button("Solve Doubt"):

    if question == "" and uploaded_image is None:
        st.warning("Please enter a question or upload an image.")

    else:

        with st.spinner("AI is solving your doubt..."):

            prompt = f"""
You are an expert tutor.

Explain the following question step-by-step in a simple way.

Question:
{question}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful teacher."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

            st.session_state.history.append({
                "question": question,
                "answer": answer
            })

            st.success("Solution Generated!")

            st.markdown("### 📘 Answer")
            st.write(answer)

# -----------------------------
# History Section
# -----------------------------
st.markdown("---")
st.subheader("Previous Doubts")

for item in reversed(st.session_state.history):

    with st.expander(item["question"][:80] + "..."):

        st.markdown("**Question:**")
        st.write(item["question"])

        st.markdown("**Answer:**")
        st.write(item["answer"])
