import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np

# UI styling
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e3a8a); color:white;}
.solution-box {background:#111827; padding:15px; border-radius:10px; font-family:monospace;}
</style>
""", unsafe_allow_html=True)

# Load dataset
with open("chemistry_sample.json") as f:
    dataset = json.load(f)

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'], gpu=False)

# Embeddings
questions = [d["question"] for d in dataset]
emb = model.encode(questions, convert_to_tensor=True)

# Semantic search
def solve(q):
    qe = model.encode(q, convert_to_tensor=True)
    sc = util.cos_sim(qe, emb)[0]
    idx = sc.argsort(descending=True)[:3]
    return [dataset[int(i)] for i in idx]

# Display block
def show(r,i):
    st.subheader(f"Similar Question {i+1}")
    st.write("### Question")
    st.write(r["question"])

    if any(r["options"].values()):
        st.write("### Options")
        for k,v in r["options"].items():
            if v.strip():
                st.write(f"{k}) {v}")

    st.success(f"Correct Answer: {r['correct_answer']}")

    st.write("### Stepwise Solution")
    lines = r["solution"].split("\n")
    st.markdown("<div class='solution-box'>", unsafe_allow_html=True)
    for l in lines:
        st.write(l)
    st.markdown("</div>", unsafe_allow_html=True)

# App UI
st.title("🚀 AI Doubt Solver")

q = st.text_input("Ask doubt")
if q:
    for i,r in enumerate(solve(q)):
        show(r,i)

img = st.file_uploader("Upload image")
if img:
    im = Image.open(img)
    st.image(im)
    text = " ".join(reader.readtext(np.array(im), detail=0))
    st.info(text)
    for i,r in enumerate(solve(text)):
        show(r,i)
