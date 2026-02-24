import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np
import os
import difflib

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

# Models
model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'], gpu=False)

questions = [d["question"] for d in dataset]
emb = model.encode(questions, convert_to_tensor=True)

# Exact match
def exact_match(q):
    scores = [difflib.SequenceMatcher(None,q.lower(),d["question"].lower()).ratio() for d in dataset]
    best = max(scores)
    if best > 0.75:
        return dataset[scores.index(best)]
    return None

# Semantic
def semantic(q):
    qe = model.encode(q, convert_to_tensor=True)
    sc = util.cos_sim(qe, emb)[0]
    idx = sc.argsort(descending=True)[:3]
    return [dataset[int(i)] for i in idx]

# Display
def show(r,i=None):

    if i is None:
        st.subheader("Exact Match Found")
    else:
        st.subheader(f"Similar Question {i+1}")

    # Question
    st.write("### Question")
    st.write(r["question"])

    # Question image
    if "question_image" in r and os.path.exists("structures/"+r["question_image"]):
        st.image("structures/"+r["question_image"])

    # Options
    if any(r["options"].values()):
        st.write("### Options")

        for k,v in r["options"].items():

            # TEXT OPTION
            if isinstance(v,str) and v.strip():
                st.write(f"{k}) {v}")

            # IMAGE OPTION
            elif isinstance(v,dict) and "image" in v:
                st.write(f"{k})")
                path = "structures/"+v["image"]
                if os.path.exists(path):
                    st.image(path)

    # Answer
    st.success(f"Correct Answer: {r['correct_answer']}")

    # Solution
    st.write("### Stepwise Solution")
    st.markdown("<div class='solution-box'>", unsafe_allow_html=True)
    for l in r["solution"].split("\n"):
        st.write(l)
    st.markdown("</div>", unsafe_allow_html=True)

    # Solution image
    if "solution_image" in r and os.path.exists("structures/"+r["solution_image"]):
        st.image("structures/"+r["solution_image"])

    st.markdown("---")

# App
st.title("🚀 AI Doubt Solver")

q = st.text_input("Ask doubt")

if q:
    ex = exact_match(q)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(q)):
            show(r,i)

img = st.file_uploader("Upload image")

if img:
    im = Image.open(img)
    st.image(im)

    text = " ".join(reader.readtext(np.array(im), detail=0))
    st.info(text)

    ex = exact_match(text)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(text)):
            show(r,i)
