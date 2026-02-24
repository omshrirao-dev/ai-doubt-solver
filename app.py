import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np
import os
import difflib

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#1e3a8a); color:white;}
.solution-box {background:#111827; padding:15px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

with open("chemistry_sample.json") as f:
    dataset = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'], gpu=False)

questions = [d["question"] for d in dataset]
emb = model.encode(questions, convert_to_tensor=True)

def exact_match(q):
    scores = [difflib.SequenceMatcher(None,q.lower(),d["question"].lower()).ratio() for d in dataset]
    best = max(scores)
    if best > 0.75:
        return dataset[scores.index(best)]
    return None

def semantic(q):
    qe = model.encode(q, convert_to_tensor=True)
    sc = util.cos_sim(qe, emb)[0]
    idx = sc.argsort(descending=True)[:3]
    return [dataset[int(i)] for i in idx]

def show(r,i=None):

    st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")

    st.write("### Question")
    st.write(r["question"])

    if "question_image" in r and os.path.exists("structures/"+r["question_image"]):
        st.image("structures/"+r["question_image"])

    if any(r["options"].values()):
        st.write("### Options")
        for k,v in r["options"].items():

            if isinstance(v,str) and v.strip():
                st.write(f"{k}) {v}")

            elif isinstance(v,dict) and "image" in v:
                st.write(f"{k})")
                path="structures/"+v["image"]
                if os.path.exists(path):
                    st.image(path)

    st.success(f"Correct Answer: {r['correct_answer']}")

    st.write("### Solution")

    # ⭐ KEY LOGIC
    if "solution_image" in r and os.path.exists("structures/"+r["solution_image"]):
        st.image("structures/"+r["solution_image"])

    elif r["solution"]:
        st.markdown("<div class='solution-box'>", unsafe_allow_html=True)
        for l in r["solution"].split("\n"):
            st.write(l)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

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
