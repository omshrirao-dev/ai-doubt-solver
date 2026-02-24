import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np
import difflib

# UI
st.set_page_config(page_title="AI Doubt Solver")
st.title("🚀 AI Doubt Solver")

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
def show(r, i=None):

    st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")

    # Question
    st.write("### Question")
    st.write(r.get("question",""))

    # Question image
    qi = r.get("question_image")
    if qi:
        path="structures/"+qi
        if os.path.exists(path):
            st.image(path)

    # Detect real options
    options=r.get("options",{})
    show_options=False

    for v in options.values():
        if isinstance(v,str) and v.strip():
            show_options=True
        elif isinstance(v,dict) and "image" in v:
            show_options=True

    if show_options:
        st.write("### Options")
        for k,v in options.items():

            if isinstance(v,str) and v.strip():
                st.write(f"{k}) {v}")

            elif isinstance(v,dict) and "image" in v:
                st.write(f"{k})")
                path="structures/"+v["image"]
                if os.path.exists(path):
                    st.image(path)

    # Answer
    st.success(f"Correct Answer: {r.get('correct_answer','')}")

    # Solution
    st.write("### Solution")

    si=r.get("solution_image")
    if si:
        path="structures/"+si
        if os.path.exists(path):
            st.image(path)
        else:
            st.warning(f"Missing image: {path}")

    elif r.get("solution"):
        for line in r["solution"].split("\n"):
            st.write(line)

    st.markdown("---")


# TEXT INPUT
q=st.text_input("Ask doubt")

if q:
    ex=exact_match(q)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(q)):
            show(r,i)

# IMAGE INPUT
img=st.file_uploader("Upload image")

if img:
    im=Image.open(img)
    st.image(im)

    text=" ".join(reader.readtext(np.array(im),detail=0))
    st.info(text)

    ex=exact_match(text)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(text)):
            show(r,i)
