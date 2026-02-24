import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np
import difflib

st.set_page_config(layout="wide")

# ======================
# 🎨 THEME ROTATION
# ======================
themes = [
"linear-gradient(-45deg,#0f172a,#1e1b4b,#312e81,#0f766e)",
"linear-gradient(-45deg,#020617,#6d28d9,#ec4899,#0ea5e9)",
"linear-gradient(-45deg,#022c22,#0f766e,#06b6d4,#1e3a8a)",
"linear-gradient(-45deg,#0f172a,#1e40af,#7c3aed,#0ea5e9)",
"linear-gradient(-45deg,#022c22,#065f46,#1e293b,#0f172a)"
]

if "theme_index" not in st.session_state:
    st.session_state.theme_index = 0

def next_theme():
    st.session_state.theme_index = (st.session_state.theme_index + 1) % len(themes)

# Button
col1,col2=st.columns([9,1])
with col2:
    st.button("🎨", on_click=next_theme)

bg = themes[st.session_state.theme_index]

# ======================
# CSS
# ======================
st.markdown(f"""
<style>

.stApp {{
    background: {bg};
    background-size:400% 400%;
    animation: gradientBG 14s ease infinite;
}}

@keyframes gradientBG{{
0%{{background-position:0% 50%;}}
50%{{background-position:100% 50%;}}
100%{{background-position:0% 50%;}}
}}

.hero {{
text-align:center;
font-size:48px;
font-weight:800;
color:white;
padding:20px;
}}

.card {{
background: rgba(255,255,255,0.07);
padding:20px;
border-radius:16px;
margin-bottom:20px;
backdrop-filter: blur(12px);
box-shadow:0 8px 20px rgba(0,0,0,0.4);
transition:0.3s;
}}

.card:hover {{
transform:translateY(-5px);
}}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero'>🚀 AI Doubt Solver</div>", unsafe_allow_html=True)

# ======================
# DATA
# ======================
with open("chemistry_sample.json") as f:
    dataset=json.load(f)

model=SentenceTransformer('all-MiniLM-L6-v2')
reader=easyocr.Reader(['en'],gpu=False)

questions=[d["question"] for d in dataset]
emb=model.encode(questions,convert_to_tensor=True)

# ======================
# SEARCH
# ======================
def exact_match(q):
    scores=[difflib.SequenceMatcher(None,q.lower(),d["question"].lower()).ratio() for d in dataset]
    best=max(scores)
    if best>0.75:
        return dataset[scores.index(best)]
    return None

def semantic(q):
    qe=model.encode(q,convert_to_tensor=True)
    sc=util.cos_sim(qe,emb)[0]
    idx=sc.argsort(descending=True)[:3]
    return [dataset[int(i)] for i in idx]

# ======================
# DISPLAY
# ======================
def show(r,i=None):

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write("### Question")
    st.write(r.get("question",""))

    options=r.get("options",{})
    for k,v in options.items():
        if isinstance(v,str) and v.strip():
            st.write(f"{k}) {v}")

    st.success(f"Correct Answer: {r.get('correct_answer','')}")

    if r.get("solution"):
        for l in r["solution"].split("\n"):
            st.write(l)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# INPUT
# ======================
q=st.text_input("💬 Ask doubt")
img=st.file_uploader("📷 Upload image")

if q:
    ex=exact_match(q)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(q)):
            show(r,i)

if img:
    im=Image.open(img)
    st.image(im,width=300)
    text=" ".join(reader.readtext(np.array(im),detail=0))
    st.info(text)

    ex=exact_match(text)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(text)):
            show(r,i)
