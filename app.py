import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util
import easyocr
from PIL import Image
import numpy as np
import difflib

st.set_page_config(page_title="AI Doubt Solver", layout="wide")

# ======================
# 🎨 PREMIUM ANIMATED CSS
# ======================
st.markdown("""
<style>

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg,#0f172a,#1e1b4b,#312e81,#0f766e);
    background-size:400% 400%;
    animation: gradientBG 18s ease infinite;
}

@keyframes gradientBG{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

/* Floating glow blobs */
.stApp:before {
    content:'';
    position:fixed;
    width:300px;
    height:300px;
    background:#4f46e5;
    filter:blur(140px);
    top:10%;
    left:10%;
    opacity:0.4;
}

.stApp:after {
    content:'';
    position:fixed;
    width:300px;
    height:300px;
    background:#06b6d4;
    filter:blur(140px);
    bottom:10%;
    right:10%;
    opacity:0.4;
}

/* Hero */
.hero {
    text-align:center;
    padding:30px;
    font-size:56px;
    font-weight:800;
    color:white;
    animation: fadeIn 1.5s ease;
}

/* Card */
.card {
    background: rgba(255,255,255,0.07);
    padding:22px;
    border-radius:18px;
    margin-bottom:20px;
    backdrop-filter: blur(14px);
    box-shadow:0 8px 25px rgba(0,0,0,0.4);
    transition:0.35s;
    animation: fadeInUp 0.8s ease;
}

.card:hover{
    transform:translateY(-6px) scale(1.01);
    box-shadow:0 15px 40px rgba(0,0,0,0.6);
}

/* Titles */
.section {
    color:#e0f2fe;
    font-weight:600;
    font-size:22px;
    margin-top:12px;
}

/* Solution */
.solution {
    background:#020617;
    padding:15px;
    border-radius:10px;
    margin-top:10px;
}

/* Animations */
@keyframes fadeIn{
    from{opacity:0}
    to{opacity:1}
}

@keyframes fadeInUp{
    from{opacity:0; transform:translateY(20px)}
    to{opacity:1; transform:translateY(0)}
}

</style>
""", unsafe_allow_html=True)

# HERO
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

    st.markdown("<div class='section'>Question</div>", unsafe_allow_html=True)
    st.write(r.get("question",""))

    options=r.get("options",{})
    show_options=False
    for v in options.values():
        if isinstance(v,str) and v.strip():
            show_options=True

    if show_options:
        st.markdown("<div class='section'>Options</div>", unsafe_allow_html=True)
        for k,v in options.items():
            if isinstance(v,str) and v.strip():
                st.write(f"{k}) {v}")

    st.success(f"Correct Answer: {r.get('correct_answer','')}")

    st.markdown("<div class='section'>Solution</div>", unsafe_allow_html=True)

    if r.get("solution"):
        st.markdown("<div class='solution'>", unsafe_allow_html=True)
        for line in r["solution"].split("\n"):
            st.write(line)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# INPUT
# ======================
col1,col2=st.columns(2)

with col1:
    q=st.text_input("💬 Ask doubt")

with col2:
    img=st.file_uploader("📷 Upload image")

# TEXT
if q:
    ex=exact_match(q)
    if ex:
        show(ex)
    else:
        for i,r in enumerate(semantic(q)):
            show(r,i)

# IMAGE
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
