import streamlit as st
import json
import difflib
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")

# ======================
# LOAD USERS
# ======================
with open("users.json") as f:
    users = json.load(f)

# ======================
# SESSION INIT
# ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None


# ======================
# LOGIN FUNCTION
# ======================
def login(username, password):
    for u in users:
        if u["username"] == username and u["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
    return False


# ======================
# LOGOUT
# ======================
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None


# ======================
# LOGIN PAGE
# ======================
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .login-box {
        background: rgba(255,255,255,0.08);
        padding:40px;
        border-radius:16px;
        backdrop-filter: blur(12px);
        width:400px;
        margin:auto;
        margin-top:120px;
        text-align:center;
        box-shadow:0 10px 30px rgba(0,0,0,0.4);
    }
    .title{
        text-align:center;
        font-size:42px;
        font-weight:800;
        margin-top:80px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🚀 AI Doubt Solver</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# DASHBOARD
# ======================
else:

    # TOP BAR
    col1, col2 = st.columns([8,2])

    with col1:
        st.title(f"Welcome {st.session_state.username}")

    with col2:
        st.button("Logout", on_click=logout)

    # ======================
    # LOAD DATASET
    # ======================
    with open("chemistry_sample.json") as f:
        dataset = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    reader = easyocr.Reader(['en'], gpu=False)

    questions = [d["question"] for d in dataset]
    embeddings = model.encode(questions, convert_to_tensor=True)

    # ======================
    # SEARCH
    # ======================
    def exact_match(q):
        scores = [difflib.SequenceMatcher(None, q.lower(), d["question"].lower()).ratio() for d in dataset]
        best = max(scores)
        if best > 0.75:
            return dataset[scores.index(best)]
        return None

    def semantic(q):
        qe = model.encode(q, convert_to_tensor=True)
        sc = util.cos_sim(qe, embeddings)[0]
        idx = sc.argsort(descending=True)[:3]
        return [dataset[int(i)] for i in idx]

    # ======================
    # DISPLAY
    # ======================
    def show(r, i=None):

        st.markdown("---")
        st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")

        st.write("### Question")
        st.write(r.get("question", ""))

        options = r.get("options", {})
        for k, v in options.items():
            if isinstance(v, str) and v.strip():
                st.write(f"{k}) {v}")

        st.success(f"Correct Answer: {r.get('correct_answer','')}")

        if r.get("solution"):
            for line in r["solution"].split("\n"):
                st.write(line)

    # ======================
    # INPUTS
    # ======================
    col1, col2 = st.columns(2)

    with col1:
        q = st.text_input("Ask doubt")

    with col2:
        img = st.file_uploader("Upload image")

    # TEXT
    if q:
        ex = exact_match(q)
        if ex:
            show(ex)
        else:
            for i, r in enumerate(semantic(q)):
                show(r, i)

    # IMAGE
    if img:
        im = Image.open(img)
        st.image(im)

        text = " ".join(reader.readtext(np.array(im), detail=0))
        st.info(text)

        ex = exact_match(text)
        if ex:
            show(ex)
        else:
            for i, r in enumerate(semantic(text)):
                show(r, i)
