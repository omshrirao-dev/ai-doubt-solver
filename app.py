import streamlit as st
import json
import difflib
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")

# ======================
# 🎨 THEME SYSTEM
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

bg = themes[st.session_state.theme_index]

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    background-size:400% 400%;
    animation: gradientBG 15s ease infinite;
}}

@keyframes gradientBG{{
0%{{background-position:0% 50%;}}
50%{{background-position:100% 50%;}}
100%{{background-position:0% 50%;}}
}}

.login-box {{
    background: rgba(255,255,255,0.08);
    padding:40px;
    border-radius:16px;
    backdrop-filter: blur(12px);
    width:400px;
    margin:auto;
    margin-top:120px;
    text-align:center;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}}

.title{{
    text-align:center;
    font-size:42px;
    font-weight:800;
    margin-top:80px;
}}

.card {{
    background: rgba(255,255,255,0.07);
    padding:20px;
    border-radius:16px;
    margin-bottom:20px;
    backdrop-filter: blur(12px);
    box-shadow:0 8px 20px rgba(0,0,0,0.4);
}}

</style>
""", unsafe_allow_html=True)

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
# LOGIN
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

    # HEADER
    col1, col2, col3 = st.columns([7,1,2])

    with col1:
        st.title(f"Welcome {st.session_state.username}")

    with col2:
        st.button("🎨", on_click=next_theme)

    with col3:
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

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")
        st.write(r.get("question", ""))

        options = r.get("options", {})
        for k, v in options.items():
            if isinstance(v, str) and v.strip():
                st.write(f"{k}) {v}")

        st.success(f"Correct Answer: {r.get('correct_answer','')}")

        if r.get("solution"):
            for line in r["solution"].split("\n"):
                st.write(line)

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # INPUTS
    # ======================
    col1, col2 = st.columns(2)

    with col1:
        q = st.text_input("Ask doubt")

    with col2:
        img = st.file_uploader("Upload image")

    # TEXT FLOW
    if q:
        ex = exact_match(q)
        if ex:
            show(ex)
        else:
            for i, r in enumerate(semantic(q)):
                show(r, i)

    # IMAGE FLOW
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

    # ======================
    # FOOTER
    # ======================
    st.markdown("---")

    col1, col2 = st.columns([2,3])

    with col1:
        st.markdown("### 🚀 AI Doubt Solver")
        st.markdown("**Created by DM sir**")
        st.markdown("📩 Contact: dmsir.ai@gmail.com")

    with col2:
        feedback = st.text_area("💬 Feedback / Suggestions")

        if st.button("Submit Feedback"):
            if feedback.strip():
                with open("feedback.txt","a") as f:
                    f.write(f"{st.session_state.username}: {feedback}\n")
                st.success("Thank you for feedback 🙌")
            else:
                st.warning("Write something first")

    st.markdown(
        "<center>© 2026 AI Doubt Solver — All rights reserved</center>",
        unsafe_allow_html=True
    )
