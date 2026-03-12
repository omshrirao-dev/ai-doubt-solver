import streamlit as st
import json
import difflib
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer, util
import datetime

st.set_page_config(layout="wide")

# ======================
# THEME SYSTEM
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

# ======================
# CSS + SOUND
# ======================

st.markdown(f"""
<style>

.stApp {{
background:{bg};
background-size:400% 400%;
animation:gradientBG 15s ease infinite;
}}

@keyframes gradientBG{{
0%{{background-position:0% 50%;}}
50%{{background-position:100% 50%;}}
100%{{background-position:0% 50%;}}
}}

.title{{
text-align:center;
font-size:42px;
font-weight:800;
margin-top:60px;
}}

.login-box {{
background:rgba(255,255,255,0.08);
padding:40px;
border-radius:16px;
backdrop-filter:blur(12px);
width:420px;
margin:auto;
margin-top:40px;
text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.4);
}}

.card {{
background:rgba(255,255,255,0.07);
padding:20px;
border-radius:16px;
margin-bottom:20px;
backdrop-filter:blur(12px);
}}

.typewriter {{
overflow:hidden;
border-right:.15em solid white;
white-space:nowrap;
margin:0 auto;
letter-spacing:.08em;
animation:typing 5s steps(40,end), blink .75s step-end infinite;
width:fit-content;
}}

@keyframes typing {{
from {{width:0}}
to {{width:100%}}
}}

@keyframes blink {{
from,to {{border-color:transparent}}
50% {{border-color:white}}
}}

</style>

<audio id="clickSound" src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>
<script>
document.addEventListener('click', function() {{
document.getElementById('clickSound').play();
}});
</script>

""", unsafe_allow_html=True)

# ======================
# LOAD USERS
# ======================

with open("users.json") as f:
    users = json.load(f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# ======================
# STREAK SYSTEM
# ======================

if "last_visit" not in st.session_state:
    st.session_state.last_visit = None

if "streak" not in st.session_state:
    st.session_state.streak = 0

today = datetime.date.today()

if st.session_state.last_visit != today:

    if st.session_state.last_visit == today - datetime.timedelta(days=1):
        st.session_state.streak += 1
    else:
        st.session_state.streak = 1

    st.session_state.last_visit = today

# ======================
# LOGIN FUNCTIONS
# ======================

def login(username,password):

    for u in users:

        if u["username"]==username and u["password"]==password:
            st.session_state.logged_in=True
            st.session_state.username=username
            return True

    return False


def logout():
    st.session_state.logged_in=False
    st.session_state.username=None

# ======================
# LOGIN PAGE
# ======================

if not st.session_state.logged_in:

    st.markdown("<div class='title'>🚀 AI Doubt Solver</div>",unsafe_allow_html=True)

    st.markdown("""
    <center>
    <h3 class='typewriter'>Welcome 👋 This is the platform for all your doubts</h3>
    </center>
    """,unsafe_allow_html=True)

    st.markdown("<div class='login-box'>",unsafe_allow_html=True)

    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if st.button("Login"):

        if login(username,password):
            st.rerun()

        else:
            st.error("Invalid credentials")

    st.markdown("</div>",unsafe_allow_html=True)

# ======================
# DASHBOARD
# ======================

else:

    col1,col2,col3=st.columns([7,1,2])

    with col1:
        st.title(f"Welcome {st.session_state.username}")
        st.success(f"🔥 Streak : {st.session_state.streak} days")

    with col2:
        st.button("🎨",on_click=next_theme)

    with col3:
        st.button("Logout",on_click=logout)

    # ======================
    # LOAD DATASET
    # ======================

    with open("chemistry_sample.json") as f:
        dataset=json.load(f)

    model=SentenceTransformer('all-MiniLM-L6-v2')
    reader=easyocr.Reader(['en'],gpu=False)

    questions=[d["question"] for d in dataset]

    embeddings=model.encode(questions,convert_to_tensor=True)

    def exact_match(q):

        scores=[difflib.SequenceMatcher(None,q.lower(),d["question"].lower()).ratio() for d in dataset]

        best=max(scores)

        if best>0.75:
            return dataset[scores.index(best)]

        return None


    def semantic(q):

        qe=model.encode(q,convert_to_tensor=True)

        sc=util.cos_sim(qe,embeddings)[0]

        idx=sc.argsort(descending=True)[:3]

        return [dataset[int(i)] for i in idx]


    def show(r,i=None):

        st.markdown("<div class='card'>",unsafe_allow_html=True)

        st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")

        st.write(r.get("question",""))

        options=r.get("options",{})

        for k,v in options.items():
            if isinstance(v,str) and v.strip():
                st.write(f"{k}) {v}")

        st.success(f"Correct Answer: {r.get('correct_answer','')}")

        if r.get("solution"):
            for line in r["solution"].split("\n"):
                st.write(line)

        st.markdown("</div>",unsafe_allow_html=True)

    # ======================
    # CHAT INPUT
    # ======================

    st.markdown("---")

    q=st.chat_input("💬 Ask your doubt")

    img=st.file_uploader("Upload doubt image")

    if q:

        with st.spinner("Solving your doubt..."):

            ex=exact_match(q)

            if ex:
                show(ex)

            else:
                for i,r in enumerate(semantic(q)):
                    show(r,i)


    if img:

        with st.spinner("Reading image..."):

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

    # ======================
    # FEEDBACK SYSTEM
    # ======================

    st.markdown("---")

    st.subheader("⭐ Give Feedback")

    rating=st.slider("Rate the platform",1,5)

    comment=st.text_area("Your feedback")

    if st.button("Submit Feedback"):

        data={
        "user":st.session_state.username,
        "rating":rating,
        "comment":comment,
        "time":str(datetime.datetime.now())
        }

        try:
            with open("feedback.json") as f:
                feedback=json.load(f)
        except:
            feedback=[]

        feedback.append(data)

        with open("feedback.json","w") as f:
            json.dump(feedback,f,indent=4)

        st.success("Thank you for your feedback!")

    # ======================
    # FOOTER
    # ======================

    st.markdown("---")

    st.markdown("### 🚀 AI Doubt Solver")

    st.markdown("Created by **Om**")

    st.markdown("📩 Contact: om.ai@gmail.com")

    st.markdown("<center>© 2026 AI Doubt Solver - All rights reserved</center>",unsafe_allow_html=True)
