import streamlit as st
import json
import difflib
import numpy as np
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer, util
import datetime
import time

st.set_page_config(layout="wide")

# -------------------------
# PREMIUM CSS
# -------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#020617);
color:white;
}

header{visibility:hidden;}

.block-container{
padding-top:1rem;
}

.chat-container{
max-width:900px;
margin:auto;
}

.chat-bubble-user{
background:#2563eb;
padding:12px 18px;
border-radius:18px;
margin-bottom:10px;
width:fit-content;
margin-left:auto;
}

.chat-bubble-ai{
background:#1e293b;
padding:14px 18px;
border-radius:18px;
margin-bottom:10px;
width:fit-content;
}

.footer{
text-align:center;
opacity:0.6;
margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# SESSION
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "streak" not in st.session_state:
    st.session_state.streak = 1

if "last_visit" not in st.session_state:
    st.session_state.last_visit = str(datetime.date.today())

today = str(datetime.date.today())

if st.session_state.last_visit != today:
    st.session_state.streak += 1
    st.session_state.last_visit = today

# -------------------------
# LOAD USERS
# -------------------------

with open("users.json") as f:
    users = json.load(f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# -------------------------
# LOGIN
# -------------------------

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

# -------------------------
# LOGIN PAGE
# -------------------------

if not st.session_state.logged_in:

    st.title("🚀 AI Doubt Solver")

    st.markdown(
    "<h3 style='text-align:center;'>Welcome 👋 This is the platform for all your doubts</h3>",
    unsafe_allow_html=True
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if login(username,password):
            st.rerun()

        else:
            st.error("Invalid credentials")

# -------------------------
# MAIN APP
# -------------------------

else:

    st.title("🚀 AI Doubt Solver")

    st.write(f"Welcome **{st.session_state.username}**")

    st.success(f"🔥 Study Streak: {st.session_state.streak} days")

    if st.button("Logout"):
        logout()
        st.rerun()

    # -------------------------
    # LOAD DATASET
    # -------------------------

    with open("chemistry_sample.json") as f:
        dataset = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    reader = easyocr.Reader(['en'],gpu=False)

    questions=[d["question"] for d in dataset]

    embeddings=model.encode(questions,convert_to_tensor=True)

    # -------------------------
    # SEARCH
    # -------------------------

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

    # -------------------------
    # CHAT HISTORY
    # -------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # -------------------------
    # CHAT INPUT
    # -------------------------

    prompt = st.chat_input("Type your doubt here...")

    uploaded_image = st.file_uploader("Upload doubt image", label_visibility="collapsed")

    # -------------------------
    # TEXT DOUBT
    # -------------------------

    if prompt:

        st.session_state.messages.append(
        {"role":"user","content":prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Solving your doubt..."):

                time.sleep(1)

                ex = exact_match(prompt)

                if ex:

                    answer = ex["solution"]

                else:

                    results = semantic(prompt)

                    answer = results[0]["solution"]

                st.markdown(answer)

        st.session_state.messages.append(
        {"role":"assistant","content":answer}
        )

    # -------------------------
    # IMAGE DOUBT
    # -------------------------

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(image)

        with st.spinner("Reading image..."):

            text=" ".join(reader.readtext(np.array(image),detail=0))

        st.session_state.messages.append(
        {"role":"user","content":text}
        )

        with st.chat_message("assistant"):

            ex=exact_match(text)

            if ex:

                answer=ex["solution"]

            else:

                results=semantic(text)

                answer=results[0]["solution"]

            st.markdown(answer)

        st.session_state.messages.append(
        {"role":"assistant","content":answer}
        )

    # -------------------------
    # FEEDBACK
    # -------------------------

    st.divider()

    st.subheader("⭐ Feedback")

    rating=st.slider("Rate platform",1,5)

    comment=st.text_area("Write feedback")

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

        st.success("Thank you for feedback!")

    # -------------------------
    # FOOTER
    # -------------------------

    st.markdown(
    "<div class='footer'>Created by Om | © 2026 AI Doubt Solver</div>",
    unsafe_allow_html=True
    )
