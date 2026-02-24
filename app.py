import streamlit as st
st.write("APP STARTED")
def show(r, i=None):

    if i is None:
        st.subheader("Exact Match")
    else:
        st.subheader(f"Similar Question {i+1}")

    # Question
    st.write("### Question")
    st.write(r["question"])

    # Question image
    if "question_image" in r:
        path = "structures/" + r["question_image"]
        if os.path.exists(path):
            st.image(path)

    # ⭐ SAFE OPTIONS CHECK
    show_options = False

    for v in r.get("options", {}).values():
        if isinstance(v, str) and v.strip():
            show_options = True
        elif isinstance(v, dict) and "image" in v:
            show_options = True

    if show_options:
        st.write("### Options")

        for k, v in r.get("options", {}).items():

            if isinstance(v, str) and v.strip():
                st.write(f"{k}) {v}")

            elif isinstance(v, dict) and "image" in v:
                st.write(f"{k})")
                img_path = "structures/" + v["image"]
                if os.path.exists(img_path):
                    st.image(img_path)

    # Answer
    st.success(f"Correct Answer: {r['correct_answer']}")

    # Solution
    st.write("### Solution")

    if "solution_image" in r:
        sol_path = "structures/" + r["solution_image"]
        if os.path.exists(sol_path):
            st.image(sol_path)
        else:
            st.warning(f"Missing image: {sol_path}")

    elif r.get("solution"):
        for line in r["solution"].split("\n"):
            st.write(line)

    st.markdown("---")


