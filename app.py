def show(r,i=None):

    st.subheader("Exact Match" if i is None else f"Similar Question {i+1}")

    st.write("### Question")
    st.write(r["question"])

    if "question_image" in r and os.path.exists("structures/"+r["question_image"]):
        st.image("structures/"+r["question_image"])

    # ⭐ FIXED OPTIONS LOGIC
    show_options=False
    for v in r["options"].values():
        if isinstance(v,str) and v.strip():
            show_options=True
        elif isinstance(v,dict) and "image" in v:
            show_options=True

    if show_options:
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

    if "solution_image" in r and os.path.exists("structures/"+r["solution_image"]):
        st.image("structures/"+r["solution_image"])

    elif r["solution"]:
        for l in r["solution"].split("\n"):
            st.write(l)

    st.markdown("---")
