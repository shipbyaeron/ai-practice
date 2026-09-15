import streamlit as st
from tool_calling import run_agent

st.title("Crypto Research Assistant")

st.subheader("What do you want to ask today?")

question = st.text_input("Question ⬇")

if st.button("Search"):
    answer, sources_list = run_agent(question, return_resources=True)
    if "The program has stopped" in answer:
        st.error(f"{answer}")
    else:
        st.success(f"Searched successfully!")
        st.write(f"{answer}")
        with st.expander("Click here to view the article sources"):
            clean_sources_list = [source.replace("\n", " ") for source in sources_list]
            source_string = " | ".join(clean_sources_list)
            st.write(f"Sources: {source_string}")