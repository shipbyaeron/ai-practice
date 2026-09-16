import streamlit as st
from tool_calling import run_agent

st.title("Crypto Research Assistant")

st.subheader("What do you want to ask today?")

question = st.text_input("Question ⬇")

error_list = ["refusal", "pause_turn", "model_context_window_exceeded"]

if st.button("Search"):
    if question.strip():
        answer, sources_list = run_agent(question, return_resources=True)
        if answer == "max_tokens":
            st.error(f"Sorry :( I hit my token limit for this question. Pls try asking another one.")
        elif answer in error_list:
            st.error(f"Searched failed :( Something went wrong during my searching session. Pls try again later!")
        elif "There has been too many loops already :(" in answer:
            st.error(f"Searched failed :( I tried searching for answer more than 10 times but couldn't find it. Pls ask another question.")
        elif "do not have enough context" in answer:
            st.info(f"This question is outside the scope of the provided sources. Pls try asking another question.")
        else:
            st.success(f"Searched successfully!")
            st.write(f"{answer}")
            if sources_list:  
                with st.expander("Click here to view the article sources"):
                    clean_sources_list = [source.replace("\n", " ") for source in sources_list]
                    source_string = " | ".join(clean_sources_list)
                    st.write(f"Sources: {source_string}")
    else:
        st.warning(f"Oops! You forgot to type the question. Pls ask me a question first before searching.")
    