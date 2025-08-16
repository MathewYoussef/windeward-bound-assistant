from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from wba.local_rag import LocalRAG

rag = LocalRAG()

st.title("Windeward Bound Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask the assistant"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply, _ = rag.answer(
        prompt,
        history=st.session_state.messages[-4:],  # last 2 prior turns
        sticky_topic=None,
        top_k=8,
    )
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

