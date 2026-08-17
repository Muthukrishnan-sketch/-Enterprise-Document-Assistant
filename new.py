
import streamlit as st

st.title("My RAG Test App")

st.write("Streamlit is working")

name = st.text_input("Enter your name")

if name:
    st.success(f"Hello {name}")