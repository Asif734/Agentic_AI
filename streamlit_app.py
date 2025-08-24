#streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

st.set_page_config(page_title="Public RAG Agent", layout="wide")
st.title("Public Knowledge RAG Agent")
st.markdown("Upload text or PDF, and query your knowledge base in real-time.")

# ---------------- Add Text ----------------
st.header("Add Text")
with st.form("text_form"):
    text_input = st.text_area("Enter text to add to RAG database:")
    source_input = st.text_input("Source (optional)", value="manual")
    submitted_text = st.form_submit_button("Add Text")
    if submitted_text:
        if text_input.strip():
            resp = requests.post(f"{API_URL}/add-text", data={"text": text_input, "source": source_input})
            st.success(resp.json().get("message"))
        else:
            st.warning("Please enter some text.")

# ---------------- Upload PDF ----------------
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    resp = requests.post(f"{API_URL}/add-pdf", files={"file": uploaded_file})
    st.success(resp.json().get("message"))

# ---------------- Query ----------------
st.header("Query RAG Agent")
with st.form("query_form"):
    user_query = st.text_input("Ask a question:")
    submitted_query = st.form_submit_button("Get Answer")
    if submitted_query:
        if user_query.strip():
            resp = requests.post(f"{API_URL}/query", data={"query": user_query})
            data = resp.json()
            if data.get("status") == "success":
                st.markdown(f"**Answer:** {data.get('answer')}")
            else:
                st.error(data.get("message"))
        else:
            st.warning("Please type a query.")
