
import streamlit as st
import requests
from dotenv import load_dotenv
import os
load_dotenv()
# Get API URL from environment, default to localhost for development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(page_title="General & Document Chatbot")
st.title("📄 Conversational Chatbot with PDF Support")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf:
    with st.spinner("Uploading and processing..."):
        #res = requests.post("http://localhost:8000/upload/", files={"file": (pdf.name, pdf, "application/pdf")})
        res = requests.post(f"{API_BASE_URL}/upload/", files={"file": (pdf.name, pdf, "application/pdf")})
        st.success(res.json().get("message", "Uploaded"))

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        res = requests.post(f"{API_BASE_URL}/chat/", data={"query": query})
        #res = requests.post("http://localhost:8000/chat/", data={"query": query})
        answer = res.json().get("response", "Sorry, no answer.")
    
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
