<!-- python -m venv venv
venv\Scripts\activate
pip freeze > requirements.txt -->
# 🤖 Conversational Chatbot with PDF Support

## 📘 Overview
This project is a conversational chatbot that supports both general queries and document-based questions. Users can upload PDF documents, and the chatbot will answer questions based on the content of those documents. It uses the **LLaMA 2** model via **Ollama** for generating responses and **FAISS** for document similarity search.

## ✨ Features
- **General Chat**: Answer questions on a wide range of topics.
- **PDF Support**: Upload and process PDF documents for context-aware answers.

## 🔧 Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com) (run `ollama pull llama2`)
- Docker (optional)
- Python 3.8+
## 🧠 LLaMA 2 Model via Ollama

This project uses [Ollama](https://ollama.com) to run the LLaMA 2 language model locally.

### 1. Install Ollama

Follow the installation guide here: https://ollama.com/download

### 2. Pull the LLaMA 2 Model

Before running the app, you must pull the model:

ollama pull llama2
## Installation

git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt

# Create .env file
echo "API_BASE_URL=http://localhost:8000" > .env
echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env

## 2. Set up environment variables

- Create a `.env` file in the root directory.
- Add the following variables (adjust as needed):


  API_BASE_URL=http://backend:8000
  "HF_HOME" = "D:/cache"
  HF_TOKEN=...
  HF_API_KEY=...
  OLLAMA_BASE_URL=http://host.docker.internal:11434

### 1. Activate Virtual Environment
venv\Scripts\activate
## 🚀 Running the Application (Development Mode)


```bash


1.	Start the FastAPI backend:
uvicorn backend:app --reload --host 127.0.0.1 --port 8000
2.	Start the Streamlit frontend in a separate terminal:
streamlit run app.py
Access the application at http://localhost:8501
3. Access the Application
Open your browser and go to:
http://localhost:8501











⚠️ Important: The application sends requests to the local Ollama server at http://localhost:11434. Make sure it's running before starting the app.
