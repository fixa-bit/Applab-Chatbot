<!-- python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
pip install fastapi uvicorn streamlit requests pdfplumber transformers sentence-transformers
pip freeze > requirements.txt -->


## 🔧 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed locally (required for LLaMA 2 model)
- Git, Docker

## 🧠 LLaMA 2 Model via Ollama

This project uses [Ollama](https://ollama.com) to run the LLaMA 2 language model locally.

### 1. Install Ollama

Follow the installation guide here: https://ollama.com/download

### 2. Pull the LLaMA 2 Model

Before running the app, you must pull the model:

```bash
ollama pull llama2


⚠️ Important: The application sends requests to the local Ollama server at http://localhost:11434. Make sure it's running before starting the app.