from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from chatbot import get_answer, process_pdf
import shutil

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/chat/")
async def chat(query: str = Form(...)):
    response = get_answer(query)
    return {"response": response}

# @app.post("/upload/")
# async def upload(file: UploadFile = File(...)):
#     file_path = f"temp_{file.filename}"
#     with open(file_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)
#     process_pdf(file_path)
#     return {"status": "success", "message": "PDF processed successfully."}

import tempfile, os

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    process_pdf(tmp_path)
    os.remove(tmp_path)
    return {"status": "success", "message": "PDF processed successfully."}
