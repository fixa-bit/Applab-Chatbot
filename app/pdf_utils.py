import pdfplumber
from io import BytesIO

def extract_text_from_pdf(file_bytes):
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
