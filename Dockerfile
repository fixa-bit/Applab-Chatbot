# FROM python:3.10-slim

# WORKDIR /app

# # Avoid interactive prompts
# ENV PYTHONUNBUFFERED=1

# # Copy requirements first for caching
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]


# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy application files
#COPY . /app

# Copy only necessary files
COPY ./app/backend.py ./
COPY ./app/app.py ./
COPY ./app/chatbot.py ./
COPY ./app/.env ./
COPY ./app/pdf_utils.py ./
# COPY ./app/backend.py ./app
# COPY ./app/app.py ./app
# COPY ./app/chatbot.py ./app
# COPY ./app/.env ./app
# COPY ./app/.env .
# COPY ./app/pdf_utils.py ./app
COPY requirements.txt .




# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Set environment variables for Streamlit (optional)
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false

# Start both FastAPI and Streamlit using a shell script
CMD ["bash", "-c", "uvicorn backend:app --host 0.0.0.0 --port 8000 & streamlit run app.py"]
