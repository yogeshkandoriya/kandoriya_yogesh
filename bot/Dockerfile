FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss/scikit-learn runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (default wheels bundle CUDA and are ~4GB larger)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Docker Spaces route traffic to app_port (7860)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT}"]
