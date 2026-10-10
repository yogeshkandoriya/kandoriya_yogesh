FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss/scikit-learn runtime.
# curl is used by Render's health checks / debugging.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Keep RAM low on Render free (512MB): single-threaded math libs,
# no tokenizer fork warnings, no HF telemetry, unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=0 \
    PORT=10000

# CPU-only torch first (default wheels bundle CUDA and are ~4GB larger).
# sentence-transformers will reuse this install instead of pulling CUDA torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects $PORT at runtime (typically 10000). Default keeps local runs working.
EXPOSE 10000

# Single uvicorn worker = lowest RAM. Do NOT raise workers on free tier.
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
