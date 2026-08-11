# ---- LLMGuardian: Render-compatible Dockerfile ----
# Expects this repo layout:
#   app/api.py           <- FastAPI entrypoint (this file's directory)
#   requirements.txt
#   Dockerfile
#   models/               <- one level ABOVE app/, matches api.py's
#                            MODELS_DIR = BASE_DIR.parent / "models"
#
# If your repo layout differs, adjust the COPY paths below to match.

FROM python:3.10-slim

# Prevents Python from writing .pyc files / buffering stdout (cleaner logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install system deps some ML wheels occasionally need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY app ./app
COPY models ./models

WORKDIR /code/app

# Render injects $PORT at runtime — do not hardcode a port here.
CMD uvicorn api:app --host 0.0.0.0 --port $PORT