FROM python:3.12-slim

# Install uv and Tesseract OCR
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (cached layer — only re-runs if lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY config.py main.py ./
COPY src/ src/

# Pre-download ML models into the image so cold starts skip the download
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
print('Models cached.')"

# Cloud Run injects PORT env var (defaults to 8080)
ENV PORT=8080

EXPOSE ${PORT}

# Run via uv so it uses the correct venv
CMD ["uv", "run", "python", "main.py"]
