# ---- Stage 1: Base image ----
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.pt

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Stage 2: Install Python dependencies (slim, inference only) ----
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ---- Stage 3: Copy application code ----
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY params.yaml .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
