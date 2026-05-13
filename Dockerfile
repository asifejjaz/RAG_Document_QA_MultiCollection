# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (CPU-only base)
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Local embeddings (BGE-M3) require sentence-transformers which pulls PyTorch.
# On GPU builds this may download NVIDIA/CUDA libraries and slow the build.
# Uncomment only if you need local embeddings:
# RUN pip install --no-cache-dir -r requirements-local.txt

# Copy application code
COPY . .

# Set working directory to where app.py is
WORKDIR /app/frontend

# Create necessary directories and non-root user (Issue 10 / prod hardening)
RUN mkdir -p /state/reports && \
    mkdir -p /app/sessions && \
    chmod -R 755 /state && \
    chmod -R 755 /app/sessions && \
    useradd --create-home --uid 10001 --shell /bin/bash raguser && \
    chown -R raguser:raguser /app /state

# Expose ports
EXPOSE 8507
EXPOSE 8000

# Environment variables (can be overridden in docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_ROOT=/data \
    STATE_ROOT=/state \
    PYTHONPATH=/app

USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8507/_stcore/health || exit 1

# Default command (run Streamlit)
CMD ["streamlit", "run", "app.py", "--server.port=8507", "--server.address=0.0.0.0"]
