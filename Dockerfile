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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set working directory to where app.py is
WORKDIR /app/frontend

# Create necessary directories
RUN mkdir -p /state/reports && \
    mkdir -p /app/frontend/sessions && \
    chmod -R 755 /state && \
    chmod -R 755 /app/frontend/sessions

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Environment variables (can be overridden in docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_ROOT=/data \
    STATE_ROOT=/state \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (run Streamlit)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]