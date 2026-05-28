# Dockerfile
# XAItan Module - Production-ready container

# === Stage 1: Builder ===
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with hashes verification
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY demo/ ./demo/
COPY data/ ./data/
COPY models/ ./models/

# === Stage 2: Runtime ===
FROM python:3.10-slim as runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --from=builder --chown=appuser:appuser /app/src ./src
COPY --from=builder --chown=appuser:appuser /app/data ./data
COPY --from=builder --chown=appuser:appuser /app/models ./models

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Mount model as read-only, restrict filesystem
RUN chmod -R 555 ./models ./data && \
    chmod -R 755 ./src && \
    mkdir -p /tmp/logs && chmod 777 /tmp/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
