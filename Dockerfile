FROM python:3.12-slim

# System deps needed by scikit-learn / xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY model/ ./model/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
RUN chmod +x scripts/entrypoint.sh
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1'

ENTRYPOINT ["scripts/entrypoint.sh"]
