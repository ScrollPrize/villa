FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default envs (override in runtime or Kubernetes manifests)
ENV AWS__S3_BUCKET="ink-recognition-tasks" \
    AWS__AWS_REGION="us-east-1" \
    AWS__AWS_ACCESS_KEY_ID="" \
    AWS__AWS_SECRET_ACCESS_KEY="" \
    SERVICE__API_PREFIX="/api/v1" \
    SERVICE__LOG_LEVEL="INFO" \
    ARGO__ARGO_SERVER_BASE_URL="https://workflows.aws.ash2txt.org/" \
    ARGO__ARGO_NAMESPACE="argo"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
