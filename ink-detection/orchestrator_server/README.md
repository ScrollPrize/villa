Ink Detection Service (Orchestrator)

Endpoints
- GET /api/v1/models
- POST /api/v1/tasks
- GET /api/v1/tasks/{task_id}

Environment Variables (nested with __)
- SERVICE__API_PREFIX: default /api/v1
- SERVICE__LOG_LEVEL: default INFO
- AWS__AWS_REGION: default us-east-1
- AWS__AWS_ACCESS_KEY_ID: optional (or AWS_ACCESS_KEY_ID)
- AWS__AWS_SECRET_ACCESS_KEY: optional (or AWS_SECRET_ACCESS_KEY)
- AWS__S3_BUCKET: required (e.g., ink-recognition-tasks)
- ARGO__ARGO_SERVER_BASE_URL: default https://workflows.aws.ash2txt.org/
- ARGO__ARGO_NAMESPACE: default argo
- ARGO__ARGO_AUTH_TOKEN: optional

Local Run
- uv sync
- uv run uvicorn main:app --host 0.0.0.0 --port 8000

Docker
- Build: docker build -t ink-orchestrator:latest .
- Run:
  docker run --rm -p 8000:8000 \
    -e AWS__S3_BUCKET=ink-recognition-tasks \
    -e AWS__AWS_REGION=us-east-1 \
    -e AWS__AWS_ACCESS_KEY_ID=... \
    -e AWS__AWS_SECRET_ACCESS_KEY=... \
    -e ARGO__ARGO_SERVER_BASE_URL=https://workflows.aws.ash2txt.org/ \
    -e ARGO__ARGO_NAMESPACE=argo \
    ink-orchestrator:latest

Example
- Create task:
  curl -s -X POST http://localhost:8000/api/v1/tasks \
    -H 'Content-Type: application/json' \
    -d '{"model_selected":"timesformer","source_uri":"/abs/path/to/data"}'
- Get status:
  curl -s http://localhost:8000/api/v1/tasks/<task_id>
