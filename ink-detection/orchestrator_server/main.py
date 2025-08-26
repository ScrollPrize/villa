from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uuid
from config import settings
from schemas import ModelsResponse, ModelItem, CreateTaskRequest, CreateTaskResponse, TaskStatusResponse
from model_registry import list_models, get_model
from s3_utils import ensure_s3_uri
from workflow_templates import build_inference_workflow
from argo_client import argo_client


app = FastAPI(title="Ink Detection Service", version="0.1.0")


@app.get("/api/v1/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    models = [
        ModelItem(name=m.name, description=m.description) for m in list_models()
    ]
    return ModelsResponse(models=models)


@app.post("/api/v1/tasks", response_model=CreateTaskResponse)
def create_task(req: CreateTaskRequest) -> CreateTaskResponse:
    model = get_model(req.model_selected)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model_selected}")

    # Deterministic task id used for workflow name and S3 prefixes
    task_id = f"{req.model_selected.replace('_','-')}-{uuid.uuid4().hex[:8]}"

    # Arrange S3 prefixes per task
    bucket = settings.aws.s3_bucket
    if not bucket:
        raise HTTPException(status_code=400, detail="S3 bucket is not configured")

    input_prefix = f"{task_id}/input/"

    # Ensure data is on S3 if local path; upload into the per-task input prefix without random suffix
    try:
        source_s3_uri = ensure_s3_uri(req.source_uri, upload_prefix=input_prefix, add_unique_suffix=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build workflow spec (container will read/write using S3 URIs directly)
    wf_spec = build_inference_workflow(
        model=model,
        task_id=task_id,
        input_s3_uri=source_s3_uri,
    )

    try:
        workflow_id = argo_client.submit_workflow(wf_spec)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to submit Argo workflow: {e}")

    return CreateTaskResponse(task_id=workflow_id)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    try:
        wf = argo_client.get_workflow(task_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {e}")

    phase = argo_client.get_workflow_phase(wf)

    # Output location is deterministic; expose when Succeeded
    s3_uri = None
    if phase == "Succeeded":
        s3_uri = f"s3://{settings.aws.s3_bucket}/{task_id}/output/"

    return TaskStatusResponse(task_id=task_id, phase=phase, s3_result_uri=s3_uri, raw_status=wf.get("status"))


def run():
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.service.log_level.lower(),
    )


if __name__ == "__main__":
    run()
