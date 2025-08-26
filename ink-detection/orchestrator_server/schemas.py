from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class ModelItem(BaseModel):
    name: str
    description: str


class ModelsResponse(BaseModel):
    models: List[ModelItem]


class CreateTaskRequest(BaseModel):
    model_selected: str = Field(description="Model name, e.g. timesformer")
    source_uri: str = Field(description="s3:// path or local path to data. Local will be uploaded to S3.")


class CreateTaskResponse(BaseModel):
    task_id: str
    status_url: Optional[HttpUrl] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    phase: Optional[str]
    s3_result_uri: Optional[str] = None
    raw_status: Optional[dict] = None
