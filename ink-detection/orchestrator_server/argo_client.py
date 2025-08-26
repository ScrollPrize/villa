import requests
from typing import Any, Dict, Optional
from config import settings


class ArgoClient:
    def __init__(self, base_url: Optional[str] = None, namespace: Optional[str] = None, token: Optional[str] = None):
        self.base_url = (base_url or str(settings.argo.argo_server_base_url)).rstrip("/")
        self.namespace = namespace or settings.argo.argo_namespace
        self.token = token or settings.argo.argo_auth_token

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def submit_workflow(self, workflow_spec: Dict[str, Any]) -> str:
        # POST /api/v1/workflows/{namespace}
        url = f"{self.base_url}/api/v1/workflows/{self.namespace}"
        resp = requests.post(url, json=workflow_spec, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # workflow metadata.name is the workflow id
        return data["metadata"]["name"]

    def get_workflow(self, workflow_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/workflows/{self.namespace}/{workflow_name}"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_workflow_phase(self, workflow: Dict[str, Any]) -> Optional[str]:
        return workflow.get("status", {}).get("phase")

    def get_artifact_s3_uri(self, workflow: Dict[str, Any]) -> Optional[str]:
        status = workflow.get("status", {})
        outputs = status.get("outputs") or {}
        artifacts = outputs.get("artifacts") or []
        for art in artifacts:
            s3 = art.get("s3")
            if s3:
                bucket = s3.get("bucket")
                key = s3.get("key")
                if bucket and key:
                    return f"s3://{bucket}/{key}"
        return None


argo_client = ArgoClient()
