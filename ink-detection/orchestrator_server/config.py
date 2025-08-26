from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl, Field, AliasChoices
from typing import Optional


class ServiceSettings(BaseSettings):
    api_prefix: str = "/api/v1"
    log_level: str = Field(default="INFO")
    model_config = SettingsConfigDict(extra="ignore")


class AWSSettings(BaseSettings):
    aws_access_key_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AWS__AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
        description="AWS Access Key ID",
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AWS__AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
        description="AWS Secret Access Key",
    )
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS__AWS_REGION", "AWS_REGION"),
        description="AWS Region",
    )
    s3_bucket: str = Field(
        default="ink-recognition-tasks",
        validation_alias=AliasChoices("AWS__S3_BUCKET", "S3_BUCKET"),
        description="Default bucket for results",
    )
    s3_results_prefix: str = Field(
        default="results",
        validation_alias=AliasChoices("AWS__S3_RESULTS_PREFIX", "S3_RESULTS_PREFIX"),
        description="Prefix under the bucket for results",
    )
    model_config = SettingsConfigDict(extra="ignore")


class ArgoSettings(BaseSettings):
    # We'll support Argo Server REST.
    argo_server_base_url: AnyUrl = Field(
        default="https://workflows.aws.ash2txt.org/", description="Base URL of Argo Workflows server"
    )
    argo_namespace: str = Field(default="argo", description="Namespace where workflows run")
    # Token for Argo server auth (e.g., sso or service account). If None, assume no auth.
    argo_auth_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ARGO__ARGO_AUTH_TOKEN", "ARGO_AUTH_TOKEN", "ARGO_AUTH_TOKEN_VENV"),
        description="Bearer token for Argo server"
    )
    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    service: ServiceSettings = ServiceSettings()
    aws: AWSSettings = AWSSettings()
    argo: ArgoSettings = ArgoSettings()
    # Support nested env like AWS__S3_BUCKET and ARGO__ARGO_SERVER_BASE_URL
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")


settings = Settings()
