import os
import uuid
from typing import Optional, Tuple
import boto3
from config import settings


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    path = uri[5:]
    bucket, _, key = path.partition("/")
    if not bucket:
        raise ValueError("S3 URI missing bucket")
    return bucket, key


def ensure_s3_uri(path_or_uri: str, upload_prefix: Optional[str] = None, add_unique_suffix: bool = True) -> str:
    """
    If input is a local path, upload it to default bucket/prefix and return s3:// URI.
    If already s3://, return as-is.

    If upload_prefix is provided, use it as the base key prefix. When add_unique_suffix
    is True, a UUID folder is appended to avoid collisions; when False, the prefix
    is used exactly as provided (suitable for per-task deterministic paths).
    """
    if path_or_uri.startswith("s3://"):
        return path_or_uri

    if not os.path.exists(path_or_uri):
        raise FileNotFoundError(f"Local path not found: {path_or_uri}")

    s3 = boto3.client(
        "s3",
        region_name=settings.aws.aws_region,
        aws_access_key_id=settings.aws.aws_access_key_id,
        aws_secret_access_key=settings.aws.aws_secret_access_key,
    )

    bucket = settings.aws.s3_bucket
    if not bucket:
        raise ValueError("AWS s3_bucket is not configured")

    base_prefix = upload_prefix or settings.aws.s3_results_prefix
    if not base_prefix.endswith("/"):
        base_prefix = base_prefix + "/"

    final_prefix = base_prefix
    if add_unique_suffix:
        final_prefix = f"{base_prefix}{uuid.uuid4()}/"

    if os.path.isdir(path_or_uri):
        for root, _, files in os.walk(path_or_uri):
            for fname in files:
                local_fp = os.path.join(root, fname)
                rel = os.path.relpath(local_fp, path_or_uri)
                key = f"{final_prefix}{rel}"
                s3.upload_file(local_fp, bucket, key)
        return f"s3://{bucket}/{final_prefix}"

    # single file upload
    fname = os.path.basename(path_or_uri)
    key = f"{final_prefix}{fname}"
    s3.upload_file(path_or_uri, bucket, key)
    return f"s3://{bucket}/{key}"
