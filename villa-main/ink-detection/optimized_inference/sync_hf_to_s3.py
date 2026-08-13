import os
import uuid
import argparse
import logging
import concurrent.futures
from typing import Tuple, List

import boto3
from botocore.config import Config
from huggingface_hub import snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("optimized_inference.sync_hf_to_s3")


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[len("s3://") :]
    parts = path.split("/", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def discover_files(root_dir: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
    if not files:
        raise RuntimeError(f"No files found under {root_dir}")
    return files


def upload_files_to_s3(
    files: List[str],
    local_root: str,
    bucket: str,
    base_prefix: str,
) -> None:
    config = Config(max_pool_connections=32, retries={"max_attempts": 3, "mode": "adaptive"})
    s3_client = boto3.client("s3", config=config)

    def _to_s3_key(local_path: str) -> str:
        rel = os.path.relpath(local_path, start=local_root)
        rel_posix = rel.replace(os.sep, "/")
        if base_prefix:
            return f"{base_prefix.rstrip('/')}/{rel_posix}"
        return rel_posix

    def _upload_one(path: str) -> str:
        key = _to_s3_key(path)
        s3_client.upload_file(path, bucket, key)
        return key

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_upload_one, p): p for p in files}
        uploaded = 0
        for future in concurrent.futures.as_completed(futures):
            src = futures[future]
            try:
                key = future.result()
                uploaded += 1
                if uploaded % 50 == 0:
                    logger.info(f"Uploaded {uploaded} files. Latest: s3://{bucket}/{key}")
            except Exception as e:
                logger.error(f"Failed to upload {src}: {e}")
                raise
        logger.info(f"Uploaded {uploaded} files to s3://{bucket}/{base_prefix}")


def resolve_default_prefix(hf_repo: str) -> str:
    # Use the repository name (after '/') as folder name by default
    # e.g., 'org/model-name' -> 'model-name'
    if "/" in hf_repo:
        return hf_repo.split("/", 1)[1]
    return hf_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror a Hugging Face repo into an S3 bucket/prefix.")
    parser.add_argument("--hf-repo", required=True, help="Hugging Face repo id, e.g. 'org/model-name'")
    parser.add_argument(
        "--s3-uri",
        required=True,
        help="Destination S3 URI, e.g. 's3://my-bucket' or 's3://my-bucket/some/prefix'",
    )
    parser.add_argument(
        "--subfolder",
        default=None,
        help=(
            "Optional subfolder under the provided S3 URI. Defaults to the repo name (text after '/'). "
            "For example, 'org/model' -> 'model'."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp",
        help="Local working directory for temporary download (default: /tmp)",
    )
    args = parser.parse_args()

    repo_id = args.hf_repo.strip()
    s3_uri = args.s3_uri.strip()
    bucket, prefix = parse_s3_uri(s3_uri)

    if args.subfolder is not None and args.subfolder.strip() != "":
        base_prefix = f"{prefix.rstrip('/')}/{args.subfolder.strip()}" if prefix else args.subfolder.strip()
    else:
        default_folder = resolve_default_prefix(repo_id)
        base_prefix = f"{prefix.rstrip('/')}/{default_folder}" if prefix else default_folder

    work_dir = os.path.join(args.work_dir, f"hf_mirror_{uuid.uuid4()}")
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"Downloading Hugging Face repo '{repo_id}' to {work_dir}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=work_dir,
        local_dir_use_symlinks=False,
    )
    logger.info(f"Repo downloaded to {local_dir}")

    logger.info("Discovering files to upload...")
    files = discover_files(local_dir)
    logger.info(f"Found {len(files)} files. Beginning upload to s3://{bucket}/{base_prefix}")

    upload_files_to_s3(files, local_root=local_dir, bucket=bucket, base_prefix=base_prefix)

    logger.info(
        f"Completed mirroring of '{repo_id}' to s3://{bucket}/{base_prefix} (source: {local_dir})"
    )


if __name__ == "__main__":
    main()


