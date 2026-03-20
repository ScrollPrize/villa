from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path


def slugify_name(name: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"\s+", "_", str(name).strip())
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug)
    slug = slug.strip("._-")
    if not slug:
        return "run"
    return slug[:max_len]


def build_run_id(
    experiment_name: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
    suffix_length: int = 6,
) -> str:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d")
    run_name = slugify_name(experiment_name)
    run_suffix = slugify_name(suffix, max_len=suffix_length) if suffix is not None else uuid.uuid4().hex[:suffix_length]
    return f"{timestamp}_{run_name}_{run_suffix}"


def build_run_dir(
    root: str | Path,
    experiment_name: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
    suffix_length: int = 6,
) -> Path:
    return Path(root) / build_run_id(
        experiment_name,
        now=now,
        suffix=suffix,
        suffix_length=suffix_length,
    )
