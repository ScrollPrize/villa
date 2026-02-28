from __future__ import annotations

import hashlib
import os
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _env_list(value: str, default="*") -> list[str]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts if parts else [default]


DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/data/images")).resolve()
DB_PATH = Path(os.getenv("DATABASE_PATH", "/app/data/preferences.db")).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CATALOG_CACHE_TTL_SECONDS = 30
MANIFEST_CACHE_TTL_SECONDS = 30

_catalog_cache: dict[str, tuple[float, list[dict[str, str | list[str]]]]] = {}
_manifest_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}

app = FastAPI(title="Pairwise Preference Collector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_list(os.getenv("ALLOWED_ORIGINS", "*")),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@dataclass
class ImagePair:
    pair_id: str
    fold: str
    sample: str
    left_image: str
    right_image: str
    left_display: str
    right_display: str


class PreferenceIn(BaseModel):
    user_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    fold: str = Field(min_length=1)
    sample: str = Field(min_length=1)
    left_image: str = Field(min_length=1)
    right_image: str = Field(min_length=1)
    preference: Literal["left", "right"]


class PreferenceOut(BaseModel):
    id: int
    message: str


class PairOut(BaseModel):
    pair_id: str
    fold: str
    sample: str
    left: dict
    right: dict



def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_id TEXT NOT NULL,
                pair_id TEXT NOT NULL,
                fold TEXT NOT NULL,
                sample TEXT NOT NULL,
                left_image TEXT NOT NULL,
                right_image TEXT NOT NULL,
                preference TEXT NOT NULL CHECK(preference IN ('left','right'))
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pair_id ON preference_logs(pair_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_id ON preference_logs(user_id)"
        )
        conn.commit()


def _collect_image_files(sample_dir: Path) -> list[Path]:
    return sorted(
        [
            file
            for file in sample_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def _collect_samples(fold_filter: str | None, sample_filter: str | None) -> list[tuple[str, str, list[Path]]]:
    samples: list[tuple[str, str, list[Path]]] = []

    if not DATASET_ROOT.exists():
        return samples

    for fold_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()):
        if fold_filter and fold_dir.name != fold_filter:
            continue

        for sample_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            if sample_filter and sample_dir.name != sample_filter:
                continue
            images = _collect_image_files(sample_dir)
            if len(images) >= 2:
                samples.append((fold_dir.name, sample_dir.name, images))

    return samples


def _load_manifest_cached(sample_dir: Path) -> list[dict[str, object]]:
    if not sample_dir.exists():
        return []

    manifest_path = sample_dir / "pairs.jsonl"
    key = str(sample_dir)
    now = time.time()
    expires, rows = _manifest_cache.get(key, (0.0, []))
    if now < expires and rows:
        return rows

    cached: list[dict[str, object]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    cached.append(payload)

    _manifest_cache[key] = (now + MANIFEST_CACHE_TTL_SECONDS, cached)
    return cached


def _manifest_images_from_entry(
    fold_name: str, sample_name: str, entry: dict[str, object]
) -> tuple[str | None, str | None]:
    left = entry.get("left_image")
    if left is None:
        left = entry.get("left")

    right = entry.get("right_image")
    if right is None:
        right = entry.get("right")

    def normalize(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().replace("\\", "/").lstrip("/")
        if normalized.startswith("images/"):
            normalized = normalized[len("images/") :]
        if normalized.startswith(f"{fold_name}/{sample_name}/"):
            return normalized
        if "/" not in normalized:
            return f"{fold_name}/{sample_name}/{normalized}"
        return f"{fold_name}/{sample_name}/{Path(normalized).name}" if not normalized.startswith(
            fold_name + "/"
        ) else normalized

    return normalize(left), normalize(right)


def _get_manifest_entries_for_sample(
    fold_name: str, sample_name: str
) -> list[dict[str, object]]:
    sample_dir = DATASET_ROOT / fold_name / sample_name
    entries = _load_manifest_cached(sample_dir)
    pairs: list[dict[str, object]] = []

    for entry in entries:
        left_rel, right_rel = _manifest_images_from_entry(
            fold_name, sample_name, entry
        )
        if left_rel is None or right_rel is None:
            continue
        left_path = DATASET_ROOT / left_rel
        right_path = DATASET_ROOT / right_rel
        if not left_path.exists() or not right_path.exists():
            continue
        if not (
            left_path.suffix.lower() in IMAGE_EXTENSIONS
            and right_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            continue
        pairs.append(entry)

    return pairs


def _manifest_pair_id_for_paths(
    fold_name: str, sample_name: str, left_rel: str, right_rel: str
) -> str | None:
    for entry in _get_manifest_entries_for_sample(fold_name, sample_name):
        candidate_left, candidate_right = _manifest_images_from_entry(
            fold_name, sample_name, entry
        )
        if candidate_left is None or candidate_right is None:
            continue
        if (candidate_left == left_rel and candidate_right == right_rel) or (
            candidate_left == right_rel and candidate_right == left_rel
        ):
            value = entry.get("pair_id")
            if isinstance(value, str) and value:
                return value
    return None


def _build_pair_from_manifest(
    fold_filter: str | None, sample_filter: str | None
) -> ImagePair | None:
    if not DATASET_ROOT.exists():
        return None

    matching_samples: list[tuple[str, str]] = []
    for fold_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()):
        if fold_filter and fold_dir.name != fold_filter:
            continue
        for sample_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            if sample_filter and sample_dir.name != sample_filter:
                continue
            pairs = _get_manifest_entries_for_sample(fold_dir.name, sample_dir.name)
            if pairs:
                matching_samples.append((fold_dir.name, sample_dir.name))

    if not matching_samples:
        return None

    fold_name, sample_name = random.choice(matching_samples)
    manifest_entries = _get_manifest_entries_for_sample(fold_name, sample_name)
    if not manifest_entries:
        return None

    entry = random.choice(manifest_entries)
    left_rel, right_rel = _manifest_images_from_entry(
        fold_name, sample_name, entry
    )
    if left_rel is None or right_rel is None:
        return None

    pair_id = entry.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        pair_id = _pair_id(
            fold_name,
            sample_name,
            DATASET_ROOT / left_rel,
            DATASET_ROOT / right_rel,
        )

    return ImagePair(
        pair_id=pair_id,
        fold=fold_name,
        sample=sample_name,
        left_image=left_rel,
        right_image=right_rel,
        left_display=_to_public_url(DATASET_ROOT / left_rel),
        right_display=_to_public_url(DATASET_ROOT / right_rel),
    )


def _collect_catalog() -> list[dict[str, str | list[str]]]:
    if not DATASET_ROOT.exists():
        return []

    catalog: list[dict[str, str | list[str]]] = []
    for fold_dir in sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir()):
        sample_names: list[str] = []
        for sample_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            images = _collect_image_files(sample_dir)
            if len(images) < 2:
                continue
            sample_names.append(sample_dir.name)

        if sample_names:
            catalog.append({"name": fold_dir.name, "samples": sample_names})

    return catalog


def _get_catalog_cached() -> list[dict[str, str | list[str]]]:
    key = str(DATASET_ROOT)
    now = time.time()
    expiry, catalog = _catalog_cache.get(key, (0.0, []))
    if now >= expiry or not catalog:
        catalog = _collect_catalog()
        _catalog_cache[key] = (now + CATALOG_CACHE_TTL_SECONDS, catalog)
    return catalog


@app.get("/api/catalog")
def get_catalog() -> dict[str, list[dict[str, str | list[str]]]]:
    return {"folds": _get_catalog_cached()}


def _pair_id(fold: str, sample: str, left: Path, right: Path) -> str:
    parts = sorted(
        [str(left.relative_to(DATASET_ROOT)), str(right.relative_to(DATASET_ROOT))]
    )
    token = f"{fold}|{sample}|{parts[0]}|{parts[1]}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def _to_public_url(path: Path) -> str:
    rel = path.relative_to(DATASET_ROOT)
    return f"/images/{rel.as_posix()}"


def _safe_rel_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if normalized.startswith("/images/"):
        normalized = normalized[len("/images/"):]
    candidate = (DATASET_ROOT / normalized).resolve()
    if not str(candidate).startswith(str(DATASET_ROOT) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not candidate.exists() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Image not found")
    return candidate.relative_to(DATASET_ROOT).as_posix()


def _build_pair() -> ImagePair:
    candidates = _collect_samples(None, None)
    if not candidates:
        raise HTTPException(
            status_code=404, detail="No sample with at least two images found"
        )

    fold_name, sample_name, images = random.choice(candidates)
    left, right = random.sample(images, 2)
    pair_id = _pair_id(fold_name, sample_name, left, right)

    return ImagePair(
        pair_id=pair_id,
        fold=fold_name,
        sample=sample_name,
        left_image=left.relative_to(DATASET_ROOT).as_posix(),
        right_image=right.relative_to(DATASET_ROOT).as_posix(),
        left_display=_to_public_url(left),
        right_display=_to_public_url(right),
    )


@app.on_event("startup")
def on_startup() -> None:
    _init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return health()


@app.get("/api/pairs")
def get_pair(fold: str | None = None, sample: str | None = None) -> PairOut:
    if not DATASET_ROOT.exists():
        raise HTTPException(status_code=404, detail="Dataset root not found")

    if fold is not None or sample is not None:
        manifest_pair = _build_pair_from_manifest(fold, sample)
        if manifest_pair is None:
            # Fallback to legacy random image pairing for backwards compatibility.
            samples = _collect_samples(fold, sample)
            if not samples:
                raise HTTPException(status_code=404, detail="No matching sample found")
            fold_name, sample_name, images = random.choice(samples)
            left, right = random.sample(images, 2)
            pair = ImagePair(
                pair_id=_pair_id(fold_name, sample_name, left, right),
                fold=fold_name,
                sample=sample_name,
                left_image=left.relative_to(DATASET_ROOT).as_posix(),
                right_image=right.relative_to(DATASET_ROOT).as_posix(),
                left_display=_to_public_url(left),
                right_display=_to_public_url(right),
            )
        else:
            pair = manifest_pair
    else:
        # Random pair from manifest when available, with legacy fallback.
        manifest_pair = _build_pair_from_manifest(None, None)
        if manifest_pair is not None:
            pair = manifest_pair
        else:
            pair = _build_pair()

    return PairOut(
        pair_id=pair.pair_id,
        fold=pair.fold,
        sample=pair.sample,
        left={
            "id": pair.left_image,
            "url": pair.left_display,
        },
        right={
            "id": pair.right_image,
            "url": pair.right_display,
        },
    )


@app.post("/api/preferences", response_model=PreferenceOut)
def log_preference(payload: PreferenceIn) -> PreferenceOut:
    left_rel = _safe_rel_path(payload.left_image)
    right_rel = _safe_rel_path(payload.right_image)

    expected_id = _pair_id(
        payload.fold, payload.sample, DATASET_ROOT / left_rel, DATASET_ROOT / right_rel
    )

    manifest_id = _manifest_pair_id_for_paths(payload.fold, payload.sample, left_rel, right_rel)
    if manifest_id:
        expected_id = manifest_id

    # pair_id can come from client, but we keep a stable canonical id for this image pair.
    pair_id = expected_id

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO preference_logs (ts, user_id, pair_id, fold, sample, left_image, right_image, preference)\n             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                payload.user_id,
                pair_id,
                payload.fold,
                payload.sample,
                left_rel,
                right_rel,
                payload.preference,
            ),
        )
        conn.commit()
        return PreferenceOut(id=cur.lastrowid, message="logged")


@app.get("/api/preferences")
def list_preferences(limit: int = 100) -> list[dict]:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM preference_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    return [dict(row) for row in rows]
