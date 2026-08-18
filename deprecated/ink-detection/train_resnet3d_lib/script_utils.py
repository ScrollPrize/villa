import hashlib
import json
import os
from typing import Any, Mapping, Sequence


def _slugify(value: Any) -> str:
    text = str(value)
    slug = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)
    return slug or 'unknown'


def build_tile_cache_key(cache_context: Mapping[str, Any]) -> str:
    serialized = json.dumps(cache_context, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]
    return f"{_slugify(cache_context.get('valid_id', 'unknown'))}_{digest}"


def build_tile_cache_dir(cache_root: str, cache_context: Mapping[str, Any]) -> str:
    os.makedirs(cache_root, exist_ok=True)
    cache_dir = os.path.join(cache_root, build_tile_cache_key(cache_context))
    os.makedirs(cache_dir, exist_ok=True)

    metadata_path = os.path.join(cache_dir, 'cache_context.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(cache_context, metadata_file, indent=2, sort_keys=True)

    return cache_dir


def resolve_fragment_base_path(fragment_id: str, candidate_roots: Sequence[str], fallback_root: str) -> str:
    for root in candidate_roots:
        if os.path.isdir(os.path.join(root, fragment_id)):
            return root
    return fallback_root
