"""Parameter-based prediction cache lookup, including old code-hashed directories."""
import json
from pathlib import Path

from lasagna.demand_predict3d import cache_identity
from lasagna.inference_provenance import json_digest


def model_parameters(model):
    result = dict(model)
    result.pop("implementation", None)
    result.setdefault("embedding_channels", None)
    return result


def resolve_prediction_cache(source_root, source, model, geometry, settings):
    """Return a compatible directory and its unchanged recorded model identity.

    Only small identity files are inspected, never chunk payloads. The store
    still takes its writer lock and revalidates the complete recorded identity.
    """
    source_root = Path(source_root)
    model = model_parameters(model)
    canonical = source_root / json_digest(model)
    expected = cache_identity({"source": source, "model": model}, geometry,
        settings.get("accumulator_dtype", "float16"),
        settings.get("accumulator_layout", "flat"), settings.get("accumulator_prefault", False))
    candidates = [canonical]
    if source_root.exists():
        candidates += sorted(p for p in source_root.iterdir() if p != canonical and p.is_dir())
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError("unsafe inference cache directory symlink")
        path = candidate / "cache_identity.json"
        if path.is_symlink():
            raise ValueError("unsafe cache identity symlink")
        if not path.exists():
            continue
        recorded = json.loads(path.read_text())
        identity = recorded.get("identity", {})
        old_model = identity.get("model")
        if not isinstance(old_model, dict):
            continue
        comparable = {**recorded, "identity": {**identity, "model": model_parameters(old_model)}}
        if comparable == expected:
            return candidate, old_model
    return canonical, model
