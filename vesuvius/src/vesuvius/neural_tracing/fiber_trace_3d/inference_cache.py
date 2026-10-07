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


def validate_cached_model(model, checkpoint_digest, settings):
    """Validate explicit cache extension by numerical parameters, not code hashes."""
    model = model_parameters(model)
    if model["checkpoint_sha256"] != checkpoint_digest:
        raise ValueError("cannot extend cache with different checkpoint")
    settings = dict(settings)
    keys = {"precision": "precision", "tile_size": "tile", "overlap": "overlap",
            "border": "border", "chunk_size": "chunk", "accumulator_dtype": "accumulator_dtype",
            "compressor": "compressor", "max_batch_size": "max_batch_size"}
    for setting, key in keys.items():
        # Batch size controls transient GPU memory and throughput; it does not
        # change published prediction values or the cache identity.  Allow a
        # smaller value when extending a cache for a constrained inspection
        # fill (the historical producer may have used a larger batch).
        if setting == "max_batch_size" and setting in settings:
            if int(settings[setting]) < 1:
                raise ValueError("max_batch_size must be positive")
            continue
        if setting == "precision" and settings.get(setting) == "auto":
            settings.pop(setting)
        if setting in settings and settings[setting] != model.get(key):
            raise ValueError(f"cannot extend cache with different {setting}")
        if setting not in settings and key in model:
            settings[setting] = model[key]
    if "scaledown_power" in settings and 1 << int(settings["scaledown_power"]) != model["scale"]:
        raise ValueError("cannot extend cache with different scale")
    if "emit_embeddings" in settings and bool(settings["emit_embeddings"]) != bool(model["embedding_channels"]):
        raise ValueError("cannot extend cache with different embedding channels")
    settings.setdefault("emit_embeddings", bool(model["embedding_channels"]))
    if 'emit_width' in settings and bool(settings['emit_width']) != bool(model.get('width_channel')):
        raise ValueError('cannot extend cache with different width channels')
    settings.setdefault('emit_width', bool(model.get('width_channel')))
    return settings


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
