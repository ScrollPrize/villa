"""Automatic operational settings, separate from the numerical training config."""

from copy import deepcopy
import json
from pathlib import Path

from .memory_calibration import calibration_report_path
from .training_state import config_fingerprint


def task_priority(settings, override=None):
    if not isinstance(settings, dict):
        raise ValueError("training.cooperative must be an object")
    request = settings.get("request", {})
    if not isinstance(request, dict):
        raise ValueError("cooperative resource request must be an object")
    value = override if override is not None else settings.get("priority", request.get("priority", 0))
    if type(value) is not int:
        raise ValueError("cooperative priority must be an integer")
    return value


def load_calibration(config_path, raw, *, allow_missing=False):
    settings = raw.get("training", {}).get("cooperative", {})
    if not isinstance(settings, dict):
        raise ValueError("training.cooperative must be an object")
    value = settings.get("memory_calibration")
    if value is None:
        path = calibration_report_path(config_path, raw)
        if not path.is_file():
            if allow_missing:
                return None, None
            raise ValueError(
                f"No measured memory calibration for this config. Run "
                f"python -m vesuvius.neural_tracing.fiber_trace_3d.memory_calibration {config_path} "
                "with permitted --gpu-uuids, --microbatch-sizes, --cpu-workers and --vram-limit-bytes. "
                f"Omit --output; training will automatically read {path}. No manager config edits needed.")
        value = json.loads(path.read_text())
    return value, validate_calibration(raw, value, check_environment=True)


def validate_calibration(raw, value, *, check_environment=False):
    """Shared validation for explicit reports and automatically discovered ones."""
    if not isinstance(value, dict) or value.get("config_fingerprint") != config_fingerprint(raw):
        raise ValueError("memory calibration is missing or belongs to a different training config")
    if set(value) != {"config_fingerprint", "estimate", "observations"} or not value["observations"]:
        raise ValueError("memory calibration requires measured observations and estimate")
    from .memory_calibration import calibration_from_observations
    from .cooperative_resources import MemoryCalibration
    estimate = dict(value["estimate"])
    estimate["measured_peaks"] = tuple(tuple(row) for row in estimate.get("measured_peaks", ()))
    calibration = MemoryCalibration(**estimate)
    checked = calibration_from_observations(config_fingerprint(raw), value["observations"],
                                            reserve_bytes=calibration.reserve_bytes)
    if tuple(tuple(row) for row in checked["estimate"]["measured_peaks"]) != calibration.measured_peaks:
        raise ValueError("memory estimate peaks differ from recorded observations")
    from dataclasses import asdict
    if asdict(calibration) != checked["estimate"]:
        raise ValueError("memory estimate differs from recorded observations and explicit overhead")
    if check_environment:
        import torch
        if any(row["torch_version"] != str(torch.__version__) or row["cuda_version"] != torch.version.cuda
               for row in value["observations"]):
            raise ValueError("memory calibration software differs; rerun calibration with this environment")
    return calibration


def derive_request(training, inventory, visible, observation, calibration):
    eligible = sorted(set(visible) & inventory["gpus"].keys())
    measured = [size for size, _ in calibration.measured_peaks if size <= calibration.max_microbatch]
    if not eligible or not measured:
        raise ValueError("no visible managed GPUs or usable calibrated microbatches")
    # Measurements size each device's microbatch, not the number of replicas.
    # In particular, a partial startup grant must not permanently cap training
    # to that grant's world size. Compatibility is checked by prepare_settings.
    minimum = min(calibration.estimated_bytes(size) for size in measured)
    eligible = [gpu for gpu in eligible if inventory["gpus"][gpu] >= minimum]
    if not eligible:
        raise ValueError("no visible managed GPU can fit a calibrated microbatch")
    cores = min(int(training.get("sample_pipeline_workers", 1)), inventory["cpu_cores"])
    return {"max_cpu_cores": cores, "max_gpus": len(eligible),
            "min_vram_bytes_per_gpu": minimum,
            "max_vram_bytes_per_gpu": max(inventory["gpus"][gpu] for gpu in eligible),
            "allowed_gpu_uuids": eligible, "priority": 0, "exclusive": False}


def prepare_settings(config_path, raw, *, socket=None):
    # Invalid/stale reports still fail; a missing report is measured by real
    # startup work, never by a synthetic or discarded probe run.
    observation, calibration = load_calibration(config_path, raw, allow_missing=True)
    from villa_gpu_manager.bootstrap import ensure_manager, compatible_gpu_uuids
    from villa_gpu_manager.paths import default_socket_path
    prepared = deepcopy(raw)
    training = prepared.setdefault("training", {})
    settings = training.setdefault("cooperative", {})
    calibration_scope = settings.get('startup_calibration_scope', 'global')
    if calibration_scope not in ('global', 'visible_gpus'):
        raise ValueError('startup_calibration_scope must be global or visible_gpus')
    selected = socket if socket is not None else settings.get("socket")
    selected, inventory = ensure_manager(selected if selected is not None else default_socket_path())
    if calibration is None:
        if not raw.get("test_datasets") or int(training.get("test_interval", 0)) <= 0:
            raise ValueError("automatic startup calibration requires configured test_datasets and a positive test_interval")
        from villa_gpu_manager.bootstrap import visible_gpu_uuids
        visible = set(visible_gpu_uuids())
        managed = set(inventory["gpus"])
        if not visible or not visible <= managed:
            raise ValueError(
                "automatic startup calibration requires a nonempty managed GPU subset"
            )
        budget = min(inventory["gpus"][gpu] for gpu in visible)
        if calibration_scope == 'visible_gpus' and any(inventory['gpus'][gpu] != budget for gpu in visible):
            raise ValueError('visible_gpus calibration needs equal-capacity GPUs to reserve each in full')
        reserve = 4 * 1024**3
        if budget <= reserve:
            raise ValueError("managed GPU capacity is too small for startup driver reserve")
        settings["socket"] = selected
        settings["startup_calibration"] = True
        settings["request"] = {"max_cpu_cores": min(int(training.get("sample_pipeline_workers", 1)), inventory["cpu_cores"]),
            "max_gpus": len(visible), "min_vram_bytes_per_gpu": budget,
            "max_vram_bytes_per_gpu": budget, "allowed_gpu_uuids": sorted(visible),
            "priority": task_priority(settings) + 100, "exclusive": calibration_scope == 'global'}
        settings["startup_inventory"] = inventory
        print(f"fiber cooperative: automatic real-update startup calibration socket={selected} "
              f"scope={calibration_scope} full_capacity_gpus={len(visible)}", flush=True)
        return prepared
    visible = compatible_gpu_uuids({row["gpu_name"] for row in observation["observations"]})
    derived = derive_request(training, inventory, visible, observation, calibration)
    if "request" not in settings:
        settings["request"] = derived
    else:
        request = settings["request"]
        if not isinstance(request, dict):
            raise ValueError("cooperative resource request must be an object")
        allowed = request.get("allowed_gpu_uuids") or list(inventory["gpus"])
        eligible = sorted(set(allowed) & set(derived["allowed_gpu_uuids"]))
        if not eligible:
            raise ValueError("explicit request has no visible compatible managed GPUs")
        request["allowed_gpu_uuids"] = eligible
        request["max_gpus"] = min(request["max_gpus"], len(eligible), derived["max_gpus"])
        request["max_cpu_cores"] = min(request["max_cpu_cores"], inventory["cpu_cores"])
    settings["socket"] = selected
    settings["request"]["priority"] = task_priority(settings)
    settings["memory_calibration"] = observation
    print(f"fiber cooperative: socket={selected} request={json.dumps(settings['request'], sort_keys=True)}", flush=True)
    return prepared
