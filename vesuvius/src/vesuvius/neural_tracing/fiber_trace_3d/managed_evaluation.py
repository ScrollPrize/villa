"""CPU-only launch helpers; acquire manager ownership before importing CUDA work."""
import os
from pathlib import Path
import subprocess


def resolve_gpu_uuids(devices, *, inventory_text=None, visibility=None):
    """Numeric IDs use nvidia-smi indices, or indices into CUDA_VISIBLE_DEVICES.

    Full/unique UUID prefixes are also accepted. MIG is deliberately unsupported.
    The child receives full UUIDs and contiguous local ordinals, never guessed
    CUDA enumeration order.
    """
    if inventory_text is None:
        inventory_text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10)
    physical = dict(tuple(x.strip() for x in line.split(",")) for line in inventory_text.splitlines() if line.strip())
    def physical_uuid(token):
        if token in physical:
            return physical[token]
        matches = [g for g in physical.values() if token.startswith("GPU-") and g.startswith(token)]
        if len(matches) != 1:
            raise ValueError(f"unknown/ambiguous GPU or unsupported MIG identifier: {token}")
        return matches[0]
    if visibility is None:
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = None if visibility is None else [physical_uuid(t.strip()) for t in visibility.split(",")]
    selected = []
    for token in str(devices).split(","):
        token = token.strip()
        if visible is not None and token.isdigit():
            if int(token) >= len(visible):
                raise ValueError("GPU index exceeds CUDA_VISIBLE_DEVICES")
            gpu = visible[int(token)]
        else:
            gpu = physical_uuid(token)
        if visible is not None and gpu not in visible:
            raise ValueError("requested GPU is hidden by CUDA_VISIBLE_DEVICES")
        selected.append(gpu)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select distinct visible GPUs")
    return selected


def launch_managed(command, *, devices, settings, socket=None, priority=None):
    from villa_gpu_manager.exclusive import run_exclusive
    selected = resolve_gpu_uuids(devices)
    command = [*command, "--unmanaged", "--devices", ",".join(str(i) for i in range(len(selected)))]
    environment = dict(os.environ)
    # Preserve explicit local import roots in the exec child, including when
    # invoked from a different working directory or an older editable install.
    repo = Path(__file__).resolve().parents[5]
    environment["PYTHONPATH"] = os.pathsep.join([
        str(repo / "vesuvius/src"), str(repo), str(repo / "gpu_manager/src"),
        environment.get("PYTHONPATH", "")])
    return run_exclusive(command, socket=socket or settings.get("socket") or
                         str(Path.home() / ".local/run/villa-gpu-manager/manager.sock"),
                         gpu_uuids=selected, priority=settings.get("priority", 100) if priority is None else priority,
                         label=settings.get("label", "fiber-evaluation"), environment=environment)
