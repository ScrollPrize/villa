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


def assert_gpus_idle(gpu_uuids):
    """Fail closed on unmanaged CUDA processes before benchmark authorization."""
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    occupied = [line.strip() for line in output.splitlines()
                if line.split(",", 1)[0].strip() in gpu_uuids]
    if occupied:
        raise RuntimeError("external GPU processes prevent an isolated benchmark: " + "; ".join(occupied))


def launch_managed(command, *, devices, settings, socket=None, priority=None):
    from villa_gpu_manager.exclusive import run_exclusive
    from villa_gpu_manager.bootstrap import ensure_manager
    selected = resolve_gpu_uuids(devices)
    exclusive = settings.get("exclusive", True)
    if type(exclusive) is not bool:
        raise ValueError("gpu_manager.exclusive must be boolean")
    socket, _ = ensure_manager(socket or settings.get("socket"))
    command = [*command, "--unmanaged", "--devices", ",".join(str(i) for i in range(len(selected)))]
    environment = dict(os.environ)
    if not exclusive:
        budget = settings.get("vram_bytes_per_gpu")
        if type(budget) is not int or budget <= 2 * 1024**3:
            raise ValueError("shared evaluation needs a VRAM budget larger than the 2GiB context reserve")
        environment["FIBER_INFERENCE_VRAM_BUDGET_BYTES"] = str(budget)
    # Preserve explicit local import roots in the exec child, including when
    # invoked from a different working directory or an older editable install.
    repo = Path(__file__).resolve().parents[5]
    environment["PYTHONPATH"] = os.pathsep.join([
        str(repo / "vesuvius/src"), str(repo), str(repo / "gpu_manager/src"),
        environment.get("PYTHONPATH", "")])
    return run_exclusive(command, socket=socket,
                         gpu_uuids=selected, priority=settings.get("priority", 100) if priority is None else priority,
                         label=settings.get("label", "fiber-evaluation"), environment=environment,
                         before_authorize=(lambda: assert_gpus_idle(selected)) if exclusive else None,
                         exclusive=exclusive, cpu_cores=settings.get("cpu_cores"),
                         vram_bytes_per_gpu=settings.get("vram_bytes_per_gpu"))
