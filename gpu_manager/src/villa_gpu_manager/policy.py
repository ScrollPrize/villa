"""Pure deterministic policy. The ownership engine separately authorizes grants."""

from .protocol import fields, integer, require


def empty():
    return {"cpu_cores": 0, "gpus": {}}


def validate_inventory(inventory):
    fields(inventory, ("cpu_cores", "gpus"))
    integer(inventory["cpu_cores"], 1)
    require(type(inventory["gpus"]) is dict and inventory["gpus"], "GPU inventory is empty")
    for gpu, size in inventory["gpus"].items():
        require(type(gpu) is str and gpu.startswith("GPU-"), "expected NVIDIA GPU UUID")
        integer(size, 1)


def validate_request(request, inventory):
    fields(request, ("max_cpu_cores", "max_gpus", "min_vram_bytes_per_gpu",
                     "max_vram_bytes_per_gpu", "allowed_gpu_uuids", "priority", "exclusive"))
    for name in ("max_cpu_cores", "max_gpus", "min_vram_bytes_per_gpu", "max_vram_bytes_per_gpu"):
        integer(request[name], 1)
    require(type(request["priority"]) is int, "priority must be integer")
    require(type(request["exclusive"]) is bool, "exclusive must be boolean")
    require(request["min_vram_bytes_per_gpu"] <= request["max_vram_bytes_per_gpu"], "invalid VRAM bounds")
    allowed = request["allowed_gpu_uuids"]
    require(type(allowed) is list and all(type(x) is str for x in allowed), "expected GPU UUID list")
    require(len(set(allowed)) == len(allowed) and set(allowed) <= inventory["gpus"].keys(), "invalid GPU allowlist")


def cores_for(request, gpus):
    # Integer arithmetic avoids floating-point rounding at exact core boundaries.
    numerator = request["max_cpu_cores"] * sum(gpus.values())
    denominator = request["max_gpus"] * request["max_vram_bytes_per_gpu"]
    return (numerator + denominator - 1) // denominator


def validate_allocation(allocation, request, inventory):
    fields(allocation, ("cpu_cores", "gpus"))
    integer(allocation["cpu_cores"])
    require(type(allocation["gpus"]) is dict, "gpus must be an object")
    require(allocation["cpu_cores"] <= request["max_cpu_cores"], "CPU request exceeded")
    require(len(allocation["gpus"]) <= request["max_gpus"], "GPU request exceeded")
    allowed = request["allowed_gpu_uuids"] or list(inventory["gpus"])
    for gpu, size in allocation["gpus"].items():
        integer(size, 1)
        require(gpu in allowed and gpu in inventory["gpus"], "GPU not allowed")
        require(request["min_vram_bytes_per_gpu"] <= size <= request["max_vram_bytes_per_gpu"], "VRAM outside request")
    require(bool(allocation["gpus"]) == bool(allocation["cpu_cores"]), "GPU and CPU allocation must both be nonzero or zero")


def fits(allocations, inventory):
    return (sum(a["cpu_cores"] for a in allocations) <= inventory["cpu_cores"]
            and all(sum(a["gpus"].get(g, 0) for a in allocations) <= size
                    for g, size in inventory["gpus"].items()))


def targets(tasks, inventory, override=None):
    result = {key: empty() for key in tasks}
    active = [key for key, t in tasks.items() if not t["terminal"] and not any(t["paused"].values())]
    active.sort(key=lambda key: (-tasks[key]["request"]["priority"], tasks[key]["order"]))
    if override is not None:
        for key, allocation in override["allocations"].items():
            require(key in active or allocation == empty(), "override conflicts with inactive task")
            validate_allocation(allocation, tasks[key]["request"], inventory)
            result[key] = allocation
        require(fits(list(result.values()), inventory), "override overcommits inventory")
        return result

    def options(key):
        request = tasks[key]["request"]
        allowed = request["allowed_gpu_uuids"] or list(inventory["gpus"])
        held = tasks[key]["granted"]["gpus"]
        return sorted(allowed, key=lambda g: (any(a["gpus"].get(g, 0) for a in result.values()), g not in held, g))

    def assign(key, gpu, size):
        gpus = dict(result[key]["gpus"], **{gpu: size})
        candidate = {"cpu_cores": cores_for(tasks[key]["request"], gpus), "gpus": gpus}
        others = [a for k, a in result.items() if k != key]
        if fits(others + [candidate], inventory):
            result[key] = candidate
            return True
        return False

    exclusive = [key for key in active if tasks[key]["request"]["exclusive"]]
    if exclusive:
        # Skip impossible exclusive requests, not all other useful work.
        for key in exclusive:
            for gpu in options(key):
                if assign(key, gpu, tasks[key]["request"]["min_vram_bytes_per_gpu"]):
                    active = [key]
                    break
            else:
                continue
            break
        else:
            active = [key for key in active if key not in exclusive]
    for key in active:
        if result[key]["gpus"]:
            continue
        for gpu in options(key):
            if assign(key, gpu, tasks[key]["request"]["min_vram_bytes_per_gpu"]):
                break
    for key in active:
        request = tasks[key]["request"]
        if not result[key]["gpus"]:
            continue
        for gpu in sorted(result[key]["gpus"]) + [g for g in options(key) if g not in result[key]["gpus"]]:
            if gpu not in result[key]["gpus"] and len(result[key]["gpus"]) >= request["max_gpus"]:
                continue
            lo = result[key]["gpus"].get(gpu, request["min_vram_bytes_per_gpu"])
            hi = min(request["max_vram_bytes_per_gpu"], inventory["gpus"][gpu])
            # Monotone integer search also respects proportional CPU rounding.
            while lo <= hi:
                mid = (lo + hi) // 2
                if assign(key, gpu, mid):
                    lo = mid + 1
                else:
                    hi = mid - 1
    return result
