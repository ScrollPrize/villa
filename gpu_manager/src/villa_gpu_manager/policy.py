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


def targets(tasks, inventory, override=None, reasons=None):
    reasons = {} if reasons is None else reasons
    result = {key: empty() for key in tasks}
    active = [key for key, t in tasks.items() if not t["terminal"] and not any(t["paused"].values())]
    active.sort(key=lambda key: (-tasks[key]["request"]["priority"], tasks[key]["order"]))
    pinned = {key for key in active if not tasks[key].get("pausable", True)
              and tasks[key]["granted"] != empty()}
    for key in pinned:
        result[key] = tasks[key]["granted"]
    if override is not None:
        result = {key: empty() for key in tasks}
        for key, allocation in override["allocations"].items():
            require(key in active or allocation == empty(), "override conflicts with inactive task")
            validate_allocation(allocation, tasks[key]["request"], inventory)
            result[key] = allocation
        for key in active:
            require(key not in pinned or result[key] == tasks[key]["granted"],
                    "override would change non-pausable allocation")
            require(tasks[key].get("pausable", True) or not tasks[key]["request"]["exclusive"]
                    or result[key] == empty()
                    or all(a == empty() for k, a in result.items() if k != key),
                    "override would share non-pausable exclusive allocation")
            fixed = tasks[key].get("fixed_allocation")
            require(not fixed or result[key] in (empty(), fixed),
                    "override would resize non-adaptable allocation")
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

    def minimum(key):
        fixed = tasks[key].get("fixed_allocation")
        if fixed:
            if fits([a for k, a in result.items() if k != key] + [fixed], inventory):
                result[key] = fixed
                return True
            return False
        return any(assign(key, gpu, tasks[key]["request"]["min_vram_bytes_per_gpu"])
                   for gpu in options(key))

    # Existing non-preemptible exclusive work owns the entire pool, including
    # unused capacity. Neither priority nor a new registration revokes it.
    pinned_exclusive = [key for key in pinned if tasks[key]["request"]["exclusive"]]
    if pinned_exclusive:
        for key in active:
            if key not in pinned:
                reasons[key] = "blocked_by_nonpausable_task:" + ",".join(sorted(pinned_exclusive))
        return result

    admitted = []
    pending_exclusive = False
    for key in active:
        if tasks[key]["request"]["exclusive"]:
            # Any pending exclusive blocks new lower-priority rigid work,
            # regardless of why this exclusive cannot yet be admitted.
            pending_exclusive = True
            blockers = sorted(pinned - {key})
            if blockers:
                reasons[key] = "blocked_by_nonpausable_task:" + ",".join(blockers)
                continue
            if admitted:
                reasons[key] = "blocked_by_higher_priority_task:" + ",".join(admitted)
                continue
            if minimum(key):
                admitted = [key]
                break
            reasons[key] = "insufficient_resources"
            continue
        # Do not start new non-preemptible work behind a blocked exclusive job.
        if pending_exclusive and key not in pinned and not tasks[key].get("pausable", True):
            reasons[key] = "waiting_for_higher_priority_exclusive"
            continue
        if key in pinned or minimum(key):
            admitted.append(key)
        else:
            reasons[key] = "insufficient_resources"
    for key in admitted:
        if key in pinned or tasks[key].get("fixed_allocation"):
            continue
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
