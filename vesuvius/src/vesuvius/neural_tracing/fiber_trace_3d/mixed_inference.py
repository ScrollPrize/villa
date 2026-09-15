"""Channel-role composition; payloads and numerical workers remain unchanged."""
from pathlib import Path
import json
import os

from lasagna.inference_provenance import atomic_write, json_digest


def request_store_key(keys, array, channel_indices):
    """Authorize each requested array within this source, not across all sources."""
    keys = [keys] if isinstance(keys, str) else keys
    owners = [key for key in dict.fromkeys(keys) if array in channel_indices[key]]
    if len(owners) != 1:
        raise ValueError("native request escapes its inference cache or has ambiguous ownership")
    return owners[0]


def write_mixed_manifest(root, prediction_path, embedding_path):
    """Compose ordinary scalar-channel manifests without copying any array."""
    paths = [Path(prediction_path).resolve(), Path(embedding_path).resolve()]
    manifests = [json.loads(p.read_text()) for p in paths]
    identities = [json.loads((p.parent / "cache_identity.json").read_text()) for p in paths]
    if identities[0]["identity"]["source"] != identities[1]["identity"]["source"]:
        raise ValueError("mixed inference requires the identical physical source/coordinate lattice")
    for field, default in (("base_shape_zyx", None), ("source_to_base", 1), ("crops", [])):
        if manifests[0].get(field, default) != manifests[1].get(field, default):
            raise ValueError(f"mixed inference incompatible {field}")
    groups = {}
    for role, names in ((0, ("presence", "nx", "ny")), (1, None)):
        for group in manifests[role]["groups"].values():
            selected = [c for c in group["channels"]
                        if c in names] if names is not None else [
                            c for c in group["channels"] if c.startswith("embedding_")]
            if not selected:
                continue
            if len(group["channels"]) != 1:
                raise ValueError("mixed inference requires scalar channel groups")
            channel, = selected
            if channel in groups:
                raise ValueError(f"duplicate mixed channel {channel}")
            groups[channel] = {**group, "zarr": str((paths[role].parent / group["zarr"]).resolve())}
    if not all(name in groups for name in ("presence", "nx", "ny")):
        raise ValueError("mixed inference requires canonical presence/nx/ny")
    count = len(groups) - 3
    if not 1 <= count <= 256 or not all(f"embedding_{i:03d}" in groups for i in range(count)):
        raise ValueError("mixed inference requires contiguous embedding_NNN channels")
    if len({g["scaledown"] for g in groups.values()}) != 1:
        raise ValueError("mixed inference channels must share the output lattice")
    # Keep role identities explicit; never present a composite as an ordinary
    # cache with its own payload ownership or writer lock.
    provenance = {"schema": 1, "presence_nx_ny": str(paths[0]), "embedding": str(paths[1]),
                  "identities": identities}
    destination = Path(root) / json_digest(provenance) / "inference.lasagna.json"
    document = {**manifests[0], "groups": groups, "mixed_inference": provenance}
    document.pop("provenance", None)  # The old relative inference.json belongs to the base cache.
    if destination.exists():
        if json.loads(destination.read_text()) != document:
            raise ValueError("existing mixed manifest changed")
    else:
        atomic_write(destination, document)
    return destination


def partition_devices(devices, count):
    if not devices:
        raise ValueError("mixed inference needs CUDA devices")
    return [list(devices[i::count]) or [devices[i % len(devices)]] for i in range(count)]


class ModelWorkerPools:
    """Reuse one ordinary pool per model, under one coordinated manager lease.

    GPUs are partitioned across pools. On a single GPU the two workers share
    a divided memory budget. CPU/RAM queue limits are divided, not duplicated.
    """
    def __init__(self, stores, readers, devices, *, groups, **options):
        from lasagna.demand_predict3d import DemandWorkerPool
        self.pools = []
        groups = list(groups.values())
        assignments = partition_devices(devices, len(groups))
        deferred = options.pop("deferred_gpu", False)
        try:
            for group, selected in zip(groups, assignments):
                settings = group["settings"]
                kwargs = dict(options)
                for name in ("input_workers", "input_slots", "result_slots", "accumulator_workers", "flush_workers"):
                    if kwargs.get(name) is not None:
                        kwargs[name] = max(1, kwargs[name] // len(groups))
                # These two affect numerics/cache identity and belong to each model.
                kwargs["max_batch_size"] = settings.get("max_batch_size", 1)
                keys = group["keys"]
                self.pools.append(DemandWorkerPool({k: stores[k] for k in keys},
                    {k: readers[k] for k in keys}, selected, deferred_gpu=True, **kwargs))
            if not deferred:
                self.start_gpu(devices)
                self.authorize_gpu()
        except BaseException:
            self.close()
            raise

    @property
    def stats(self):
        result = {"models": [p.stats for p in self.pools]}
        for name in ("shared_slot_bytes", "input_slots", "input_workers", "result_slots", "tiles"):
            result[name] = sum(p.stats.get(name, 0) for p in self.pools)
        result["max_batch_size"] = max(p.stats["max_batch_size"] for p in self.pools)
        return result

    def queue_snapshot(self):
        rows = [p.queue_snapshot() for p in self.pools]
        return {key: sum(row[key] for row in rows) for key in rows[0]}

    @property
    def busy(self):
        return any(p.busy for p in self.pools)

    @property
    def processes(self):
        return [proc for p in self.pools for proc in p.processes]

    @property
    def active(self):
        return {(i, k): v for i, p in enumerate(self.pools) for k, v in p.active.items()}

    @property
    def loading(self):
        return {(i, k): v for i, p in enumerate(self.pools) for k, v in p.loading.items()}

    @property
    def input_owners(self):
        return {(i, k): v for i, p in enumerate(self.pools) for k, v in p.input_owners.items()}

    @property
    def last_gpu_activity(self):
        return max(p.last_gpu_activity for p in self.pools)

    @property
    def gpu_authorized(self):
        return all(p.gpu_authorized for p in self.pools)

    @gpu_authorized.setter
    def gpu_authorized(self, value):
        for p in self.pools:
            p.gpu_authorized = value

    def start_gpu(self, devices, *, vram_budget=None):
        import torch
        if vram_budget is None and os.environ.get("FIBER_INFERENCE_VRAM_BUDGET_BYTES"):
            vram_budget = int(os.environ["FIBER_INFERENCE_VRAM_BUDGET_BYTES"])
        assignments = partition_devices(devices, len(self.pools))
        for pool, selected in zip(self.pools, assignments):
            sharing = max(sum(d in group for group in assignments) for d in selected)
            budget = vram_budget
            if sharing > 1:
                if budget is None:
                    budget = min(torch.cuda.get_device_properties(d).total_memory for d in selected)
                budget = int(budget * .9 / sharing)
            pool.start_gpu(selected, vram_budget=budget)

    def authorize_gpu(self):
        for p in self.pools:
            p.authorize_gpu()

    def poll_stop_gpu(self):
        # Do not short circuit: advance every pool's nonblocking shutdown.
        return all([p.poll_stop_gpu() for p in self.pools])

    def pump(self):
        for p in self.pools:
            p.pump()

    def close(self):
        error = None
        for p in reversed(self.pools):
            try:
                p.close()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error
