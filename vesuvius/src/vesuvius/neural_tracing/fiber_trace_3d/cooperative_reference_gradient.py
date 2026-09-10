"""Test-only frozen B128 gradient oracle; production remains FP32 accumulation.

Observe the actual contributions entering the shared accumulator, sum them in
FP64, and bound FP32 rounding independently of Adam's subsequent amplification.
Uses the existing capped probe launcher and existing training implementation.
"""

import json
from pathlib import Path

from .memory_calibration import ProbeFactory


class GradientProbeFactory(ProbeFactory):
    def __call__(self, rank, grant, state):
        import torch
        from . import train
        from .gradient_accumulation import ComponentGradientAccumulator, COMPONENTS
        from .loss_statistics import LossStatistics

        current = []
        original_forward = train._forward_loss_statistics
        output = Path(self.output_dir)

        class ObservedAccumulator(ComponentGradientAccumulator):
            def __init__(self, model, **kwargs):
                super().__init__(model, gradient_observer=self.observe, **kwargs)
                self.shadow = [torch.zeros_like(bank, dtype=torch.float64) for bank in self.flat_banks]
                self.absolute = [torch.zeros_like(bank, dtype=torch.float64) for bank in self.flat_banks]
                sizes = [p.numel() for p in self.parameters]
                self.shadow_views = [[v.view_as(p) for v, p in zip(bank.split(sizes), self.parameters)]
                                     for bank in self.shadow]
                self.absolute_views = [[v.view_as(p) for v, p in zip(bank.split(sizes), self.parameters)]
                                       for bank in self.absolute]
                self.names = [name for name, p in model.named_parameters() if p.requires_grad]
                self.ids = []
                self.shadow_statistics = torch.zeros_like(self.statistics, dtype=torch.float64)
                current[:] = [self]

            def observe(self, component, parameter, contribution):
                self.shadow_views[component][parameter].add_(contribution)
                self.absolute_views[component][parameter].add_(contribution.abs())

            def add(self, statistics):
                self.shadow_statistics.add_(statistics.packed().detach())
                super().add(statistics)

            def reduce_and_assign(self, **kwargs):
                result = super().reduce_and_assign(**kwargs)
                distributed = kwargs.get("distributed", False)
                if distributed:
                    for tensor in self.shadow + self.absolute + [self.shadow_statistics]:
                        torch.distributed.all_reduce(tensor)
                    rows = [None] * len(grant["gpus"])
                    torch.distributed.all_gather_object(rows, self.ids)
                    ids = sorted(i for row in rows for i in row)
                else:
                    ids = sorted(self.ids)
                assert ids == list(range(128)), "frozen update changed canonical sample membership"
                fields = list(LossStatistics.__dataclass_fields__)
                counts = {name: self.shadow_statistics[i].item()
                          for i, name in enumerate(fields) if name.endswith("_count")}
                for i, name in enumerate(fields):
                    if name.endswith("_count"):
                        assert self.statistics[i].double() == self.shadow_statistics[i], name
                epsilon = torch.finfo(torch.float32).eps
                # A sequential sum of 128 contributions is a conservative
                # bound for either one rank or two groups of 64 plus reduction.
                gamma = 128 * epsilon / (1 - 128 * epsilon)
                reports = []
                for ci, name in enumerate(COMPONENTS):
                    for pi, parameter_name in enumerate(self.names):
                        actual = self.banks[ci][pi].double()
                        expected = self.shadow_views[ci][pi]
                        absolute = self.absolute_views[ci][pi]
                        error = (actual - expected).abs()
                        bound = gamma * absolute
                        assert bool((error <= bound).all()), f"roundoff bound exceeded: {name}/{parameter_name}"
                        reports.append({"component": name, "parameter": parameter_name,
                            "max_abs": error.max().item(),
                            "relative_l2": ((error.square().sum() / expected.square().sum().clamp_min(1e-300)) ** .5).item(),
                            "max_bound_fraction": (error / bound.clamp_min(1e-300)).max().item()})
                if rank == 0:
                    output.mkdir(parents=True, exist_ok=True)
                    torch.save({"reference": [bank.cpu() for bank in self.shadow], "counts": counts,
                                "sample_ids": ids}, output / "gradient-reference.pt")
                    (output / "gradient-roundoff.json").write_text(json.dumps(reports, indent=2) + "\n")
                    print(f"gradient oracle: world={len(grant['gpus'])} samples={len(ids)} "
                          f"layers={len(reports)} max_bound_fraction="
                          f"{max(row['max_bound_fraction'] for row in reports):.6f}", flush=True)
                return result

        def forward(model, batch, **kwargs):
            if current and not current[0].finished:
                current[0].ids.extend(batch.stream_indices.detach().cpu().tolist())
            return original_forward(model, batch, **kwargs)

        def accumulator(model, **kwargs):
            # Probe validation requires two optimizer updates. Only the first
            # is the frozen-model oracle; do not compare a changed model next.
            return (ComponentGradientAccumulator if current else ObservedAccumulator)(model, **kwargs)

        train.ComponentGradientAccumulator = accumulator
        train._forward_loss_statistics = forward
        return super().__call__(rank, grant, state)


def validate(config, output, devices):
    import torch
    from . import memory_calibration
    from .train import _load_raw_config, _normalize_mixed_precision_mode

    raw = _load_raw_config(config)
    if len(devices) != 2 or _normalize_mixed_precision_mode(raw["training"].get("mixed_precision")) != "off":
        raise ValueError("the frozen gradient oracle requires FP32 and exactly two available devices")
    if raw["training"].get("optimizer_global_batch_size") != 128:
        raise ValueError("the frozen gradient oracle requires effective batch 128")

    original = memory_calibration.ProbeFactory
    memory_calibration.ProbeFactory = GradientProbeFactory
    try:
        for count in (1, 2):
            result = memory_calibration.main([str(config), "--gpu-uuids", *devices[:count],
                "--microbatch-sizes", "1", "--cpu-workers", "8", "--steps", "2",
                "--vram-limit-bytes", str(16 * 1024**3), "--reserve-bytes", str(4 * 1024**3),
                "--allow-shared", "--output", str(output / f"world{count}" / "probe.json")])
            assert result == 0
    finally:
        memory_calibration.ProbeFactory = original
    a, b = [torch.load(output / f"world{count}/memory-probes/gradient-reference.pt",
                       map_location="cpu", weights_only=True) for count in (1, 2)]
    assert a["sample_ids"] == b["sample_ids"] and a["counts"] == b["counts"]
    for x, y in zip(a["reference"], b["reference"]):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-13)
    print("gradient oracle: one/two-GPU FP64 contribution sums match", flush=True)
