"""Repeated reference measurements through the production cohort and trainer.

The unmanaged comparison removes manager admission/RPC only, not bounded
accumulation. Permission to share GPUs is not evidence of actual contention.
"""

import json
from pathlib import Path
import statistics
import time


def timing_summary(values):
    values = sorted(float(value) for value in values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("positive timing samples required")
    def percentile(p):
        position = (len(values) - 1) * p
        low = int(position)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
    return dict(n=len(values), mean=statistics.mean(values), p50=percentile(.5), p95=percentile(.95),
                minimum=values[0], maximum=values[-1])


def run(calibration_path, output, devices, *, warmup=10, measured=100, repeats=3, timeout=7200,
        previous=None):
    import torch
    from .cooperative_reference_lifecycle import validate, describe_state_differences
    if warmup < 5 or measured < 100 or repeats < 3:
        raise ValueError("reference benchmark needs >=5 warmup, >=100 measured updates, >=3 repeats")
    config = Path(calibration_path).with_name("reference-config.json")
    report = {"warmup_updates": warmup, "measured_updates": measured, "repeats": repeats,
              "effective_batch": 128, "devices": devices, "shared_gpu_allowed": True,
              "baseline": "same bounded trainer/cohort without manager admission or RPC",
              "cases": []}
    modes = ("unmanaged", "fixed", "dynamic")
    started = time.monotonic()
    previous_report = json.loads((previous / "benchmark.json").read_text()) if previous else None
    if previous_report:
        previous_control = json.loads((previous / "control.json").read_text())["reference"]
        previous_calibration = json.loads(Path(previous_control["calibration"]).read_text())
        current_calibration = json.loads(Path(calibration_path).read_text())
        assert previous_calibration == current_calibration, "completed benchmark used different calibration/config"
        for key in ("warmup_updates", "measured_updates", "repeats", "effective_batch", "devices", "baseline"):
            assert previous_report[key] == report[key], f"incompatible completed benchmark {key}"
    for repeat in range(repeats):
        # Counterbalance execution order; never run GPU cases concurrently.
        order = modes[repeat % 3:] + modes[:repeat % 3]
        states = {}
        for mode in order:
            old = next((row for row in previous_report["cases"]
                        if row["repeat"] == repeat + 1 and row["mode"] == mode), None) if previous_report else None
            if old is not None:
                folder = Path(old.get("artifact_dir", previous / f"repeat-{repeat + 1}-{mode}"))
                state_path = folder / f"{mode}-validation-state.pt"
                assert state_path.is_file() and (folder / "lifecycle.json").is_file()
                report["cases"].append(dict(old, artifact_dir=str(folder), reused_completed=True))
                states[mode] = state_path
                print(f"benchmark reuse completed repeat={repeat + 1} mode={mode} from {folder}", flush=True)
                continue
            folder = output / f"repeat-{repeat + 1}-{mode}"
            folder.mkdir()
            print(f"benchmark repeat={repeat + 1}/{repeats} mode={mode} start", flush=True)
            validate(config, calibration_path, folder, devices, steps=warmup + measured,
                     modes=(mode,), benchmark=True, timeout=timeout)
            result = json.loads((folder / "lifecycle.json").read_text())[mode]
            units = [row for row in result["units"] if row["update_completed"] and row["step"] > warmup]
            wall = [row["seconds"] for row in result["updates"] if row["step"] > warmup]
            if len(units) != measured or len(wall) != measured:
                raise AssertionError("benchmark did not measure every canonical update")
            active = timing_summary(row["seconds"] for row in units)
            elapsed = timing_summary(wall)
            case = dict(repeat=repeat + 1, mode=mode, artifact_dir=str(folder), active_update_seconds=active,
                        wall_update_seconds=elapsed, global_samples_per_second=128 / elapsed["mean"],
                        active_global_samples_per_second=128 / active["mean"],
                        total_seconds=result["seconds"], transitions=result["transitions"],
                        by_gpu_count={str(count): timing_summary(row["seconds"] for row in units if row["gpus"] == count)
                                      for count in sorted({row["gpus"] for row in units})})
            report["cases"].append(case)
            states[mode] = folder / f"{mode}-validation-state.pt"
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"benchmark repeat={repeat + 1} mode={mode} global_samples/s={case['global_samples_per_second']:.3f} "
                  f"update_mean={elapsed['mean']:.3f}s p50={elapsed['p50']:.3f}s p95={elapsed['p95']:.3f}s", flush=True)
        fixed = torch.load(states["fixed"], map_location="cpu", weights_only=False)
        unmanaged = torch.load(states["unmanaged"], map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "grad_scaler", "progress"):
            # Progress includes per-mode run paths/timing, so compare training identity only.
            if key == "progress":
                for name in ("step", "sample_frontier", "best_metric", "pending_test"):
                    assert fixed[key][name] == unmanaged[key][name], name
            else:
                torch.testing.assert_close(fixed.get(key), unmanaged.get(key), rtol=0, atol=0)
        del unmanaged
        dynamic = torch.load(states["dynamic"], map_location="cpu", weights_only=False)
        comparison = output / f"repeat-{repeat + 1}-comparison"
        comparison.mkdir()
        describe_state_differences(comparison, payloads={"fixed": fixed, "dynamic": dynamic})
        assert fixed["progress"]["sample_frontier"] == dynamic["progress"]["sample_frontier"] == (warmup + measured) * 128
        assert dynamic["progress"]["pending_test"] is None
        del fixed, dynamic
    report["elapsed_seconds"] = time.monotonic() - started
    report["exact_unmanaged_fixed_state"] = True
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
