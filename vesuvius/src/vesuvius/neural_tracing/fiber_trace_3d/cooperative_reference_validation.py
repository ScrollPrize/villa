"""Opt-in reference-data checks called by the fixed cooperative test launcher.

Uses the archived S1A NML data/test selection, current GroupNorm quarter-output
model and effective B=128. No training algorithm is duplicated here.
"""

import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[5]
    control = json.loads(Path(os.environ["FIBER_COOPERATIVE_TEST_CONTROL"]).read_text())["reference"]
    output = Path(os.environ["FIBER_COOPERATIVE_TEST_OUTPUT"])
    if control["phase"] == "benchmark-check":
        from .cooperative_reference_lifecycle import validate_benchmark_transitions
        run = Path(control["run"])
        report = json.loads((run / "benchmark.json").read_text())
        assert report.get("exact_unmanaged_fixed_state") is True, "benchmark incomplete"
        evidence = {str(repeat): validate_benchmark_transitions(
            run / f"repeat-{repeat}-dynamic/lifecycle.json")
            for repeat in range(1, report["repeats"] + 1)}
        (output / "transition-validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"validated applied resize sequences for {len(evidence)} repetitions", flush=True)
        return 0
    if control["phase"] == "compare":
        from .cooperative_reference_lifecycle import describe_state_differences
        describe_state_differences(Path(control["run"]))
        return 0
    devices = os.environ["FIBER_COOPERATIVE_TEST_GPUS"].split(",")
    uuids = subprocess.check_output(["nvidia-smi", f"--id={','.join(devices)}",
        "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines()
    uuids = [value.strip() for value in uuids]
    if control["phase"] == "benchmark":
        from .cooperative_reference_benchmark import run
        run(Path(control["calibration"]), output, uuids,
            warmup=int(control.get("warmup", 10)), measured=int(control.get("measured", 100)),
            repeats=int(control.get("repeats", 3)), timeout=int(control.get("case_timeout", 7200)))
        return 0
    if control["phase"] == "gradient":
        from .cooperative_reference_gradient import validate
        validate(Path(control["config"]), output, uuids)
        return 0
    if control["phase"] in ("lifecycle", "fixed-repeat", "restart", "profile"):
        from .cooperative_reference_lifecycle import validate
        calibration = Path(control["calibration"])
        validate(calibration.with_name("reference-config.json"), calibration,
                 output, uuids, steps=int(control.get("steps", 6)),
                 modes=("fixed",) if control["phase"] in ("fixed-repeat", "profile") else
                       ("dynamic",) if control["phase"] == "restart" else ("fixed", "dynamic"),
                 resize=control["phase"] != "restart",
                 fixed_reference=Path(control["reference_run"]) if control["phase"] == "restart" else None,
                 profile=control["phase"] == "profile")
        if control["phase"] == "fixed-repeat":
            from .cooperative_reference_lifecycle import describe_state_differences
            describe_state_differences(output, fixed_reference=Path(control["reference_run"]))
        return 0
    if control["phase"] != "calibrate":
        raise ValueError("unsupported reference validation phase")
    baseline = root / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json"
    raw = json.loads(baseline.read_text())["config"]
    raw.pop("_config_dir", None)
    raw["model_3d"].update(normalization="group", output_stride=4)
    if "precision" in control:
        if control["precision"] not in ("off", "bf16", "fp16"):
            raise ValueError("reference precision must be off, bf16, or fp16")
        raw["training"]["mixed_precision"] = control["precision"]
    raw["training"].update(device="cuda", bounded_gradient_accumulation=True,
        sync_batchnorm=False, ddp_broadcast_buffers=False, initialization_seed=0,
        optimizer_global_batch_size=128, max_steps=control["steps"],
        run_path=str(output / "runs"), run_name="s1a-cooperative-reference", run_datestr="validation",
        compile_mode="off", loader_workers=0, sample_pipeline_workers=control["cpu_workers"],
        sample_pipeline_slots=128, sample_pipeline_initial_in_flight=8,
        sample_pipeline_cuda_prefetch_depth=2, sample_pipeline_cuda_value_augmentation="all",
        scalar_log_interval=1, sample_vis_interval=control["steps"],
        test_interval=control["steps"], checkpoint_interval=control["steps"], kept_snapshot_interval=0)
    config = output / "reference-config.json"
    config.write_text(json.dumps(raw, indent=2) + "\n")
    from .memory_calibration import main as calibrate
    return calibrate([str(config), "--gpu-uuids", *uuids,
        "--microbatch-sizes", *map(str, control["microbatch_sizes"]),
        "--cpu-workers", str(control["cpu_workers"]), "--steps", str(control["steps"]),
        "--vram-limit-bytes", str(16 * 1024**3), "--reserve-bytes", str(4 * 1024**3),
        "--allow-shared", "--output", str(output / "calibration.json")])


if __name__ == "__main__":
    raise SystemExit(main())
