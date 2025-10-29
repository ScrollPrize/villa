#!/usr/bin/env python3
"""
Surface tracer evaluation harness

This script orchestrates a multi-stage pipeline:

  1) Seed discovery
  2) Seeding (process-managed with a watchdog)
  3) Expansion (process-managed with a watchdog)
  4) Patch selection for tracing
  5) Tracing (two flip_x variants per patch)
  6) Winding number computation
  7) Metrics calculation
  8) Optional W&B logging (scalar-only) + watchdog kill counters

"""

import os
import json
import time
import math
import click
import logging
import subprocess
import numbers
from statistics import median
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterable

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    path: Path
    area: float


class SurfaceTracerEvaluation:
    """
    End-to-end harness (see module docstring for overview).
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.config_path = Path(config_path)

        # Output roots
        self.out_dir = Path(self.config["out_path"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.patches_dir = self.out_dir / "patches"
        self.traces_dir = self.out_dir / "traces"
        self.patches_dir.mkdir(exist_ok=self.config["use_existing_patches"])
        self.traces_dir.mkdir(exist_ok=True)

        # Dedicated process logs (stdout/stderr per child process)
        self.proc_logs_dir = self.out_dir / "proc_logs"
        self.proc_logs_dir.mkdir(exist_ok=True)

        # Dedicated scratch root for TMPDIR (unique per child)
        # NOTE: Scratch must never be inside traces/patches to avoid confusing scanners.
        self.scratch_dir = self.out_dir / "scratch"
        self.scratch_dir.mkdir(exist_ok=True)

        # Resolve bin dir once (stable absolute paths for all tools)
        self.bin_dir = Path(self.config["bin_path"]).resolve()

        # Watchdog controls (overridable in config)
        self._watch_check_period = int(self.config.get("watchdog_check_period_sec", 1800))  # default: 30 min
        self._watch_trigger_fraction = float(self.config.get("watchdog_trigger_fraction", 0.8))
        self._watch_min_samples = int(self.config.get("watchdog_min_samples", 12))  # retained for compatibility
        self._watch_grace_seconds = int(self.config.get("watchdog_grace_seconds", 30))

        # Watchdog kill counters (included in W&B summary if enabled)
        self.watchdog_kills = {"seeding": 0, "expansion": 0}

    # -------------------------------
    # Helpers
    # -------------------------------
    def _exec_env(self) -> Dict[str, str]:
        """
        Return a copy of os.environ with deterministic low-thread settings.
        IMPORTANT: we intentionally override (not setdefault) to guarantee stability.
        """
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        return env

    def _unique_tmp(self, tag: str) -> Path:
        """
        Create a unique scratch directory under out/scratch and return its path.
        The tag helps identify the stage (seed/expand/trace/winding/metrics).
        """
        tmp = self.scratch_dir / f"{tag}_{int(time.time_ns())}"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    @staticmethod
    def _is_finite_number(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    @staticmethod
    def _has_nonempty_tifxyz(td: Path, min_bytes: int = 1) -> bool:
        """
        Return True iff x/y/z.tif exist and are non-zero in size.
        We purposely only require > 0 bytes. Valid traces can be small/very compressible.
        """
        try:
            sizes = [(td / f).stat().st_size for f in ("x.tif", "y.tif", "z.tif")]
        except FileNotFoundError:
            return False
        return all(s > 0 for s in sizes)

    @staticmethod
    def _trace_complete(td: Path) -> bool:
        """
        A "complete" trace directory has meta.json and non-empty tifXYZ.
        """
        if not (td / "meta.json").exists():
            return False
        return SurfaceTracerEvaluation._has_nonempty_tifxyz(td)

    @staticmethod
    def _trace_mtime(td: Path) -> float:
        """
        Representative mtime for a trace directory: max mtime across the key files.
        Used to pick the newest "complete" candidate.
        """
        mtimes = []
        for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
            p = td / f
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0

    def _robust_threshold(self, durations: List[float]) -> Optional[float]:
        """
        Robust wall-clock duration threshold in seconds from completed jobs:
          thr = max(2*median, median + 3*MAD, floor), where floor defaults to 300s.
        No fixed minimum-sample requirement; computes with whatever data is available.
        """
        if not durations:
            return None
        m = float(median(durations))
        mad = float(median([abs(x - m) for x in durations])) if len(durations) > 1 else 0.0
        floor = float(self.config.get("watchdog_floor_seconds", 300.0))
        thr = max(2.0 * m, m + 3.0 * mad, floor)
        return thr

    # -------------------------------
    # Seed discovery
    # -------------------------------
    def find_seed_points(self) -> List[Tuple[float, float, float]]:
        """
        Discover seed points from either:
          • a JSON file with seeds grouped by mode, taking "explicit_seed"
          • a directory of patch subfolders with meta.json containing vc_gsfs_mode == explicit_seed
        Always filters seeds by z-range from config.
        """
        patches_path = Path(self.config["existing_patches_for_seeds"])
        z_min, z_max = self.config["z_range"]

        if patches_path.is_file() and patches_path.suffix == ".json":
            with open(patches_path, "r") as f:
                seeds_by_mode = json.load(f)
            seed_points = [
                (x, y, z)
                for (x, y, z) in seeds_by_mode.get("explicit_seed", [])
                if z_min <= z <= z_max
            ]
            logger.info(f"Loaded {len(seed_points)} seeds from JSON")
            return seed_points

        elif patches_path.is_dir():
            seed_points = []
            failed_count = 0
            for patch_dir in patches_path.iterdir():
                if not patch_dir.is_dir():
                    continue
                try:
                    meta_file = patch_dir / "meta.json"
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    if meta.get("vc_gsfs_mode") != "explicit_seed":
                        continue
                    seed = meta.get("seed")
                    if not seed or len(seed) != 3:
                        continue
                    x, y, z = seed
                    if z_min <= z <= z_max:
                        seed_points.append((x, y, z))
                except Exception:
                    failed_count += 1
                    continue

            if failed_count:
                logger.warning(f"Failed to read meta.json from {failed_count} patches")
            logger.info(
                f"Found {len(seed_points)} explicit_seed seed points in z-range [{z_min}, {z_max}]"
            )
            return seed_points

        else:
            logger.error(
                f"existing_patches_for_seeds path {patches_path} is neither a valid JSON file nor a directory"
            )
            return []

    # -------------------------------
    # Seeding / Expansion (watchdog Popen manager)
    # -------------------------------
    def _launch_seed_proc(self, seeding_params_file: Path, seed_point: Tuple[float, float, float], idx: int):
        """
        Launch a single seeding process with its own logfile and scratch TMPDIR.
        """
        env = self._exec_env()
        # Unique logfile and scratch
        log_path = self.proc_logs_dir / f"seed_{idx}_{int(time.time_ns())}.log"
        logf = open(log_path, "wb")
        tmpdir = self._unique_tmp(f"seed_{idx}")
        env["TMPDIR"] = str(tmpdir)  # FORCE override

        cmd = [
            str(self.bin_dir / "vc_grow_seg_from_seed"),
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(seeding_params_file),
            str(int(seed_point[0])),
            str(int(seed_point[1])),
            str(int(seed_point[2])),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        return {"p": p, "start": time.time(), "logf": logf, "idx": idx}

    def _launch_expand_proc(self, expansion_params_file: Path, idx: int):
        """
        Launch a single expansion process with its own logfile and scratch TMPDIR.
        """
        env = self._exec_env()
        log_path = self.proc_logs_dir / f"expand_{idx}_{int(time.time_ns())}.log"
        logf = open(log_path, "wb")
        tmpdir = self._unique_tmp(f"expand_{idx}")
        env["TMPDIR"] = str(tmpdir)  # FORCE override

        cmd = [
            str(self.bin_dir / "vc_grow_seg_from_seed"),
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(expansion_params_file),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        return {"p": p, "start": time.time(), "logf": logf, "idx": idx}

    def _run_watchdog_loop(
        self,
        tasks,      # list of (idx, payload) or idx
        launcher,   # function to create a proc dict
        parallel: int,
        stage_key: str,
    ) -> int:
        """
        Generic manager for seeding/expansion using Popen.
        Returns number of successful runs (rc==0).
        """
        active = []
        finished_durations: List[float] = []
        completed = 0
        successful_runs = 0
        total = max(1, len(tasks))
        next_watch = None
        last_status = 0.0
        status_period = float(self.config.get("status_log_period_sec", 60.0))
        watch_armed_logged = False

        def _close_logf(t):
            try:
                t["logf"].close()
            except Exception:
                pass

        while tasks or active:
            # Fill available slots up to 'parallel'
            while tasks and len(active) < parallel:
                item = tasks.pop(0)
                t = launcher(item[0], item[1]) if isinstance(item, tuple) else launcher(item)
                active.append(t)

            # Poll actives
            now = time.time()
            still = []
            for t in active:
                rc = t["p"].poll()
                if rc is None:
                    still.append(t)
                    continue
                dur = now - t["start"]
                finished_durations.append(dur)
                completed += 1
                _close_logf(t)
                if rc == 0:
                    successful_runs += 1
            active = still

            # Periodic status
            if now - last_status >= status_period:
                progress_total = (completed / total) if total > 0 else 0.0
                logger.info(
                    f"[{stage_key}] active={len(active)} completed={completed}/{total} "
                    f"queued={len(tasks)} (parallel={parallel}) "
                    f"progress_total={progress_total:.1%}"
                )
                last_status = now

            # Watchdog: only after the requested fraction of TOTAL work is done
            trigger_completed = max(1, math.ceil(total * self._watch_trigger_fraction))
            if completed >= trigger_completed:
                if not watch_armed_logged:
                    logger.info(
                        f"[watchdog] Activated for {stage_key}: "
                        f"completed={completed}/{total} (>= {trigger_completed}); "
                        f"threshold_fraction={self._watch_trigger_fraction:.2f}"
                    )
                    watch_armed_logged = True
                if next_watch is None:
                    next_watch = now  # fire immediately at first trigger
                if now >= next_watch:
                    thr = self._robust_threshold(finished_durations)
                    if thr is not None and active:
                        kills = 0
                        for t in list(active):
                            elapsed = now - t["start"]
                            if elapsed > thr:
                                logger.warning(
                                    f"[watchdog] Killing slow {stage_key} task idx={t['idx']} "
                                    f"elapsed={elapsed:.1f}s > thr={thr:.1f}s"
                                )
                                try:
                                    t["p"].terminate()
                                    try:
                                        t["p"].wait(self._watch_grace_seconds)
                                    except subprocess.TimeoutExpired:
                                        t["p"].kill()
                                    _close_logf(t)
                                    kills += 1
                                except Exception as e:
                                    logger.warning(f"[watchdog] terminate failed: {e}")
                        if kills:
                            self.watchdog_kills[stage_key] += kills
                            logger.info(f"[watchdog] Killed {kills} {stage_key} task(s) this check")
                    next_watch = now + self._watch_check_period

            time.sleep(0.5)

        return successful_runs

    def run_seeding(self, seed_points: List[Tuple[float, float, float]]) -> List[Path]:
        """
        Run vc_grow_seg_from_seed for explicit seeds with watchdog supervision,
        then harvest any patch directories created under patches_dir.
        """
        logger.info(f"Running vc_grow_seg_from_seed seeding for {len(seed_points)} seed points")

        seeding_params = self.config["vc_grow_seg_from_seed_params"]["seeding"].copy()
        seeding_params["mode"] = "seed"
        seeding_params_file = self.out_dir / "seeding_params.json"
        with open(seeding_params_file, "w") as f:
            json.dump(seeding_params, f, indent=2)

        max_num = int(self.config.get("max_num_seeds", len(seed_points)))
        parallel = int(self.config["seeding_parallel_processes"])
        tasks = [(i, sp) for i, sp in enumerate(seed_points[:max_num])]

        successful = self._run_watchdog_loop(
            tasks,
            lambda idx, sp: self._launch_seed_proc(seeding_params_file, sp, idx),
            parallel,
            "seeding",
        )

        # Collect all created patches from patches directory (meta.json is the marker)
        created_patches = []
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists():
                created_patches.append(patch_dir)

        logger.info(f"Completed {successful} seed runs, found {len(created_patches)} patches in results directory")
        return created_patches

    def run_expansion(self, existing_patches: List[Path]) -> List[Path]:
        """
        Run vc_grow_seg_from_seed in expansion mode N times with watchdog supervision,
        then return only the *new* patch directories (not in existing_patches).
        """
        num_expansion_patches = int(self.config.get("num_expansion_patches", 1))
        logger.info(f"Running vc_grow_seg_from_seed in expansion mode {num_expansion_patches} times")

        expansion_params = self.config["vc_grow_seg_from_seed_params"]["expansion"].copy()
        expansion_params["mode"] = "expansion"
        expansion_params_file = self.out_dir / "expansion_params.json"
        with open(expansion_params_file, "w") as f:
            json.dump(expansion_params, f, indent=2)

        parallel = int(self.config.get("expansion_parallel_processes", self.config.get("seeding_parallel_processes", 1)))
        tasks = list(range(num_expansion_patches))

        successful = self._run_watchdog_loop(
            tasks, lambda idx: self._launch_expand_proc(expansion_params_file, idx), parallel, "expansion"
        )

        # Collect new patches created by expansion runs
        all_patches = []
        existing_patches_set = set(existing_patches)
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists() and patch_dir not in existing_patches_set:
                all_patches.append(patch_dir)

        logger.info(f"Completed {successful} successful expansion runs, found {len(all_patches)} new patches")
        return all_patches

    # -------------------------------
    # Patch selection for tracing
    # -------------------------------
    def get_trace_starting_patches(self, patches: List[Path]) -> List[PatchInfo]:
        """
        Read meta.json for each patch, compute areas, filter by min_size,
        then select num_trace_starting_patches patches either via 'top_k' or
        evenly spaced 'quantiles' selection strategy.
        """
        patch_infos = []
        for patch_dir in patches:
            try:
                with open(patch_dir / "meta.json", "r") as f:
                    meta = json.load(f)
                area = float(meta.get("area_vx2", 0.0) or 0.0)
                patch_infos.append(PatchInfo(path=patch_dir, area=area))
            except Exception as e:
                logger.warning(f"Error reading meta.json from {patch_dir}: {e}")
                continue

        min_size = float(self.config.get("min_trace_starting_patch_size", 0.0))
        filtered = [p for p in patch_infos if p.area >= min_size]
        if not filtered:
            logger.warning(f"No patches found with area >= {min_size}")
            return []

        k = max(0, int(self.config.get("num_trace_starting_patches", 1)))
        filtered.sort(key=lambda p: p.area, reverse=True)
        n = len(filtered)

        if k <= 1:
            return filtered[:min(1, n)]
        if k >= n:
            return filtered

        # selection strategy: "quantiles" (default) or "top_k"
        strategy = str(self.config.get("trace_starting_selection", "quantiles")).lower()
        if strategy == "top_k":
            # strictly take the k largest patches
            return filtered[:k]
        # default: evenly spaced indices across [0, n-1] (quantiles)
        idxs = [(i * (n - 1)) // (k - 1) for i in range(k)]
        return [filtered[i] for i in idxs]

    # -------------------------------
    # Tracing (both flip_x variants)
    # -------------------------------
    def run_tracer(self, source_patches: List[PatchInfo]) -> List[Path]:
        """
        For each source patch, run vc_grow_seg_from_segments twice:
        • flip_x = 0
        • flip_x = 1
        We run sequentially for determinism and simpler resource behavior.
        Each run writes into its own run_traces_dir and uses an isolated TMPDIR
        under out/scratch to avoid temp-file collisions.

        RESTORED BEHAVIOR:
        • Final trace selection is the LEXICOGRAPHICALLY LAST complete candidate
            (ignoring *_opt), matching the old harness behavior.
        • Do NOT bail out on non-zero return code; we still scan the run folder
            and salvage a complete trace if present (old harness tolerance).
        """
        logger.info(f"Running vc_grow_seg_from_segments (both flip_x variants) for {len(source_patches)} source patches")

        # Base params (optionally inject z_range)
        base_params = self.config["vc_grow_seg_from_segments_params"].copy()
        if "z_range" in self.config:
            base_params["z_range"] = self.config["z_range"]

        # Materialize params per flip
        param_files: Dict[bool, Path] = {}
        for fv in (False, True):
            params = base_params.copy()
            params["flip_x"] = 1 if fv else 0
            pf = self.out_dir / f"tracer_params_fx{int(fv)}.json"
            with open(pf, "w") as f:
                json.dump(params, f, indent=2)
            param_files[fv] = pf

        trace_paths: List[Path] = []
        logger.info("Tracing sequentially (parallelism disabled)")
        base_env = self._exec_env()

        def _run_one(source_patch: PatchInfo, tracer_params_file: Path, run_tag: str) -> Optional[Path]:
            """
            Launch one vc_grow_seg_from_segments run with its own output folder and scratch.
            Select the *final* trace using the EXACT OLD rule: pick the lexicographically
            last trace directory name among complete candidates (ignoring *_opt).
            If the subprocess exits non-zero, we still scan/salvage any complete trace.
            """
            ts = time.time_ns()
            tag = f"_{run_tag}" if run_tag else ""
            run_traces_dir = self.traces_dir / f"from_{source_patch.path.name}{tag}_{ts}"
            run_traces_dir.mkdir(exist_ok=True)

            # Per-run env to prevent temp-file collisions across processes
            env = base_env.copy()
            env["TMPDIR"] = str(self._unique_tmp(f"trace_{source_patch.path.name}_{run_tag}"))

            cmd = [
                str(self.bin_dir / "vc_grow_seg_from_segments"),
                self.config["surface_zarr_volume"],
                str(self.patches_dir),
                str(run_traces_dir),
                str(tracer_params_file),
                str(source_patch.path),
            ]

            # Log stdout/stderr to a dedicated file, not to memory
            log_path = self.proc_logs_dir / f"trace_{source_patch.path.name}_{run_tag}_{ts}.log"
            logger.info(f"Starting vc_grow_seg_from_segments run from {source_patch.path.name} ({run_tag})")
            with open(log_path, "wb") as lf:
                result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            logger.info(f"Finished vc_grow_seg_from_segments run (return code {result.returncode})")

            if result.returncode != 0:
                logger.error(
                    f"vc_grow_seg_from_segments exited with rc={result.returncode} "
                    f"(attempting to salvage any complete traces); see log: {log_path}"
                )

            # Enumerate candidate traces: require complete tifXYZ + meta.json, ignore *_opt folders
            candidates: List[Path] = []
            for td in run_traces_dir.iterdir():
                if td.is_dir() and not td.name.endswith("_opt") and self._trace_complete(td):
                    candidates.append(td)

            if not candidates:
                # Nothing to keep (even if rc==0 or rc!=0). This mirrors the old harness’ tolerance:
                # we only reject the run if no complete outputs exist.
                logger.warning(f"No trace produced starting from patch {source_patch.path.name} ({run_tag})")
                return None

            # ---- RESTORED: exact old final-choice rule (lexicographic last by name) ----
            candidates.sort(key=lambda td: td.name)
            final_td = candidates[-1]

            if result.returncode != 0:
                logger.warning(
                    f"Using salvaged trace {final_td.name} from a non-zero exit run "
                    f"(patch {source_patch.path.name}, {run_tag})"
                )
            else:
                logger.info(f"Selected final trace {final_td.name} for patch {source_patch.path.name} ({run_tag})")

            return final_td

        # Strictly sequential: iterate patches, and for each patch run fx0 then fx1.
        for patch_info in source_patches:
            for fv in (False, True):  # deterministic order
                pf = param_files[fv]
                tag = f"fx{int(fv)}"
                try:
                    res = _run_one(patch_info, pf, tag)
                    if res:
                        trace_paths.append(res)
                except Exception as e:
                    logger.error(f"Error in tracer run for {patch_info.path.name} ({tag}): {e}")

        logger.info(f"Created {len(trace_paths)} valid traces")
        return trace_paths

    # -------------------------------
    # Winding numbers
    # -------------------------------
    def run_winding_numbers(self, traces: List[Path]) -> List[Path]:
        """
        Run vc_tifxyz_winding within each trace directory.
        Preflight ensures we never call winding on an incomplete/empty trace.
        """
        logger.info(f"Running vc_tifxyz_winding for {len(traces)} traces")
        env_base = self._exec_env()

        successful = []
        for trace_dir in traces:
            if not self._trace_complete(trace_dir):
                logger.error(f"Skipping winding: incomplete or empty tifXYZ in {trace_dir.name}")
                continue
            env = env_base.copy()
            env["TMPDIR"] = str(self._unique_tmp(f"winding_{trace_dir.name}"))
            cmd = [str(self.bin_dir / "vc_tifxyz_winding"), "."]
            logger.info(f"Starting vc_tifxyz_winding for {trace_dir.name}")
            result = subprocess.run(cmd, cwd=trace_dir, env=env, capture_output=True, text=True)
            logger.info(f"Finished vc_tifxyz_winding (return code {result.returncode})")

            winding_file = trace_dir / "winding.tif"
            if result.returncode == 0 and winding_file.exists():
                successful.append(trace_dir)
            else:
                logger.error(f"Failed to calculate winding numbers for {trace_dir.name}")
                if result.stdout:
                    logger.error(f"vc_tifxyz_winding STDOUT:\n{result.stdout}")
                if result.stderr:
                    logger.error(f"vc_tifxyz_winding STDERR:\n{result.stderr}")
        logger.info(f"Completed winding calculation for {len(successful)} traces")
        return successful

    # -------------------------------
    # Metrics
    # -------------------------------
    def run_metrics(self, traces: List[Path]) -> Dict[Path, Dict]:
        """
        Run vc_calc_surface_metrics for each trace with winding.tif present.
        """
        logger.info(f"Running vc_calc_surface_metrics for {len(traces)} traces")
        env = self._exec_env()

        results: Dict[Path, Dict] = {}
        for trace_dir in traces:
            metrics_file = trace_dir / "metrics.json"
            z_range = self.config.get("z_range", [-1, -1])
            cmd = [
                str(self.bin_dir / "vc_calc_surface_metrics"),
                "--collection",
                self.config["wrap_labels"],
                "--surface",
                str(trace_dir),
                "--winding",
                str(trace_dir / "winding.tif"),
                "--output",
                str(metrics_file),
                "--z_min",
                str(z_range[0]),
                "--z_max",
                str(z_range[1]),
            ]
            # Isolate metrics temp as well (cheap + consistent)
            env["TMPDIR"] = str(self._unique_tmp(f"metrics_{trace_dir.name}"))
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0 and metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    results[trace_dir] = metrics
                    logger.info(f"Successfully calculated metrics for {trace_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to parse metrics.json for {trace_dir.name}: {e}")
            else:
                logger.error(f"Failed to calculate metrics for {trace_dir.name}")
                if result.stdout:
                    logger.error(f"vc_calc_surface_metrics STDOUT:\n{result.stdout}")
                if result.stderr:
                    logger.error(f"vc_calc_surface_metrics STDERR:\n{result.stderr}")

        logger.info(f"Completed metrics calculation for {len(results)} traces")
        return results

    # -------------------------------
    # Collate & W&B logging (scalar-only)
    # -------------------------------
    def _wandb_safe_config(self) -> Dict[str, str]:
        """
        Convert non-scalar config values into compact JSON strings
        to avoid implicit Tables and IMMUTABLE warnings.
        """
        safe = {}
        for k, v in self.config.items():
            if isinstance(v, (str, bool, numbers.Number)) or v is None:
                safe[k] = v
            else:
                try:
                    safe[k] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)[:10000]
                except Exception:
                    safe[k] = str(v)
        return safe

    def collate_and_log_metrics(self, metrics_results: Dict[Path, Dict]):
        """
        Build scalar means across the top-K (by ranking metric) traces and, if configured,
        log them to Weights & Biases along with watchdog kill counters.
        """
        logger.info("Collating metrics and logging to wandb")

        # Prepare scalar summary (may be empty if no metrics)
        metric_to_mean: Dict[str, float] = {}

        if metrics_results:
            ranking_metric = self.config["trace_ranking_metric"]

            def safe_metric(trace: Path) -> float:
                v = metrics_results.get(trace, {}).get(ranking_metric, None)
                if self._is_finite_number(v):
                    return float(v)  # type: ignore[arg-type]
                logger.debug(
                    f"Trace {trace.name} missing/non-numeric ranking metric '{ranking_metric}': {v}"
                )
                return float("-inf")

            ranked = [t for t in metrics_results.keys() if safe_metric(t) != float("-inf")]
            if not ranked:
                logger.warning(
                    f"No traces contained a numeric '{ranking_metric}'. Skipping summarization."
                )
            else:
                ranked.sort(key=safe_metric, reverse=True)
                best_traces = ranked[: int(self.config["num_best_traces_to_average"])]

                metric_to_values = defaultdict(list)
                for trace in best_traces:
                    for metric_name, value in metrics_results[trace].items():
                        if self._is_finite_number(value):
                            metric_to_values[metric_name].append(float(value))

                metric_to_mean = {
                    metric_name: (sum(values) / len(values))
                    for metric_name, values in metric_to_values.items()
                    if values
                }

                logger.info(
                    f'final metrics, average over best {self.config["num_best_traces_to_average"]} traces:'
                )
                for metric_name, mean_val in metric_to_mean.items():
                    logger.info(f"  {metric_name}: {mean_val}")

        # ---- W&B (always attempt to log watchdog counters if project set) ----
        if "wandb_project" in self.config:
            try:
                # Env for W&B service in multiprocess envs
                os.environ.setdefault("WANDB_START_METHOD", "thread")
                os.environ.setdefault("WANDB_SILENT", "true")

                import wandb  # local import

                # Ignore heavy paths
                default_ignores = [
                    "patches/**",
                    "traces/**",
                    "**/*.tif",
                    "**/*.tiff",
                    "**/*.png",
                    "**/*.jpg",
                    "**/*.jpeg",
                    "**/*.zarr",
                    "**/*.npz",
                    "**/*.npy",
                    "**/*.h5",
                    "**/*.zip",
                    "**/*.tar",
                    "**/*.gz",
                ]
                ignore_globs = self.config.get("wandb_ignore_globs", default_ignores)
                os.environ.setdefault("WANDB_IGNORE_GLOBS", ",".join(ignore_globs))

                # Keep W&B files contained
                wandb_dir = self.out_dir / self.config.get("wandb_run_dir_name", "wandb_runs")
                wandb_dir.mkdir(parents=True, exist_ok=True)

                settings = wandb.Settings(
                    ignore_globs=tuple(ignore_globs),
                    save_code=False,
                    disable_code=True,
                    disable_git=True,
                    root_dir=str(wandb_dir),
                    mode=self.config.get("wandb_mode", "online"),
                )

                # Derive an informative run name
                run_name = os.environ.get("VC3D_RUN_NAME")
                if not run_name:
                    def _clean(s: str) -> str:
                        return "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in s.lower()).strip("-")

                    cfg_stem = _clean(self.config_path.stem)
                    tags_raw = self.config.get("wandb_tags", self.config.get("tags", []))
                    if not isinstance(tags_raw, list):
                        tags_raw = [str(tags_raw)]
                    tags_clean = "-".join(_clean(str(t)) for t in tags_raw if str(t).strip())
                    run_name = f"{cfg_stem}--{tags_clean}" if tags_clean else cfg_stem

                run = wandb.init(
                    project=self.config["wandb_project"],
                    config=self._wandb_safe_config(),
                    name=run_name,
                    tags=self.config.get("wandb_tags", self.config.get("tags", [])),
                    dir=str(wandb_dir),
                    settings=settings,
                )

                # Scalar row (metrics if any) + watchdog kill counters
                scalar_row = {k: float(v) for k, v in metric_to_mean.items() if self._is_finite_number(v)}
                scalar_row.update(
                    {
                        "watchdog_kills_seeding": float(self.watchdog_kills.get("seeding", 0)),
                        "watchdog_kills_expansion": float(self.watchdog_kills.get("expansion", 0)),
                        "watchdog_kills_total": float(
                            self.watchdog_kills.get("seeding", 0) + self.watchdog_kills.get("expansion", 0)
                        ),
                    }
                )
                # Always log at least the watchdog counts
                run.log(scalar_row or {"watchdog_kills_total": 0.0}, step=0, commit=True)
                run.finish()
            except Exception as e:
                logger.warning(f"wandb logging skipped: {e}")

    # -------------------------------
    # Driver
    # -------------------------------
    def run(self):
        """
        Main orchestration:
          • Optionally use existing patches (if configured) or run seeding+expansion
          • Choose starting patches
          • Run tracer, winding, metrics
          • Collate metrics and optionally log to W&B
        """
        try:
            if self.config.get("use_existing_patches", False):
                existing_patches = []
                for patch_dir in self.patches_dir.iterdir():
                    if patch_dir.is_dir() and all(
                        (patch_dir / filename).exists() for filename in ["meta.json", "x.tif", "y.tif", "z.tif"]
                    ):
                        existing_patches.append(patch_dir)
                logger.info(f"Using {len(existing_patches)} existing patches")
                all_patches = existing_patches
            else:
                seed_points = self.find_seed_points()
                if len(seed_points) == 0:
                    raise RuntimeError("No seed points found")
                seeding_patches = self.run_seeding(seed_points)
                expansion_patches = self.run_expansion(seeding_patches)
                all_patches = seeding_patches + expansion_patches

            top_patches = self.get_trace_starting_patches(all_patches)
            traces = self.run_tracer(top_patches)
            traces = self.run_winding_numbers(traces)

            metrics_results = self.run_metrics(traces)
            self.collate_and_log_metrics(metrics_results)

        except Exception as e:
            logger.error(f"Error: {e}")
            raise


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file: str):
    harness = SurfaceTracerEvaluation(config_file)
    harness.run()


if __name__ == "__main__":
    main()
