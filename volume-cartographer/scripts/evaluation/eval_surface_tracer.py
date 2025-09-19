import os
import json
import time
import click
import logging
import subprocess
import secrets
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional: wandb (the script still runs if it's not installed)
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    path: Path
    area: float


class SurfaceTracerEvaluation:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.out_dir = Path(self.config["out_path"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir = self.out_dir / "patches"
        self.traces_dir = self.out_dir / "traces"
        self.patches_dir.mkdir(exist_ok=self.config.get("use_existing_patches", False))
        self.traces_dir.mkdir(exist_ok=True)

        # --- W&B wiring (main process only) ---
        self.wandb_enabled = bool(self.config.get("wandb_project")) and (wandb is not None)

        # Unique suffix for run naming (UTC timestamp + short random hex)
        ts_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rand4 = secrets.token_hex(2)  # 4 hex chars

        default_base = "surface_tracer"
        default_id = f"{default_base}_{ts_str}_{rand4}"

        # id and name both get a unique suffix; if user provides a base, we append the suffix
        self.run_id = str(self.config.get("wandb_run_id", default_id))
        base_name = self.config.get("wandb_run_name", default_base)
        self.run_name = f"{base_name}_{ts_str}_{rand4}"

        self.run_group = self.config.get("wandb_group")
        self.run_tags = list(self.config.get("wandb_tags", []))
        self.job_type = self.config.get("wandb_job_type", "evaluation")
        self.commit_every = int(self.config.get("wandb_commit_every", 10))  # <— throttle knob


        # one global step across the whole pipeline
        self._global_step = 0

        # per-trace table state
        self._metrics_table = None
        self._metrics_table_columns: List[str] = []  # ["trace", ...metric names...]

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _init_wandb(self) -> None:
        if not self.wandb_enabled:
            return
        wandb.init(
            project=self.config["wandb_project"],
            config=self.config,
            name=self.run_name,
            dir=str(self.out_dir),
            id=self.run_id,
            resume="allow",
            group=self.run_group,
            job_type=self.job_type,
            tags=self.run_tags,
        )
        self._define_metrics_schema()
        # Log that we started
        self._log({"stage": "startup", "started_at": self._now_iso()})

    def _define_metrics_schema(self) -> None:
        if not self.wandb_enabled:
            return
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    def _log(self, data: Dict, commit: bool = True) -> None:
        """Centralized logging that adds a global step; no-op if wandb is disabled."""
        if not self.wandb_enabled:
            return
        if commit:
            self._global_step += 1
        payload = {"global_step": self._global_step}
        payload.update(data or {})
        wandb.log(payload, commit=commit)

    def _maybe_commit(self, i_completed: int, total: int, force: bool = False) -> bool:
        """
        Decide whether to commit this log based on the throttle.
        - commit every N completions (N = self.commit_every)
        - always commit when force=True or at the end
        """
        if force or i_completed >= total:
            return True
        return (i_completed % max(1, self.commit_every)) == 0

    def _ensure_metrics_table(self, metric_names: List[str]) -> None:
        """Initialize the per-trace metrics table with stable columns."""
        if not self.wandb_enabled:
            return
        # Initialize once with union of seen metrics
        if self._metrics_table is None:
            cols = ["trace"] + sorted(set(metric_names))
            self._metrics_table = wandb.Table(columns=cols)
            self._metrics_table_columns = cols
            wandb.log({"trace_metrics/init_time": self._now_iso(), "trace_metrics/table": self._metrics_table})
        else:
            # If new metric names appear later, rebuild with extended columns
            existing = set(self._metrics_table_columns[1:])
            new = set(metric_names) - existing
            if new:
                all_cols = ["trace"] + sorted(existing.union(new))
                # rebuild table with old data (W&B Tables are append-only; we re-create)
                old_rows = list(self._metrics_table.data)
                new_table = wandb.Table(columns=all_cols)
                # map old rows to new schema
                for row in old_rows:
                    # row layout matches previous columns
                    row_dict = dict(zip(self._metrics_table_columns, row))
                    new_row = [row_dict.get("trace")]
                    for c in all_cols[1:]:
                        new_row.append(row_dict.get(c))
                    new_table.add_data(*new_row)
                self._metrics_table = new_table
                self._metrics_table_columns = all_cols
                wandb.log({"trace_metrics/table": self._metrics_table}, commit=True)

    def _append_metrics_row(self, trace_name: str, metrics: Dict) -> None:
        if not self.wandb_enabled:
            return
        self._ensure_metrics_table(list(metrics.keys()))
        row = [trace_name]
        for c in self._metrics_table_columns[1:]:
            row.append(metrics.get(c))
        self._metrics_table.add_data(*row)
        # log the updated table without bumping global_step too often
        wandb.log({"trace_metrics/table": self._metrics_table}, commit=False)

    def _log_artifacts(self) -> None:
        if not self.wandb_enabled:
            return
        # Params artifacts — add any that exist
        params_art = wandb.Artifact(f"{self.run_id}-params", type="config")
        for fp in [
            self.out_dir / "seeding_params.json",
            self.out_dir / "expansion_params.json",
            self.out_dir / "tracer_params.json",
        ]:
            if fp.exists():
                params_art.add_file(str(fp))
        # log config artifact if it has at least one file
        if params_art.manifest.entries:
            wandb.log_artifact(params_art)

        # Directories (use references for remote stores instead)
        if self.patches_dir.exists():
            patches_art = wandb.Artifact(f"{self.run_id}-patches", type="dataset")
            patches_art.add_dir(str(self.patches_dir))
            wandb.log_artifact(patches_art)
        if self.traces_dir.exists():
            traces_art = wandb.Artifact(f"{self.run_id}-traces", type="dataset")
            traces_art.add_dir(str(self.traces_dir))
            wandb.log_artifact(traces_art)

    # -----------------------------
    # Pipeline bits
    # -----------------------------
    def find_seed_points(self) -> List[Tuple[float, float, float]]:
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

            logger.warning(f"Failed to read meta.json from {failed_count} patches")
            logger.info(f"Found {len(seed_points)} explicit_seed seed points in z-range [{z_min}, {z_max}]")
            return seed_points

        else:
            logger.error(
                f"existing_patches_for_seeds path {patches_path} is neither a valid JSON file nor a directory"
            )
            return []

    def _run_vc_grow_seg_from_seed(
        self, mode: str, params_file: Path, seed_point: Tuple[float, float, float] = None
    ) -> bool:
        cmd = [
            f"{self.config['bin_path']}/vc_grow_seg_from_seed",
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(params_file),
        ]
        if seed_point:
            cmd.extend([str(int(seed_point[0])), str(int(seed_point[1])), str(int(seed_point[2]))])

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        logger.info(f"Starting {mode} run of vc_grow_seg_from_seed")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Finished {mode} run")
            return True
        else:
            logger.error(f"Failed {mode} run")
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            return False

    def run_seeding(self, seed_points: List[Tuple[float, float, float]]) -> List[Path]:
        logger.info(f"Running vc_grow_seg_from_seed seeding for {len(seed_points)} seed points")

        seeding_params = self.config["vc_grow_seg_from_seed_params"]["seeding"].copy()
        seeding_params["mode"] = "seed"
        seeding_params_file = self.out_dir / "seeding_params.json"
        with open(seeding_params_file, "w") as f:
            json.dump(seeding_params, f, indent=2)

        max_num_seeds = self.config.get("max_num_seeds", len(seed_points))
        seeds_total = min(max_num_seeds, len(seed_points))

        successful_runs = 0
        with ProcessPoolExecutor(max_workers=self.config["seeding_parallel_processes"]) as executor:
            futures = []
            for i, seed_point in enumerate(seed_points[:seeds_total]):
                futures.append(
                    executor.submit(
                        self._run_vc_grow_seg_from_seed, "seeding", seeding_params_file, seed_point
                    )
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_runs += 1
                    # Throttled progress logging
                    commit = self._maybe_commit(successful_runs, seeds_total)
                    self._log(
                        {
                            "stage": "seeding",
                            "seeds_completed": successful_runs,
                            "seeds_total": seeds_total,
                        },
                        commit=commit,
                    )
                except Exception as e:
                    logger.error(f"Error in seed growth: {e}")
            # Force a commit at the end
            self._log({}, commit=True)

        # Collect all created patches from patches directory
        created_patches = []
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists():
                created_patches.append(patch_dir)

        logger.info(
            f"Completed {successful_runs} seed runs, found {len(created_patches)} patches in results directory"
        )
        self._log({"stage": "seeding", "created_patches": len(created_patches)}, commit=True)
        return created_patches

    def run_expansion(self, existing_patches: List[Path]) -> List[Path]:
        num_expansion_patches = int(self.config.get("num_expansion_patches", 1))
        logger.info(
            f"Running vc_grow_seg_from_seed in expansion mode {num_expansion_patches} times"
        )

        expansion_params = self.config["vc_grow_seg_from_seed_params"]["expansion"].copy()
        expansion_params["mode"] = "expansion"
        expansion_params_file = self.out_dir / "expansion_params.json"
        with open(expansion_params_file, "w") as f:
            json.dump(expansion_params, f, indent=2)

        successful_runs = 0
        with ProcessPoolExecutor(max_workers=self.config.get("seeding_parallel_processes", 1)) as executor:
            futures = []
            for _ in range(num_expansion_patches):
                futures.append(
                    executor.submit(
                        self._run_vc_grow_seg_from_seed, "expansion", expansion_params_file, None
                    )
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_runs += 1
                    commit = self._maybe_commit(successful_runs, num_expansion_patches)
                    self._log(
                        {
                            "stage": "expansion",
                            "expansion_runs_completed": successful_runs,
                            "expansion_runs_total": num_expansion_patches,
                        },
                        commit=commit,
                    )
                except Exception as e:
                    logger.error(f"Error in expansion run: {e}")
            self._log({}, commit=True)

        # Collect new patches created by expansion runs
        all_patches = []
        existing_patches_set = set(existing_patches)
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists() and patch_dir not in existing_patches_set:
                all_patches.append(patch_dir)

        logger.info(
            f"Completed {successful_runs} successful expansion runs, found {len(all_patches)} new patches"
        )
        self._log({"stage": "expansion", "new_patches": len(all_patches)}, commit=True)
        return all_patches

    def get_trace_starting_patches(self, patches: List[Path]) -> List[PatchInfo]:
        patch_infos = []
        for patch_dir in patches:
            meta_file = patch_dir / "meta.json"
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                area = meta.get("area_vx2", 0)
                patch_infos.append(PatchInfo(path=patch_dir, area=area))
            except Exception as e:
                logger.warning(f"Error reading meta.json from {patch_dir}: {e}")
                continue

        min_size = float(self.config.get("min_trace_starting_patch_size", 0.0))
        filtered = [p for p in patch_infos if p.area >= min_size]
        if not filtered:
            logger.warning(f"No patches found with area >= {min_size}")
            return []
        target_count = int(self.config["num_trace_starting_patches"])
        if len(filtered) <= target_count:
            selected = filtered
        else:
            filtered.sort(key=lambda p: p.area, reverse=True)
            step = max(1, len(filtered) // target_count)
            selected = filtered[::step][:target_count]

        self._log({"stage": "selection", "trace_candidates": len(selected)}, commit=True)
        return selected

    def _run_vc_grow_seg_from_segments(
        self, source_patch: PatchInfo, tracer_params_file: Path
    ) -> Optional[Path]:
        run_traces_dir = self.traces_dir / f"from_{source_patch.path.name}_{int(time.time())}"
        run_traces_dir.mkdir(exist_ok=True)

        cmd = [
            f"{self.config['bin_path']}/vc_grow_seg_from_segments",
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(run_traces_dir),
            str(tracer_params_file),
            str(source_patch.path),
        ]

        logger.info(f"Starting vc_grow_seg_from_segments run from {source_patch.path.name}")
        result = subprocess.run(cmd)
        logger.info(f"Finished vc_grow_seg_from_segments run (return code {result.returncode})")

        # Find the final trace
        trace_paths = []
        for trace_dir in run_traces_dir.iterdir():
            if all(
                (trace_dir / fname).exists()
                for fname in ["meta.json", "x.tif", "y.tif", "z.tif"]
            ) and not trace_dir.name.endswith("_opt"):
                trace_paths.append(trace_dir)
        if not trace_paths:
            logger.warning(f"No trace produced starting from patch {source_patch.path.name}")
            return None
        trace_paths.sort(key=lambda p: p.name)
        last_trace_path = trace_paths[-1]
        logger.info(f"Selected final trace {last_trace_path.name} for patch {source_patch.path.name}")
        return last_trace_path

    def run_tracer(self, source_patches: List[PatchInfo]) -> List[Path]:
        logger.info(f"Running vc_grow_seg_from_segments for {len(source_patches)} source patches")

        tracer_params = self.config["vc_grow_seg_from_segments_params"].copy()
        if "z_range" in self.config:
            tracer_params["z_range"] = self.config["z_range"]
        tracer_params_file = self.out_dir / "tracer_params.json"
        with open(tracer_params_file, "w") as f:
            json.dump(tracer_params, f, indent=2)

        trace_paths = []
        total = len(source_patches)
        for i, patch_info in enumerate(source_patches, start=1):
            result = self._run_vc_grow_seg_from_segments(patch_info, tracer_params_file)
            if result:
                trace_paths.append(result)
            commit = self._maybe_commit(i, total)
            self._log(
                {"stage": "tracing", "traces_attempted": i, "traces_completed": len(trace_paths), "traces_total": total},
                commit=commit,
            )

        logger.info(f"Created {len(trace_paths)} valid traces")
        self._log({"stage": "tracing", "traces_created": len(trace_paths)}, commit=True)
        return trace_paths

    def run_winding_numbers(self, traces: List[Path]) -> List[Path]:
        logger.info(f"Running vc_tifxyz_winding for {len(traces)} traces")

        successful_traces = []
        total = len(traces)
        for i, trace_dir in enumerate(traces, start=1):
            if self._run_vc_tifxyz_winding(trace_dir):
                successful_traces.append(trace_dir)
            commit = self._maybe_commit(i, total)
            self._log(
                {
                    "stage": "winding",
                    "winding_attempted": i,
                    "winding_completed": len(successful_traces),
                    "winding_total": total,
                },
                commit=commit,
            )

        logger.info(f"Completed winding calculation for {len(successful_traces)} traces")
        self._log({"stage": "winding", "winding_success": len(successful_traces)}, commit=True)
        return successful_traces

    def _run_vc_tifxyz_winding(self, trace_dir: Path) -> bool:
        cmd = [str(Path(self.config["bin_path"]).resolve() / "vc_tifxyz_winding"), "."]
        logger.info(f"Starting vc_tifxyz_winding for {trace_dir.name}")
        result = subprocess.run(cmd, cwd=trace_dir)
        logger.info(f"Finished vc_tifxyz_winding (return code {result.returncode})")

        winding_file = trace_dir / "winding.tif"
        if result.returncode == 0 and winding_file.exists():
            return True
        else:
            logger.error(f"Failed to calculate winding numbers for {trace_dir.name}")
            return False

    def run_metrics(self, traces: List[Path]) -> Dict[Path, Dict]:
        logger.info(f"Running vc_calc_surface_metrics for {len(traces)} traces")

        metrics_results: Dict[Path, Dict] = {}
        total = len(traces)
        for i, trace_dir in enumerate(traces, start=1):
            result = self._run_vc_calc_surface_metrics(trace_dir)
            if result:
                metrics_results[trace_dir] = result
                self._append_metrics_row(trace_dir.name, result)
            commit = self._maybe_commit(i, total)
            self._log(
                {
                    "stage": "metrics",
                    "metrics_attempted": i,
                    "metrics_completed": len(metrics_results),
                    "metrics_total": total,
                },
                commit=commit,
            )

        logger.info(f"Completed metrics calculation for {len(metrics_results)} traces")
        self._log({"stage": "metrics", "traces_with_metrics": len(metrics_results)}, commit=True)
        return metrics_results

    def _run_vc_calc_surface_metrics(self, trace_dir: Path) -> Optional[Dict]:
        metrics_file = trace_dir / "metrics.json"

        z_range = self.config.get("z_range", [-1, -1])  # -1 means entire surface bbox is used
        cmd = [
            f"{self.config['bin_path']}/vc_calc_surface_metrics",
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

        result = subprocess.run(cmd)

        if result.returncode == 0 and metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            logger.info(f"Successfully calculated metrics for {trace_dir.name}")
            return metrics
        else:
            logger.error(f"Failed to calculate metrics for {trace_dir.name}")
            return None

    def collate_and_log_metrics(self, metrics_results: Dict[Path, Dict]) -> None:
        logger.info("Collating metrics and logging to wandb")

        if not metrics_results:
            logger.warning("No metrics to collate.")
            return

        ranking_metric = self.config["trace_ranking_metric"]
        best_traces = sorted(
            metrics_results,
            key=lambda trace: metrics_results[trace][ranking_metric],
            reverse=True,
        )[: self.config["num_best_traces_to_average"]]

        metric_to_values = defaultdict(list)
        for trace in best_traces:
            for metric_name in metrics_results[trace].keys():
                metric_to_values[metric_name].append(metrics_results[trace][metric_name])
        metric_to_mean = {k: (sum(v) / max(1, len(v))) for k, v in metric_to_values.items()}

        logger.info(
            f'final metrics, average over best {self.config["num_best_traces_to_average"]} traces:'
        )
        for metric_name, mean in metric_to_mean.items():
            logger.info(f"  {metric_name}: {mean}")

        # Log aggregates
        self._log({f"final/{k}": v for k, v in metric_to_mean.items()}, commit=True)

        # Optional distribution insight for the ranking metric
        try:
            ranking_values = [metrics_results[t][ranking_metric] for t in metrics_results]
            if self.wandb_enabled and ranking_values:
                wandb.log({"final/ranking_hist": wandb.Histogram(ranking_values)}, commit=True)
        except Exception:
            pass

        # Upload artifacts (params, patches, traces)
        self._log_artifacts()

    # -----------------------------
    # Entrypoint
    # -----------------------------
    def run(self) -> None:
        try:
            self._init_wandb()

            if self.config.get("use_existing_patches", False):
                existing_patches = []
                for patch_dir in self.patches_dir.iterdir():
                    if patch_dir.is_dir() and all(
                        (patch_dir / filename).exists()
                        for filename in ["meta.json", "x.tif", "y.tif", "z.tif"]
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
        finally:
            if self.wandb_enabled and wandb.run is not None:
                wandb.finish()


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file: str):
    test_harness = SurfaceTracerEvaluation(config_file)
    test_harness.run()


if __name__ == "__main__":
    main()
