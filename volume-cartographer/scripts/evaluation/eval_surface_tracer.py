
import os
import json
import time
import click
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    path: Path
    area: float


class SurfaceTracerEvaluation:
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.out_dir = Path(self.config["out_path"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir = self.out_dir / "patches"
        self.traces_dir = self.out_dir / "traces"
        self.patches_dir.mkdir(exist_ok=self.config["use_existing_patches"])
        self.traces_dir.mkdir(exist_ok=True)

    def find_seed_points(self) -> List[Tuple[float, float, float]]:
        # Find seed points from existing patches, filtering by z-range and vc_gsfs_mode = explicit_seed
        patches_path = Path(self.config["existing_patches_for_seeds"])
        z_min, z_max = self.config["z_range"]
        
        if patches_path.is_file() and patches_path.suffix == '.json':

            with open(patches_path, 'r') as f:
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
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    if meta.get("vc_gsfs_mode") != "explicit_seed":
                        continue
                    seed = meta.get("seed")
                    if not seed or len(seed) != 3:
                        continue
                    x, y, z = seed
                    if z_min <= z <= z_max:
                        seed_points.append((x, y, z))
                except Exception as e:
                    failed_count += 1
                    continue
            
            logger.warning(f"Failed to read meta.json from {failed_count} patches")
            logger.info(f"Found {len(seed_points)} explicit_seed seed points in z-range [{z_min}, {z_max}]")
            return seed_points
        
        else:
            logger.error(f"existing_patches_for_seeds path {patches_path} is neither a valid JSON file nor a directory")
            return []
    
    def _run_vc_grow_seg_from_seed(self, mode: str, params_file: Path, seed_point: Tuple[float, float, float] = None) -> bool:
        
        cmd = [
            f"{self.config['bin_path']}/vc_grow_seg_from_seed",
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(params_file)
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
        with open(seeding_params_file, 'w') as f:
            json.dump(seeding_params, f, indent=2)

        max_num_seeds = self.config.get("max_num_seeds", len(seed_points))
        
        successful_runs = 0
        with ProcessPoolExecutor(max_workers=self.config["seeding_parallel_processes"]) as executor:
            futures = []
            
            for i, seed_point in enumerate(seed_points[:max_num_seeds]):
                future = executor.submit(
                    self._run_vc_grow_seg_from_seed,
                    "seeding", seeding_params_file, seed_point
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_runs += 1
                except Exception as e:
                    logger.error(f"Error in seed growth: {e}")
        
        # Collect all created patches from patches directory
        created_patches = []
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists():
                created_patches.append(patch_dir)
        
        logger.info(f"Completed {successful_runs} seed runs, found {len(created_patches)} patches in results directory")
        return created_patches
    
    def run_expansion(self, existing_patches: List[Path]) -> List[Path]:

        num_expansion_patches = self.config.get("num_expansion_patches", 1)
        logger.info(f"Running vc_grow_seg_from_seed in expansion mode {num_expansion_patches} times")
        
        expansion_params = self.config["vc_grow_seg_from_seed_params"]["expansion"].copy()
        expansion_params["mode"] = "expansion"
        
        expansion_params_file = self.out_dir / "expansion_params.json"
        with open(expansion_params_file, 'w') as f:
            json.dump(expansion_params, f, indent=2)
        
        successful_runs = 0
        with ProcessPoolExecutor(max_workers=self.config.get("seeding_parallel_processes")) as executor:
            futures = []
            
            for i in range(num_expansion_patches):
                future = executor.submit(
                    self._run_vc_grow_seg_from_seed,
                    "expansion", expansion_params_file, None
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_runs += 1
                except Exception as e:
                    logger.error(f"Error in expansion run: {e}")
        
        # Collect new patches created by expansion runs
        all_patches = []
        existing_patches = set(existing_patches)
        for patch_dir in self.patches_dir.iterdir():
            if patch_dir.is_dir() and (patch_dir / "meta.json").exists() and patch_dir not in existing_patches:
                all_patches.append(patch_dir)
        
        logger.info(f"Completed {successful_runs} successful expansion runs, found {len(all_patches)} new patches")
        return all_patches
    
    def get_trace_starting_patches(self, patches: List[Path]) -> List[PatchInfo]:
        patch_infos = []
        for patch_dir in patches:
            meta_file = patch_dir / "meta.json"
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                area = meta.get("area_vx2", 0)
                patch_infos.append(PatchInfo(
                    path=patch_dir,
                    area=area
                ))
            except Exception as e:
                logger.warning(f"Error reading meta.json from {patch_dir}: {e}")
                continue
        
        # Filter by minimum size, sort by area, then subsample deterministically
        min_size = self.config.get("min_trace_starting_patch_size", 0.0)
        filtered_patches = [p for p in patch_infos if p.area >= min_size]
        if not filtered_patches:
            logger.warning(f"No patches found with area >= {min_size}")
            return []
        target_count = self.config["num_trace_starting_patches"]
        if len(filtered_patches) <= target_count:
            return filtered_patches
        filtered_patches.sort(key=lambda p: p.area, reverse=True)
        return filtered_patches[::len(filtered_patches) // target_count][:target_count]
    
    def run_tracer(self, source_patches: List[PatchInfo]) -> List[Path]:

        logger.info(f"Running vc_grow_seg_from_segments for {len(source_patches)} source patches")
        
        tracer_params = self.config["vc_grow_seg_from_segments_params"].copy()
        if "z_range" in self.config:
            tracer_params["z_range"] = self.config["z_range"]
        tracer_params_file = self.out_dir / "tracer_params.json"
        with open(tracer_params_file, 'w') as f:
            json.dump(tracer_params, f, indent=2)
        
        trace_paths = []
        for patch_info in source_patches:
            result = self._run_vc_grow_seg_from_segments(patch_info, tracer_params_file)
            if result:
                trace_paths.append(result)
        
        logger.info(f"Created {len(trace_paths)} valid traces")
        
        return trace_paths
    
    def _run_vc_grow_seg_from_segments(self, source_patch: PatchInfo, tracer_params_file: Path) -> Optional[Tuple[Path, float]]:

        run_traces_dir = self.traces_dir / f"from_{source_patch.path.name}_{int(time.time())}"
        run_traces_dir.mkdir(exist_ok=True)

        cmd = [
            f"{self.config['bin_path']}/vc_grow_seg_from_segments",
            self.config["surface_zarr_volume"],
            str(self.patches_dir),
            str(run_traces_dir),
            str(tracer_params_file),
            str(source_patch.path)
        ]
        
        logger.info(f"Starting vc_grow_seg_from_segments run from {source_patch.path.name}")
        result = subprocess.run(cmd)
        logger.info(f"Finished vc_grow_seg_from_segments run (return code {result.returncode})")
        
        # Find the final trace
        trace_paths = []
        for trace_dir in run_traces_dir.iterdir():
            if all(os.path.exists(trace_dir / filename) for filename in ["meta.json", "x.tif", "y.tif", "z.tif"]) and not trace_dir.name.endswith("_opt"):
                trace_paths.append(trace_dir)
        if not trace_paths:
            logger.warning(f"No trace produced starting from patch {source_patch.path.name}")
            return None
        trace_paths.sort(key=lambda p: p.name)
        last_trace_path = trace_paths[-1]
        logger.info(f"Selected final trace {last_trace_path.name} for patch {source_patch.path.name}")
        
        return last_trace_path
    
    def run_winding_numbers(self, traces: List[Path]) -> List[Path]:

        logger.info(f"Running vc_tifxyz_winding for {len(traces)} traces")
        
        successful_traces = []
        for trace_dir in traces:
            if self._run_vc_tifxyz_winding(trace_dir):
                successful_traces.append(trace_dir)
        
        logger.info(f"Completed winding calculation for {len(successful_traces)} traces")
        return successful_traces
    
    def _run_vc_tifxyz_winding(self, trace_dir: Path) -> bool:
        
        cmd = [
            str(Path(self.config['bin_path']).resolve() / "vc_tifxyz_winding"),
            "."
        ]
        
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
        
        metrics_results = {}
        for trace_dir in traces:
            result = self._run_vc_calc_surface_metrics(trace_dir)
            if result:
                metrics_results[trace_dir] = result
        
        logger.info(f"Completed metrics calculation for {len(metrics_results)} traces")
        return metrics_results
    
    def _run_vc_calc_surface_metrics(self, trace_dir: Path) -> Optional[Dict]:
        
        metrics_file = trace_dir / "metrics.json"

        z_range = self.config.get("z_range", [-1, -1])  # -1 means entire surface bbox is used
        cmd = [
            f"{self.config['bin_path']}/vc_calc_surface_metrics",
            "--collection", self.config["wrap_labels"],
            "--surface", str(trace_dir),
            "--winding", str(trace_dir / "winding.tif"),
            "--output", str(metrics_file),
            "--z_min", str(z_range[0]),
            "--z_max", str(z_range[1]),
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0 and metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Successfully calculated metrics for {trace_dir.name}")
            return metrics
        else:
            logger.error(f"Failed to calculate metrics for {trace_dir.name}")
            return None
    
    def collate_and_log_metrics(self, metrics_results: Dict[Path, Dict]):

        logger.info("Collating metrics and logging to wandb")
        
        ranking_metric = self.config["trace_ranking_metric"]
        best_traces = sorted(
            metrics_results,
            key=lambda trace: metrics_results[trace][ranking_metric],
            reverse=True
        )[:self.config["num_best_traces_to_average"]]
        
        metric_to_values = defaultdict(list)
        for trace in best_traces:
            for metric_name in metrics_results[trace].keys():
                metric_to_values[metric_name].append(metrics_results[trace][metric_name])
        metric_to_mean = {
            metric_name: sum(values) / len(values)
            for metric_name, values in metric_to_values.items()
        }

        logger.info(f'final metrics, average over best {self.config["num_best_traces_to_average"]} traces:')
        for metric_name, mean in metric_to_mean.items():
            logger.info(f'  {metric_name}: {mean}')

        if "wandb_project" in self.config:
            import wandb
            wandb.init(
                project=self.config["wandb_project"],
                config=self.config,
                name=f"surface_tracer_{int(time.time())}",
                dir=self.out_dir,
            )
            wandb.summary.update(metric_to_mean)
            wandb.finish()
    
    def run(self):
        
        try:

            if self.config.get("use_existing_patches", False):
                existing_patches = []
                for patch_dir in self.patches_dir.iterdir():
                    if patch_dir.is_dir() and all((patch_dir / filename).exists() for filename in ["meta.json", "x.tif", "y.tif", "z.tif"]):
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
@click.argument('config_file', type=click.Path(exists=True))
def main(config_file: str):
    test_harness = SurfaceTracerEvaluation(config_file)
    test_harness.run()


if __name__ == '__main__':
    main()
