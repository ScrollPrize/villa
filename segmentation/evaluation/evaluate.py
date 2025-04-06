import argparse
import pandas as pd
import numpy as np
import tifffile
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from PIL import Image

def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    with config_file.open('r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config

def load_3d_tif(file_path: str) -> np.ndarray:
    return tifffile.imread(str(file_path))

def get_files_dict(folder: str, extension: str = ".tif") -> Dict[str, str]:
    folder_path = Path(folder)
    file_dict: Dict[str, str] = {}
    for file in folder_path.glob(f"*{extension}"):
        file_dict[file.stem] = str(file)
    return file_dict

def compute_metrics(label: np.ndarray, prediction: np.ndarray, metrics_config: List[Dict[str, Any]]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for metric in metrics_config:
        metric_type: str = metric["type"]
        hyperparams: Dict[str, Any] = metric.get("hyperparameters", {})
        try:
            mod = __import__(f"metrics.{metric_type}", fromlist=["compute"])
            metric_value = mod.compute(label, prediction, **hyperparams)
            if isinstance(metric_value, dict):
                results.update(metric_value)
            else:
                results[metric_type] = float(metric_value)
        except ModuleNotFoundError:
            print(f"Module for metric '{metric_type}' not found in the metrics folder. Skipping.")
        except AttributeError:
            print(f"Module for metric '{metric_type}' does not have a 'compute' function. Skipping.")
    return results

def log_histogram_for_metric(
    df: pd.DataFrame,
    metric: str,
    output_folder: str,
    bins: int = 20,
    color: str = "skyblue"
) -> None:
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in the results. Skipping histogram plot.")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[metric], bins=bins, kde=True, color=color)
    ax.set_title(f"Distribution of {metric}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    output_folder_path = Path(output_folder)
    plot_path = output_folder_path / f"{metric}_hist.png"
    plt.savefig(str(plot_path.resolve()))
    plt.close()
    # Open the saved image with PIL and log it with wandb.
    with Image.open(str(plot_path.resolve())) as img:
         wandb.log({f"{metric}_hist": wandb.Image(img)})

def compute_statistics(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    values = df[metric]
    stats = {
        f"{metric}/avg": values.mean(),
        f"{metric}/median": values.median(),
        f"{metric}/std": values.std(),
        f"{metric}/q1": values.quantile(0.25),
        f"{metric}/q3": values.quantile(0.75),
        f"{metric}/min": values.min(),
        f"{metric}/max": values.max(),
    }
    return stats

def main(args: argparse.Namespace) -> None:
    config: Dict[str, Any] = load_config(args.config)
    labels_folder: str = config["labels_folder"]
    predictions_folder: str = config["predictions_folder"]
    output_csv: str = config["output_csv"]
    output_folder: str = config["output_folder"]
    wandb_project: str = config["wandb_project"]
    run_name: str = config["run_name"]
    metrics_config: List[Dict[str, Any]] = config.get("metrics", [])
    
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project=wandb_project, name=run_name, config=config)
    
    label_files: Dict[str, str] = get_files_dict(labels_folder)
    pred_files: Dict[str, str] = get_files_dict(predictions_folder)
    
    common_bases = set(label_files.keys()).intersection(set(pred_files.keys()))
    if not common_bases:
        print("No matching files found between the two folders.")
        return
    
    results: List[Dict[str, Any]] = []
    
    for base in tqdm(common_bases, desc="Processing files"):
        label_path: str = label_files[base]
        pred_path: str = pred_files[base]
        
        label_data: np.ndarray = load_3d_tif(label_path)
        pred_data: np.ndarray = load_3d_tif(pred_path)
        
        metrics: Dict[str, float] = compute_metrics(label_data, pred_data, metrics_config)
        metrics_with_file = metrics.copy()
        metrics_with_file["file"] = base
        results.append(metrics_with_file)
        
        wandb.log(metrics)
    
    df = pd.DataFrame(results)
    output_csv_path = Path(output_csv)
    df.to_csv(output_csv_path, index=False)
    print(f"Metrics saved to {output_csv_path}")
    
    stats_summary: Dict[str, float] = {}
    # Compute statistics for every metric column (except "file")
    for col in df.columns:
        if col != "file":
            stats_summary.update(compute_statistics(df, col))
    wandb.log(stats_summary)
    
    # Log histograms for every metric column (except "file")
    for col in df.columns:
        if col != "file":
            log_histogram_for_metric(df, metric=col, output_folder=str(output_folder_path))
    
    for metric in metrics_config:
        metric_type: str = metric["type"]
        try:
            mod = importlib.import_module(f"metrics.{metric_type}")
            if hasattr(mod, "finalize"):
                bins = metric.get("hyperparameters", {}).get("bins", 20)
                mod.finalize(output_folder=str(output_folder_path), bins=bins)
        except Exception as e:
            print(f"Error finalizing metric '{metric_type}': {e}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for 3D .tif files using YAML configuration, dynamic metric import, and robust statistical logging."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args)
