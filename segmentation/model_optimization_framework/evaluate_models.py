#!/usr/bin/env python3
import os
import yaml
import argparse
import subprocess

def evaluate_model(trainer_name, labels_folder, predictions_folder, output_folder, metrics_config, 
                   wandb_project, evaluation_script_dir, run_name=None):
    """Evaluate a model using the segmentation/evaluation framework"""
    model_output_folder = os.path.join(output_folder, trainer_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Create a configuration file for this specific evaluation
    model_run_name = run_name if run_name else trainer_name
    model_predictions_folder = os.path.join(predictions_folder, trainer_name)
    
    eval_config = {
        "labels_folder": labels_folder,
        "predictions_folder": model_predictions_folder,
        "output_csv": os.path.join(model_output_folder, f"{trainer_name}_metrics.csv"),
        "output_folder": model_output_folder,
        "wandb_project": wandb_project,
        "run_name": model_run_name,
        "metrics": metrics_config
    }
    
    # For each metric that has an output_folder parameter, set it to the model's output folder
    for metric in eval_config["metrics"]:
        if "hyperparameters" in metric and metric["hyperparameters"] is not None:
            if "output_folder" in metric["hyperparameters"]:
                metric["hyperparameters"]["output_folder"] = model_output_folder
    
    config_path = os.path.join(output_folder, f"{trainer_name}_eval_config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(eval_config, f)
    
    main_script = os.path.join(evaluation_script_dir, "evaluate.py")
    cmd = [
        "python", 
        main_script,
        "--config", config_path
    ]
    
    print(f"\nEvaluating model {trainer_name}")
    print(f"Config file: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {trainer_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--config", type=str, default="config.yml", 
                        help="Path to the model optimization configuration file")
    parser.add_argument("--evaluation-script-dir", type=str, required=True,
                        help="Path to the directory containing custom evaluation scripts")
    parser.add_argument("--labels-folder", type=str, required=True,
                        help="Path to the ground truth labels")
    parser.add_argument("--predictions-folder", type=str, default=None,
                        help="Base path to the predictions (default: from config)")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Base path where evaluation results will be stored (default: from config)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: from config)")
    parser.add_argument("--variants", type=str, default=None, 
                        help="Comma-separated list of variant names to evaluate (without base trainer prefix)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    base_trainer = config['base_trainer']
    
    experiment_name = config.get('experiment_name', 'default')
    
    eval_config = config.get('evaluation', {})
    predictions_base_folder = args.predictions_folder or eval_config.get('predictions_base_folder')
    output_base_folder = args.output_folder or eval_config.get('output_folder', 'evaluation_results')
    wandb_project = args.wandb_project or eval_config.get('wandb_project', 'nnunet-model-optimization')
    metrics_config = eval_config.get('metrics', [])
    
    if not predictions_base_folder:
        raise ValueError("Predictions folder not specified. Either provide it via --predictions-folder or in the config file.")
    
    dataset_id = config['dataset_id']
    plans_name = config.get('plans', None)
    plans_identifier = plans_name if plans_name else 'default_plans'
    experiment_identifier = f"{experiment_name}_{dataset_id}_{plans_identifier}"
    
    predictions_folder = os.path.join(predictions_base_folder, experiment_identifier)
    output_folder = os.path.join(output_base_folder, experiment_identifier)
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(predictions_folder):
        print(f"Warning: Predictions folder {predictions_folder} does not exist. Creating it.")
        os.makedirs(predictions_folder, exist_ok=True)
    
    if args.variants:
        selected_variants = args.variants.split(',')
        configurations = [cfg for cfg in config['configurations'] if cfg['name'] in selected_variants]
    else:
        configurations = config['configurations']
    
    eval_config['labels_folder'] = args.labels_folder
    eval_config['predictions_base_folder'] = predictions_base_folder
    eval_config['output_folder'] = output_base_folder
    config['evaluation'] = eval_config
    
    with open(args.config, 'w') as f:
        yaml.dump(config, f)
    
    # Run evaluation for each configuration
    successful_evaluations = []
    failed_evaluations = []
    
    for trainer_config in configurations:
        trainer_name = f"{base_trainer}_{trainer_config['name']}"
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating {trainer_name}")
        print(f"{'=' * 80}")
        
        success = evaluate_model(
            trainer_name=trainer_name,
            labels_folder=args.labels_folder,
            predictions_folder=predictions_folder,
            output_folder=output_folder,
            metrics_config=metrics_config,
            wandb_project=wandb_project,
            evaluation_script_dir=args.evaluation_script_dir
        )
        
        if success:
            successful_evaluations.append(trainer_name)
        else:
            failed_evaluations.append(trainer_name)
    
    print(f"\n{'=' * 80}")
    print(f"Evaluation Summary for Experiment: {experiment_name}")
    print(f"Dataset ID: {dataset_id}, Plans: {plans_identifier}")
    print(f"{'=' * 80}")
    print(f"Successful evaluations ({len(successful_evaluations)}):")
    for trainer in successful_evaluations:
        print(f"  - {trainer}")
    
    if failed_evaluations:
        print(f"\nFailed evaluations ({len(failed_evaluations)}):")
        for trainer in failed_evaluations:
            print(f"  - {trainer}")
    
    print(f"\nEvaluation results saved to {output_folder}")
    print(f"Next step: Run compare_models.py to analyze results")


if __name__ == "__main__":
    main()
