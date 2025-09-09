#!/usr/bin/env python3
import os
import yaml
import argparse
import subprocess
import multiprocessing
import math
import time

def run_inference(dataset_id, configuration, fold, trainer_name, plans_name, input_folder, output_folder, gpu_id=0):
    """Run inference for a specific trainer configuration"""
    trainer_output_folder = os.path.join(output_folder, f"{trainer_name}")
    os.makedirs(trainer_output_folder, exist_ok=True)
    
    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", trainer_output_folder,
        "-d", dataset_id,
        "-c", configuration,
        "-f", str(fold),
        "-tr", trainer_name
    ]
    
    if plans_name is not None:
        cmd.extend(["-p", plans_name])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\nRunning: {' '.join(cmd)} on GPU {gpu_id}")
    
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run inference for all trained models")
    parser.add_argument("--config", type=str, default="config.yml", 
                        help="Path to the model optimization configuration file")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Path to the input folder containing images to predict")
    parser.add_argument("--output-folder", type=str, default="predictions",
                        help="Base path where predictions will be stored")
    parser.add_argument("--variants", type=str, default=None, 
                        help="Comma-separated list of variant names to run (without base trainer prefix)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    base_trainer = config['base_trainer']
    dataset_id = config['dataset_id']
    configuration = config['configuration']
    fold = config['fold']
    plans_name = config.get('plans', None)
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.variants:
        selected_variants = args.variants.split(',')
        configurations = [cfg for cfg in config['configurations'] if cfg['name'] in selected_variants]
    else:
        configurations = config['configurations']
    
    experiment_name = config.get('experiment_name', 'default')
    num_gpus = config.get('num_gpus', 1)
    
    plans_identifier = plans_name if plans_name else 'default_plans'
    experiment_output_folder = os.path.join(args.output_folder, f"{experiment_name}_{dataset_id}_{plans_identifier}")
    os.makedirs(experiment_output_folder, exist_ok=True)
    
    successful_runs = []
    failed_runs = []
    
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel inference")
        
        num_batches = math.ceil(len(configurations) / num_gpus)
        
        for batch in range(num_batches):
            print(f"\nProcessing batch {batch+1}/{num_batches}")
            
            start_idx = batch * num_gpus
            end_idx = min(start_idx + num_gpus, len(configurations))
            batch_configs = configurations[start_idx:end_idx]
            
            processes = []
            batch_results = multiprocessing.Manager().dict()
            
            for i, trainer_config in enumerate(batch_configs):
                gpu_id = i
                trainer_name = f"{base_trainer}_{trainer_config['name']}"
                
                print(f"\n{'=' * 80}")
                print(f"Starting inference for {trainer_name} on GPU {gpu_id}")
                print(f"{'=' * 80}")
                
                p = multiprocessing.Process(
                    target=lambda name, success_dict: success_dict.update({
                        name: run_inference(
                            dataset_id=dataset_id,
                            configuration=configuration,
                            fold=fold,
                            trainer_name=name,
                            plans_name=plans_name,
                            input_folder=args.input_folder,
                            output_folder=experiment_output_folder,
                            gpu_id=gpu_id
                        )
                    }),
                    args=(trainer_name, batch_results)
                )
                processes.append(p)
                p.start()
                time.sleep(1)
            
            for p in processes:
                p.join()
            
            for name, success in batch_results.items():
                if success:
                    successful_runs.append(name)
                else:
                    failed_runs.append(name)
            
            print(f"Batch {batch+1}/{num_batches} completed")
    else:
        print("Using a single GPU for sequential inference")
        
        for trainer_config in configurations:
            trainer_name = f"{base_trainer}_{trainer_config['name']}"
            
            print(f"\n{'=' * 80}")
            print(f"Running inference for {trainer_name} on GPU 0")
            print(f"{'=' * 80}")
            
            success = run_inference(
                dataset_id=dataset_id,
                configuration=configuration,
                fold=fold,
                trainer_name=trainer_name,
                plans_name=plans_name,
                input_folder=args.input_folder,
                output_folder=experiment_output_folder,
                gpu_id=0
            )
            
            if success:
                successful_runs.append(trainer_name)
            else:
                failed_runs.append(trainer_name)
    
    if 'evaluation' in config:
        config['evaluation']['predictions_base_folder'] = experiment_output_folder
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    print(f"\n{'=' * 80}")
    print(f"Inference Summary for Experiment: {experiment_name}")
    print(f"Dataset ID: {dataset_id}, Plans: {plans_identifier}")
    print(f"{'=' * 80}")
    print(f"Successful runs ({len(successful_runs)}):")
    for trainer in successful_runs:
        print(f"  - {trainer}")
    
    if failed_runs:
        print(f"\nFailed runs ({len(failed_runs)}):")
        for trainer in failed_runs:
            print(f"  - {trainer}")
    
    print(f"\nPredictions saved to {experiment_output_folder}")
    print(f"Next step: Run evaluation with evaluate_models.py")


if __name__ == "__main__":
    main()
