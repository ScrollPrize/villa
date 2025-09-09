#!/usr/bin/env python3
import os
import argparse
import subprocess
import yaml
import multiprocessing
import math
import time

def run_training_on_gpu(dataset_id, configuration, fold, trainer_name, gpu_id, 
                       plans_name=None, pretrained_weights=None, continue_training=False):
    """Run training on a specific GPU"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "nnUNetv2_train", dataset_id, configuration, str(fold),
        "-tr", trainer_name
    ]
    
    if plans_name is not None:
        cmd.extend(["-p", plans_name])
    
    if pretrained_weights is not None:
        cmd.extend(["-pretrained_weights", pretrained_weights])
    
    if continue_training:
        cmd.append("--c")
    
    print(f"\n{'='*80}")
    print(f"Running on GPU {gpu_id}: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running on GPU {gpu_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run training for multiple configurations")
    parser.add_argument("--config", type=str, default="config.yml", 
                        help="Path to the model optimization configuration file")
    parser.add_argument("--pretrained-weights", type=str, default=None,
                        help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--continue-training", action="store_true",
                        help="Continue training from the latest checkpoint")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on the trained model")
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
    
    num_gpus = config.get('num_gpus', 1)
    
    if args.variants:
        selected_variants = args.variants.split(',')
        configurations = [cfg for cfg in config['configurations'] if cfg['name'] in selected_variants]
    else:
        configurations = config['configurations']
    
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel training")
        
        # Group configurations for batch processing
        num_batches = math.ceil(len(configurations) / num_gpus)
        
        for batch in range(num_batches):
            print(f"\nProcessing batch {batch+1}/{num_batches}")
            
            start_idx = batch * num_gpus
            end_idx = min(start_idx + num_gpus, len(configurations))
            batch_configs = configurations[start_idx:end_idx]
            
            processes = []
            for i, trainer_config in enumerate(batch_configs):
                gpu_id = i
                trainer_name = f"{base_trainer}_{trainer_config['name']}"
                
                p = multiprocessing.Process(
                    target=run_training_on_gpu,
                    args=(dataset_id, configuration, fold, trainer_name, gpu_id),
                    kwargs={
                        "plans_name": plans_name,
                        "pretrained_weights": args.pretrained_weights,
                        "continue_training": args.continue_training
                    }
                )
                processes.append(p)
                p.start()
                # Small delay to prevent race conditions
                time.sleep(1)
            
            for p in processes:
                p.join()
            
            print(f"Batch {batch+1}/{num_batches} completed")
    else:
        print("Using a single GPU for sequential training")
        
        for trainer_config in configurations:
            trainer_name = f"{base_trainer}_{trainer_config['name']}"
            success = run_training_on_gpu(
                dataset_id, configuration, fold, trainer_name, 0,
                plans_name=plans_name,
                pretrained_weights=args.pretrained_weights,
                continue_training=args.continue_training
            )
            if not success:
                print(f"Training failed for {trainer_name}")
    
    print("\nAll training processes completed!")


if __name__ == "__main__":
    main()
