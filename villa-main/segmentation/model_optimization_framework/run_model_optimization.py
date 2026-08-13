#!/usr/bin/env python3
import argparse
import subprocess
import yaml

def run_step(script_name, args=None):
    """Run a specific step in the model optimization process"""
    cmd = ["python", script_name]
    
    if args is not None:
        cmd.extend(args)
    
    print(f"\n{'=' * 80}")
    print(f"Running {script_name} {' '.join(args if args else [])}")
    print(f"{'=' * 80}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run model optimization process end-to-end")
    parser.add_argument("--config", type=str, default="configs/config.yml", 
                        help="Path to the model optimization configuration file")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Path to the input folder containing images to predict")
    parser.add_argument("--labels-folder", type=str, required=True,
                        help="Path to the ground truth labels folder")
    parser.add_argument("--predictions-folder", type=str, default="predictions",
                        help="Base path where predictions will be stored")
    parser.add_argument("--evaluation-folder", type=str, default="evaluation_results",
                        help="Base path where evaluation results will be stored")
    parser.add_argument("--pretrained-weights", type=str, default=None,
                        help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--evaluation-script-dir", type=str, default=None,
                        help="Path to the directory containing custom evaluation scripts")
    parser.add_argument("--variants", type=str, default=None, 
                        help="Comma-separated list of variant names to run (without base trainer prefix)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for training (default: 1)")
    parser.add_argument("--skip-generate", action="store_true", help="Skip trainer generation step")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'evaluation' not in config:
        config['evaluation'] = {}
    
    config['evaluation']['labels_folder'] = args.labels_folder
    config['evaluation']['predictions_base_folder'] = args.predictions_folder
    config['evaluation']['output_folder'] = args.evaluation_folder
    
    if args.evaluation_script_dir:
        config['evaluation']['evaluation_script_dir'] = args.evaluation_script_dir
    
    config['num_gpus'] = args.num_gpus
    
    with open(args.config, 'w') as f:
        yaml.dump(config, f)
    
    variant_args = ["--variants", args.variants] if args.variants else []
    config_args = ["--config", args.config]
    
    steps = [
        {
            "name": "Generate trainer variants",
            "script": "generate_trainers.py",
            "args": config_args,
            "skip": args.skip_generate
        },
        {
            "name": "Run training",
            "script": "run_training.py",
            "args": config_args + variant_args + (["--pretrained-weights", args.pretrained_weights] if args.pretrained_weights else []),
            "skip": args.skip_training
        },
        {
            "name": "Run inference",
            "script": "run_inference.py",
            "args": config_args + variant_args + ["--input-folder", args.input_folder, "--output-folder", args.predictions_folder],
            "skip": args.skip_inference
        },
        {
            "name": "Evaluate models",
            "script": "evaluate_models.py",
            "args": config_args + variant_args + ["--labels-folder", args.labels_folder, "--predictions-folder", args.predictions_folder, "--output-folder", args.evaluation_folder] + (["--evaluation-script-dir", args.evaluation_script_dir] if args.evaluation_script_dir else []),
            "skip": args.skip_evaluation
        }
    ]
    
    for step in steps:
        if not step["skip"]:
            print(f"\nRunning step: {step['name']}")
            success = run_step(step["script"], step["args"])
            
            if not success:
                print(f"\nError in step: {step['name']}. Stopping execution.")
                break
        else:
            print(f"\nSkipping step: {step['name']}")
    
    print("\nModel optimization process completed!")
    print(f"Check {args.evaluation_folder} for evaluation results")


if __name__ == "__main__":
    main()
