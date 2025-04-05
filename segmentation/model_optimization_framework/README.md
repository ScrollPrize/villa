# Model Optimization Framework for nnUNet

This framework provides a structured approach to model optimization for nnUNet, including hyperparameter tuning and other optimization techniques. It automates the process of creating trainer variants, running training, performing inference, and evaluating models.

## Components

The framework consists of the following components:

1. **Configuration file (`configs/config.yml`)**: Defines the hyperparameter space to explore
2. **Trainer generator (`generate_trainers.py`)**: Creates trainer variant classes
3. **Training script (`run_training.py`)**: Executes training for all configurations
4. **Inference script (`run_inference.py`)**: Runs inference on test data
5. **Evaluation script (`evaluate_models.py`)**: Computes metrics using the evaluation framework
6. **Master script (`run_model_optimization.py`)**: Orchestrates the entire process

## Setup

### Prerequisites

- nnUNet installed and configured
- Dataset preprocessed and ready for training
- Evaluation metrics in the `segmentation/evaluation` directory

### Configuration

Edit the `configs/config.yml` file to define:

1. The base trainer (e.g., `nnUNetTrainer`)
2. Dataset information (dataset ID, configuration, fold)
3. Hyperparameter configurations to explore
4. Evaluation settings
5. Number of GPUs to use for parallel training/inference

Example configuration:

```yaml
base_trainer: "nnUNetTrainer"
dataset_id: "Dataset003_sk-fibers_hzvt-augmented-2"
configuration: "3d_fullres"
fold: 0
plans: "nnUNetResEncUNetPlans_48G"
experiment_name: "my_experiment"  # Used for organizing output folders
num_gpus: 2  # Number of GPUs to use for parallel execution

configurations:
  - name: "baseline"
    params:
      initial_lr: 1e-2
      weight_decay: 3e-5
      num_epochs: 1000
      oversample_foreground_percent: 0.33
  
  - name: "low_lr"
    params:
      initial_lr: 5e-3
      weight_decay: 3e-5
      num_epochs: 1000
      oversample_foreground_percent: 0.33

evaluation:
  labels_folder: ""  # To be set at runtime
  predictions_base_folder: ""  # To be set at runtime
  output_folder: "evaluation_results"

  metrics:
    - type: "dice_multiclass"
      hyperparameters:
        num_classes: 4
        ignore_index: 0
    - type: "skeleton_distance_length"
      hyperparameters:
        distance_metric: "wasserstein"
        bins: 30
        epsilon: 1e-8
    - type: "critical_components_multiclass"
      hyperparameters: {}
    - type: "connected_components"
      hyperparameters:
        connectivity: 26
        num_classes: 4
        ignore_index: 0
```

## Usage

### Option 1: Run the entire pipeline

Use the master script to run the entire hyperparameter tuning process:

```bash
python run_model_optimization.py \
  --input-folder /path/to/input/data \
  --labels-folder /path/to/ground/truth \
  --predictions-folder predictions \
  --evaluation-folder evaluation_results \
  --evaluation-script-dir /path/to/evaluation/scripts
```

Optional arguments:

- `--config`: Path to the configuration file (default: configs/config.yml)
- `--pretrained-weights`: Path to pretrained weights for fine-tuning
- `--evaluation-script-dir`: Path to the directory containing custom evaluation scripts (required)
- `--variants`: Comma-separated list of variant names to run (e.g., "baseline,low_lr")
- `--num-gpus`: Number of GPUs to use for training (default: 1)
- `--skip-generate`: Skip trainer generation step
- `--skip-training`: Skip training step
- `--skip-inference`: Skip inference step
- `--skip-evaluation`: Skip evaluation step

### Option 2: Run steps individually

#### 1. Generate trainer variants

```bash
python generate_trainers.py --config configs/config.yml [--output-dir /path/to/output]
```

This will create trainer classes in the specified output directory. If `--output-dir` is not provided, it will try to find the nnUNet installation path and create the trainers in the appropriate subdirectory.

#### 2. Run training

```bash
python run_training.py --config configs/config.yml [--pretrained-weights /path/to/weights] [--variants baseline,low_lr] [--continue-training] [--validate-only]
```

Arguments:

- `--config`: Path to the configuration file (default: configs/config.yml)
- `--pretrained-weights`: Path to pretrained weights for fine-tuning
- `--variants`: Comma-separated list of variant names to run
- `--continue-training`: Continue training from the latest checkpoint
- `--validate-only`: Only run validation on the trained model

The script will use the `num_gpus` parameter from the config file to determine whether to run training in parallel across multiple GPUs.

#### 3. Run inference

```bash
python run_inference.py --config configs/config.yml --input-folder /path/to/input/data --output-folder predictions [--variants baseline,low_lr]
```

Arguments:

- `--config`: Path to the configuration file (default: configs/config.yml)
- `--input-folder`: Path to the folder containing input images to predict (required)
- `--output-folder`: Base path where predictions will be stored (default: predictions)
- `--variants`: Comma-separated list of variant names to run

The script will use the `num_gpus` parameter from the config file to determine whether to run inference in parallel across multiple GPUs.

#### 4. Evaluate models

```bash
python evaluate_models.py --config configs/config.yml --labels-folder /path/to/ground/truth --predictions-folder predictions --output-folder evaluation_results --evaluation-script-dir /path/to/evaluation/scripts [--wandb-project project_name] [--variants baseline,low_lr]
```

Arguments:

- `--config`: Path to the configuration file (default: configs/config.yml)
- `--labels-folder`: Path to the ground truth labels (required)
- `--evaluation-script-dir`: Path to the directory containing custom evaluation scripts (required)
- `--predictions-folder`: Base path to the predictions (default: from config)
- `--output-folder`: Base path where evaluation results will be stored (default: from config)
- `--wandb-project`: Weights & Biases project name for logging (default: from config)
- `--variants`: Comma-separated list of variant names to evaluate

This step computes metrics for each model using the evaluation framework in `segmentation/evaluation`.
