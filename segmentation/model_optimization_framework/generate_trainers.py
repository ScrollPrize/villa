#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
import argparse
import sys
import importlib.util
import inspect
import logging
from typing import Dict, Any, Optional, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_nnunet_path() -> Optional[str]:
    """Determine the path to the nnunetv2 package installation
    
    Returns:
        Optional[str]: Path to nnUNetv2 installation or None if not found
    """
    try:
        spec = importlib.util.find_spec('nnunetv2')
        if spec and spec.origin:
            nnunet_path = str(Path(spec.origin).parent.parent)
            logger.info(f"Found nnUNetv2 at: {nnunet_path}")
            return nnunet_path
        
        dlami_path = '/opt/dlami/nvme/nnUNet'
        if Path(dlami_path).exists():
            logger.info(f"Found nnUNetv2 at DLAMI path: {dlami_path}")
            return dlami_path
        
        logger.error("nnUNetv2 package not found. Please ensure it is installed correctly.")
        return None
    except ImportError as e:
        logger.error(f"Error locating nnunetv2: {e}")
        return None


def generate_optimizer_code(optimizer_config: Dict[str, Any]) -> str:
    """Generate code for the optimizer configuration.
    
    Args:
        optimizer_config: Dictionary containing optimizer configuration with the following structure:
            optimizer:
              type: <optimizer_type>  # e.g., SGD, Adam, AdamW, RMSprop
              params:
                param1: value1
                param2: value2
    
    Default optimizer parameters by type:
        SGD:
            momentum: 0.99
            nesterov: True
            weight_decay: self.weight_decay
        
        Adam/AdamW:
            betas: (0.9, 0.999)
            eps: 1e-8
            amsgrad: False
            weight_decay: self.weight_decay
    
    Example configurations:
        # SGD with momentum
        optimizer:
          type: SGD
          params:
            momentum: 0.9
            nesterov: True
            weight_decay: 0.0001
        
        # AdamW with custom betas
        optimizer:
          type: AdamW
          params:
            betas: [0.9, 0.999]
            eps: 1e-8
            weight_decay: 0.01
    
    Returns:
        str: Python code string for instantiating the optimizer
    """
    # Return default SGD optimizer if no config provided
    if not optimizer_config:
        return "torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)"
    
    optimizer_type = optimizer_config.get('type', 'SGD')
    optimizer_params = optimizer_config.get('params', {})
    
    param_list = ["self.network.parameters()", "lr=self.initial_lr"]
    
    if optimizer_type == 'SGD':
        param_list.extend([
            f"weight_decay={repr(optimizer_params.get('weight_decay', 'self.weight_decay'))}",
            f"momentum={repr(optimizer_params.get('momentum', 0.99))}",
            f"nesterov={repr(optimizer_params.get('nesterov', True))}"
        ])
    
    elif optimizer_type in ['Adam', 'AdamW']:
        beta1 = optimizer_params.get('beta1', 0.9)
        beta2 = optimizer_params.get('beta2', 0.999)
        
        param_list.extend([
            f"weight_decay={repr(optimizer_params.get('weight_decay', 'self.weight_decay'))}",
            f"betas=({repr(beta1)}, {repr(beta2)})",
            f"eps={repr(optimizer_params.get('eps', 1e-8))}",
            f"amsgrad={repr(optimizer_params.get('amsgrad', False))}"
        ])
    
    else:
        for k, v in optimizer_params.items():
            param_list.append(f"{k}={repr(v)}")
    
    # Remove first element (network parameters) to handle it separately
    params_str = param_list[0]
    options_str = ", ".join(param_list[1:])
    
    return f"torch.optim.{optimizer_type}({params_str}, {options_str})"


def generate_lr_scheduler_code(scheduler_config: Dict[str, Any]) -> Tuple[str, str]:
    """Generate code for the learning rate scheduler configuration.
    
    Args:
        scheduler_config: Dictionary containing scheduler configuration
        
    Returns:
        Tuple[str, str]: Import statement and scheduler initialization code
    """
    # Default scheduler if no config provided
    if not scheduler_config:
        scheduler_import = "from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler"
        scheduler_code = "PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)"
        return scheduler_import, scheduler_code
    
    scheduler_type = scheduler_config.get('type', 'PolyLR')
    scheduler_params = scheduler_config.get('params', {})
    
    if scheduler_type == 'PolyLR':
        scheduler_import = "from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler"
        
        param_list = [
            "optimizer",
            "self.initial_lr",
            "self.num_epochs",
            f"{scheduler_params.get('exponent', 0.9)}"
        ]
        
        scheduler_code = f"PolyLRScheduler({', '.join(param_list)})"
    
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler_import = "from torch.optim.lr_scheduler import CosineAnnealingLR"
        
        param_list = [
            "optimizer",
            f"T_max={scheduler_params.get('T_max', 'self.num_epochs')}",
            f"eta_min={scheduler_params.get('eta_min', 0)}"
        ]
        
        scheduler_code = f"CosineAnnealingLR({', '.join(param_list)})"
    
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler_import = "from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts"
        
        param_list = [
            "optimizer",
            f"T_0={scheduler_params.get('T_0', 'self.num_epochs // 3')}",
            f"T_mult={scheduler_params.get('T_mult', 1)}",
            f"eta_min={scheduler_params.get('eta_min', 0)}"
        ]
        
        scheduler_code = f"CosineAnnealingWarmRestarts({', '.join(param_list)})"
    
    elif scheduler_type == 'OneCycleLR':
        scheduler_import = "from torch.optim.lr_scheduler import OneCycleLR"
        
        # Use num_iterations_per_epoch instead of dataloader length to avoid None reference
        param_list = [
            "optimizer",
            f"max_lr=self.initial_lr",
            f"total_steps=self.num_epochs * self.num_iterations_per_epoch",
            f"pct_start={scheduler_params.get('pct_start', 0.3)}",
            f"div_factor={scheduler_params.get('div_factor', 25.0)}",
            f"final_div_factor={scheduler_params.get('final_div_factor', 10000.0)}"
        ]
        
        scheduler_code = f"OneCycleLR({', '.join(param_list)})"
    
    else:
        scheduler_import = f"from torch.optim.lr_scheduler import {scheduler_type}"
        
        param_list = ["optimizer"]
        for k, v in scheduler_params.items():
            param_list.append(f"{k}={v}")
        
        scheduler_code = f"{scheduler_type}({', '.join(param_list)})"
    
    return scheduler_import, scheduler_code


def generate_warmup_wrapper_code(scheduler_code: str, warmup_config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Generate code to wrap a scheduler with warmup.
    
    Args:
        scheduler_code: Code string for the main scheduler
        warmup_config: Dictionary containing warmup configuration
        
    Returns:
        Optional[Tuple[str, str]]: Import statement and wrapped scheduler code, or None if no warmup
    """
    if not warmup_config:
        return None
    
    warmup_type = warmup_config.get('type', 'linear')
    warmup_epochs = warmup_config.get('epochs', 5)
    start_factor = warmup_config.get('start_factor', 0.001)  # Start from 0.1% of base learning rate
    
    if warmup_type == 'linear':
        import_line = "from nnunetv2.training.lr_scheduler.linear_warmup import LinearWarmUpScheduler"
    elif warmup_type == 'exponential':
        import_line = "from nnunetv2.training.lr_scheduler.exponential_warmup import ExponentialWarmUpScheduler"
    else:
        logger.warning(f"Unknown warmup type {warmup_type}, defaulting to linear")
        import_line = "from nnunetv2.training.lr_scheduler.linear_warmup import LinearWarmUpScheduler"
    
    base_scheduler_code = f"base_scheduler = {scheduler_code}"
    
    if warmup_type == 'linear':
        warmup_scheduler = f"{base_scheduler_code}\n        lr_scheduler = LinearWarmUpScheduler(optimizer, warmup_steps={warmup_epochs}, base_scheduler=base_scheduler)"
    elif warmup_type == 'exponential':
        warmup_scheduler = f"{base_scheduler_code}\n        lr_scheduler = ExponentialWarmUpScheduler(optimizer, warmup_steps={warmup_epochs}, base_scheduler=base_scheduler, start_factor={start_factor})"
    else:
        logger.warning(f"Unknown warmup type {warmup_type}, defaulting to linear")
        warmup_scheduler = f"{base_scheduler_code}\n        lr_scheduler = LinearWarmUpScheduler(optimizer, warmup_steps={warmup_epochs}, base_scheduler=base_scheduler)"
    
    return import_line, warmup_scheduler


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that the configuration has the required structure and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(config, dict):
        logger.error(f"Configuration must be a dictionary, got {type(config)}")
        return False
    
    if "name" not in config:
        logger.error(f"Configuration missing required 'name' key: {config}")
        return False
    
    if "optimizer" in config:
        optimizer = config["optimizer"]
        if not isinstance(optimizer, dict):
            logger.error(f"Optimizer config must be a dictionary: {optimizer}")
            return False
        if "type" not in optimizer:
            logger.warning(f"Optimizer missing 'type' key, will use default: {optimizer}")
    
    if "scheduler" in config:
        scheduler = config["scheduler"]
        if not isinstance(scheduler, dict):
            logger.error(f"Scheduler config must be a dictionary: {scheduler}")
            return False
        if "type" not in scheduler:
            logger.warning(f"Scheduler missing 'type' key, will use default: {scheduler}")
    
    if "warmup" in config:
        warmup = config["warmup"]
        if not isinstance(warmup, dict):
            logger.error(f"Warmup config must be a dictionary: {warmup}")
            return False
        if "epochs" in warmup and not isinstance(warmup["epochs"], (int, float)):
            logger.error(f"Warmup epochs must be a number: {warmup['epochs']}")
            return False
    
    if "params" in config and not isinstance(config["params"], dict):
        logger.error(f"Parameters must be a dictionary: {config['params']}")
        return False
    
    return True


def generate_configure_optimizers_method(config: Dict[str, Any]) -> Optional[Tuple[List[str], str]]:
    """Generate the configure_optimizers method based on config.
    
    Args:
        config: Configuration dictionary with optimizer and scheduler settings
        
    Returns:
        Optional[Tuple[List[str], str]]: Import statements and method code, or None if no customization
    """

    optimizer_config = config.get('optimizer', None)
    scheduler_config = config.get('scheduler', None)

    warmup_config = scheduler_config.get('warmup', None) if scheduler_config else None
    
    if not any([optimizer_config, scheduler_config, warmup_config]):
        return None
    
    optimizer_code = generate_optimizer_code(optimizer_config) if optimizer_config else None
    
    scheduler_import = None
    scheduler_code = None
    if scheduler_config:
        scheduler_import, scheduler_code = generate_lr_scheduler_code(scheduler_config)
    
    warmup_import = None
    warmup_code = None
    if warmup_config and scheduler_code:
        warmup_result = generate_warmup_wrapper_code(scheduler_code, warmup_config)
        if warmup_result:
            warmup_import, warmup_code = warmup_result
    
    # Build the method using list of lines
    method_lines = [
        "    def configure_optimizers(self):"
    ]
    
    if optimizer_code:
        method_lines.append(f"        optimizer = {optimizer_code}")
    else:
        method_lines.append("        optimizer = super().configure_optimizers()[0]")
    
    if scheduler_code:
        if warmup_code:
            for line in warmup_code.split('\n'):
                method_lines.append(f"        {line.strip()}")
        else:
            method_lines.append(f"        lr_scheduler = {scheduler_code}")
    else:
        method_lines.append("        lr_scheduler = super().configure_optimizers()[1]")
    
    method_lines.append("        return optimizer, lr_scheduler")
    
    imports = set()
    if optimizer_config:
        imports.add("import torch")
    if scheduler_import:
        imports.add(scheduler_import)
    if warmup_import:
        imports.add(warmup_import)
    
    return list(imports), "\n".join(method_lines)


def find_base_trainer_class(base_trainer: str) -> Tuple[str, Any]:
    """Find the base trainer class and generate appropriate import statement.
    
    Args:
        base_trainer: Name of the base trainer class
        
    Returns:
        Tuple[str, Any]: Import statement and class object if found
    """
    base_trainer_import = ""
    base_trainer_class = None
    
    try:
        nnunet_path = get_nnunet_path()
        if nnunet_path:
            sys.path.insert(0, nnunet_path)
            from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
            
            base_trainer_class = recursive_find_python_class(
                folder=os.path.join(nnunet_path, 'nnunetv2'),
                class_name=base_trainer,
                current_module='nnunetv2'
            )
            
            if base_trainer_class:
                base_trainer_module = base_trainer_class.__module__
                base_trainer_import = f"from {base_trainer_module} import {base_trainer}"
                logger.info(f"Found {base_trainer} in {base_trainer_module}")
                return base_trainer_import, base_trainer_class
    except Exception as e:
        logger.warning(f"Could not find nnUNet installation: {e}")
    
    # If we couldn't find the class dynamically, use a basic import
    base_trainer_import = f"from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import {base_trainer}"
    logger.info("Using default import path for base trainer")
    return base_trainer_import, base_trainer_class


def extract_constructor_signature(base_trainer_class: Any) -> Tuple[str, str, bool]:
    """Extract constructor arguments and super call from base trainer class.
    
    Args:
        base_trainer_class: The base trainer class object
        
    Returns:
        Tuple[str, str, bool]: Constructor args, super call, and whether torch is needed
    """
    constructor_args = ""
    super_call = ""
    need_torch_import = False
    
    try:
        if base_trainer_class:
            init_signature = inspect.signature(base_trainer_class.__init__)
            
            # Format the constructor parameters - skip 'self'
            params = list(init_signature.parameters.items())[1:]  # Skip 'self'
            
            args_list = []
            param_annotations = {}
            
            for name, param in params:
                # Get the parameter annotation if available
                if param.annotation is not param.empty:
                    if 'device' in str(param.annotation):
                        param_annotations[name] = 'torch.device'
                        need_torch_import = True
                    else:
                        param_annotations[name] = str(param.annotation).replace('<class ', '').replace('>', '').replace("'", "")
                
                if param.default is param.empty:
                    # Required parameter without default
                    if name in param_annotations:
                        args_list.append(f"{name}: {param_annotations[name]}")
                    else:
                        args_list.append(f"{name}")
                else:
                    default_str = str(param.default)
                    
                    # Special handling for device parameter
                    if name == "device" or "device" in default_str:
                        args_list.append(f"{name}: torch.device = torch.device('cuda')")
                        need_torch_import = True
                    else:
                        default_repr = repr(param.default)
                        if name in param_annotations:
                            args_list.append(f"{name}: {param_annotations[name]} = {default_repr}")
                        else:
                            args_list.append(f"{name}={default_repr}")
            
            constructor_args = ", ".join(args_list)
            
            # Create the super() call with argument names only
            super_args = [name.split(':')[0].strip() for name in args_list]
            super_call = f"super().__init__({', '.join(super_args)})"
            
            logger.info(f"Generated constructor with arguments: {constructor_args}")
            logger.debug(f"Generated super call: {super_call}")
            return constructor_args, super_call, need_torch_import
    except Exception as e:
        logger.error(f"Error getting constructor signature: {e}")
    
    # Fallback to hardcoded constructor
    constructor_args = "plans, configuration, fold, dataset_json, device: torch.device = torch.device('cuda')"
    super_call = "super().__init__(plans, configuration, fold, dataset_json, device)"
    need_torch_import = True
    return constructor_args, super_call, need_torch_import


def generate_parameter_assignments(params_dict: Dict[str, Any]) -> List[str]:
    """Generate code lines for parameter assignments.
    
    This function handles the conversion of configuration parameters to Python code,
    with special handling for different types:
    - Numeric strings are converted to floats
    - Non-numeric strings are preserved as strings
    - Other types (bool, list, etc.) are formatted using repr()
    
    Args:
        params_dict: Dictionary of parameter names and values
    
    Returns:
        List[str]: List of code lines for parameter assignments
    
    Example:
        Input: {"batch_size": "16", "use_bias": True, "dims": [1, 2, 3]}
        Output: [
            "        self.batch_size = 16",
            "        self.use_bias = True",
            "        self.dims = [1, 2, 3]"
        ]
    """
    param_lines = []
    
    for param, value in params_dict.items():
        try:
            if isinstance(value, str):
                try:
                    float_value = float(value)
                    param_lines.append(f"        self.{param} = {float_value}")
                except ValueError:
                    param_lines.append(f"        self.{param} = {repr(value)}")
            else:
                # Use repr for proper formatting of lists, tuples, booleans, etc.
                param_lines.append(f"        self.{param} = {repr(value)}")
        except Exception as e:
            logger.debug(f"Error formatting parameter {param!r} with value {value!r}: {e}")
            param_lines.append(f"        self.{param} = {repr(str(value))}")
    
    return param_lines


def generate_trainer_class(config: Dict[str, Any], base_trainer: str, output_dir: str) -> Optional[str]:
    """Generate a variant trainer class based on the provided config.
    
    Args:
        config: Configuration dictionary
        base_trainer: Name of the base trainer class
        output_dir: Directory where the generated class file will be saved
        
    Returns:
        Optional[str]: Name of the generated trainer class, or None if failed
    """

    if not validate_config(config):
        logger.error(f"Invalid configuration: {config}")
        return None
    
    if 'params' not in config:
        config['params'] = {}
    
    trainer_name = f"{base_trainer}_{config['name']}"
    output_file = os.path.join(output_dir, f"{trainer_name}.py")
    
    if os.path.exists(output_file):
        logger.info(f"Trainer {trainer_name} already exists at {output_file}")
        return trainer_name
    
    base_trainer_import, base_trainer_class = find_base_trainer_class(base_trainer)
    
    constructor_args, super_call, need_torch_import = extract_constructor_signature(base_trainer_class)
    
    optimizer_imports = []
    configure_optimizers_method = None
    
    if any(key in config for key in ['optimizer', 'scheduler', 'warmup']):
        optimizer_result = generate_configure_optimizers_method(config)
        if optimizer_result:
            optimizer_imports, configure_optimizers_method = optimizer_result
            need_torch_import = True
    
    imports = set([base_trainer_import])
    if need_torch_import:
        imports.add("import torch")
    imports.update(optimizer_imports)
    
    code_lines = []
    
    code_lines.extend(sorted(imports))
    
    if imports:
        code_lines.append("")
    
    code_lines.append(f"class {trainer_name}({base_trainer}):")
    code_lines.append(f"    def __init__(self, {constructor_args}):")
    code_lines.append(f"        {super_call}")
    
    param_lines = generate_parameter_assignments(config['params'])
    code_lines.extend(param_lines)
    
    if configure_optimizers_method:
        if param_lines:
            code_lines.append("")
        code_lines.append(configure_optimizers_method)
    
    code = "\n".join(code_lines)
    
    try:
        with open(output_file, 'w') as f:
            f.write(code)
        logger.info(f"Generated trainer class: {trainer_name} at {output_file}")
        return trainer_name
    except Exception as e:
        logger.error(f"Error writing trainer class file: {e}")
        return None


def validate_main_config(config: Dict[str, Any]) -> bool:
    """Validate the main configuration file.
    
    Args:
        config: The main configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """

    required_keys = ['base_trainer', 'configurations']
    for key in required_keys:
        if key not in config:
            logger.error(f"Configuration missing required key: '{key}'")
            return False
    
    if not isinstance(config['configurations'], list):
        logger.error(f"'configurations' must be a list of trainer configurations")
        return False
    
    if not config['configurations']:
        logger.error(f"'configurations' list is empty, no trainers to generate")
        return False
    
    if not isinstance(config['base_trainer'], str):
        logger.error(f"'base_trainer' must be a string")
        return False
    
    optional_keys = ['dataset_id', 'configuration', 'fold', 'plans']
    for key in optional_keys:
        if key in config and not isinstance(config[key], (str, int)):
            logger.warning(f"Optional key '{key}' should be a string or integer")
    
    return True


def create_init_files(output_dir: str, experiment_dir: str, experiment_name: str) -> None:
    """Create necessary __init__.py files for Python package structure.
    
    Args:
        output_dir: Main output directory
        experiment_dir: Experiment-specific directory
        experiment_name: Name of the experiment
    """

    init_file = os.path.join(output_dir, "__init__.py")
    if not os.path.exists(init_file):
        try:
            with open(init_file, 'w') as f:
                f.write("# Auto-generated by model optimization framework\n")
            logger.debug(f"Created __init__.py in {output_dir}")
        except Exception as e:
            logger.error(f"Error creating __init__.py in {output_dir}: {e}")
    
    exp_init_file = os.path.join(experiment_dir, "__init__.py")
    if not os.path.exists(exp_init_file):
        try:
            with open(exp_init_file, 'w') as f:
                f.write(f"# Auto-generated by model optimization framework for {experiment_name}\n")
            logger.debug(f"Created __init__.py in {experiment_dir}")
        except Exception as e:
            logger.error(f"Error creating __init__.py in {experiment_dir}: {e}")


def main() -> None:
    """Main function to generate trainer variants based on configuration."""
    parser = argparse.ArgumentParser(description="Generate trainer variants for model optimization")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml",
        help="Path to the model optimization configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory where trainer variants will be stored. If not specified, will use nnunet path."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    args = parser.parse_args()
    
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration file {args.config}: {e}")
        sys.exit(1)
    
    if not validate_main_config(config):
        logger.error("Invalid configuration. Please fix the errors and try again.")
        sys.exit(1)
    
    base_trainer = config['base_trainer']
    experiment_name = config.get('experiment_name', 'default')
    
    if args.output_dir is None:
        nnunet_path = get_nnunet_path()
        if nnunet_path is None:
            logger.error("Could not determine nnunet path and no --output-dir provided")
            sys.exit(1)
            
        output_dir = os.path.join(nnunet_path, "nnunetv2", "training", "nnUNetTrainer", "variants")
        logger.info(f"Using nnunet path for output: {output_dir}")
    else:
        output_dir = args.output_dir
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")
        
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        logger.debug(f"Created experiment directory: {experiment_dir}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        sys.exit(1)
    
    create_init_files(output_dir, experiment_dir, experiment_name)
    
    trainers = []
    success_count = 0
    total_count = len(config['configurations'])
    
    logger.info(f"Generating {total_count} trainer variants for experiment '{experiment_name}'")
    
    for trainer_config in config['configurations']:
        logger.debug(f"Generating trainer for config: {trainer_config}")
        trainer_name = generate_trainer_class(trainer_config, base_trainer, experiment_dir)
        if trainer_name:
            trainers.append(trainer_name)
            success_count += 1
    
    if success_count == 0:
        logger.error("No trainer variants could be generated. Please check the error messages above.")
        sys.exit(1)
    
    logger.info(f"Generated {success_count} trainer variants out of {total_count} configurations:")
    for trainer in trainers:
        logger.info(f"  - {trainer}")
    
    if all(key in config for key in ['dataset_id', 'configuration', 'fold', 'plans']):
        usage_cmd = f"nnUNetv2_train {config['dataset_id']} {config['configuration']} {config['fold']} -tr <trainer_name>"
        if 'plans' in config:
            usage_cmd += f" -p {config['plans']}"
        logger.info(f"\nTo use these trainers, run:\n{usage_cmd}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
