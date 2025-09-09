"""
Test trainer for quick validation of training pipelines with limited epochs and iterations.

This module provides TestTrainer as a configurable class for rapid testing of different loss functions,
datasets, and training configurations without the overhead of full training runs.
"""

from pathlib import Path
import sys
from models.run.train import BaseTrainer


class TestTrainer(BaseTrainer):
    """
    Configurable test trainer that runs training for only 5 epochs with 50 iterations per epoch.
    
    This class subclasses BaseTrainer and overrides key training parameters to enable
    rapid testing of training pipelines, loss functions, and datasets. It uses the
    real ConfigManager and all existing functionality while dramatically reducing
    training time.
    
    Features:
    - 5 epochs maximum training
    - 50 iterations per epoch  
    - 10 validation iterations per epoch
    - Configurable loss function and dataset format
    - Preserves all BaseTrainer functionality (checkpoints, validation, etc.)
    - Uses real ConfigManager (not mocked)
    
    Parameters
    ----------
    mgr : ConfigManager, optional
        If provided, use this config manager instance instead of creating a new one
    loss_function : str, optional
        Loss function to use for all targets. Options: "DC_and_CE_loss", "DC_and_BCE_loss", 
        "MemoryEfficientSoftDiceLoss", "DC_and_topk_loss". Default: None (use config)
    dataset_format : str, optional
        Dataset format to use. Options: "tif", "zarr", "napari". Default: None (use config)
    verbose : bool, default=True
        Whether to print verbose output
        
    Examples
    --------
    Basic usage:
    >>> from models.configuration.config_manager import ConfigManager
    >>> mgr = ConfigManager(verbose=True)
    >>> mgr.load_config("path/to/config.yaml")
    >>> trainer = TestTrainer(mgr=mgr)
    >>> trainer.train()
    
    With specific loss and dataset:
    >>> trainer = TestTrainer(mgr=mgr, loss_function="MemoryEfficientSoftDiceLoss", dataset_format="zarr")
    >>> trainer.train()
    """
    
    def __init__(self, mgr=None, loss_function=None, dataset_format=None, verbose=True):
        """
        Initialize the test trainer with a config manager instance.
        
        Parameters
        ----------
        mgr : ConfigManager, optional
            If provided, use this config manager instance instead of creating a new one
        loss_function : str, optional
            Loss function to use for all targets
        dataset_format : str, optional
            Dataset format to use ("tif", "zarr", "napari")
        verbose : bool, default=True
            Whether to print verbose output
        """
        # Store verbose for use in our methods before calling super()
        self.verbose = verbose
        
        super().__init__(mgr=mgr, verbose=verbose)
        
        # Configure loss function if specified
        if loss_function is not None:
            self._configure_loss_function(loss_function)
        
        # Configure dataset format if specified
        if dataset_format is not None:
            self._configure_dataset_format(dataset_format)
        
        # Apply test configuration overrides
        self._override_test_configs()
        
        if self.verbose:
            print("=" * 60)
            print("TEST TRAINER INITIALIZED")
            print("=" * 60)
            print(f"Max epochs: {self.mgr.max_epoch}")
            print(f"Max steps per epoch: {self.mgr.max_steps_per_epoch}")
            print(f"Max validation steps per epoch: {self.mgr.max_val_steps_per_epoch}")
            print(f"Model name: {self.mgr.model_name}")
            if loss_function:
                print(f"Loss function: {loss_function}")
            if dataset_format:
                print(f"Dataset format: {dataset_format}")
            print("=" * 60)
    
    def _configure_loss_function(self, loss_function):
        """
        Configure all targets to use the specified loss function.
        
        Parameters
        ----------
        loss_function : str
            Loss function to apply to all targets
        """
        valid_losses = [
            "DC_and_CE_loss", 
            "DC_and_BCE_loss", 
            "MemoryEfficientSoftDiceLoss", 
            "DC_and_topk_loss"
        ]
        
        if loss_function not in valid_losses:
            raise ValueError(f"Invalid loss function: {loss_function}. "
                           f"Valid options: {valid_losses}")
        
        if hasattr(self.mgr, 'targets') and self.mgr.targets:
            for target_name in self.mgr.targets:
                self.mgr.targets[target_name]["loss_fn"] = loss_function
            
            if self.verbose:
                print(f"TestTrainer: Configured all targets to use {loss_function}")
        else:
            # Store for later application when targets are available
            self.mgr.selected_loss_function = loss_function
            if self.verbose:
                print(f"TestTrainer: Set default loss function to {loss_function}")
    
    def _configure_dataset_format(self, dataset_format):
        """
        Configure the dataset format.
        
        Parameters
        ----------
        dataset_format : str
            Dataset format to use ("tif", "zarr", "napari")
        """
        valid_formats = ["tif", "zarr", "napari"]
        
        if dataset_format not in valid_formats:
            raise ValueError(f"Invalid dataset format: {dataset_format}. "
                           f"Valid options: {valid_formats}")
        
        self.mgr.data_format = dataset_format
        
        # Add to dataset config for persistence
        if not hasattr(self.mgr, 'dataset_config'):
            self.mgr.dataset_config = {}
        self.mgr.dataset_config['data_format'] = dataset_format
        
        if self.verbose:
            print(f"TestTrainer: Configured dataset format to use {dataset_format}")
    
    def _override_test_configs(self):
        """
        Override configuration parameters for quick testing.
        
        This method modifies the ConfigManager to use test-appropriate values:
        - 5 epochs maximum
        - 50 training iterations per epoch
        - 10 validation iterations per epoch
        """
        # Override training duration
        self.mgr.max_epoch = 5
        self.mgr.tr_configs["max_epoch"] = 5
        
        # Override iterations per epoch
        self.mgr.max_steps_per_epoch = 10
        self.mgr.tr_configs["max_steps_per_epoch"] = 10
        
        # Override validation iterations per epoch (proportionally smaller)
        self.mgr.max_val_steps_per_epoch = 2
        self.mgr.tr_configs["max_val_steps_per_epoch"] = 2
        
        # Ensure the model name reflects this is a test
        if not self.mgr.model_name.startswith("Test_"):
            self.mgr.model_name = f"Test_{self.mgr.model_name}"
            self.mgr.tr_info["model_name"] = self.mgr.model_name
        
        if self.verbose:
            print("Test configuration overrides applied:")
            print(f"  - Max epochs: {self.mgr.max_epoch}")
            print(f"  - Max steps per epoch: {self.mgr.max_steps_per_epoch}")
            print(f"  - Max validation steps per epoch: {self.mgr.max_val_steps_per_epoch}")
            print(f"  - Model name: {self.mgr.model_name}")

    def _create_training_transforms(self):
        return 

def create_test_trainer(mgr=None, loss_function=None, dataset_format=None, verbose=True):
    """
    Factory function to create a test trainer with specified configuration.
    
    Parameters
    ----------
    mgr : ConfigManager, optional
        Configuration manager instance
    loss_function : str, optional
        Loss function to use for all targets
    dataset_format : str, optional
        Dataset format to use ("tif", "zarr", "napari")
    verbose : bool, default=True
        Whether to print verbose output
        
    Returns
    -------
    TestTrainer
        An instance of the configured test trainer
        
    Examples
    --------
    >>> trainer = create_test_trainer(mgr=my_config_manager, 
    ...                               loss_function="MemoryEfficientSoftDiceLoss",
    ...                               dataset_format="zarr")
    >>> trainer.train()
    """
    return TestTrainer(mgr=mgr, loss_function=loss_function, 
                      dataset_format=dataset_format, verbose=verbose)


def main():
    """
    Main entry point for running test trainers from command line.
    
    This function provides a simple CLI interface for testing different
    trainer configurations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run test trainers for quick validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config", required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--loss-function", 
                       choices=["DC_and_CE_loss", "DC_and_BCE_loss", 
                               "MemoryEfficientSoftDiceLoss", "DC_and_topk_loss"],
                       help="Loss function to use for all targets")
    parser.add_argument("--dataset-format",
                       choices=["tif", "zarr", "napari"],
                       help="Dataset format to use")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Import ConfigManager here to avoid circular imports
    from models.configuration.config_manager import ConfigManager
    
    # Load configuration
    mgr = ConfigManager(verbose=args.verbose)
    if not Path(args.config).exists():
        raise ValueError(f"Config file does not exist: {args.config}")
    
    mgr.load_config(args.config)
    
    # Create and run trainer
    trainer = TestTrainer(
        mgr=mgr,
        loss_function=args.loss_function,
        dataset_format=args.dataset_format,
        verbose=args.verbose
    )
    
    config_str = []
    if args.loss_function:
        config_str.append(f"loss={args.loss_function}")
    if args.dataset_format:
        config_str.append(f"format={args.dataset_format}")
    
    config_desc = f" ({', '.join(config_str)})" if config_str else ""
    print(f"Starting test trainer{config_desc}...")
    
    trainer.train()
    print("Test training completed!")


if __name__ == "__main__":
    main()
