from configuration.config_manager import ConfigManager
from training.trainers.basetrainer import BaseTrainer
from dataloading.wk2_dataset import wkDataset
import argparse

class WKTrainer(BaseTrainer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.mgr = ConfigManager(config_file)

    def _configure_dataset(self):
        dataset = wkDataset(mgr=self.mgr)
        return dataset

def main():
    """Main entry point for the script when run via `python -m training.trainers.wk_trainer`."""
    parser = argparse.ArgumentParser(description="Train script for WKTrainer.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--debug_dataloader", action="store_true", help="Debug dataloader, then exit.")
    parser.add_argument("--verbose", action="store_true", help="Print extra details.")
    args = parser.parse_args()

    # Instantiate your trainer
    trainer = WKTrainer(config_file=args.config_path)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()