from configuration.config_manager import ConfigManager
from training.trainers.basetrainer import BaseTrainer
from dataloading.dataset import MultiTask3dDataset
import argparse

class Trainer2(BaseTrainer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.mgr = ConfigManager(config_file)

    def _configure_dataset(self):
        dataset = MultiTask3dDataset(mgr=self.mgr)
        return dataset

def main():
    parser = argparse.ArgumentParser(description="Train script for WKTrainer.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--debug_dataloader", action="store_true", help="Debug dataloader, then exit.")
    parser.add_argument("--verbose", action="store_true", help="Print extra details.")
    args = parser.parse_args()


    trainer = Trainer2(config_file=args.config_path)

    trainer.train()


if __name__ == "__main__":
    main()