from torch.utils.data import DataLoader

from train_resnet3d_lib.config import CFG


def build_eval_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
