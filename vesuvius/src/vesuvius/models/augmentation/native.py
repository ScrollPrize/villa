"""Dataset integration for vc_reader.augment (no augmentation kernels in Python).

Epoch and draw identity travel with sampler indices, so persistent DataLoader
workers see epoch updates without shared mutable RNG or worker-local counters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationIndex:
    index: int
    epoch: int = 0
    draw: int = 0
    view: int = 0


class AugmentationSampler:
    """Wrap an existing sampler, including DistributedSampler.

    Call set_epoch(epoch) in the main process before creating each epoch's
    DataLoader iterator. The wrapped sampler's set_epoch is forwarded.
    """
    def __init__(self, sampler, *, view=0):
        self.sampler = sampler
        self.epoch = 0
        self.view = int(view)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)

    def __iter__(self):
        for draw, index in enumerate(self.sampler):
            yield AugmentationIndex(int(index), self.epoch, draw, self.view)

    def __len__(self):
        return len(self.sampler)


def apply_native(pipeline, sample, index, run_seed, *, profile=False, inplace=False):
    """Adapt sampler identity to the native pipeline's explicit seed argument."""
    from vc_reader.augment import seed

    token = index if isinstance(index, AugmentationIndex) else AugmentationIndex(int(index))
    return pipeline(sample, seed=seed(int(run_seed), token.epoch, token.index, token.draw, token.view),
                    profile=profile, inplace=inplace)


class AugmentedDataset:
    """Add a native pipeline to any dictionary-returning, unaugmented dataset.

    The wrapped dataset must return channel-first arrays/tensors in a shared
    grid. Disable its existing augmentation before wrapping it. Set inplace=True
    only when the dataset returns fresh writable buffers, such as reader crops;
    cached samples must use the default to preserve their contents.
    """
    def __init__(self, dataset, pipeline, *, seed=0, profile=False, inplace=False):
        self.dataset = dataset
        self.pipeline = pipeline
        self.seed = int(seed)
        self.profile = bool(profile)
        self.inplace = bool(inplace)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw_index = index.index if isinstance(index, AugmentationIndex) else index
        return apply_native(self.pipeline, self.dataset[raw_index], index, self.seed, profile=self.profile, inplace=self.inplace)
