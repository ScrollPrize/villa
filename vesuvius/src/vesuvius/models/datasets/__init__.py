"""Dataset exports, loaded independently of other datasets' optional dependencies."""

__all__ = ["ZarrDataset", "MutexAffinityDataset"]


def __getattr__(name):
    if name == "ZarrDataset":
        from .zarr_dataset import ZarrDataset
        return ZarrDataset
    if name == "MutexAffinityDataset":
        from .mutex_affinity_dataset import MutexAffinityDataset
        return MutexAffinityDataset
    raise AttributeError(name)
