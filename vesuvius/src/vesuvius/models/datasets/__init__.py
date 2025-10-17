# Dataset exports
from .base_dataset import BaseDataset
from .mutex_affinity_dataset import MutexAffinityDataset
from .orchestrator import DatasetOrchestrator

__all__ = ["BaseDataset", "DatasetOrchestrator", "MutexAffinityDataset"]


_LEGACY_DATASETS = {
    "ImageDataset": "Use DatasetOrchestrator with adapter='image'.",
    "ZarrDataset": "Use DatasetOrchestrator with adapter='zarr'.",
    "NapariDataset": "Use DatasetOrchestrator with adapter='napari'.",
}


def __getattr__(name: str):  # pragma: no cover - defensive
    message = _LEGACY_DATASETS.get(name)
    if message is not None:
        raise ImportError(f"{name} has been removed. {message}")
    raise AttributeError(name)
