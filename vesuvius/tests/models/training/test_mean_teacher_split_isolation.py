from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


if os.environ.get("VESUVIUS_FALLBACK_SITE_PACKAGES"):
    site.addsitedir(os.environ["VESUVIUS_FALLBACK_SITE_PACKAGES"])

import pytest
import torch
from torch.utils.data import Dataset


VILLA_ROOT = Path(__file__).parents[3]
ROOT = (
    VILLA_ROOT
    / "src/vesuvius/models/training/trainers/semi_supervised"
)


class _BaseTrainer:
    def __init__(self, mgr=None, verbose=True):
        self.mgr = mgr
        self.verbose = verbose
        self.device = torch.device("cpu")


class _Dataset(Dataset):
    def __init__(self, labeled=8, unlabeled=4, data_path="source-a"):
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.data_path = data_path

    def __len__(self):
        return self.labeled + self.unlabeled

    def __getitem__(self, index):
        return {"index": index}

    def get_labeled_unlabeled_patch_indices(self):
        return (
            list(range(self.labeled)),
            list(range(self.labeled, len(self))),
        )


def _load_modules():
    replacements = {}

    def install(name, module):
        replacements[name] = sys.modules.get(name)
        sys.modules[name] = module

    train_module = ModuleType("vesuvius.models.training.train")
    train_module.BaseTrainer = _BaseTrainer
    install(train_module.__name__, train_module)

    ramps_name = "vesuvius.models.training.trainers.semi_supervised.ramps"
    ramps_spec = importlib.util.spec_from_file_location(
        ramps_name, ROOT / "ramps.py"
    )
    assert ramps_spec is not None and ramps_spec.loader is not None
    ramps = importlib.util.module_from_spec(ramps_spec)
    install(ramps_name, ramps)
    ramps_spec.loader.exec_module(ramps)

    package = ModuleType("vesuvius.models.training.trainers.semi_supervised")
    package.__path__ = [str(ROOT)]
    package.ramps = ramps
    install(package.__name__, package)

    sampler_name = (
        "vesuvius.models.training.trainers.semi_supervised."
        "two_stream_batch_sampler"
    )
    sampler_spec = importlib.util.spec_from_file_location(
        sampler_name, ROOT / "two_stream_batch_sampler.py"
    )
    assert sampler_spec is not None and sampler_spec.loader is not None
    sampler = importlib.util.module_from_spec(sampler_spec)
    install(sampler_name, sampler)
    sampler_spec.loader.exec_module(sampler)

    modules = {"sampler": sampler}
    for key, filename in (
        ("mean_teacher", "train_mean_teacher.py"),
        ("uamt", "train_uncertainty_aware_mean_teacher.py"),
    ):
        name = f"mean_teacher_split_{key}_target"
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        install(name, module)
        spec.loader.exec_module(module)
        modules[key] = module
    return modules


@pytest.fixture(scope="module")
def modules():
    return _load_modules()


def _manager(split=0.75, seed=17):
    return SimpleNamespace(
        train_batch_size=4,
        labeled_batch_size=2,
        labeled_ratio=1.0,
        num_labeled=None,
        train_num_dataloader_workers=0,
        tr_val_split=split,
        seed=seed,
        verbose=False,
        out_channels=2,
        uncertainty_T=2,
    )


def _trainer(modules, name, split=0.75):
    cls = (
        modules[name].TrainMeanTeacher
        if name == "mean_teacher"
        else modules[name].TrainUncertaintyAwareMeanTeacher
    )
    return cls(_manager(split=split), verbose=False)


def _configure(modules, name, train, val=None, split=0.75):
    trainer = _trainer(modules, name, split=split)
    train_loader, _, train_indices, val_indices = trainer._configure_dataloaders(
        train, val
    )
    return (
        list(train_loader.batch_sampler.primary_indices),
        list(train_indices),
        list(val_indices),
    )


def test_shared_split_helper_partitions_at_floor(modules):
    train, val = modules["sampler"].split_labeled_indices(
        list(range(8)), 0.75, 2
    )

    assert train == list(range(6))
    assert val == [6, 7]


def test_shared_split_helper_allows_empty_validation(modules):
    train, val = modules["sampler"].split_labeled_indices(
        list(range(8)), 1.0, 2
    )

    assert train == list(range(8))
    assert val == []


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_shared_split_helper_rejects_invalid_fraction(modules, fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        modules["sampler"].split_labeled_indices(list(range(8)), fraction, 2)


def test_shared_split_helper_rejects_too_few_training_labels(modules):
    with pytest.raises(ValueError, match="too few labeled training"):
        modules["sampler"].split_labeled_indices(list(range(3)), 0.5, 2)


@pytest.mark.parametrize("name", ["mean_teacher", "uamt"])
def test_internal_training_and_validation_are_disjoint_partitions(modules, name):
    primary, returned_train, val = _configure(modules, name, _Dataset())

    assert primary == returned_train
    assert len(primary) == 6
    assert len(val) == 2
    assert set(primary).isdisjoint(val)
    assert sorted(primary + val) == list(range(8))


@pytest.mark.parametrize("name", ["mean_teacher", "uamt"])
def test_distinct_objects_with_same_source_split_internally(modules, name):
    primary, _, val = _configure(
        modules,
        name,
        _Dataset(data_path="shared"),
        _Dataset(data_path="shared"),
    )

    assert set(primary).isdisjoint(val)
    assert sorted(primary + val) == list(range(8))


@pytest.mark.parametrize("name", ["mean_teacher", "uamt"])
def test_external_validation_preserves_full_training_pool(modules, name):
    primary, _, val = _configure(
        modules,
        name,
        _Dataset(data_path="train"),
        _Dataset(labeled=5, unlabeled=0, data_path="external"),
    )

    assert len(primary) == 8
    assert val == list(range(5))


@pytest.mark.parametrize("name", ["mean_teacher", "uamt"])
def test_split_one_has_empty_internal_validation(modules, name):
    primary, _, val = _configure(modules, name, _Dataset(), split=1.0)

    assert len(primary) == 8
    assert val == []


def test_fixed_seed_produces_identical_splits_for_both_trainers(modules):
    mean = _configure(modules, "mean_teacher", _Dataset())
    uncertainty = _configure(modules, "uamt", _Dataset())

    assert mean == uncertainty
