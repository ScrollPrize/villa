from __future__ import annotations

from dataclasses import replace

from ink.experiments.erm import EXPERIMENT as _ERM_EXPERIMENT
from ink.recipes.data.patch_bundle import PatchBundleDataRecipe

_BUNDLE_ROOT = ".tmp/patch_bundles/erm"
_SOURCE_DATA = _ERM_EXPERIMENT.data


EXPERIMENT = replace(
    _ERM_EXPERIMENT,
    name="erm_patch_bundle",
    data=PatchBundleDataRecipe(
        bundle_root=_BUNDLE_ROOT,
        train_batch_size=int(_SOURCE_DATA.train_batch_size),
        valid_batch_size=_SOURCE_DATA.valid_batch_size,
        num_workers=int(_SOURCE_DATA.num_workers),
        shuffle=bool(_SOURCE_DATA.shuffle),
        normalization=_SOURCE_DATA.normalization,
        extras=dict(_SOURCE_DATA.extras or {}),
    ),
)
