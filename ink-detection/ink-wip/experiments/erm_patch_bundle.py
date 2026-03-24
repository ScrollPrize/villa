from __future__ import annotations

from dataclasses import replace

from ink.experiments.erm import EXPERIMENT as _ERM_EXPERIMENT
from ink.recipes.data.patch_bundle import GeneratedPatchBundleDataRecipe

_BUNDLE_ROOT = ".tmp/patch_bundles/erm"
_SOURCE_DATA = _ERM_EXPERIMENT.data


EXPERIMENT = replace(
    _ERM_EXPERIMENT,
    name="erm_patch_bundle",
    data=GeneratedPatchBundleDataRecipe(
        bundle_root=_BUNDLE_ROOT,
        source=_SOURCE_DATA,
    ),
)
