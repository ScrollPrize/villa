import numpy as np
import pytest

from koine_machines.data.patch_finding.default import (
    _combined_patch_discovery_support,
)


def test_patch_discovery_includes_disjoint_validation_support():
    supervision = np.zeros((4, 5), dtype=np.uint8)
    validation = np.zeros((4, 5), dtype=np.uint8)
    supervision[1, 1] = 1
    validation[3, 4] = 1

    support = _combined_patch_discovery_support(supervision, validation)

    assert set(zip(*np.nonzero(support))) == {(1, 1), (3, 4)}


def test_patch_discovery_rejects_misaligned_validation_support():
    with pytest.raises(ValueError, match="validation support"):
        _combined_patch_discovery_support(
            np.zeros((4, 5), dtype=np.uint8),
            np.zeros((5, 4), dtype=np.uint8),
        )
