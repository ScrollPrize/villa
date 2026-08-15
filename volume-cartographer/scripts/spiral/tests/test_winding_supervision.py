import torch

from config import Config
from winding_supervision import get_winding_inference_losses


class _RecordingStore:
    def __init__(self):
        self.relative_calls = []
        self.density_calls = []

    @staticmethod
    def _empty():
        return {
            "points": torch.empty((0, 2, 3)),
            "target": torch.empty((0,)),
        }

    def sample_relative(self, count, min_delta, max_delta, *, generator=None):
        self.relative_calls.append((count, min_delta, max_delta))
        return self._empty()

    def sample_adjacent(self, count, *, generator=None):
        self.density_calls.append(count)
        return self._empty()


def test_shared_pair_count_drives_both_winding_model_components():
    cfg = Config({"sample_count_winding_model_pairs": 17}).as_dict()
    store = _RecordingStore()

    get_winding_inference_losses(
        None, torch.tensor(1.0), store, cfg, 0, 1,
    )

    assert store.relative_calls == [(17, 3, 15)]
    assert store.density_calls == [17]


def test_shared_pair_count_respects_disabled_component():
    cfg = Config({
        "sample_count_winding_model_pairs": 17,
        "loss_weight_dense_spacing_density": 0.0,
    }).as_dict()
    store = _RecordingStore()

    get_winding_inference_losses(
        None, torch.tensor(1.0), store, cfg, 0, 1,
    )

    assert store.relative_calls == [(17, 3, 15)]
    assert store.density_calls == [0]
