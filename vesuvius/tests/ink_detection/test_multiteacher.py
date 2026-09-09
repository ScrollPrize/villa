"""Regression tests for the explicit ink bottleneck and independent supervision."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vesuvius.ink_detection.training.multiteacher_loss import (
    FrozenInkTeachers, joint_loss, masked_sample_mean, finish_loss_metrics, suppression_statistics,
)
from vesuvius.ink_detection.data.multiteacher import (
    RankDrawSampler, max_pool_mask, patch_coordinates, spatial_holdout, select_assets,
    aligned_extent, FlatDistillationDataset,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(1, 3, 3, padding=1), nn.GELU())
        self.decoder = nn.Conv3d(3, 1, 1)

    def forward(self, x):
        return {"ink": self.decoder(self.encoder(x))}


def example():
    torch.manual_seed(27)
    image = torch.randn(2, 1, 8, 12, 12)
    return {"image": image, "labels_2d": torch.randint(0, 2, (2, 1, 12, 12)).float(),
            "mask_2d": torch.ones(2, 1, 12, 12), "valid_3d": torch.ones_like(image)}


def test_removing_depth_prior_requires_explicit_and_exclusive_resume_change():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    old = {"model_config": {"projection": {"channels": 16, "prior_sigma": 8.}},
           "learning_rate": .01, "manifest_sha256": "dataset"}
    new = deepcopy(old)
    new["model_config"]["projection"]["prior_sigma"] = None
    assert not validate_resume_signature(old, old)
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new)
    assert validate_resume_signature(old, new, remove_depth_prior=True)
    assert old["model_config"]["projection"]["prior_sigma"] == 8.
    for key, value in (("learning_rate", .02), ("manifest_sha256", "other")):
        bad = deepcopy(new)
        bad[key] = value
        with pytest.raises(ValueError, match="Resume configuration"):
            validate_resume_signature(old, bad, remove_depth_prior=True)
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(new, old, remove_depth_prior=True)


def test_coarse_teacher_transition_changes_only_checkpoint_identity():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    old = {"learning_rate": .01, "optimizer_updates": 50000, "distillation": {
        "coarse": {"checkpoint": "old.pth", "sha256": "old", "weight": .2,
                   "background_thresholds": {"814": 86}},
        "paris4": {"checkpoint": "precise.pth", "sha256": "precise", "weight": 1.}}}
    new = deepcopy(old)
    new["distillation"]["coarse"].update(checkpoint="selfteacher.pth", sha256="new")
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new)
    assert validate_resume_signature(old, new, replace_coarse_teacher=True)
    assert not validate_resume_signature(new, new, replace_coarse_teacher=True)
    for path,value in ((["learning_rate"], .02), (["optimizer_updates"], 70000),
                       (["distillation", "coarse", "weight"], .3),
                       (["distillation", "coarse", "background_thresholds"], {"814": 80}),
                       (["distillation", "paris4", "sha256"], "changed")):
        bad = deepcopy(new)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="Resume configuration"):
            validate_resume_signature(old, bad, replace_coarse_teacher=True)


def test_soft_bce_optimum_is_teacher_probability_and_masks_ignore_voxels():
    target = torch.tensor([[[[[0.1, 0.35, 0.8]]]]])
    logits = torch.logit(target).requires_grad_()
    value = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    masked_sample_mean(value, torch.ones_like(target)).sum().backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits), atol=1e-7)


@pytest.mark.parametrize("ids", [[0, 1], [0, 0], [1, 1]])
def test_teacher_routing_background_and_human_labels_are_independent(monkeypatch, ids):
    import vesuvius.ink_detection.training.multiteacher_loss as module
    class Constant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(value), requires_grad=False)
        def forward(self, image):
            return {"ink": self.value.expand_as(image)}
    monkeypatch.setattr(module, "load_frozen_ink_model",
                        lambda p, **kw: Constant(0.0 if p == "paris" else 2.0).eval())
    generator = FrozenInkTeachers({"paris4": {"checkpoint": "paris", "weight": 1},
                                  "coarse": {"checkpoint": "coarse", "weight": .25}}, "cpu")
    raw = torch.tensor([0., 50., 51., 100.]).reshape(1, 1, 1, 1, 4).repeat(2, 1, 1, 1, 1)
    batch = {"raw": raw, "teacher_image": raw/100, "teacher_id": torch.tensor(ids),
             "labels_2d": torch.ones(2, 1, 1, 4)}
    saved = batch["labels_2d"].clone()
    targets, weights, probabilities = generator.generate(batch)
    for i, teacher_id in enumerate(ids):
        if teacher_id == 0:
            assert (targets[i] == .5).all() and weights[i] == 1
        else:
            assert (targets[i, ..., :2] == 0).all()
            assert torch.allclose(targets[i, ..., 2:], torch.sigmoid(torch.tensor(2.)))
            assert weights[i] == .25
    assert torch.equal(batch["labels_2d"], saved)
    assert not targets.requires_grad and not probabilities.requires_grad
    assert all(p.grad is None for model in generator.models for p in model.parameters())


def test_validation_exclusion_preserves_single_pixel_and_guard():
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[1001, 1003] = True
    heldout = max_pool_mask(mask)
    assert heldout.sum() == 1
    train, val = patch_coordinates(np.ones_like(heldout), heldout, mask.shape)
    assert len(train) and len(val)
    for y, x in train:
        assert not (y-256 <= 1001 < y+256+256 and x-256 <= 1003 < x+256+256)


def test_spatial_blocks_are_repeatable_and_rank_resume_skips_completed_draws():
    support = np.ones((256, 256), dtype=bool)
    a, blocks = spatial_holdout(support, 27, "paris4/w00")
    b, again = spatial_holdout(support, 27, "paris4/w00")
    assert np.array_equal(a, b) and blocks == again
    full = [list(RankDrawSampler(0, 160, rank, 4)) for rank in range(4)]
    assert len(set(sum(full, []))) == 160
    for rank in range(4):
        assert list(RankDrawSampler(80, 160, rank, 4)) == full[rank][20:]


def test_1667_uses_matching_v2_pair_and_validation_with_short_prefix(tmp_path):
    directory = tmp_path/"w018_2um"
    directory.mkdir()
    for name in ("w018_2um_inklabels.zarr", "w018_2um_supervision_mask.zarr",
                 "w018_inklabels_v2.zarr", "w018_supervision_mask_v2.zarr",
                 "w018_validation_mask_v2.zarr"):
        (directory/name).mkdir()
    assets, version = select_assets(directory)
    assert version == 2
    assert assets["inklabels"].endswith("w018_inklabels_v2.zarr")
    assert assets["validation_mask"].endswith("w018_validation_mask_v2.zarr")


def test_optimizer_update_schedule_is_independent_of_accumulation_microsteps():
    from types import SimpleNamespace
    from vesuvius.ink_detection.config import OptimizerConfig, SchedulerConfig
    from vesuvius.ink_detection.training.train import create_training_scheduler
    cfg = {"learning_rate": .01, "warmup_steps": 1000}
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=.01)
    views = SimpleNamespace(optimizer=OptimizerConfig.from_mapping(cfg), max_steps=50000,
                            scheduler=SchedulerConfig.from_mapping(cfg, max_steps=50000))
    scheduler = create_training_scheduler(optimizer, views)
    # The 60k teacher step is deliberately absent from a new optimizer/scheduler.
    assert scheduler.last_epoch == 0
    for _ in range(2):
        (parameter.sum()/2).backward()
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1


def test_known_man5_extent_preserves_coordinates_and_rejects_other_mismatches():
    assert aligned_extent("man5/MAN5_outer_3", (65, 24120, 29900),
                          [(65, 24061, 29841)] * 3) == (24061, 29841)
    with pytest.raises(ValueError, match="geometry mismatch"):
        aligned_extent("unknown", (65, 24120, 29900), [(65, 24061, 29841)])
    with pytest.raises(ValueError, match="geometries disagree"):
        aligned_extent("man5/MAN5_outer_3", (65, 24120, 29900),
                       [(65, 24061, 29841), (65, 24062, 29841)])


def test_validation_only_scroll_is_never_sampled_for_training(tmp_path):
    records = [{"scroll": "train"}, {"scroll": "heldout"}]
    (tmp_path/"manifest.json").write_text(json.dumps({"seed": 27, "segments": records}))
    np.savez(tmp_path/"patches.npz", train_0=np.array([[0, 0]]),
             train_1=np.empty((0, 2), dtype=int), val_0=np.array([[256, 256]]),
             val_1=np.array([[0, 0]]), heldout_0=np.zeros((64, 64)),
             heldout_1=np.ones((64, 64)))
    train = FlatDistillationDataset(tmp_path/"manifest.json", ["percentile_minmax"]*2)
    val = FlatDistillationDataset(tmp_path/"manifest.json", ["percentile_minmax"]*2,
                                  validation=True)
    assert train.scrolls == ["train"]
    assert all(train.locate(i)[0] == 0 for i in range(100))
    assert val.scrolls == ["heldout", "train"]


def test_source_rgba_mask_uses_the_label_converter_channel_semantics(tmp_path, monkeypatch):
    import tifffile
    import vesuvius.ink_detection.data.multiteacher as module
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[8:16, 8:16, :3] = 255
    path = tmp_path/"mask.zarr"
    tifffile.imwrite(path.with_suffix(".tif"), rgba, photometric="rgb")
    rendered = np.zeros((65, 32, 32), dtype=np.uint8)
    rendered[32] = rgba[..., 0]
    monkeypatch.setattr(module, "open_volume", lambda *args: rendered)
    assert np.array_equal(module.read_mask(path), rgba[..., 0])


def test_rebalance_and_ct_context_transitions_require_explicit_flags():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    old = {"model_config": {"projection": {"channels": 16, "prior_sigma": None}},
           "paris4_per_batch": 1, "learning_rate": .01}
    new = deepcopy(old)
    new.update(paris4_per_batch=0, paris4_per_rank_update=1, human_2d_weight=2.)
    new["model_config"]["projection"]["ct_context"] = True
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new, rebalance=True)
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new, add_ct_context=True)
    assert validate_resume_signature(old, new, rebalance=True, add_ct_context=True)
    new["learning_rate"] = .02
    with pytest.raises(ValueError, match="Resume configuration"):
        validate_resume_signature(old, new, rebalance=True, add_ct_context=True)


def test_eighth_ph4_global_quota_is_exact_rotating_and_resume_stable(tmp_path):
    records = [{'scroll':s} for s in ('phercparis4','0009b','814')]
    (tmp_path/'manifest.json').write_text(json.dumps({'seed':27,'segments':records}))
    arrays = {}
    for i in range(3):
        arrays[f'train_{i}'] = np.array([[0,0],[256,256]])
        arrays[f'heldout_{i}'] = np.zeros((64,64))
    np.savez(tmp_path/'patches.npz', **arrays)
    data = FlatDistillationDataset(tmp_path/'manifest.json', ['percentile_minmax']*2,
        world_size=4, batch_size=2, grad_acc_steps=2, paris4_per_global_update=2)
    def precise(draw):
        return records[data.locate(draw)[0]]['scroll']=='phercparis4'
    for first in (0,7,19999,20000):
        counts = [0]*4
        for update in range(first,first+4):
            assert sum(precise(d) for d in range(update*16,(update+1)*16))==2
            for micro in range(2):
                assert sum(precise(update*16+micro*8+d) for d in range(8))==1
            for rank in range(4):
                draws=list(RankDrawSampler(update*16,(update+1)*16,rank,4))
                counts[rank] += sum(map(precise,draws))
        assert counts == [2]*4
    uninterrupted = [data.locate(d)[:2] for d in range(20000*16,20002*16)]
    resumed = [data.locate(d)[:2] for d in reversed(range(20000*16,20002*16))][::-1]
    assert uninterrupted == resumed
    with pytest.raises(ValueError, match='only one'):
        FlatDistillationDataset(tmp_path/'manifest.json', ['percentile_minmax']*2,
            world_size=4,batch_size=2,grad_acc_steps=2,
            paris4_per_rank_update=1,paris4_per_global_update=2)


def test_teacher_and_objective_transition_remains_bounded():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    old={'paris4_per_batch':0,'paris4_per_rank_update':1,'learning_rate':.001,
         'distillation':{'coarse':{'checkpoint':'old','sha256':'oldhash','weight':.2,
                                    'background_thresholds':{'814':86}},
                         'paris4':{'checkpoint':'precise','weight':1.}}}
    new=deepcopy(old)
    new.update(paris4_per_rank_update=0,paris4_per_global_update=2)
    new['distillation']['coarse'].update(checkpoint='ema20k',sha256='newhash',weight=.5)
    for flags in ({}, {'rebalance':True}, {'replace_coarse_teacher':True}):
        with pytest.raises(ValueError,match='Resume configuration'):
            validate_resume_signature(old,new,**flags)
    assert validate_resume_signature(old,new,rebalance=True,replace_coarse_teacher=True)
    for path,value in ((('learning_rate',),.002),
                       (('distillation','coarse','background_thresholds'),{'814':80}),
                       (('distillation','paris4','weight'),.5)):
        bad=deepcopy(new);target=bad
        for key in path[:-1]: target=target[key]
        target[path[-1]]=value
        with pytest.raises(ValueError,match='Resume configuration'):
            validate_resume_signature(old,bad,rebalance=True,replace_coarse_teacher=True)


def test_quarter_ph4_quota_covers_each_rank_update_and_each_global_microbatch(tmp_path):
    records = [{"scroll": s} for s in ("phercparis4", "0009b", "814")]
    (tmp_path/"manifest.json").write_text(json.dumps({"seed": 27, "segments": records}))
    arrays = {}
    for i in range(3):
        arrays[f"train_{i}"] = np.array([[0, 0], [256, 256]])
        arrays[f"heldout_{i}"] = np.zeros((64, 64))
    np.savez(tmp_path/"patches.npz", **arrays)
    dataset = FlatDistillationDataset(tmp_path/"manifest.json", ["percentile_minmax"]*2,
        world_size=4, batch_size=2, grad_acc_steps=2, paris4_per_rank_update=1)
    is_precise = lambda draw: records[dataset.locate(draw)[0]]["scroll"] == "phercparis4"
    for start_update in (0, 5, 561, 3001):
        for update in range(start_update, start_update+4):
            for rank in range(4):
                draws = list(RankDrawSampler(update*16, (update+1)*16, rank, 4))
                assert sum(is_precise(draw) for draw in draws) == 1
            for microstep in range(2):
                assert sum(is_precise(draw) for draw in range(update*16+microstep*8, update*16+(microstep+1)*8)) == 2


def test_teacher_conditional_metrics_use_global_sample_counts():
    values = {"teacher/precise_fraction": .25, "teacher/coarse_fraction": .75,
              "loss/3d_precise_unweighted_contribution": .1, "loss/3d_precise_contribution": .1,
              "loss/3d_coarse_unweighted_contribution": .6, "loss/3d_coarse_contribution": .03}
    result = finish_loss_metrics(values)
    assert result["loss/3d_precise"] == pytest.approx(.4)
    assert result["loss/3d_coarse"] == pytest.approx(.8)
    assert result["loss/3d_coarse_weighted"] == pytest.approx(.04)


def test_suppression_diagnostics_distinguish_uniform_and_concentrated_depth_weights():
    batch = example()
    batch["teacher_id"] = torch.tensor([0, 1])
    probability = .2
    output = {"ink_3d_logits": torch.full_like(batch["image"], torch.logit(torch.tensor(probability))),
              "depth_weights": torch.full_like(batch["image"], 1/8)}
    targets = torch.full_like(batch["image"], .9)
    def measure():
        return finish_loss_metrics({k: v.item() for k,v in suppression_statistics(output, batch, targets).items()})
    uniform = measure()
    assert uniform["attention/effective_depth"] == pytest.approx(8)
    assert uniform["ink_retention/coarse/positive_student_probability"] == pytest.approx(.2)
    output["depth_weights"].zero_()
    output["depth_weights"][:, :, 3] = 1
    assert measure()["attention/effective_depth"] == pytest.approx(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_deterministic_pool_preserves_forward_and_backward(dtype):
    from vesuvius.ink_detection.models.deterministic_pool import DeterministicAvgPool3d
    if dtype == torch.bfloat16:
        if not torch.cuda.is_available():
            pytest.skip("CPU AvgPool3d does not support bfloat16")
        device = "cuda"
    else:
        device = "cpu"
    x = torch.randn(2, 3, 8, 12, 16, dtype=dtype, device=device, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    reference = torch.nn.functional.avg_pool3d(x, 2, 2)
    result = DeterministicAvgPool3d((2, 2, 2))(other)
    assert torch.equal(reference, result)
    gradient = torch.randn_like(result)
    reference.backward(gradient)
    result.backward(gradient)
    assert torch.equal(x.grad, other.grad)


def test_every_rank_minibatch_contains_precise_and_coarse_supervision(tmp_path):
    records = [{"scroll": "phercparis4"}, {"scroll": "0009b"}, {"scroll": "814"}]
    (tmp_path/"manifest.json").write_text(json.dumps({"seed": 27, "segments": records}))
    arrays = {}
    for i in range(3):
        arrays[f"train_{i}"] = np.array([[0, 0], [256, 256]])
        arrays[f"heldout_{i}"] = np.zeros((64, 64))
    np.savez(tmp_path/"patches.npz", **arrays)
    data = FlatDistillationDataset(tmp_path/"manifest.json", ["percentile_minmax"]*2,
                                   world_size=4, batch_size=2, paris4_per_batch=1)
    for start in (0, 80):
        for rank in range(4):
            draws = list(RankDrawSampler(start, 160, rank, 4))
            for i in range(0, len(draws), 2):
                indices = [data.locate(d)[0] for d in draws[i:i+2]]
                assert sum(records[j]["scroll"] == "phercparis4" for j in indices) == 1


def test_per_scroll_thresholds_do_not_mask_precise_teacher(monkeypatch):
    import vesuvius.ink_detection.training.multiteacher_loss as module
    from vesuvius.ink_detection.data.multiteacher import SCROLLS
    class Constant(nn.Module):
        def forward(self, image):
            return {"ink": torch.zeros_like(image)}
    monkeypatch.setattr(module, "load_frozen_ink_model", lambda *a, **kw: Constant())
    thresholds = {s: 70 for s in SCROLLS if s != "phercparis4"}
    thresholds["814"] = 35
    teachers = FrozenInkTeachers({"paris4": {"checkpoint": "p", "weight": 1},
        "coarse": {"checkpoint": "c", "weight": .05, "background_thresholds": thresholds}}, "cpu")
    raw = torch.full((3, 1, 2, 2, 2), 50.)
    batch = {"raw": raw, "teacher_image": raw, "teacher_id": torch.tensor([1, 1, 0]),
             "scroll_id": torch.tensor([SCROLLS.index(s) for s in ("0009b", "814", "phercparis4")])}
    targets, weights, _ = teachers.generate(batch)
    assert targets[0].eq(0).all()
    assert targets[1:].eq(.5).all()
    assert torch.allclose(weights, torch.tensor([.05, .05, 1.]))
