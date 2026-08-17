"""Schema and value contracts for the shipped aligned-corpus configurations."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping


_CONFIGS = (
    Path(__file__).parents[2]
    / "src"
    / "vesuvius"
    / "ink_detection"
    / "configs"
)


def _load_config(name: str) -> dict:
    return json.loads((_CONFIGS / name).read_text(encoding="utf-8"))


def test_aligned21_hybrid_recipe_parses_with_frozen_values_and_layout():
    authored = _load_config("aligned21_hybrid_3d2d.json")
    prior = _load_config("aligned21_fixed_scroll_prior.json")

    training = TrainingConfig.from_mapping(resolve_training_mapping(authored))

    assert training.ink.model.model_type == "vesuvius_unet_3d_stem_2d"
    assert training.ink.model.pretrained_backbone is None
    assert training.ink.model.freeze_encoder is False
    assert training.ink.model.spacing == (1.0, 1.0, 1.0)
    assert training.wandb_run_id is None
    assert training.wandb_resume is False
    assert training.model_crop_size == (17, 128, 128)
    assert training.ink.data.jitter.window_depth == 17
    assert training.ink.data.jitter.max_offset == 2
    assert training.ink.data.normalization.mode == "robust_mad"
    assert training.ink.data.sampling.strategy == "fixed_scroll_prior_stratified"
    assert training.ink.data.sampling.fixed_batch_quotas == {
        "0139": 29,
        "1667": 22,
        "Paris4": 11,
        "0814": 2,
    }

    sources = authored["datasets"]
    aligned = "/path/to/ink_9um/labels/aligned-scrollprizeorg-21slices"
    native = "/path/to/ink_9um/labels/native9-scrollprizeorg-21slices"
    assert [source["segments_path"] for source in sources] == [
        aligned,
        native,
        aligned,
        aligned,
        aligned,
    ]
    assert [source["segments"] for source in sources] == [
        [
            "pherc0139-w016",
            "pherc0139-w017",
            "pherc0139-w028",
            "pherc0139-w029",
            "pherc0139-w035",
            "pherc0139-w039",
            "pherc0139-w040",
            "pherc0139-w041",
            "pherc0139-w043",
        ],
        ["w035", "w039", "w040", "w041", "w044"],
        ["pherc0814-46527"],
        [
            "pherc1667-w013",
            "pherc1667-w018",
            "pherc1667-w023",
            "pherc1667-w028",
            "pherc1667-w029",
            "pherc1667-w031",
        ],
        [
            "phercparis4-w00",
            "phercparis4-w01",
            "phercparis4-w02",
            "phercparis4-w03",
            "phercparis4-w05",
            "phercparis4-w06",
            "phercparis4-w07",
            "phercparis4-w09",
        ],
    ]
    families = {aligned: "public_2p4_level2_zmean4", native: "native_9p362_level0"}
    for source in sources:
        family = families[source["segments_path"]]
        assert source["volume_scale"] == 0
        for segment in source["segments"]:
            assert source["surface_volume_paths"][segment] == (
                f"{source['segments_path']}/{segment}/surface-volume.zarr"
            )
            assert source["sampling_representation_keys"][segment] == (
                f"{family}:{segment}"
            )
    authored_representations = {
        (
            segment,
            source["sampling_scroll"],
            source["sampling_physical_segment_keys"][segment],
            source["sampling_representation_keys"][segment],
        )
        for source in sources
        for segment in source["segments"]
    }
    assert authored_representations == {
        (
            item["segment"],
            item["scroll"],
            item["physical_segment_key"],
            item["representation_key"],
        )
        for item in prior["representations"]
    }


def test_aligned21_fixed_prior_manifest_has_frozen_schema_and_corpus_values():
    prior = _load_config("aligned21_fixed_scroll_prior.json")
    representations = prior["representations"]

    assert set(prior) == {
        "batch_size",
        "description",
        "representations",
        "schema_version",
        "seed",
        "strategy",
        "target_batch_counts",
    }
    assert prior["schema_version"] == 1
    assert prior["strategy"] == "fixed_scroll_prior_stratified"
    assert prior["seed"] == 42
    assert prior["batch_size"] == 64
    assert prior["target_batch_counts"] == {
        "0139": 29,
        "1667": 22,
        "Paris4": 11,
        "0814": 2,
    }
    assert sum(prior["target_batch_counts"].values()) == prior["batch_size"]
    assert len(representations) == 29
    assert all(
        set(item)
        == {
            "source_family",
            "segment",
            "scroll",
            "physical_segment_key",
            "representation_key",
        }
        for item in representations
    )
    assert Counter(item["scroll"] for item in representations) == {
        "0139": 14,
        "1667": 6,
        "Paris4": 8,
        "0814": 1,
    }
    assert Counter(item["source_family"] for item in representations) == {
        "public_2p4_level2_zmean4": 24,
        "native_9p362_level0": 5,
    }
    assert len({item["representation_key"] for item in representations}) == 29
    assert representations[0] == {
        "source_family": "public_2p4_level2_zmean4",
        "segment": "pherc0139-w016",
        "scroll": "0139",
        "physical_segment_key": "0139:w016",
        "representation_key": "public_2p4_level2_zmean4:pherc0139-w016",
    }
    assert representations[-1] == {
        "source_family": "native_9p362_level0",
        "segment": "w044",
        "scroll": "0139",
        "physical_segment_key": "0139:w044",
        "representation_key": "native_9p362_level0:w044",
    }
