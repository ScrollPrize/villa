"""Keep the shipped training recipe and sampling manifest consistent."""

from collections import Counter
import json
from pathlib import Path

import pytest

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping


_CONFIGS = Path(__file__).parents[2] / "src/vesuvius/ink_detection/configs"


@pytest.fixture
def aligned21_configs():
    return tuple(
        json.loads((_CONFIGS / name).read_text(encoding="utf-8"))
        for name in ("aligned21_hybrid_3d2d.json", "aligned21_fixed_scroll_prior.json")
    )


def test_training_recipe_and_prior_agree(aligned21_configs):
    authored, prior = aligned21_configs
    training = TrainingConfig.from_mapping(resolve_training_mapping(authored))
    sampling = training.ink.data.sampling

    assert prior["schema_version"] == 1
    assert sampling.strategy == prior["strategy"] == "fixed_scroll_prior_stratified"
    assert dict(sampling.fixed_batch_quotas) == prior["target_batch_counts"]
    assert sum(sampling.fixed_batch_quotas.values()) == training.batch_size == prior["batch_size"]
    assert all(count > 0 for count in sampling.fixed_batch_quotas.values())
    assert training.seed == authored["fixed_scroll_prior"]["seed"] == prior["seed"]

    # The hybrid stem consumes the jittered depth window; both must fit the
    # released 21-slice volumes without padding.
    assert training.ink.model.model_type == "vesuvius_unet_3d_stem_2d"
    jitter = training.ink.data.jitter
    assert training.model_crop_size[0] == jitter.window_depth
    assert jitter.window_depth + 2 * jitter.max_offset <= 21


def test_manifest_matches_training_sources_without_duplicate_representations(aligned21_configs):
    authored, prior = aligned21_configs
    training_rows = []
    for source in authored["datasets"]:
        segments = source["segments"]
        assert segments
        assert source["volume_scale"] == 0
        for key in ("surface_volume_paths", "sampling_physical_segment_keys", "sampling_representation_keys"):
            assert set(source[key]) == set(segments)
        for segment in segments:
            assert source["surface_volume_paths"][segment] == str(
                Path(source["segments_path"]) / segment / "surface-volume.zarr"
            )
            training_rows.append((
                segment, source["sampling_scroll"],
                source["sampling_physical_segment_keys"][segment],
                source["sampling_representation_keys"][segment],
            ))

    rows = prior["representations"]
    manifest_rows = [
        (row["segment"], row["scroll"], row["physical_segment_key"], row["representation_key"])
        for row in rows
    ]
    # A set comparison alone silently accepts duplicate rows, which would bias
    # sampling toward the duplicated representation.
    assert len({row[3] for row in training_rows}) == len(training_rows)
    assert len({row[3] for row in manifest_rows}) == len(manifest_rows)
    assert Counter(training_rows) == Counter(manifest_rows)
    assert {row["scroll"] for row in rows} == set(prior["target_batch_counts"])
    for row in rows:
        assert row["representation_key"] == f"{row['source_family']}:{row['segment']}"
        assert row["physical_segment_key"].startswith(row["scroll"] + ":")


def test_duplicate_scans_share_the_published_physical_sampling_group(aligned21_configs):
    _, prior = aligned21_configs
    groups = {
        row["representation_key"]: (row["scroll"], row["physical_segment_key"])
        for row in prior["representations"]
    }
    # The published ink_9um source table maps aligned w028 to native w044:
    # PHerc0139/segments/20260115000000-w044_2026011522. Folder suffixes
    # therefore cannot determine physical identity. These corpus counts are
    # intentional regression checks, unlike incidental recipe hyperparameters.
    assert len(groups) == 29
    assert len(set(groups.values())) == 24
    assert len({physical for scroll, physical in groups.values() if scroll == "0139"}) == 9
    assert groups["public_2p4_level2_zmean4:pherc0139-w028"] == ("0139", "0139:w044")
    assert groups["native_9p362_level0:w044"] == ("0139", "0139:w044")
