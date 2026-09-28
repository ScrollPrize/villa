from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from vesuvius import frame_preflight


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def transformed_entry(
    path: str,
    target: str | None,
    *,
    creation_target: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "tifxyz-transformed",
        "origins": [{"path": path}],
    }
    if target is not None:
        entry["parameters"] = {"target_volume": target}
    if creation_target is not None:
        entry["creation_info"] = {
            "provenance": {"parameters": {"target_volume": creation_target}}
        }
    return entry


def catalog(
    entries: list[object],
    *,
    transforms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "samples": {
                "sample-a": {
                    "sample": {"volume_transforms": transforms or []},
                    "segments": {
                        "segment-a": {
                            "id": "short-segment-a",
                            "long_id": "segment-a",
                            "original_volume_id": "requested-frame",
                            "data": entries,
                        }
                    },
                }
            }
        }
    }


def transform_group(
    from_frame: str | None,
    to_frame: str | None,
    matrix: list[list[float]],
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "transforms": [{"transformation_matrix": matrix}],
    }
    if from_frame is not None:
        group["from_volume_id"] = from_frame
    if to_frame is not None:
        group["transforms"][0]["to_volume_id"] = to_frame
    return group


def inspect(
    value: dict[str, Any],
    *,
    selected: str = "selected/",
    requested: str = "requested-frame",
) -> dict[str, Any]:
    return frame_preflight.inspect_catalog(
        value,
        sample_id="sample-a",
        segment_id="segment-a",
        entry_origin_path=selected,
        target_volume_id=requested,
        catalog_fingerprint={"source": "fixture", "sha256": "fixture-sha"},
    )


def test_explicit_entry_target_is_same_frame() -> None:
    report = inspect(catalog([transformed_entry("selected/", "requested-frame")]))

    assert report["frame_relation"] == "SAME_FRAME"
    assert report["frame_reason"] == "EXPLICIT_ENTRY_TARGET_MATCHES_REQUESTED_VOLUME"
    assert report["transform"] is None
    assert report["alternative_relation"] == "NONE_CONFIRMED"
    assert "bounds_relation" not in report
    assert "mesh_quality" not in report


def test_direct_transform_is_required_with_forward_direction() -> None:
    value = catalog(
        [transformed_entry("selected/", "entry-frame")],
        transforms=[transform_group("entry-frame", "requested-frame", IDENTITY)],
    )

    report = inspect(value)

    assert report["frame_relation"] == "TRANSFORM_REQUIRED"
    assert report["transform"]["application_direction"] == "FORWARD"
    assert report["transform"]["inverse_required"] is False


def test_reverse_catalog_transform_requires_inverse() -> None:
    value = catalog(
        [transformed_entry("selected/", "entry-frame")],
        transforms=[transform_group("requested-frame", "entry-frame", IDENTITY)],
    )

    report = inspect(value)

    assert report["frame_relation"] == "TRANSFORM_REQUIRED"
    assert report["transform"]["catalog_from_volume_id"] == "requested-frame"
    assert report["transform"]["catalog_to_volume_id"] == "entry-frame"
    assert report["transform"]["application_direction"] == "INVERSE"
    assert report["transform"]["inverse_required"] is True


def test_explicit_alternative_for_target_is_reported() -> None:
    value = catalog(
        [
            transformed_entry("selected/", "entry-frame"),
            transformed_entry("same-frame/", "requested-frame"),
        ],
        transforms=[transform_group("entry-frame", "requested-frame", IDENTITY)],
    )

    report = inspect(value)

    assert report["alternative_relation"] == "SAME_FRAME_ALTERNATIVE_AVAILABLE"
    assert report["alternative_entry_ids"] == ["same-frame/"]
    assert "recommended_alternative" not in report


def test_segment_original_volume_does_not_rescue_missing_entry_frame() -> None:
    selected = {
        "type": "tifxyz",
        "origins": [{"path": "selected/"}],
    }

    report = inspect(catalog([selected]))

    assert report["frame_relation"] == "UNKNOWN"
    assert report["frame_reason"] == "MISSING_EFFECTIVE_ENTRY_FRAME_ID"
    assert any(
        item["field_path"] == "segment.original_volume_id"
        for item in report["evidence"]
    )

    raw_with_target = {
        "type": "tifxyz",
        "origins": [{"path": "selected/"}],
        "parameters": {"target_volume": "requested-frame"},
    }
    raw_report = inspect(catalog([raw_with_target]))
    assert raw_report["frame_relation"] == "UNKNOWN"
    assert raw_report["frame_reason"] == "ENTRY_TYPE_DOES_NOT_ESTABLISH_EFFECTIVE_FRAME"


def test_conflicting_strong_frame_evidence_is_unknown() -> None:
    selected = transformed_entry(
        "selected/",
        "entry-frame",
        creation_target="different-frame",
    )

    report = inspect(catalog([selected]))

    assert report["frame_relation"] == "UNKNOWN"
    assert report["frame_reason"] == "CONFLICTING_FRAME_EVIDENCE"
    assert report["contradictions"]


@pytest.mark.parametrize(
    ("transforms", "reason"),
    [
        ([], "TRANSFORM_NOT_FOUND"),
        (
            [
                transform_group(
                    "requested-frame",
                    "entry-frame",
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
            ],
            "TRANSFORM_NOT_INVERTIBLE",
        ),
        (
            [transform_group(None, "requested-frame", IDENTITY)],
            "TRANSFORM_DIRECTION_UNKNOWN",
        ),
        (
            [
                transform_group(
                    "entry-frame",
                    "requested-frame",
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                )
            ],
            "TRANSFORM_MATRIX_INVALID",
        ),
    ],
)
def test_missing_noninvertible_or_undirected_transform_is_unknown(
    transforms: list[dict[str, Any]], reason: str
) -> None:
    value = catalog(
        [transformed_entry("selected/", "entry-frame")],
        transforms=transforms,
    )

    report = inspect(value)

    assert report["frame_relation"] == "UNKNOWN"
    assert report["frame_reason"] == reason


def test_multiple_alternatives_are_sorted_without_selection() -> None:
    value = catalog(
        [
            transformed_entry("selected/", "requested-frame"),
            transformed_entry("z-alternative/", "requested-frame"),
            transformed_entry("a-alternative/", "requested-frame"),
        ]
    )

    report = inspect(value)

    assert report["alternative_relation"] == "SAME_FRAME_ALTERNATIVE_AVAILABLE"
    assert report["alternative_entry_ids"] == [
        "a-alternative/",
        "z-alternative/",
    ]
    assert "recommended_alternative" not in report


def test_conflicting_duplicate_transforms_are_unknown() -> None:
    translated = [row[:] for row in IDENTITY]
    translated[0][3] = 10.0
    value = catalog(
        [transformed_entry("selected/", "entry-frame")],
        transforms=[
            transform_group("entry-frame", "requested-frame", IDENTITY),
            transform_group("entry-frame", "requested-frame", translated),
        ],
    )

    report = inspect(value)

    assert report["frame_relation"] == "UNKNOWN"
    assert report["frame_reason"] == "CONFLICTING_TRANSFORMS"

    forward = transform_group("entry-frame", "requested-frame", IDENTITY)
    reverse = transform_group("requested-frame", "entry-frame", IDENTITY)
    forward_first = inspect(
        catalog(
            [transformed_entry("selected/", "entry-frame")],
            transforms=[forward, reverse],
        )
    )
    reverse_first = inspect(
        catalog(
            [transformed_entry("selected/", "entry-frame")],
            transforms=[reverse, forward],
        )
    )
    assert frame_preflight.canonical_json_bytes(
        forward_first
    ) == frame_preflight.canonical_json_bytes(reverse_first)


def test_ambiguity_incomplete_enumeration_and_cli_output_are_conservative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambiguous = catalog(
        [
            transformed_entry("selected/", "requested-frame"),
            transformed_entry("selected/", "requested-frame"),
        ]
    )
    ambiguous_report = inspect(ambiguous)
    assert ambiguous_report["frame_reason"] == "SELECTED_ENTRY_AMBIGUOUS"
    assert ambiguous_report["alternative_relation"] == "UNKNOWN"

    duplicate_segment_identity = catalog(
        [transformed_entry("selected/", "requested-frame")]
    )
    segments = duplicate_segment_identity["metadata"]["samples"]["sample-a"]["segments"]
    segments["other-segment"] = {
        "long_id": "segment-a",
        "data": [transformed_entry("other/", "requested-frame")],
    }
    duplicate_report = inspect(duplicate_segment_identity)
    assert duplicate_report["frame_reason"] == "SEGMENT_AMBIGUOUS"

    incomplete = catalog(
        [
            transformed_entry("selected/", "requested-frame"),
            {"type": "tifxyz-transformed", "origins": [{"path": "broken/"}]},
        ]
    )
    incomplete_report = inspect(incomplete)
    assert incomplete_report["frame_relation"] == "SAME_FRAME"
    assert incomplete_report["alternative_relation"] == "UNKNOWN"
    assert incomplete_report["alternative_reason"] == "CATALOG_STRUCTURE_INCOMPLETE"

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog([transformed_entry("selected/", "requested-frame")])),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        frame_preflight,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local catalog loading must not use the network")
        ),
    )
    loaded, _fingerprint = frame_preflight.load_catalog(str(catalog_path))
    assert loaded["metadata"]["samples"]["sample-a"]

    compressed_path = tmp_path / "catalog.json.gz"
    compressed_path.write_bytes(gzip.compress(b'{"metadata": {}}'))
    with monkeypatch.context() as limit_patch:
        limit_patch.setattr(frame_preflight, "MAX_DECODED_CATALOG_BYTES", 4)
        with pytest.raises(frame_preflight.CatalogError, match="Decoded catalog exceeds"):
            frame_preflight.load_catalog(str(compressed_path))

    source_root = Path(frame_preflight.__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    command = [
        sys.executable,
        "-m",
        "vesuvius.frame_preflight",
        "--catalog",
        str(catalog_path),
        "--sample-id",
        "sample-a",
        "--segment-id",
        "segment-a",
        "--entry-origin-path",
        "selected/",
        "--target-volume-id",
        "requested-frame",
    ]
    for output in outputs:
        completed = subprocess.run(
            [*command, "--output", str(output)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    first = outputs[0].read_bytes()
    second = outputs[1].read_bytes()
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
