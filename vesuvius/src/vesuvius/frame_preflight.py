"""Resolve TIFXYZ/volume frame provenance from the public data catalog.

This module deliberately inspects catalog metadata only. It does not open the
selected TIFXYZ surface, a Zarr store, or CT data, and it does not assess mesh
quality or spatial bounds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_DECODED_CATALOG_BYTES = 128 * 1024 * 1024
MATRIX_TOLERANCE = 1e-9


class CatalogError(ValueError):
    """Raised when the explicitly supplied catalog cannot be read or parsed."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the stable JSON representation used by the CLI."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_limited(stream: Any, limit: int, description: str) -> bytes:
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise CatalogError(f"{description} exceeds the {limit}-byte metadata-only limit")
    return payload


def load_catalog(source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one explicit local or HTTP(S) JSON catalog and fingerprint it."""

    parsed = urlparse(source)
    is_windows_path = len(parsed.scheme) == 1 and len(source) >= 2 and source[1] == ":"
    headers: Mapping[str, str] = {}
    if parsed.scheme in {"http", "https"}:
        request = Request(
            source,
            headers={"User-Agent": "vesuvius.frame_preflight/1"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = _read_limited(response, MAX_CATALOG_BYTES, "Catalog")
                headers = response.headers
        except CatalogError:
            raise
        except Exception as exc:
            raise CatalogError(f"Unable to read catalog URL: {exc}") from exc
    elif parsed.scheme and not is_windows_path:
        raise CatalogError("Catalog source must be a local path or an HTTP(S) URL")
    else:
        path = Path(source)
        try:
            with path.open("rb") as stream:
                payload = _read_limited(stream, MAX_CATALOG_BYTES, "Catalog")
        except CatalogError:
            raise
        except OSError as exc:
            raise CatalogError(f"Unable to read catalog file: {exc}") from exc

    payload_sha256 = hashlib.sha256(payload).hexdigest()
    content_encoding = headers.get("Content-Encoding", "").lower()
    compressed = content_encoding == "gzip" or payload[:2] == b"\x1f\x8b"
    if compressed:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                decoded = _read_limited(
                    stream,
                    MAX_DECODED_CATALOG_BYTES,
                    "Decoded catalog",
                )
        except OSError as exc:
            raise CatalogError(f"Unable to decompress catalog: {exc}") from exc
    else:
        decoded = payload

    try:
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Catalog is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogError("Catalog root must be a JSON object")

    fingerprint: dict[str, Any] = {
        "source": source,
        "payload_bytes": len(payload),
        "decoded_bytes": len(decoded),
        "sha256": hashlib.sha256(decoded).hexdigest(),
    }
    if compressed:
        fingerprint["payload_sha256"] = payload_sha256
    if headers:
        etag = headers.get("ETag")
        fingerprint["etag"] = etag.strip('"') if etag else None
        fingerprint["last_modified"] = headers.get("Last-Modified")
    return value, fingerprint


def _base_report(
    *,
    sample_id: str,
    segment_id: str,
    entry_origin_path: str,
    target_volume_id: str,
    catalog_fingerprint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "segment_id": segment_id,
        "selected_entry_id": entry_origin_path,
        "requested_target_volume_id": target_volume_id,
        "frame_relation": "UNKNOWN",
        "frame_reason": "NOT_EVALUATED",
        "alternative_relation": "UNKNOWN",
        "alternative_reason": "NOT_EVALUATED",
        "transform": None,
        "alternative_entry_ids": [],
        "evidence": [],
        "contradictions": [],
        "catalog_fingerprint": dict(catalog_fingerprint or {}),
    }


def _evidence(field_path: str, observed_value: Any) -> dict[str, Any]:
    return {
        "field_path": field_path,
        "source": "catalog",
        "observed_value": observed_value,
    }


def _metadata_root(catalog: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = catalog.get("metadata")
    if metadata is None:
        return catalog
    return metadata if isinstance(metadata, Mapping) else None


def _lookup_object(
    objects: Any,
    requested_id: str,
    *,
    identity_fields: tuple[str, ...],
) -> tuple[Mapping[str, Any] | None, str | None]:
    if not isinstance(objects, Mapping):
        return None, "CATALOG_STRUCTURE_INCOMPLETE"
    matches = [
        item
        for key, item in objects.items()
        if isinstance(item, Mapping)
        and (
            key == requested_id
            or any(item.get(field) == requested_id for field in identity_fields)
        )
    ]
    if not matches:
        return None, "NOT_FOUND"
    if len(matches) != 1:
        return None, "AMBIGUOUS"
    return matches[0], None


def _origin_paths(entry: Mapping[str, Any]) -> list[str] | None:
    origins = entry.get("origins")
    if not isinstance(origins, list) or not origins:
        return None
    paths: list[str] = []
    for origin in origins:
        if not isinstance(origin, Mapping):
            return None
        values = [origin.get(key) for key in ("path", "url", "uri")]
        strings = [value for value in values if isinstance(value, str) and value]
        if not strings:
            return None
        paths.extend(strings)
    return sorted(set(paths))


def _entry_id(entry: Mapping[str, Any]) -> str | None:
    paths = _origin_paths(entry)
    if paths:
        return paths[0]
    entry_id = entry.get("id")
    return entry_id if isinstance(entry_id, str) and entry_id else None


def _is_tifxyz(entry_type: Any) -> bool:
    return isinstance(entry_type, str) and "tifxyz" in entry_type.lower()


def _is_transformed_tifxyz(entry_type: Any) -> bool:
    if not isinstance(entry_type, str):
        return False
    return entry_type.lower().replace("_", "-") == "tifxyz-transformed"


def _target_at(container: Any) -> tuple[str | None, bool]:
    if not isinstance(container, Mapping):
        return None, container is not None
    value = container.get("target_volume")
    if value is None:
        return None, False
    if not isinstance(value, str) or not value:
        return None, True
    return value, False


def _entry_frame_evidence(
    entry: Mapping[str, Any],
) -> tuple[str | None, str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []

    primary, primary_malformed = _target_at(entry.get("parameters"))
    evidence.append(_evidence("selected_entry.parameters.target_volume", primary))
    if primary_malformed:
        return None, "MALFORMED_FRAME_EVIDENCE", evidence, contradictions

    creation = entry.get("creation_info")
    secondary_values: list[tuple[str, str]] = []
    malformed_secondary = False
    if creation is not None and not isinstance(creation, Mapping):
        malformed_secondary = True
    if isinstance(creation, Mapping):
        direct, malformed = _target_at(creation.get("parameters"))
        malformed_secondary = malformed_secondary or malformed
        if direct is not None:
            secondary_values.append(
                ("selected_entry.creation_info.parameters.target_volume", direct)
            )

        provenance = creation.get("provenance")
        if provenance is not None and not isinstance(provenance, Mapping):
            malformed_secondary = True
        if isinstance(provenance, Mapping):
            nested, malformed = _target_at(provenance.get("parameters"))
            malformed_secondary = malformed_secondary or malformed
            if nested is not None:
                secondary_values.append(
                    ("selected_entry.creation_info.provenance.parameters.target_volume", nested)
                )

    evidence.extend(_evidence(path, value) for path, value in secondary_values)
    if malformed_secondary:
        return None, "MALFORMED_FRAME_EVIDENCE", evidence, contradictions
    if primary is None:
        return None, "MISSING_EFFECTIVE_ENTRY_FRAME_ID", evidence, contradictions

    conflicting = [(path, value) for path, value in secondary_values if value != primary]
    if conflicting:
        contradictions.append(
            {
                "code": "CONFLICTING_FRAME_EVIDENCE",
                "primary_target_volume": primary,
                "conflicting_claims": [
                    {"field_path": path, "target_volume": value}
                    for path, value in conflicting
                ],
            }
        )
        return None, "CONFLICTING_FRAME_EVIDENCE", evidence, contradictions
    return primary, None, evidence, contradictions


def _matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) not in {3, 4}:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        converted: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            number = float(item)
            if not math.isfinite(number):
                return None
            converted.append(number)
        rows.append(converted)
    if len(rows) == 3:
        rows.append([0.0, 0.0, 0.0, 1.0])
    elif not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=MATRIX_TOLERANCE)
        for actual, expected in zip(rows[3], [0.0, 0.0, 0.0, 1.0], strict=True)
    ):
        return None
    return rows


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    size = 4
    augmented = [
        matrix[row][:] + [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]
    return [row[size:] for row in augmented]


def _matrices_equal(left: list[list[float]], right: list[list[float]]) -> bool:
    return all(
        math.isclose(a, b, rel_tol=MATRIX_TOLERANCE, abs_tol=MATRIX_TOLERANCE)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def _resolve_transform(
    sample: Mapping[str, Any],
    entry_frame: str,
    requested_frame: str,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    records: list[tuple[str, Any, Any, Any]] = []
    direction_unknown = False

    sample_properties = sample.get("sample")
    groups_path = "sample.volume_transforms"
    if not isinstance(sample_properties, Mapping) or not sample_properties:
        sample_properties = sample.get("properties")
        groups_path = "properties.volume_transforms"
    groups: Any = None
    if isinstance(sample_properties, Mapping):
        groups = sample_properties.get("volume_transforms")
        nested_properties = sample_properties.get("properties")
        if groups is None and isinstance(nested_properties, Mapping):
            groups = nested_properties.get("volume_transforms")
            groups_path = groups_path.replace(
                ".volume_transforms", ".properties.volume_transforms"
            )
    if groups is not None:
        if not isinstance(groups, list):
            direction_unknown = True
        else:
            for group in groups:
                if not isinstance(group, Mapping):
                    direction_unknown = True
                    continue
                from_frame = group.get("from_volume_id")
                transforms = group.get("transforms")
                if not isinstance(transforms, list):
                    direction_unknown = True
                    continue
                for transform in transforms:
                    if not isinstance(transform, Mapping):
                        direction_unknown = True
                        continue
                    records.append(
                        (
                            groups_path,
                            from_frame,
                            transform.get("to_volume_id"),
                            transform.get(
                                "transformation_matrix", transform.get("matrix")
                            ),
                        )
                    )

    volumes = sample.get("volumes")
    if isinstance(volumes, Mapping):
        for volume_key, volume in volumes.items():
            if not isinstance(volume, Mapping):
                continue
            explicit_id = volume.get("id")
            from_frame = explicit_id if isinstance(explicit_id, str) else volume_key
            properties = volume.get("properties")
            if not isinstance(properties, Mapping):
                continue
            transforms = properties.get("transforms")
            if transforms is None:
                continue
            if not isinstance(transforms, list):
                direction_unknown = True
                continue
            for transform in transforms:
                if not isinstance(transform, Mapping):
                    direction_unknown = True
                    continue
                records.append(
                    (
                        f"volumes[{volume_key!r}].properties.transforms",
                        from_frame,
                        transform.get("to_volume_id"),
                        transform.get(
                            "transformation_matrix", transform.get("matrix")
                        ),
                    )
                )

    candidates: list[dict[str, Any]] = []
    for field_path, from_frame, to_frame, matrix_value in records:
        if not isinstance(from_frame, str) or not isinstance(to_frame, str):
            if matrix_value is not None:
                direction_unknown = True
            continue
        is_forward = from_frame == entry_frame and to_frame == requested_frame
        is_reverse = from_frame == requested_frame and to_frame == entry_frame
        if not (is_forward or is_reverse):
            continue
        field_path = (
            f"{field_path}[from_volume_id={from_frame!r}]"
            f".transforms[to_volume_id={to_frame!r}]"
        )
        matrix = _matrix4(matrix_value)
        if matrix is None:
            return None, "TRANSFORM_MATRIX_INVALID", []
        effective = matrix if is_forward else _invert_matrix(matrix)
        if effective is None:
            return None, "TRANSFORM_NOT_INVERTIBLE", []
        candidates.append(
            {
                "catalog_from_volume_id": from_frame,
                "catalog_to_volume_id": to_frame,
                "application_direction": "FORWARD" if is_forward else "INVERSE",
                "inverse_required": is_reverse,
                "matrix": matrix,
                "effective_matrix": effective,
                "field_path": field_path,
            }
        )

    if not candidates:
        reason = "TRANSFORM_DIRECTION_UNKNOWN" if direction_unknown else "TRANSFORM_NOT_FOUND"
        return None, reason, []
    candidates.sort(
        key=lambda item: (
            item["application_direction"] != "FORWARD",
            item["field_path"],
            json.dumps(item["matrix"], separators=(",", ":")),
        )
    )
    reference = candidates[0]["effective_matrix"]
    if any(not _matrices_equal(reference, item["effective_matrix"]) for item in candidates[1:]):
        contradictions = [
            {
                "code": "CONFLICTING_TRANSFORMS",
                "field_path": item["field_path"],
                "catalog_from_volume_id": item["catalog_from_volume_id"],
                "catalog_to_volume_id": item["catalog_to_volume_id"],
            }
            for item in candidates
        ]
        return None, "CONFLICTING_TRANSFORMS", contradictions

    chosen = candidates[0]
    transform_result = {
        key: chosen[key]
        for key in (
            "catalog_from_volume_id",
            "catalog_to_volume_id",
            "application_direction",
            "inverse_required",
            "matrix",
        )
    }
    transform_evidence = [
        _evidence(item["field_path"], {
            "from_volume_id": item["catalog_from_volume_id"],
            "to_volume_id": item["catalog_to_volume_id"],
        })
        for item in candidates
    ]
    return transform_result, None, transform_evidence


def _alternatives(
    entries: list[Any],
    selected: Mapping[str, Any],
    requested_target: str,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    matches: list[str] = []
    evidence: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return "UNKNOWN", "CATALOG_STRUCTURE_INCOMPLETE", [], evidence
        entry_type = entry.get("type")
        if not isinstance(entry_type, str):
            return "UNKNOWN", "CATALOG_STRUCTURE_INCOMPLETE", [], evidence
        if entry is selected or not _is_tifxyz(entry_type):
            continue
        entry_id = _entry_id(entry)
        if entry_id is None:
            return "UNKNOWN", "CATALOG_STRUCTURE_INCOMPLETE", [], evidence
        if not _is_transformed_tifxyz(entry_type):
            continue
        target, reason, _target_evidence, contradictions = _entry_frame_evidence(entry)
        if reason is not None or contradictions:
            return "UNKNOWN", "CATALOG_STRUCTURE_INCOMPLETE", [], evidence
        if target == requested_target:
            matches.append(entry_id)
            evidence.append(
                _evidence(
                    f"segment.data[origin={entry_id!r}].parameters.target_volume",
                    target,
                )
            )
    if matches:
        return (
            "SAME_FRAME_ALTERNATIVE_AVAILABLE",
            "EXPLICIT_CATALOG_ENTRY_TARGETS_REQUESTED_VOLUME",
            sorted(set(matches)),
            evidence,
        )
    return (
        "NONE_CONFIRMED",
        "COMPLETE_SEGMENT_ENUMERATION_WITHOUT_STRONG_MATCH",
        [],
        evidence,
    )


def inspect_catalog(
    catalog: Mapping[str, Any],
    *,
    sample_id: str,
    segment_id: str,
    entry_origin_path: str,
    target_volume_id: str,
    catalog_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate frame and alternative evidence for one selected catalog entry."""

    report = _base_report(
        sample_id=sample_id,
        segment_id=segment_id,
        entry_origin_path=entry_origin_path,
        target_volume_id=target_volume_id,
        catalog_fingerprint=catalog_fingerprint,
    )
    metadata = _metadata_root(catalog)
    if metadata is None:
        report["frame_reason"] = "CATALOG_STRUCTURE_INCOMPLETE"
        report["alternative_reason"] = "CATALOG_STRUCTURE_INCOMPLETE"
        return report

    sample, sample_error = _lookup_object(
        metadata.get("samples"), sample_id, identity_fields=("id", "sample_id")
    )
    if sample is None:
        report["frame_reason"] = f"SAMPLE_{sample_error}"
        report["alternative_reason"] = f"SAMPLE_{sample_error}"
        return report

    segment, segment_error = _lookup_object(
        sample.get("segments"),
        segment_id,
        identity_fields=("id", "long_id"),
    )
    if segment is None:
        report["frame_reason"] = f"SEGMENT_{segment_error}"
        report["alternative_reason"] = f"SEGMENT_{segment_error}"
        return report

    original_volume = segment.get("original_volume_id")
    if isinstance(original_volume, str) and original_volume:
        report["evidence"].append(
            _evidence("segment.original_volume_id", original_volume)
        )

    entries = segment.get("data")
    if not isinstance(entries, list):
        report["frame_reason"] = "CATALOG_STRUCTURE_INCOMPLETE"
        report["alternative_reason"] = "CATALOG_STRUCTURE_INCOMPLETE"
        return report
    selected_matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and (_origin_paths(entry) or [])
        and entry_origin_path in (_origin_paths(entry) or [])
    ]
    if not selected_matches:
        report["frame_reason"] = "SELECTED_ENTRY_NOT_FOUND"
        report["alternative_reason"] = "SELECTED_ENTRY_NOT_FOUND"
        return report
    if len(selected_matches) != 1:
        report["frame_reason"] = "SELECTED_ENTRY_AMBIGUOUS"
        report["alternative_reason"] = "SELECTED_ENTRY_AMBIGUOUS"
        return report
    selected = selected_matches[0]
    report["selected_entry_id"] = entry_origin_path

    alternative_relation, alternative_reason, alternatives, alternative_evidence = _alternatives(
        entries, selected, target_volume_id
    )
    report["alternative_relation"] = alternative_relation
    report["alternative_reason"] = alternative_reason
    report["alternative_entry_ids"] = alternatives
    report["evidence"].extend(alternative_evidence)

    entry_frame, frame_error, frame_evidence, contradictions = _entry_frame_evidence(selected)
    report["evidence"].extend(frame_evidence)
    report["contradictions"].extend(contradictions)
    if frame_error is not None or entry_frame is None:
        report["frame_reason"] = frame_error or "MISSING_EFFECTIVE_ENTRY_FRAME_ID"
        return report
    if not _is_transformed_tifxyz(selected.get("type")):
        report["frame_reason"] = "ENTRY_TYPE_DOES_NOT_ESTABLISH_EFFECTIVE_FRAME"
        return report

    if entry_frame == target_volume_id:
        report["frame_relation"] = "SAME_FRAME"
        report["frame_reason"] = "EXPLICIT_ENTRY_TARGET_MATCHES_REQUESTED_VOLUME"
        return report

    transform, transform_error, transform_details = _resolve_transform(
        sample, entry_frame, target_volume_id
    )
    if transform_error is not None or transform is None:
        report["frame_reason"] = transform_error or "TRANSFORM_NOT_FOUND"
        if transform_error == "CONFLICTING_TRANSFORMS":
            report["contradictions"].extend(transform_details)
        else:
            report["evidence"].extend(transform_details)
        return report

    report["frame_relation"] = "TRANSFORM_REQUIRED"
    report["frame_reason"] = "EXPLICIT_DIFFERENT_TARGET_AND_DIRECTED_VOLUME_TRANSFORM"
    report["transform"] = transform
    report["evidence"].extend(transform_details)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a selected TIFXYZ entry's coordinate frame and explicit "
            "same-frame alternatives from catalog metadata only."
        )
    )
    parser.add_argument("--catalog", required=True, help="Local JSON path or explicit HTTP(S) URL")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--entry-origin-path", required=True)
    parser.add_argument("--target-volume-id", required=True)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog, fingerprint = load_catalog(args.catalog)
        report = inspect_catalog(
            catalog,
            sample_id=args.sample_id,
            segment_id=args.segment_id,
            entry_origin_path=args.entry_origin_path,
            target_volume_id=args.target_volume_id,
            catalog_fingerprint=fingerprint,
        )
    except CatalogError as exc:
        report = _base_report(
            sample_id=args.sample_id,
            segment_id=args.segment_id,
            entry_origin_path=args.entry_origin_path,
            target_volume_id=args.target_volume_id,
            catalog_fingerprint={"source": args.catalog},
        )
        report["frame_reason"] = "CATALOG_UNREADABLE"
        report["alternative_reason"] = "CATALOG_UNREADABLE"
        report["contradictions"].append(
            {"code": "CATALOG_UNREADABLE", "detail": str(exc)}
        )

    payload = canonical_json_bytes(report)
    if args.output is not None:
        args.output.write_bytes(payload)
    else:
        print(payload.decode("utf-8"), end="")
    return 0 if report["frame_relation"] == "SAME_FRAME" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
