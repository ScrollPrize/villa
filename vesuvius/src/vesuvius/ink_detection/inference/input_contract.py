"""Check that a flat-inference input matches what the released recipe trained on.

The released ``aligned21`` checkpoints read a 17-slice window cut from a
21-slice surface volume (the centred 84 planes of a level-2 render, mean-pooled
by 4, see ``prepare_9um_isotropic_input``). Inference accepts any volume that is
at least as deep as the window, so a 17-slice level-2 CT, a wrongly prepared
volume, or the wrong depth order runs to completion and returns a confident map
that scores at chance. This module reports those cases before the first patch is
read. It cannot prove an input is right; it refuses the ones that are provably
not the trained form and says what it cannot verify.

Command line (no GPU work; loads the checkpoint config only)::

    python -m vesuvius.ink_detection.inference.input_contract \\
        SURFACE_VOLUME.zarr CHECKPOINT.pth [--direction both]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

# Mirrors prepare_9um_isotropic_input (FORMAT_TAG, OUTPUT_Z * POOL_Z); kept as
# literals so the check has no zarr/numcodecs import cost.
FORMAT_TAG = "level2-zmean4-21slice-v1"
PREPARED_LEVEL = "2"
PREPARED_SOURCE_PLANES = 84

ERROR, WARN, INFO = "error", "warn", "info"
MODES = ("error", "warn", "off")


@dataclass(frozen=True)
class Finding:
    """One contract observation: severity, stable code, human message."""

    severity: str
    code: str
    message: str


class InputContractError(ValueError):
    """Raised when an input provably does not match the recipe it is run with."""


def expected_source_depth(window_depth: int, max_z_offset: int) -> int | None:
    """Depth of the volume a jittered-window recipe trained on, if declared.

    A recipe trained with ``flat_z_window_jitter`` (window ``W``, offsets
    ``+-m``) saw ``W + 2m`` slices; 17 and 2 give the 21-slice released input.
    """

    if max_z_offset <= 0:
        return None
    return int(window_depth) + 2 * int(max_z_offset)


def _describe_layers(layer_indices: np.ndarray) -> str:
    indices = np.asarray(layer_indices, dtype=np.int64)
    if indices.size == 0:
        return "no layers"
    order = "ascending" if indices[0] <= indices[-1] else "descending"
    return f"{int(indices[0])}..{int(indices[-1])} ({indices.size} layers, {order})"


def check_input_contract(
    *,
    depth: int,
    dtype: Any,
    attrs: Mapping[str, Any] | None,
    layer_indices: Sequence[int],
    direction: str,
    window_depth: int,
    preprocessing: str,
    max_z_offset: int = 0,
) -> list[Finding]:
    """Return every contract finding for one volume; pure, no I/O.

    ``direction`` is the requested direction (``forward``, ``reverse`` or
    ``both``); ``layer_indices`` are the source layers of the run being checked.
    ``max_z_offset`` is the checkpoint's z-jitter half-range; ``0`` means the
    recipe declares no source depth and the depth-mismatch rules are skipped.
    """

    source_depth = expected_source_depth(window_depth, max_z_offset)
    attrs = dict(attrs or {})
    layers = np.asarray(layer_indices, dtype=np.int64)
    depth = int(depth)
    findings: list[Finding] = []

    def add(severity: str, code: str, message: str) -> None:
        findings.append(Finding(severity, code, message))

    if depth < window_depth:
        add(
            ERROR,
            "DEPTH_TOO_SHALLOW",
            f"input has {depth} slices but the model reads a {window_depth}-slice "
            "window; the missing slices would be zero-filled",
        )
    elif source_depth is not None and depth < source_depth:
        add(
            ERROR,
            "DEPTH_NOT_TRAINED_FORM",
            f"input has {depth} slices; this recipe trained on {source_depth} "
            f"(a {window_depth}-slice window cut from it), so a {depth}-slice CT "
            "is not that representation. Prepare the input with "
            "prepare_9um_isotropic_input: centred 84 planes, mean-pooled by 4",
        )
    elif source_depth is not None and depth > source_depth:
        add(
            WARN,
            "DEPTH_DEEPER_THAN_TRAINED",
            f"input has {depth} slices, deeper than the {source_depth} this recipe "
            "trained on; only a centred window is read, which is right only if the "
            "extra slices are symmetric padding of the same surface",
        )

    tag = attrs.get("format")
    if tag is None:
        add(
            INFO,
            "NO_PROVENANCE",
            "input carries no 'format' attribute, so pooling, scale and z-window "
            "cannot be verified from metadata",
        )
    elif tag != FORMAT_TAG:
        add(
            WARN,
            "UNKNOWN_FORMAT",
            f"input format {tag!r} is not the released {FORMAT_TAG!r}",
        )
    else:
        level = attrs.get("source_level")
        if level is not None and str(level) != PREPARED_LEVEL:
            add(
                ERROR,
                "SCALE_MISMATCH",
                f"input was prepared from pyramid level {level!r}; the recipe "
                f"trains on level {PREPARED_LEVEL} (9.6 um in-plane for a 2.399 um "
                "render). Prepare it again with --level 2",
            )
        z_slice = attrs.get("source_z_slice")
        if z_slice is not None:
            span = int(z_slice[1]) - int(z_slice[0])
            shape = attrs.get("source_shape_zyx")
            if span != PREPARED_SOURCE_PLANES:
                add(
                    ERROR,
                    "Z_POOL_MISMATCH",
                    f"input pools {span} source planes; the recipe trains on "
                    f"{PREPARED_SOURCE_PLANES} (21 slices of 4)",
                )
            elif shape is not None:
                start = -(-(int(shape[0]) - span) // 2)  # centered_slice: ceil
                if int(z_slice[0]) != start:
                    add(
                        WARN,
                        "Z_WINDOW_OFF_CENTRE",
                        f"input pools source planes {list(map(int, z_slice))}, not "
                        f"the centred window starting at {start}",
                    )

    if preprocessing == "divide_255" and np.dtype(dtype) != np.uint8:
        add(
            ERROR,
            "DTYPE",
            f"input dtype is {np.dtype(dtype)} but this recipe divides by 255 and "
            "needs uint8",
        )
    elif tag == FORMAT_TAG and np.dtype(dtype) != np.uint8:
        add(WARN, "DTYPE", f"prepared input should be uint8, got {np.dtype(dtype)}")

    if layers.size != window_depth:
        add(
            ERROR,
            "WINDOW_SIZE",
            f"{layers.size} layers selected but the model reads {window_depth}; "
            "the rest would be zero-filled (check --layer-start/--layer-end)",
        )
    elif depth >= window_depth and layers.size:
        first = int(layers.min())
        centred = (depth - window_depth) // 2
        if 0 < max_z_offset < abs(first - centred):
            add(
                WARN,
                "Z_WINDOW_OUTSIDE_TRAINED_RANGE",
                f"window starts at layer {first}; training jitter kept it within "
                f"+-{max_z_offset} of the centre ({centred})",
            )

    if direction == "both":
        add(INFO, "DIRECTION", "direction=both: forward and reverse are each run")
    else:
        detail = (
            f"direction={direction}: source layers {_describe_layers(layers)}. "
            "Orientation cannot be verified from the data; a depth order opposite "
            "to the training labels returns a confident map at chance level, "
            "so run --direction both when the orientation is not known"
        )
        add(WARN if direction == "reverse" else INFO, "DIRECTION", detail)
    return findings


def enforce_input_contract(
    findings: Sequence[Finding], *, mode: str, label: str = "input"
) -> None:
    """Log findings and raise :class:`InputContractError` on errors in ``error`` mode."""

    if mode not in MODES:
        raise ValueError(f"input contract mode must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return
    for finding in findings:
        level = logging.INFO if finding.severity == INFO else logging.WARNING
        if finding.severity == ERROR and mode == "warn":
            LOGGER.warning(
                "input contract WARNING [%s] %s: %s (--input-contract error "
                "would refuse this input; continuing because the mode is warn)",
                finding.code, label, finding.message,
            )
            continue
        LOGGER.log(
            level, "input contract %s [%s] %s: %s",
            finding.severity.upper(), finding.code, label, finding.message,
        )
    errors = [f for f in findings if f.severity == ERROR]
    if errors and mode == "error":
        raise InputContractError(
            f"{label} does not match the input contract: "
            + "; ".join(f"[{f.code}] {f.message}" for f in errors)
            + " (pass --input-contract warn to run anyway)"
        )


def require_finite_probabilities(probabilities: np.ndarray) -> None:
    """Raise if a batch of probabilities has NaN/inf (typically fp16 overflow)."""

    if not np.isfinite(probabilities).all():
        raise FloatingPointError(
            "model produced non-finite probabilities; the output would be written "
            "as zeros. Half precision overflows on some GPUs: rerun in full "
            "precision (see --amp-dtype)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Check one volume against a checkpoint's contract; exit 1 on errors."""

    from vesuvius.ink_detection.inference import infer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_zarr")
    parser.add_argument("checkpoint")
    parser.add_argument("--resolution", default="0")
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument(
        "--direction", choices=("forward", "reverse", "both"), default="forward"
    )
    args = parser.parse_args(argv)
    findings = infer.contract_findings_for(
        input_zarr=args.input_zarr,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        requested_direction=args.direction,
    )
    for finding in findings:
        print(f"{finding.severity.upper():5s} [{finding.code}] {finding.message}")
    failed = any(f.severity == ERROR for f in findings)
    print("CONTRACT FAIL" if failed else "CONTRACT OK (see warnings)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
