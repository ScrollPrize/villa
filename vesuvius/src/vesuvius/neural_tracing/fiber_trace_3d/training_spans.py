"""Training-only eligibility; do not alter external tracing references."""

from dataclasses import replace

import numpy as np

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index


def training_fiber_spans(fiber: Vc3dFiber) -> list[Vc3dFiber]:
    """Return maximal eligible runs, without tails or connectors across gaps.

    NML edges are explicit annotations; their synthesized legacy Lasagna mode
    is not provenance. JSON spans require an explicit cspline/trace mode.
    """
    count = len(fiber.control_points_xyz)
    if count < 2:
        return []
    indices = [control_point_line_index(fiber, i) for i in range(count)]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"training fiber {fiber.path}: control points must follow line order")
    if len(fiber.control_point_segments) != count:
        raise ValueError(f"training fiber {fiber.path}: inconsistent segment metadata count")
    nml = fiber.metadata.get("source_format") == "nml"
    eligible = [nml or (m is not None and m.interp_mode in {"cspline", "trace"})
                for m in fiber.control_point_segments[:-1]]
    runs = []
    start = None
    for i, accepted in enumerate(eligible + [False]):
        if accepted and start is None:
            start = i
        if not accepted and start is not None:
            points = fiber.line_points_xyz[indices[start]:indices[i] + 1].copy()
            if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0):
                metadata = {**fiber.metadata, "training_control_span": (start, i)}
                if "baked_annotations" in metadata:
                    from .annotations import baked_normals
                    baked_normals(fiber)  # Validate before slicing, never truncate silently.
                    metadata["baked_annotations"] = {
                        **metadata["baked_annotations"],
                        "normals_xyz": metadata["baked_annotations"]["normals_xyz"][indices[start]:indices[i]+1],
                    }
                    for key in ('valid', 'widths_base'):
                        if key in metadata['baked_annotations']:
                            metadata['baked_annotations'][key] = metadata['baked_annotations'][key][indices[start]:indices[i]+1]
                runs.append(replace(
                    fiber,
                    line_points_xyz=points,
                    control_points_xyz=fiber.control_points_xyz[start:i + 1].copy(),
                    control_point_segments=tuple(fiber.control_point_segments[start:i]) + (None,),
                    metadata=metadata,
                ))
            start = None
    return runs
