# Task: independent fiber normal orientation

Sample Lasagna sheet normals at regular arc intervals and exact constraint base
points; sweep in original AFV point order, flipping to minimize change from the
previous valid oriented normal. First valid normal fixes an arbitrary per-fiber
gauge. Store only constraint endpoint signed normals, plus fiber normal_sign=+1;
no cross-fiber alignment or signed winding inference yet.

Implementation: native second pass over existing database, reusing AFV loading,
polyline geometry and batch normal sampler. Default native normal spacing (16 base voxels here),
configurable positive finite. Merge regular and endpoint samples in arc order;
ties preserve original sign when dot=0. Invalid samples remain null, sweep keeps
last valid sample and counts invalid gaps. No guarantee across missing data or
true rotations exceeding90degrees between samples; report these limitations.

Preserve source database, clone into a refused-if-existing .partial output,
validate AFV UUID/source IDs/lengths and sample provenance, add schema2 fiber sign,
replace only endpoint normal fields; preserve all measurement JSON unchanged.
Batch sampling across fibers with128workers; batches of256 fibers and progress
output. Sample count/memory varies with fiber length and configured spacing.

Tests: unit gauge/flip/tie/invalid cases; native suite; small existing pilot
database pass then full raw database pass. Validate unchanged events, magnitudes,
fiber sign defaults, exact endpoint axis equivalence, integrity and provenance.

Spec update: schema2 signed endpoint normals and independent mutable fiber sign
(effective normal = normal_sign * baked normal). No stored per-fiber sample array.
Docs updates: database schema, commands, results, task log/status/changelog.
Independent plan review required by fiber_trace_2d/AGENTS.md.
