# Short reproducible failure benchmark

Follow-up: run Classic on the same200pieces, then fiberlet evaluation using best
u22000 and128basevoxel construction chunks, existing inference cache and protocol.
No visualization requested. Report comparison including graph credited coverage.

Accepted benchmark now complete. Evaluate all200 exported directed pieces using
the frozen best EMA999u22000 snapshot without embedding, reusing/extending its
cache, and report overall/per-scroll metrics and runtime. Do not change the set.

Current user instruction: use a persistent work queue, retaining inference
resources between batches and using available CPU/GPU capacity. Preserve
checkpoint/cache evidence. Validate then run the extraction. Ten minutes is not
a hard timeout; avoid idle capacity and unsupported completion estimates.

Additional requirement: before failure selection, exclude entire original fibers
with nonlocal self-proximity inside the acceptance ellipsoid. Use accelerated
spatial indexing, not quadratic all-pairs checks. Record excluded locations.

Stop fiberlet run, preserve caches. Build200 directed beam-tracer failure cases: up to99 best-u22000, up to50 distinct other-model failures where Classic succeeds, Classic fills remainder (initially51). Deduplicate locations. Search starts from both sides, pick shortest reproducing direction, verify failure with start100voxels earlier. End at most100 beyond failure; match within±100 original GT arc voxels. Preserve CP/non-Lasagna boundaries and source/model/endpoint provenance. Use cached inference, not fiberlets.
