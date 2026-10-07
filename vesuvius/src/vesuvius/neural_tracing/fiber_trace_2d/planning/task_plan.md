# Plan

1. Preserve prior neural work; branch already fiber-exp5. Controls: tuned EMA999u12100 and best-loss EMA99u19100.
2. Add opt-in normal-head output: six sigmoid projection channels accumulated then shared hemisphere nx/ny encoding. Existing demand queue/cache/publication; include in cache identity/readiness. No scheduling changes.
3. Native tracing uses predicted normals, including reverse traces; reference acceptance still uses Lasagna. Reuse sampler/decoder; disabled behavior unchanged.
4. Tests for head layout/encoding, identity and separate trace/reference samplers. Approved build/test runner. Evaluate frozen checkpoints unchanged policy/precision on GPU0–5; report min0/min512/per-scroll and runtime caveats.

## Spec update
Optional normal product and tracing-only normal binding; unchanged GT ellipses, no default switch.

## Docs updates
Comparison MD/JSON, provenance/configs/commands, validation, task log/status/changelog.

## Validation
Independent plan review. Reject missing heads. Verify scales/encoding, readiness, cohort and policy. No concurrency retuning.
