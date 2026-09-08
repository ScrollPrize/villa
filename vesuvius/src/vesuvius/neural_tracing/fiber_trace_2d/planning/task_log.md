# Task log: native lower-resolution Fiber3D output

- Started from the uncommitted two-GPU profiling/normalization work on branch
  `fiber-training`; those changes are preserved.
- User selected GroupNorm as the retained fastest configuration for current
  architecture experiments.
- Initial target is native output stride 4, matching Fiber inference's current
  default factor-four output reduction relative to its selected input.
- Existing stride-1 architecture/checkpoints remain the compatibility default.
- GPU experiments remain restricted to physical GPUs 0 and 1 and use
  `/home/hendrik/.venv_las` with artifacts below
  `/ephemeral/hendrik/fiber3d_train2`.
- Independent review required an exact shared lattice definition, source-cell
  occupancy targets, conservative validity/margins, total/native/residual
  inference scale separation, all-consumer coverage, and repeated paired
  benchmarks. The plan was corrected before continuing implementation.
- Implemented a shared isotropic output-lattice helper, decoder truncation,
  direct coarse target materialization, and explicit output/target shape checks.
- Whole-volume and native Trace2CP inference now separate total requested,
  native model, and residual output scaling, with early shape and divisibility
  errors. Legacy adapters/checkpoints default to native stride 1.
- Added source-only replay hashes independent of materialized targets. Fresh
  stride-1 and stride-4 GroupNorm replay runs recorded identical source hash
  `9d4d0bb3c7d8186f49a80d9eae89ae179028b1f427994de0181c067b065f3139`.
- Three replay trials averaged 55.301 global samples/s at stride 1 and 95.714
  at stride 4 (+73.1%). Peak allocated memory fell from 27.83 to 14.91 GiB/rank.
  Real-sample throughput rose from 50.231 to 59.154 global samples/s (+17.8%);
  the stride-4 real run exposed skewed loader stalls, so input delivery is now
  the larger end-to-end limiter.
- Added the fixed `vesuvius/scripts/run_fiber3d_profile_benchmark` caller and
  line-oriented args file so future scoped benchmark variations reuse one
  approved command without shell evaluation.
- Validation: 269 Fiber3D/profile tests passed. The two directly affected
  shared Lasagna runner tests passed. The full Lasagna test module still has
  unrelated existing wrapper-mock failures (for example `_auto_download` is no
  longer exported from `preprocess_cos_omezarr`) and was not changed here.
- Final `git diff --check` and benchmark-caller shell syntax checks passed.
