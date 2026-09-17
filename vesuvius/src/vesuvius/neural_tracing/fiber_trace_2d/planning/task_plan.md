# Plan

1. Extend refinement state configuration with a frozen-update mode and an explicit
   state-source checkpoint. Restore and validate the existing fiber archive without
   loading model/optimizer/EMA progress. Use its current training fibers; evaluate
   against immutable original test GT because paired sample identities do not map
   to the archive's unpaired test patches. Make the imported store read-only and do
   not construct observations, create entries, solve, commit, or emit refinement
   metrics. Checkpoints retain immutable source path/SHA provenance rather than
   rewriting the 520MiB archive; later model resumes restore the same source.
2. Extend the existing paired local-contrastive embedding objective with an optional
   channel split. Preserve legacy behavior when omitted. For an 8/8 split, compute
   short-range positives and presence-weighted negatives only within each patch on
   independently normalized channels0:8; compute strictly cross-patch long-range
   positives and symmetric paired-region negatives on independently normalized
   channels8:16. Carry separate numerator/count statistics through accumulation,
   distributed reduction and logging, and use `0.5*(short_mean+long_mean)`.
   Split the existing configured negative budget evenly between banks so total
   sampled pairs and dot-product work remain comparable to the legacy objective.
3. Add CPU regression coverage for config validation, channel isolation, positive
   routing, negative routing/count normalization, gradients to both banks, unequal
   rank/microbatch accumulation equivalence, frozen train-only restore, original-GT
   tests, no proposal generation, fresh model state, and unchanged legacy behavior.
4. Add a fresh full-resolution five-scroll JSON paired-embedding config using the
   update17800 refinement archive, LR0.01, global batch128,16 total embedding
   channels split8/8, and no model resume.
5. Store bank semantics in config/checkpoints. Make current inference/tracing reject
   split banks unless a bank is selected explicitly; do not concatenate them into
   one cosine space. Update refinement/embedding documentation and planning records.
6. Run the
   focused embedding, paired, refinement, gradient-accumulation and config suites
   via the approved cooperative test launcher; run managed CUDA coverage if the
   suite selects it. Measure legacy versus split embedding-statistics runtime with
   fixed input/budget and report the distribution before launch.
7. Commit the implementation/config, launch with the approved managed production
   runner, and verify from logs that update0 uses fresh weights, frozen fibers are
   restored from update17800, no refinement updates occur, and both embedding banks
   report finite losses/gradients.
8. Remove the observed 128-worker AWS credential-helper stampede without restarting
   the active run: make native S3 volume, manifest, and Lasagna object access try
   anonymous HTTP first and resolve ambient AWS credentials lazily only after an
   authentication denial. Preserve private-bucket fallback and add deterministic
   native regression coverage.

## Spec update

No `planning/spec.md` exists in this checkout. Document the new opt-in contracts in
the maintained embedding/refinement docs: frozen geometry imports only fiber state,
and split-range channels retain the legacy behavior when not configured. Document
that downstream consumers must explicitly select short or long channels.

## Docs updates

Update `docs/training_refinement.md` and `docs/paired_embedding_training.md` with
the frozen-state and split-range semantics, config fields, snapshot provenance and
new production config/run command, bank metadata and downstream compatibility.

## Changelog

Add a concise entry for frozen refined-fiber training and independently supervised
short-/long-range embedding banks.
