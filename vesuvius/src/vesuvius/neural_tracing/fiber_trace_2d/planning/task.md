# Default Training Without Embedding Loss

Standard `fiber_trace_2d` training should use direction and sheet/fiber
presence supervision only. The contrastive embedding head and loss must remain
supported, but they should be explicit opt-in rather than part of the default
example training configuration.

- Disable contrastive embedding in the standard example config.
- Make disabled contrastive training instantiate no embedding head, even if a
  stale embedding-channel value is present in a config.
- Keep explicit contrastive opt-in support unchanged: when
  `training.contrastive_enabled` is true, a positive
  `training.contrastive_embedding_channels` is still required and the existing
  same-fiber grouped sampling/loss/logging path remains active.
- Update specs/docs to state the default training objective is direction plus
  optional presence, with contrastive embedding as experimental opt-in.
