# Full-gradient accumulated reference-initialized solve

User requests10 global accumulated updates, freshly initialized from reference
windings. Reuse151CCW reference initialization, fixed rejection cost, no warmup,
all global parameters trainable; no continuation of the previous saved solve.

## Implementation and spec update
Add opt-in update_mode=global_accumulated (default minibatch unchanged), requiring
full_pass traversal. Each pass evaluates every event and node once in bounded
minibatches at identical parameter values, accumulates loss/updates gradients
normalized by number of partitions, and applies SparseAdam/Adam exactly once.
Existing balanced event/node weights give true full means. Width priors and all
continuation links repeated per minibatch are divided by partitions too, hence
counted once globally. No LR retuning: .03 embedding and existing2 width default.

Use a fixed dense embedding-gradient buffer plus width-gradient buffer; ingest
each coalesced sparse minibatch gradient using unique row index copies. This
avoids growing sparse duplicate lists across millions of repeated rows. Present
the accumulated buffer as coalesced all-row sparse gradient to existing SparseAdam.
All nodes participate in the full node regularizer; optimizer state unchanged.
Recompute each minibatch graph and discard after backward; no retained graphs.

Global mode forbids changing entropy schedule within a pass. Config disables it.
Report optimizer_updates explicitly and objective as pre-update full objective;
end-of-update hard metrics audit all constraints (no100k sample for this mode).

## Tests
Compare accumulated objective, sparse/dense gradients and optimizer results with
one monolithic toy batch including repeated node IDs, uneven partitions, node
regularization and repeated global terms. Test buffer reset over2updates and
validation. Existing reference-init/fullpass/broken tests must still pass.

## Execution/docs
Independent review before implementation. New approved-runner config/output,
shared manager24GiB reservation;10globalupdates, no subsequent run. Persist all
pass metrics and current table columns (objective,sign,retained,L1,within2,both
q10/q90 ranges). Record caveat pre-update objective versus post-update metrics.
