# Combined Direction And Contrastive Embedding Tracer Plan

## Scope

- Add an optional combined scorer for Trace2CP/line tracing visualization.
- Keep the existing direction-only and median-TTA direction tracer paths intact.
- Do not change training, model architecture, loader sampling, or metric definition.
- Require checkpoints with embedding channels when the combined tracer mode is enabled; otherwise fail clearly.

## Current Code Shape

- `runner.py` traces by sampling decoded direction fields through `_bilinear_direction_sample`, orienting the ambiguous direction against the previous step, and stepping by `step_px`.
- `model.py` returns direction channels first and optional raw embedding channels after them. Direction consumers already slice via `direction_output`; embedding consumers should slice via `embedding_output`.
- `embedding.py` already has normalized CP similarity utilities for visualization, but no tracer-step embedding scorer.
- Trace2CP visualization already compares selected and reference bidirectional traces; this can be extended without changing the metric implementation.

## Implementation Plan

1. Add model-output prediction helpers for combined tracing.
   - Extend the runner inference path so one model pass can return both:
     - decoded `direction_xy`;
     - raw or normalized `embedding_chw`.
   - Keep the existing direction-only helper for callers that do not need embeddings, or refactor it into a wrapper around the richer helper.
   - Normalize embeddings only at sampling/scoring boundaries to avoid storing repeatedly normalized copies unless profiling later shows it matters.

2. Add embedding sampling helpers.
   - Implement bilinear embedding sampling at floating `xy`, mirroring direction sampling validity checks.
   - Return normalized embedding vectors.
   - Add cosine-loss helper: `1 - dot(normalized_a, normalized_b)`.
   - Invalid points, invalid mask samples, or zero-norm embeddings make that candidate invalid.

3. Add fiber CP embedding-bank construction for visualization.
   - For Trace2CP single-pair/fiber-json visualization, precompute normalized CP embeddings from the same fiber once per command.
   - Use the existing loader/model path to evaluate CP-local patches and extract the rounded/transformed CP embedding.
   - For the current inspection target, correctness and shared semantics matter more than speed; cache the bank in-memory for the command.
   - If some CP-local sample is invalid, skip that CP embedding and report the skipped count in the summary/debug output.
   - If no fiber-bank embeddings are available, either disable only the fiber-bank term with a clear summary note or fail when its weight is non-zero. Prefer failing for now so score interpretation is not silent.

4. Add candidate fan generation.
   - New defaults:
     - `candidate_max_degrees = 25.0`;
     - `candidate_step_degrees = 1.0`, producing `[-25, -24, ..., 25]` = 51 candidates.
   - Permit `candidate_step_degrees = 2.0` for every second degree.
   - Generate candidate unit vectors by rotating the oriented current direction; candidate point is `current + candidate_unit * step_px`.
   - Keep deterministic ordering and tie-break by lowest absolute angle first, then stable candidate order.

5. Add greedy combined-step scorer.
   - At each step:
     - sample/orient the current direction as before;
     - build the candidate fan around that oriented direction;
     - sample candidate embeddings;
     - compute:
       - `direction_loss = 1 - dot(candidate_unit, oriented_direction)`;
       - `last_loss = 1 - cos(candidate_embedding, previous_accepted_embedding)`;
       - `enclosing_loss = mean(loss to start CP embedding, loss to target CP embedding)`;
       - `fiber_loss = mean(loss to all normalized CP embeddings in the same fiber bank)`;
       - `total = w_dir*direction + w_last*last + w_enclosing*enclosing + w_fiber*fiber`.
   - Start `previous_accepted_embedding` from the start CP embedding.
   - After accepting a candidate, update `previous_accepted_embedding` to the accepted candidate embedding.
   - Terminate with a clear reason if no candidate remains valid.

6. Add bidirectional Trace2CP integration.
   - Add a combined variant of `_trace_direction_line_to_target`.
   - Preserve forward/reverse Trace2CP construction and existing metric calculation.
   - Use the start/target embeddings appropriate to each direction:
     - tracing start -> target uses start as previous/enclosing start;
     - tracing target -> start swaps the enclosing CP order but uses the same two embeddings.
   - Ensure the reverse trace still follows the existing ambiguous-direction orientation rule.

7. Add CLI knobs for inspection.
   - Add an explicit flag such as `--trace2cp-combined`.
   - Add optional weights:
     - `--trace2cp-combined-direction-weight`;
     - `--trace2cp-combined-last-weight`;
     - `--trace2cp-combined-enclosing-weight`;
     - `--trace2cp-combined-fiber-weight`.
   - Default all four weights to `1.0`.
   - Add:
     - `--trace2cp-candidate-max-deg` default `25.0`;
     - `--trace2cp-candidate-step-deg` default `1.0`.

8. Update visualization and summaries.
   - Show direction-only reference and combined direction+embedding traces side-by-side in Trace2CP vis.
   - Include labels with Trace2CP metric/error and average per-step score components:
     - direction;
     - last embedding;
     - enclosing embedding;
     - fiber-bank embedding;
     - total.
   - Print a clear command-line summary comparing direction-only vs combined metric error.
   - Include candidate settings, weights, embedding-bank size, and skipped CP-bank count in summary/debug text.

9. Keep TTA behavior conservative.
   - The first implementation should not change median-TTA semantics.
   - If `--med-tta` and `--trace2cp-combined` are both provided, either:
     - use median-TTA only for the direction reference and use reference-patch embeddings for candidate scoring; or
     - reject the combination with a clear message.
   - Prefer the first if it can be done without extra geometric embedding warping; otherwise reject for the initial version.

## Spec Update

- Add an optional combined Trace2CP tracer mode that greedily chooses each step from an angular candidate fan around the oriented direction prediction.
- Specify default candidate fan as -25..+25 degrees at 1 degree spacing, configurable to coarser spacing.
- Specify combined score components and default even weights:
  direction agreement, previous-step embedding agreement, enclosing-CP embedding agreement, and same-fiber CP-bank embedding agreement.
- Specify that the mode requires embedding channels and must not silently run when no embedding head is present.
- Specify that this is visualization/inspection first and does not replace the direction-only tracer or training metric unless explicitly enabled.

## Docs Updates

- Update `docs/code_structure.md` to mention the optional combined tracer path in `runner.py`.
- Update runner/Trace2CP CLI documentation if a dedicated docs file already lists trace flags.
- Update `planning/changelog.md` after implementation with one concise entry.

## Tests And Validation

- Unit-test candidate fan generation:
  - default max/step yields 51 candidates and includes 0 degrees;
  - 2 degree step yields every-second-degree candidates and stable ordering.
- Unit-test combined scorer on a small synthetic field:
  - with direction-only weight, it chooses the center direction candidate;
  - with embedding weight dominating, it chooses the candidate with best embedding cosine;
  - invalid candidates are skipped.
- Unit-test missing embedding channels:
  - `--trace2cp-combined` fails clearly when checkpoint/model output has no embedding channels.
- Run focused test suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Smoke-run the existing Trace2CP visualization command on one sample/checkpoint if the local checkpoint/data are available, verifying that both direction-only and combined columns render and summary prints both metric errors.

## Review Notes

- This plan keeps all geometric sampling/visualization semantics unchanged.
- It does not introduce image-space geometric warps.
- It uses direction and embedding slices explicitly, matching the model-output spec.
- It intentionally leaves training untouched until the visualization shows whether the combined scorer improves Trace2CP error.
