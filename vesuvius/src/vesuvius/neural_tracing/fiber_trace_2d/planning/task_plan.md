# Native 3D Trace2CP Vectorized Beam Lookahead Plan

## Current State

- `NativeTraceFieldCache` already keeps tensor operations on `cache.device`.
  With normal CLI usage this is the selected torch device, so candidate scoring
  and `grid_sample` point lookup can run on GPU.
- `sample_point_choices_torch(points)` accepts a batch of points and returns
  `directions[N,K,3]`, `presence[N,K]`, and `valid[N,K]`.
- `_score_candidate_loss_tensors(...)` scores all candidates for one beam
  state in torch, but its API takes one current direction and one previous
  direction.
- `_trace_native_3d_one_way_beam(...)` still loops in Python over:
  - lookahead depth,
  - beam/frontier states,
  - valid candidate children,
  - parent-node object creation.
- `_trace_candidate_directions(...)` returns NumPy candidate directions for one
  axis at a time. This prevents fully GPU-side candidate generation.

## Target Behavior

- Keep the current semantics:
  - `beam_width=1` remains greedy compatibility mode.
  - `beam_lookahead_steps=3` defaults to expanding short futures before
    pruning.
  - Multi-direction branch choices remain coupled direction/presence options.
  - Ambiguous axes remain sign-aligned before dot products.
  - Smoothness, target-plane crossing, and step guards keep current meaning.
- Make beam mode run the lookahead expansion mostly as GPU tensor batches.

## Implementation Plan

### 1. GPU Candidate Basis

- Add torch helpers:
  - `_orthonormal_basis_torch(axes_n3) -> (b0_n3, b1_n3)`
  - `_cone_offset_table_torch(cfg, device) -> offsets_m2`
  - `_trace_candidate_directions_torch(axes_n3, cfg) -> dirs_n_m_3`
- Cache the offset table per `(cone_angle_degrees, cone_angle_step_degrees,
  cone_grid_size, device)` if useful, but keep correctness independent of the
  cache.
- For the legacy grid fallback (`cone_angle_step_degrees <= 0`), either build
  the old square-to-disk offsets once in torch or keep the existing NumPy path
  only for `beam_width=1`/explicit legacy. Prefer torch fallback so beam mode
  stays vectorized.

### 2. Batched Current-Point Lookup

- Replace per-state `_sample_trace_point_aligned(...)` calls in beam mode with
  a batched helper:
  `_sample_trace_points_aligned_torch(cache, points_n3, references_n3)`.
- It should call `cache.sample_point_choices_torch(points_n3)` once, align all
  branches to the corresponding reference, choose branch by
  `dot(branch_dir, reference) * branch_presence`, and return
  `current_dirs_n3`, `presence_n`, `valid_n`.
- For the first depth at the start CP, use the supplied CP-local tangent as the
  current direction exactly as today.

### 3. Batched Candidate Scoring

- Generalize `_score_candidate_loss_tensors` to accept batched states:
  `current_dirs_n3`, `previous_dirs_n3`, `candidate_dirs_n_m_3`,
  `next_points_n_m_3`.
- Flatten candidate points to `[N*M,3]`, call
  `cache.sample_point_choices_torch(...)` once, then reshape returned branch
  tensors to `[N,M,K,...]`.
- Compute:
  - `current_dot[N,M]`
  - `next_dot[N,M,K]`
  - `presence[N,M,K]`
  - `smoothness[N,M]`
  - total loss `[N,M,K]`
- Reduce over branch only, not over candidate/state:
  `candidate_loss[N,M]`, `candidate_branch[N,M]`.
- Keep a wrapper for the current one-state scorer so existing tests and greedy
  path do not need to change.

### 4. Tensor Beam Frontier

- In beam mode, store frontier state as tensors on GPU:
  - `points[N,3]`
  - `previous_dirs[N,3]`
  - `cumulative_loss[N]`
  - `parent_index[N]`, `parent_generation[N]`, `step_diag` arrays for path
    reconstruction
  - `depth[N]`
- For each lookahead depth:
  - Generate `candidate_dirs[N,M,3]`.
  - Generate `next_points[N,M,3]`.
  - Score all candidates in one batched call.
  - Flatten `[N,M] -> [N*M]`, add cumulative loss, and append children to a
    compact tensor frontier for the next internal depth.
  - Keep parent references as integer tensors, not Python node objects.
- Do not prune between internal lookahead depths.

### 5. Batched Target-Plane Handling

- Compute plane crossing in torch for all `[N,M]` candidate edges:
  - `d0 = dot(current - target, plane_normal)`
  - `d1 = dot(next - target, plane_normal)`
  - crossing mask where `d0 == 0` or `d0*d1 <= 0`
  - interpolate crossing points in torch.
- If any reached candidates appear at a lookahead depth, choose the reached
  child with lowest cumulative loss and reconstruct that path.
- If none reach, continue expansion until lookahead depth or step guard.

### 6. Batched Pruning

- First implement exact top-k pruning by cumulative loss:
  `torch.topk`/`torch.argsort` over the flattened frontier, keeping
  `beam_width`.
- Then re-add near-duplicate pruning:
  - For small `beam_width`, a tiny CPU loop after top-k oversampling is
    acceptable but should only run on at most `beam_width * prune_factor`
    candidates, not all expanded candidates.
  - Better GPU option: oversample `beam_width * 4`, compute pairwise distances
    among oversampled points, and greedily mask duplicates. Because the
    oversampled set is small, either CPU or GPU is fine.
- Document if near-duplicate pruning remains a small post-processing loop; this
  does not dominate candidate evaluation.

### 7. Path Reconstruction

- Keep per-generation arrays for selected frontiers so only surviving states
  are stored after each lookahead/prune stage.
- For reached candidates, reconstruct the chosen path by following integer
  parent references through saved generations.
- Convert only the final selected path and diagnostics back to NumPy/Python
  `NativeTraceStep` objects.

### 8. Testing / Measurement

- Preserve existing unit tests.
- Add a regression that checks vectorized beam lookahead matches the previous
  Python beam result on a fake cache.
- Add a fake multi-branch case verifying branch/presence coupling under the
  batched scorer.
- Add a small benchmark-style test or diagnostic command for a synthetic cache:
  compare old Python beam path vs vectorized beam path for
  `beam_width=8`, `lookahead=3`, 81 candidates.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Spec Update

- Update specs to require native 3D beam-mode candidate generation, scoring,
  and lookahead expansion to be batched torch operations on `cache.device`.
- State that final path reconstruction may use Python because it only handles
  the selected path.
- State that any duplicate-pruning CPU fallback must operate only on a small
  oversampled top-k set, not on the full candidate frontier.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section to explain the
  tensor frontier and batched point lookup/scoring.

## Changelog

- Add a 2026-07-17 entry when implemented: native 3D Trace2CP beam lookahead
  is vectorized across active beam/frontier states and candidate directions.

## Non-Goals

- Do not change the scoring formula.
- Do not change model outputs, training losses, or fusion scoring.
- Do not introduce approximate pruning that can silently drop candidates before
  the configured lookahead expansion.
