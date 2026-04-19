# Audit of `autoreg_mesh` Codebase

**Date**: 2026-04-19
**Config reviewed**: `config_autoreg_mesh_0139_2um_r1_sigma066_axis_factorized_sd.json`
**Training state**: 100k steps, ~70% coarse 3D accuracy

---

## 1. BUG: Double Attention Scaling (model.py:115-122, 164-171)

Both `RotarySelfAttention` and `CrossAttention` pre-multiply queries by `1/sqrt(d)` **and** then call `F.scaled_dot_product_attention`, which internally applies another `1/sqrt(d)`:

```python
# RotarySelfAttention.forward (line 115-122)
q = q * self.scale                # self.scale = head_dim ** -0.5
out = F.scaled_dot_product_attention(q, k, v, ...)  # default scale = 1/sqrt(head_dim) again
```

Effective attention logits: `Q @ K^T / d` instead of `Q @ K^T / sqrt(d)`.

With head_dim=48 (768/16), the attention distributions are ~7x softer than intended. The model has compensated for this over 100k steps of training, so **fixing it would break pretrained weights** and require retraining. But it's important to know:
- Standard hyperparameter intuitions about attention temperature don't apply.
- If you ever initialize from a model trained with standard scaling, it won't work.

**Fix options** (pick one, requires retraining):
- Remove `q = q * self.scale` from both attention classes, or
- Pass `scale=1.0` to `F.scaled_dot_product_attention` so SDPA doesn't re-scale.

---

## 2. CRITICAL INFERENCE: O(n^2) Autoregressive Loop (infer.py:156-231)

The inference loop re-runs the **full transformer forward pass** over the entire prompt+history at every step:

- Step 1: forward on 1 target token
- Step 2: forward on 2 target tokens
- Step N: forward on N target tokens

For a 16x16 coarse grid, N = 256 target tokens. Total work: ~32K transformer-layer token computations instead of 256 with KV-caching. This is the single biggest inference bottleneck.

**Recommended fix**: Implement KV-caching:
1. Cache the self-attention K,V tensors from prompt tokens (computed once).
2. At each autoregressive step, only pass the new token through the transformer.
3. Append the new token's K,V to the cache.
4. Cross-attention K,V from memory tokens can be pre-computed once.

This turns O(N^2 * D) inference into O(N * D) for D decoder layers.

---

## 3. INFERENCE: No Temperature / Top-k / Nucleus Sampling (infer.py:64-68)

`_sample_from_logits` only supports greedy argmax or raw multinomial:

```python
def _sample_from_logits(logits, *, greedy):
    if greedy:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

For autoregressive generation, standard improvements include:
- **Temperature scaling**: `logits / temperature` before softmax
- **Top-k filtering**: zero-out all but top-k logits
- **Nucleus (top-p) sampling**: keep smallest set of tokens whose cumulative probability exceeds p

These are especially important for the non-greedy mode, where raw multinomial sampling can produce very noisy outputs.

---

## 4. INFERENCE: Axis-Factorized Constraint Mask is Very Weak (model.py:730-738)

The joint coarse continuation mask defines a tube + ball region in 3D. For axis-factorized mode, this mask is projected onto individual axes via `any()`:

```python
z_valid = mask_5d.any(dim=(3, 4))  # any valid (y,x) for this z?
y_valid = mask_5d.any(dim=(2, 4))  # any valid (z,x) for this y?
x_valid = mask_5d.any(dim=(2, 3))  # any valid (z,y) for this x?
```

The Cartesian product of per-axis allowed ranges is much larger than the actual tube. A prediction far outside the tube can be accepted if each axis individually falls within the tube's projection. This is a fundamental limitation of axis-factorized prediction with hard constraints.

**Potential improvement**: Predict axes sequentially (z -> y|z -> x|z,y) within each step, conditioning each axis mask on the previously-predicted axes. This would maintain the tight constraint while staying factorized. However, it requires architectural changes to add per-axis sequential conditioning within a single decoder step.

---

## 5. PERFORMANCE: Python Loops in Coarse Continuation Mask (model.py:485-549)

`_build_raw_coarse_continuation_mask` iterates over `batch_size x target_len` in Python with per-token geometry computations. For batch_size=8 and target_len=200, that's 1,600 Python iterations doing tensor ops. This is a training bottleneck on GPU because it prevents parallelism.

**Fix**: Vectorize the entire computation. The anchor/direction logic can be precomputed per-within_idx (since it only depends on the strip position and frontier geometry), then the tube + ball mask can be computed as a single batched tensor operation.

---

## 6. PERFORMANCE: Python Loops in All Geometry Losses (losses.py)

All geometry losses (`_iter_geometry_examples`, `_paired_seam_bands`, `_triangle_det_ratio_from_sequence`, `_geometry_metric_loss_from_sequence`, `_geometry_sd_loss_from_sequence`) use Python `for` loops over batch examples. Each iteration creates separate tensors and computes losses independently.

Not a correctness bug, but with batch_size=8, these losses do 8 separate GPU kernel launches per loss term. Vectorizing across the batch dimension would give meaningful speedup.

---

## 7. DESIGN: No Mixed Precision Training (config.py:401)

The config explicitly disallows mixed precision:
```python
if str(cfg["mixed_precision"]).lower() != "no":
    raise ValueError("autoreg_mesh currently does not implement mixed precision; set mixed_precision='no'")
```

With a 768d, 16-layer, 16-head transformer + DINOv2 backbone, training in fp32 uses roughly 2x the memory and is 1.5-2x slower than bf16 on modern GPUs. The backbone is already frozen, and the decoder is standard transformer architecture that works well with bf16/fp16.

**Recommendation**: Add `torch.autocast('cuda', dtype=torch.bfloat16)` around the forward pass and use `GradScaler` for the backward. The geometry losses that need numerical precision can be computed in fp32 selectively.

---

## 8. INFERENCE: Redundant Conditioning Grid Transfer (infer.py:168-181)

Inside the step loop, `build_pseudo_inference_batch` creates a new pseudo_batch dict every step. While the conditioning grid tensor itself isn't re-copied, the pseudo_batch construction allocates new tensors for target_coarse_ids, target_offset_bins, target_xyz at every step:

```python
target_coarse_ids = torch.full((1, current_len), IGNORE_INDEX, ...)
target_offset_bins = torch.full((1, current_len, 3), IGNORE_INDEX, ...)
target_xyz = torch.zeros((1, current_len, 3), ...)
```

Then copies history into them. Pre-allocating to max size and growing a view would avoid repeated allocation.

---

## 9. SUBTLE: `_soft_decode_local_xyz` Refinement Mismatch

During training, `position_refine_head` is supervised against:
```python
target_residual = batch["target_xyz"] - batch["target_bin_center_xyz"]  # offset from GT bin center
```

But `pred_xyz_soft` is computed as:
```python
expected_coarse_start + expected_offset + pred_refine_residual  # offset added to soft-decoded position
```

The soft-decoded position (weighted average over all bins) differs from the hard bin center (argmax bin). The model learns to jointly optimize the soft logits + residual to minimize `xyz_soft_loss`, so this works in practice, but it means the `position_refine_head` isn't purely learning a "bin-center residual" -- it's learning a more complex correction that depends on the distribution shape. This is fine, just worth understanding when analyzing refinement quality.

---

## 10. INFERENCE IMPROVEMENT: Beam Search

Currently only greedy/sampling is supported. For mesh completion, a beam search (width 2-4) over coarse cell predictions could significantly improve quality at modest cost. The key insight is that coarse cell errors are hard to recover from (wrong patch = wrong region of space), so exploring multiple candidates early is high-value.

---

## 11. INFERENCE IMPROVEMENT: Early Geometric Validation

The inference loop only checks the stop probability. It could also check for:
- **Out-of-bounds vertices**: Stop or re-sample if predicted xyz falls outside the volume.
- **Triangle flips**: Detect inverted triangles between the predicted vertex and its neighbors, and either stop or re-sample.
- **Excessive distance**: Flag vertices whose distance from the frontier/previous strip exceeds a threshold.

These checks are cheap to compute and would catch degenerate predictions during generation.

---

## 12. MINOR: `_build_example` Performs Redundant Full Serialization (dataset.py:714-755)

`_build_example` calls `_serialize_candidate_plan` which does a full `serialize_split_conditioning_example`, then throws away the serialized result and re-serializes after augmentation. Only the `conditioning` dict (containing split grids and corners) from the first call is used. The first full serialization could be replaced with a lighter-weight split + corner computation.

---

## Summary: Priority Ranking

| Priority | Issue | Impact |
|----------|-------|--------|
| **P0** | O(n^2) inference (no KV-cache) | Inference speed, deployment feasibility |
| **P1** | Double attention scaling | Correctness (but trained around) |
| **P1** | No mixed precision | 2x memory waste during training |
| **P2** | No temp/top-k/nucleus sampling | Inference quality for non-greedy mode |
| **P2** | Weak axis-factorized constraint mask | Inference accuracy |
| **P2** | Python loops in continuation mask | Training speed |
| **P3** | Python loops in geometry losses | Training speed |
| **P3** | Beam search / geometric validation | Inference quality |
| **P3** | Redundant serialization | Data loading speed |

The 70% coarse accuracy at 100k steps is good given these issues. Fixing the KV-cache alone would make inference practical for production. Adding temperature-controlled sampling and geometric validation during inference would likely push accuracy higher without retraining.
