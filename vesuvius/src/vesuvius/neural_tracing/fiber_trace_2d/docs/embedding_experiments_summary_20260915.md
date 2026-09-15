# Fiber-identity embeddings for 3D papyrus tracing: experimental summary

**Status: 15 September 2026.** This summarizes the recorded 3D embedding work: training objectives, representation diagnostics, tracer memory policies, full-dataset evaluations and differentiable policy fitting. It is an evidence review, not a new benchmark. Detailed reports are linked below; machine-local artifact locations are indexed at the end.

## Executive summary

Embeddings show a usable continuity signal, but we have **not demonstrated a robust, practically compelling improvement in fiber tracing**.

- Early pairwise training produced extremely consistent GT embeddings but weak identity discrimination. Contrastive training improved off-GT separation, while exposing stronger dependence on inference-tile context.
- Training on two distant patches from the same fiber improved long-range GT consistency on the diagnostic fiber. It did not reliably distinguish the correct continuation from the wrong path actually selected by the tracer.
- Extensive tuning on one difficult fiber reduced its historical failure count from **87 without embeddings to 69 with plain EMA, and 64 with a two-anchor policy**. These were repeated exactly, but are in-sample tuning results under an older evaluation protocol.
- Full-dataset tests did not reproduce that scale of benefit. Among the matched current evaluations, the strongest non-embedding model gives **113 restarts**. Supplying embeddings from a separate model gives **111**, and fitting their influence gives **110**: only three fewer restarts, or **2.65%**, on this cohort.
- Additional history/sampling/nonlinear-integration experiments did not beat 110. A learned update gate tied it while moving errors between scrolls. The held-out calibration subset does not show a clear benefit over embeddings-off.
- Embeddings add inference channels, cache traffic and tracing work; mixed models also require a second network. We have not established that the small quality gain justifies this cost. No new embedding policy was promoted automatically.

The evidence supports keeping embeddings as a research option, rather than treating the present implementation as a mature improvement. This is not proof that embeddings cannot work.

## 1. What is being predicted and how it is used?

The network predicts fiber presence, a local fiber direction and, when enabled, a **16-component embedding vector**. The intended embedding behavior is high cosine similarity along the same fiber and lower similarity at competing off-fiber locations.

The native beam tracer still follows its existing geometry, presence and direction costs. Embeddings contribute an additional weighted cosine-distance penalty against a causal reference: recent observations, an EMA, or retained representative vectors. Each beam hypothesis has its own memory. Candidates are scored against the parent's state before a selected child updates that state; memory resets on restart. GT is used for training and evaluation, not to update the deployed tracer's memory.

Two unrelated uses of “EMA” must be distinguished:

- **Model EMA99 / EMA999:** exponential averaging of neural-network weights during training (decay 0.99 / 0.999).
- **Tracer EMA:** averaging embedding observations along the traced path, with a half-life measured in **base-resolution voxels**. For elapsed distance `d` and half-life `H`, decay is `2^(-d/H)`, followed by vector normalization.

“Best snapshot” means selected by minimum recorded dense validation/test total loss, **not** minimum tracing restarts. Older evaluation labels saying “tuned” refer to **Anchor-LRU**, not plain tracer EMA. Those policies are named explicitly here.

## 2. Training experiments and representation diagnostics

All run names in this section are under `/ephemeral/hendrik/fiber3d_train2/runs/`. `uN` denotes optimizer update N, not N samples. The principal runs use 128 patches per optimizer update; paired training groups these into 64 pairs.

| ID | Training run | Purpose / evaluated snapshots |
|---|---|---|
| T0 | `s1a_nml_embedding_128_sd1_16x8_20260910_125217` | Earlier S1a embedding prototype. Run/config retained; no comparable completed embedding-quality result is asserted here. |
| T1 | `combined_5scroll_s1a_embedding_128_sd1_16x8_ema_20260913_094346` | Local pairwise metric loss; raw u5300 consistency diagnostic. |
| T2 | `combined_5scroll_s1a_contrastive_128_sd1_16x8_ema_20260913_121417` | Local contrastive competition; raw u2100 diagnostic, raw u5400 tracing/history experiments. |
| T3 | `combined_5scroll_s1a_paired_contrastive_128_sd1_16x8_ema_20260913_174112` | Long-range paired-patch contrastive training; raw u3700/u4300 and model EMA99 u6500/u9900/u32000. |
| T4 | `5scroll_json_eligible_paired_contrastive_128_sd1_16x8_ema_20260914_164404` | Eligible JSON paired-contrastive run; model EMA999 best u10800 and final/current u42000. |
| N | `5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436` | Non-embedding comparison run; final model EMA999 u11800 supplies the strongest current presence/direction baseline. |

The September 13 combined-data configs balanced five scrolls, with Paris4's JSON and S1a NML groups sharing its weight. They used fresh initialization, LR 0.01, 128³ input patches, a 16-channel embedding head, and model-weight EMAs. These were not matched-training-age ablations. Later eligible-JSON runs also changed the supervision selection; cross-run differences cannot all be attributed to the embedding loss.

### 2.1 Local pairwise loss: consistency without sufficient identity

T1 sampled same-fiber positives and presence-weighted off-GT negatives within a patch, excluding a normal-aware ellipse around GT. At 128³ it used 32,768 positive and 32,768 negative pairs. The objective attracted positives and penalized positive cosine for negatives.

On ten held-out fiber windows, overlapping-tile estimates of the same GT point had mean cosine **0.999994**. But **23.4–31.8%** of high-presence off-GT controls, depending on scroll, also exceeded cosine 0.9. Mean GT directions from different windows were almost identical. This is evidence of poor discrimination despite smoothness, not proof of a universal collapse.

Evidence: [local objective and implementation](local_fiber_embeddings.md), [T1 consistency diagnostic](embedding_consistency_20260913.md).

### 2.2 Local contrastive competition: better separation, weaker cross-tile consistency

T2 replaced the pairwise objective with local competition: each GT anchor has a positive and 32 presence-sampled negatives, using margin 0.2 and temperature 0.1. At 128³ this gives 2,048 anchors/positives and 65,536 negatives.

On the same ten diagnostic windows, **none of 7,899 off-GT controls exceeded cosine 0.9**, including 1,024 high-presence controls. However, same-point overlapping-tile cosine fell to **0.898922**, and same-fiber embeddings became more variable. Blended adjacent-point cosine was 0.957 in interiors versus 0.953 at contributor switches; only 15 switch pairs were sampled. No zero GT vectors occurred.

Thus contrastive separation improved descriptively, but local patch supervision did not establish a globally consistent identity field. The old/new checkpoints were u5300 versus u2100, and the high-presence subset changed with the model. Histogram overlap was **not** a classification error rate.

Evidence: [contrastive diagnostic and histograms](embedding_consistency_contrastive_20260913.md).

### 2.3 Paired patches: longer memory, but wrong-route ambiguity remains

T3 sampled two independently positioned patches on the same fiber, sharing photometric augmentation but applying independent spatial transforms. Positives could come from either patch; negatives were sampled across both. This explicitly encouraged identity consistency beyond one field of view.

On the hard PHerc1299 diagnostic fiber, mean GT cosine at 4,096 base voxels separation improved from **0.338** (single-patch raw u5400) to **0.892** (paired raw u3700). Nevertheless, the paired model's baseline tracing was worse: **140 → 133** failures with embeddings, versus the older model's **96 → 109**. Training age differed, so this is not a controlled objective-only comparison.

At paired-model failure locations, GT and taken-path embeddings had mean cosine **0.812**; **47.1%** exceeded 0.9. Among both-high-presence cases, **56.8%** exceeded 0.9. Long-range consistency alone therefore did not solve local route selection.

Evidence: [paired training mechanism](paired_embedding_training.md), [paired/single-patch comparison](embedding_range_paired_20260913.md).

### 2.4 Diagnose the actual wrong paths, not just random negatives

For the older raw u5400 model, native cached queries showed same-GT mean cosine falling from **0.944 at 8 voxels** to **0.338 at 4,096 voxels**. At 96 baseline failure points:

- An oracle GT history favored GT over the taken point in 71.6% of available cases, with mean cosine advantage +0.124.
- The actual traced-route history favored GT in only 54.2%, with mean advantage **−0.017**.
- 70/96 GT points had predicted presence below 0.5; among 22 both-high-presence cases, 59.1% still had GT/taken cosine above 0.9.

This suggests two difficulties: memory can adapt to the wrong route, and some failures also involve weak presence/localization rather than clean identity competition. Oracle advantages are not deployable gains. The wrong path was not independently verified to be another physical fiber in every case.

Evidence: [GT-versus-error-route analysis](embedding_route_analysis_20260913.md).

## 3. Historical single-fiber tracer experiments

**Do not compare these counts numerically with the current full-cohort tables.** Most policy tuning used both directions of one deliberately difficult PHerc1299 fiber, `lt_20260825T075316065_000069.json` (folder alias `PHerc1229`). The old reference included extrapolated tails/Lasagna spans and used earlier restart/acceptance rules. Repeated use makes this an in-sample tuning workload, even though the annotation originally belonged to the test set.

### Initial pilots and history tuning

| Model / experiment | Embeddings off | Embedding result | Finding |
|---|---:|---:|---|
| T2 raw u5400, shorter PHerc0139 pilot | 1 | 1 | Trajectories changed, failure count did not. |
| T2 raw u5400, hard PHerc1299, original mean history | 96 | 109 | Untuned history worsened failures. |
| Same model, 36 spacing/window/falloff settings at weight 0.25 | 96 | **88** | Best spacing8/window256/Gaussian/count64. Exact repeat. |
| Same model, another 36-setting grid at weight 0.30 | 96 | 93 | Worse than the best weight-0.25 setting; weight neighbors did not help. |
| T3 raw u3700, original history | 140 | 133 | Improvement within this model, but poorer underlying trace performance. |
| T3 raw u4300, history/weight/count refinement | 86 | **82** | Weight0.25, spacing128/window1024/Gaussian/count8; exact repeat. |
| T3 model EMA99 u6500, original mean / transferred raw winner | 87 | 82 / 87 | The raw-snapshot winner did not transfer as a winner. |
| Same EMA99 u6500, plain tracer EMA | 87 | **69** | Weight0.25, half-life512; repeated exactly. Weights0.20/0.30 gave80/84. |

History sweeps tested uniform, linear and Gaussian age weighting, admission spacing, finite range, count caps and nearby weights. Some dense/long-range combinations were count-limited, not true full-window tests. Geometric tracing step was unchanged.

Evidence: [initial pilots](embedding_tracing.md), [history sweeps](embedding_history_sweep_20260913.md), [raw paired tuning](embedding_paired_tracer_tuning_20260913.md), [EMA99 comparison](embedding_ema99_references_20260913.md).

### Memory-policy families and refinements on EMA99 u6500

The following summarizes the best reported outcomes in each family, not an equal-budget global search. Searches used grids and strict-improvement coordinate refinement, multiple seeds where recorded, and exact replay checks. Full parameter listings remain in the linked reports.

| Family | What it changes | Best recorded failures |
|---|---|---:|
| Plain EMA | One continuously adapted reference | 69 |
| Hard-gated EMA | Update only at presence ≥0.5 and cosine ≥0.5 | 75 |
| EMA + initial seed | Average EMA/seed penalties | 78 |
| Hard-gated EMA + seed | Combine both mechanisms | 73 |
| Seed only | Never update initial identity | 72 |
| Delayed EMA | Delay observations before integrating | 69 at delay0; nonzero-delay second-seed result71 |
| Finite mean / best history | Average reference distance / nearest retained reference | 70 / 74 |
| Windowed representative anchors + EMA mixture | Cover recent appearance modes | 67 |
| Plain LRU, nearest / mean aggregation | Fixed representatives, recency-based replacement | 66 / 68 |
| Novelty LRU, nearest / mean aggregation | Insert only sufficiently distinct observations | 68 / **64** |
| Hit-only recency, nearest / mean | Refresh only qualifying matches | 68 / 66 |
| Whole-path appearance / support-weighted soft minimum | Retain broader appearance modes and support | 70 / 67 |
| Normalized-mean LRU | Normalize anchor mean before scoring | 68–71 after refinement; did not beat64 |

The **64-failure policy** used two anchors, observations every512 base voxels, novelty threshold0.10, nearest-anchor recency refresh, least-recently-used eviction, ordinary mean cosine distance, weight0.10 and no EMA mixture or age expiry. It reduced failures by 26.4% versus87off on this tuned fiber, with exact-repeat evidence. “Hit-only” recency was a different, inferior variant.

Further boundary/multi-start searches initially stopped on missing cache coverage; these were marked incomplete rather than treated as negative results. Subsequent cached verification and demand-inference continuation verified the alternative67 result and completed the remaining four families; none beat64. Normalizing the anchor mean gave75 at three borrowed settings, improving only to68/71/68 after its own refinements. Near-cancellation was not observed in the anchor-coherence profiles.

Comparing LRU64 with support-softmin67 found **60 matched, 4 LRU-only and 7 softmin-only failures** within64 base voxels. Most failures were shared. Different restart histories confound apparent local recoveries; no GT-free switching rule or demonstrated mixture benefit resulted.

Evidence: [multi-anchor experiments](embedding_multi_anchor_20260913.md), [consolidated refinements and completed continuations](embedding_refinement_summary_20260914.md).

## 4. Full-cohort transfer under the older protocol

T3's two-anchor winner was applied without retuning to all50 fibers, **100 whole-fiber directions / 8,081.558 directed mm**, before current eligibility and restart filtering. The same heavily tuned fiber was included. These historical counts are retained for provenance, not pooled with current counts; old mean formulas are not reproduced here.

| T3 model EMA99 snapshot | Embeddings off | Anchor-LRU | Difference |
|---|---:|---:|---:|
| Reference u6500 | 2200 | 2181 | −19 |
| Validation-best u9900 | 2491 | 2520 | +29 |
| Current u32000 | 1958 | 1958 | 0 |

The single-fiber gain did not transfer as a reliable full-cohort improvement. Dense validation-best also was not tracing-best. Off/on pairs are the useful comparison; mixed historical inference settings and protocol changes prevent attributing all cross-run differences to embeddings.

Evidence: [legacy full comparison](benchmarks/embedding_full_comparison_complete_20260914.md), [snapshot manifest](benchmarks/embedding_full_matrix_20260914_manifest.json). Counts above were checked against the saved full summaries.

## 5. Current evaluation protocol and paired/mixed-model results

Current matched evaluations use **50 fibers across PHerc0139, PHerc0332, PHerc1299, PHerc1451 and PHercParis4**, including both Paris4 annotation collections. CP-bounded eligible geometry excludes extrapolation and Lasagna interpolation, leaving90 spans / **180 run-directions / 6,867.627 directed mm**.

Acceptance uses a normal-aware ellipse with thickness radius40 and tangential radius80 base voxels, a160-base-voxel excursion grace distance, and restart-episode filtering over80 reference voxels. Both directions are evaluated. The current reported mean is **combined directed reference distance / (counted restarts +1)** at each aggregation level; it is not the arithmetic average of all episode lengths. Median/P05/P95 include terminal segments.

### T4 paired model versus separate-model embeddings

All following results use the current matched protocol. “Mixed” means presence/direction from N's EMA999 u11800 and embeddings from T4, with separate immutable inference caches. No channels are relabeled or substituted by convenience.

| Presence/direction model | Embedding source / tracer policy | Restarts | Mean mm |
|---|---|---:|---:|
| Classic | Off | 292 | 23.439 |
| N, EMA999 u11800 | Off | 113 | 60.242 |
| T4, best EMA999 u10800 | Off | 135 | 50.497 |
| T4, best EMA999 u10800 | Same model, Anchor-LRU | 139 | 49.054 |
| T4, best EMA999 u10800 | Same model, plain EMA | 133 | 51.251 |
| T4, current EMA999 u42000 | Off | 205 | 33.338 |
| T4, current EMA999 u42000 | Same model, Anchor-LRU | 208 | 32.859 |
| T4, current EMA999 u42000 | Same model, plain EMA | 211 | 32.394 |
| N, EMA999 u11800 | T4 best u10800, plain EMA | 123 | 55.384 |
| N, EMA999 u11800 | T4 current u42000, plain EMA | **111** | **61.318** |

Plain EMA here is weight0.25/half-life512, with no anchors, seed mixture or delay. The first T4 “tuned” tests mistakenly used Anchor-LRU when EMA was intended; the four explicit EMA-only evaluations corrected that experiment, and both sets are retained above.

This separates two effects: the paired run's own presence/direction performance deteriorated at u42000 (particularly PHerc0332), while its embeddings were slightly useful when transplanted onto stronger predictions. It does **not** prove that embedding training caused the deterioration; the runs are not a controlled loss-only ablation.

Evidence: [paired off/Anchor-LRU matrix](benchmarks/paired_ema999_complete_20260915.md), [corrected EMA-only and mixed-model matrix](benchmarks/ema_only_matrix_20260915.md), [mixed inference implementation](mixed_inference.md).

## 6. Differentiable fitting of a fixed-path scoring objective

Using the mixed N-u11800/T4-u42000 setup, we extracted common-history GT and erroneous continuations from saved traces and fit scoring parameters on GPU without retraining either network. Whole fibers were grouped into **30 fit /9 selection /11 calibration-heldout**, retaining duplicates together. There were148 accepted windows:29 failure comparisons and119 successful controls.

The loss combined paired route ranking, preservation of successful-control score differences and parameter regularization. It fitted weight and half-life, then also a zero-anchored smooth threshold/temperature mapping. Shared warm-up was recomputed and detached before backpropagating through32-step/~256-base-voxel branches. Selection chose parameters; heldout did not.

| Policy | Full restarts | Mean mm | Fit / selection / heldout restarts |
|---|---:|---:|---|
| Embeddings off | 113 | 60.242 | 59 /36 /18 |
| Original mixed EMA, weight0.25/H512 | 111 | 61.318 | 55 /35 /21 |
| Fitted linear EMA, weight0.0454013/H510.432 | **110** | **61.871** | 56 /34 /20 |
| Fitted smooth mapping | 117 | 58.200 | 55 /38 /24 |

The smooth mapping lowered surrogate loss but worsened actual tracing. Fitting mostly reduced embedding influence; it did not demonstrate reliable GT/error discrimination. A longer512-voxel diagnostic retained zero heldout full-window GT-preference wins for both fitted models. It mined longer windows separately, so was not a strictly paired truncation ablation.

The fixed GT connector need not belong to the native beam's discrete candidate set; warm-up truncates ancestry and long-window ranking is not depth-two beam search. Thus a better surrogate loss is not necessarily a better tracer. These are important limitations, not merely optimizer issues.

Evidence: [fixed-path calibration results, parameters and caveats](benchmarks/fixed_path_ema_calibration_20260915.md).

## 7. Sampling, history range and nonlinear integration

We next fitted45 candidates:20 ordinary EMA candidates (spacing8/16/32/64 × half-life starts64/128/256/512/1024),16 finite-window candidates (same spacings × ranges256/512/1024/2048), and three initializations each for a learned gate, angular cap and dual EMA at spacing8. Weight/half-life and applicable nonlinear coefficients were learned; spacing/window were discrete choices. GPU fitting used150 Adam updates per family with candidates vectorized together.

- **Gate:** modulate update strength from presence and disagreement with current memory; rejected observations still consume the admission interval.
- **Angular cap:** limit the angular motion of each ordinary EMA update, scaled by elapsed physical distance.
- **Dual EMA:** independently adapt short/long memories and learn how to mix their candidate distances; these are not anchors.
- **Finite window:** average distance to admitted observations within the candidate's physical history range.

Scoring happens at every candidate; sampling controls only memory updates. Warm-up increased to4096voxels, while all148 window identities and scored branch embeddings/distances/base costs remained unchanged. Each family's selection-set winner received a full50fiber native evaluation.

| Policy | Restarts | Mean mm | Median mm | P05 mm | P95 mm | Heldout restarts |
|---|---:|---:|---:|---:|---:|---:|
| Embeddings off | 113 | 60.242 | 9.337 | 0.233 | 94.960 | 18 |
| Original mixed EMA | 111 | 61.318 | 10.512 | 0.236 | 96.233 | 21 |
| Previous fitted linear EMA | **110** | **61.871** | 10.506 | 0.232 | 89.634 | 20 |
| Cadenced EMA | 117 | 58.200 | 10.011 | 0.232 | 102.469 | 21 |
| Finite window | 111 | 61.318 | 10.490 | 0.251 | 89.563 | 19 |
| Learned gate | **110** | **61.871** | 10.685 | 0.228 | 96.830 | 20 |
| Angular-limited EMA | 115 | 59.204 | 10.507 | 0.262 | 102.700 | 22 |
| Dual EMA | 113 | 60.242 | 9.984 | 0.237 | 108.852 | 22 |

All finalists selected spacing8; the finite-window winner used range256. The angular and dual winners retained their initial coefficients rather than finding a selection improvement. No new family improved overall restart count beyond110.

### Where the gains and regressions occur

| Policy | PHerc0139 | PHerc0332 | PHerc1299 | PHerc1451 | Paris4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Embeddings off | 50 | 4 | 7 | 0 | 52 | 113 |
| Original mixed EMA | 49 | 4 | 9 | 0 | 49 | 111 |
| Previous fitted linear | 46 | 4 | 7 | 0 | 53 | 110 |
| Cadenced EMA | 51 | 4 | 6 | 0 | 56 | 117 |
| Finite window | 49 | 3 | 7 | 0 | 52 | 111 |
| Learned gate | 50 | 4 | 8 | 0 | 48 | 110 |
| Angular limit | 51 | 4 | 7 | 0 | 53 | 115 |
| Dual EMA | 49 | 4 | 8 | 0 | 52 | 113 |

Gate saves five Paris4 restarts against the prior fit but adds four in0139 and one in1299. Its selection-subset count worsens34→37; heldout ties20. The finite window has19heldout versus20previous fit, but embeddings-off has18. These small differences do not establish significance.

Native/PyTorch replay parity passed for all45 fitted settings on270 real paths, maximum absolute weighted-score difference **2.13×10⁻⁷**. The final suite passed107 native and231 Python tests (eight optional skips). The five native evaluations took286–300s each, approximately356–372s including coordination; variable extra inference makes these unsuitable as isolated overhead comparisons.

Evidence: [all45 candidates, exact settings, per-scroll and split metrics](benchmarks/embedding_integration_20260915.md), [integration definitions](embedding_integration.md).

## 8. Practical interpretation and next research questions

**Demonstrated:** a learnable continuity signal; better local contrastive separation; improved long-range consistency with pairing; reproducible single-fiber policy gains; a small aggregate gain from transplanting embeddings onto a stronger predictor.

**Not demonstrated:** stable fiber identity across contexts; robust discrimination of actual competing routes; consistent transfer of a tuned policy across checkpoints/scrolls; statistical significance of113→110; or a favorable quality/runtime/storage trade-off.

Likely explanations, rather than proven causes, are weak identity separation at actual failures, route-memory contamination, conflicts with presence/direction scores, and mismatch between fixed-path optimization and beam decisions. Improved continuity and lower dense loss are insufficient proxies for the final tracing task.

If revisited, the most informative next steps would be:

1. Train/evaluate against **actual competing beam continuations**, including hard nearby high-presence alternatives, rather than rely primarily on random off-GT negatives or constructed GT connectors.
2. Preserve the stronger presence/direction model while testing embedding-specific training changes under a matched initialization/data/training-budget protocol.
3. Test cross-context consistency together with discrimination; neither near-constant embeddings nor indiscriminate smoothing is a solution.
4. Select policies on development fibers and confirm them on a genuinely untouched cohort, reporting scroll-level trade-offs and inference/cache cost.

These are recommendations, not completed experiments. Defaults remain unchanged. There was no new training, tracing or rendering for this summary.

## 9. Reproducibility and artifact index

Repository links above are relative so the document works when viewed in the repository. The paths below are **machine-local**, not public download links. Let `E = /ephemeral/hendrik/fiber3d_train2/evaluations/`; directory names below are relative to E. Full-evaluation results are in `full/summary.json`, with individual trajectories under `full/cases/`; diagnostic/sweep directories retain their own `report.json` or `results.json`.

| Experiment | Artifact directory / detailed source |
|---|---|
| Pairwise consistency | `embedding_consistency_20260913_v1` |
| Contrastive consistency/histograms | `embedding_consistency_20260913_contrastive_v1` |
| First short / hard tracer pilots | `embedding_trace_20260913_v2`, `embedding_trace_hard_20260913_v1` |
| Raw5400 history/weight grids | `embedding_trace_sweep_20260913_v4`, `embedding_weight_sweep_20260913_v1`, `embedding_history_weight030_20260913_v1` |
| Raw5400 error-route analysis | `embedding_route_analysis_20260913_v3` |
| Paired3700 trace / route diagnostics | `embedding_trace_paired_current_20260913_v2`, `embedding_route_paired_current_20260913_v2` |
| Paired4300 tuning | `embedding_paired_tune_history_20260913_v1`, `embedding_paired_tune_weight_20260913_v1`, `embedding_paired_tune_refine_20260913_v1` |
| EMA6500 memory/weight tuning | `embedding_ema99_modes_20260913_v1`, `embedding_ema99_weight_20260913_v1` |
| Anchors/mechanisms/refinement | `embedding_ema99_anchors_20260913_v1`, `embedding_ema99_refinement_20260913_v1`, `embedding_ema99_mechanisms_20260913_v1`, `embedding_ema99_all_refine_20260913_v1` |
| Seed/delay/touch and normalization | `embedding_ema99_seed_delay_touch_20260914_v1`, `embedding_ema99_new_method_refine_20260914_v1`, `embedding_ema99_normalized_mean_20260914_v1` |
| Boundary / repeat verification / continuations | `embedding_ema99_anchor_boundaries_20260914_v1`, `embedding_ema99_boundary_cpu_verify_20260914_v1`, `embedding_ema99_remaining_cpu_refine_20260914_v1`, `embedding_ema99_remaining_adaptive_refine_20260914_v1` |
| Legacy full T3 off/Anchor-LRU | `embedding_full_{reference6500,best9900,current32000}_{off,tuned}_20260914_v2` (six directories) |
| Current Classic / N baseline | `classic_thickness40_grace160_20260914_v2`, `eligible_ema999_final_vis_20260914_v1` |
| Current T4 off/Anchor-LRU | `paired_ema999_{best,current}_{off,tuned}_20260915_v1` (four directories) |
| Current T4 plain EMA | `paired_{best,current}_ema_only_20260915_v1` |
| Mixed models, plain EMA | `mixed_{best,current}_ema_only_20260915_v1` |
| Fixed-path extraction/fitting + full finalists | `fixed_path_ema_fit_20260915_v1`, `fixed_path_ema_fit_long_20260915_v1`, `fixed_path_ema_{linear,smooth}_20260915_v1` |
| Integration extraction/fitting + five full finalists | `embedding_integration_data_20260915_v1`, `embedding_integration_fit_20260915_v1`, `embedding_integration_{ema,window,gate,angular,dual}_20260915_v1` |

Important immutable checkpoint SHA256 identities:

| Snapshot | SHA256 |
|---|---|
| T1 raw u5300 | `f4e261f81de165a76bde523b7d864b6001ddefa543037ce958590cbf6e8ee258` |
| T2 raw u2100 | `46ff8c8320a7787d1d4209fe0dbf8b9e3aea8f1e40a2314b9be708ded7489eb6` |
| T2 raw u5400 | `7ea0f9c89e4bd4ce8c48dbae920b7ed24c810c178ab34576767e30b930057b35` |
| T3 raw u3700 / u4300 | `c412ccf6dd9a95562af370b99a92b7595aadd343f689d90bea088cee8415f505` / `ee272228afaee9f4163338b5d803338f834ddd7bdfcbdb6e34c03a08e76b6f0f` |
| T3 EMA99 u6500 | `0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417` |
| T3 EMA99 best u9900 / current u32000 | `8f2b330d96401cd3fcb8cde232bebad1dec988b7c58e98a1c7f9d7885c6a9d91` / `d1c74163d47e331ec2e0235ec41060f49ddf363a69e8b4c896644aa5032a8e2f` |
| N EMA999 u11800 | `4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82` |
| T4 EMA999 best u10800 | `440d8689455023b03993b1f1c8aa5cf47e0b7f02f4523104b85ab6114e8e9dcf` |
| T4 EMA999 current u42000 | `7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15` |

The linked benchmark reports contain exact evaluation commands, pinned configs and additional hashes. Historical September13 commands used `/home/hendrik/villa`; current commands use `/home/hendrik/villa2`. Reproduction should use the recorded frozen snapshots/configs, not a mutable `current` checkpoint or today's mutable sweep settings, and should write to a fresh evaluation output directory without deleting reusable caches.

**Scope limitation:** this inventory covers recorded 3D embedding-quality experiments, plus the identifiable S1a prototype. Older 2D aspirations and unmeasured/incomplete launch attempts are not presented as successful quality experiments. Interrupted searches are distinguished from their verified continuations. No claim of an exhaustive search over all possible embedding models or policies is intended.
