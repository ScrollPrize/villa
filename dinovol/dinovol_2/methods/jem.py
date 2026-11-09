"""JEM: per-depth categorical alignment, logical-batch global collapse, structural collapse.

One optimizer update uses the configured global batch of independent sources. Each
rank processes its sources in chunks. The global-collapse marginal of every depth,
entropy group and position is formed over the whole logical batch:

* One chunk: forward once, reduce detached statistics across ranks, backpropagate.
  (With global_marginal "running", after the first update, d comes from the running
  marginal instead, for one chunk as for several.)
* Several chunks: pass 1 runs the student without gradients on every chunk to
  collect statistics (RNG state saved per chunk); after the cross-rank reduction,
  pass 2 replays each chunk with the same RNG state (drop-path), adds the teacher,
  and backpropagates its share using the fixed marginal. Only the last chunk's
  backward synchronizes DDP gradients.

The surrogate W/M * (C + S . stop_gradient(log mu + 1)) gives, after DDP's gradient
average, the exact first-order gradient of the full-batch objective; the logged
global-collapse value is the true objective computed from the reduced statistics.

method.global_marginal selects the marginal behind d = log mu + 1:
"exact" is the two-pass scheme above. "running" (default) runs one pass per chunk with d
from the running marginal r of past updates (momentum 0: the previous update's); the
coefficients W/M come from this batch's validity masks and stay exact, and the same pass
accumulates this batch's statistics for the logged loss and the next r. This is the
gradient of max_r [C/M + sum mu log r] at a stale r, not the exact gradient. The first
update (no r yet) and evaluation use the exact scheme.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._functorch import config as functorch_config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from dinovol_2.loss.jem_losses import (
    EntropyStatistics, accumulation_terms, alignment_from_means, distance_metric, entropy_attribution,
    entropy_groups, entropy_surrogate, fp32_reductions, global_collapse_weight, nonaffine_target,
    overlap_capacity, overlap_order, slot_group_index, structural_band_counts, structural_distance_stratified,
    structural_uniform_contraction_rows, structural_uniform_tiled, structural_weight,
)
from dinovol_2.dataset.jem_views import zyx
from dinovol_2.model.jem import JemStudentTeacher
from dinovol_2.ops.precision import resolve_amp_dtype
from dinovol_2.training.compilation import resolve_compile_settings
from .base import StepOutput, TrainingMethod

KINDS = ("global", "local")


class JemMethod(TrainingMethod):
    name = "jem"

    def __init__(self, config: Mapping[str, Any], *, device, is_distributed=False, rank=0) -> None:
        super().__init__()
        self.recipe = dict(config["method"])
        runtime = config["runtime"]
        self.data_config = config["data"]
        self.backbone_config = dict(config["backbone"])
        self.device = torch.device(device)
        self.is_distributed = bool(is_distributed)
        self.rank = rank
        self.world_size = dist.get_world_size() if self.is_distributed and dist.is_initialized() else 1
        self.total_updates = int(runtime.get("max_iterations", runtime.get("num_iterations", 1000000)))
        if int(runtime.get("grad_accumulation_steps", 1)) != 1:
            raise ValueError("JEM uses one microbatch per update; split sources with method.source_chunk_size")
        self.local_batch_size = int(self.data_config.get("batch_size", 2))
        global_batch = int(self.recipe["global_batch_size"])
        if self.local_batch_size * self.world_size != global_batch:
            raise ValueError(f"JEM global_batch_size {global_batch} must equal data.batch_size "
                             f"{self.local_batch_size} x world size {self.world_size}")
        chunk = self.recipe["source_chunk_size"]
        self.source_chunk_size = self.local_batch_size if chunk is None else min(int(chunk), self.local_batch_size)
        self.compile_settings = resolve_compile_settings(runtime.get("compile", False),
                                                         ("student", "teacher", "objective"))

        self.patch = zyx(self.backbone_config["patch_size"], "patch_size")  # ZYX token cell
        heads = {
            "alignment_depths": self.recipe["alignment_depths"],
            "structural_depths": self.recipe["structural_depths"],
            "categories": self.recipe["categories"], "hidden_dim": self.recipe["head_hidden_dim"],
            "bottleneck_dim": self.recipe["head_bottleneck_dim"], "structural_dim": self.recipe["structural_dim"],
            "nlayers": self.recipe["head_nlayers"], "last_layer_type": self.recipe["last_layer_type"],
            "checkpoint_heads": self.recipe["checkpoint_heads"],
        }
        self.model = JemStudentTeacher(self.backbone_config, heads).to(self.device)
        self.init_seed = int(torch.initial_seed())
        if self.is_distributed:
            kwargs: dict[str, Any] = {"broadcast_buffers": False, "find_unused_parameters": False}
            if self.device.type == "cuda":
                kwargs.update(device_ids=[self.device.index], output_device=self.device.index)
            self.model = DDP(self.model, **kwargs)
            # DDP broadcast the student from rank 0; start the EMA teacher from it too.
            self.model_module.synchronize_teacher_from_student()

        self.depth_weights = {int(k): float(v) for k, v in self.recipe["depth_weights"].items()}
        self.alignment_depths = [int(d) for d in self.recipe["alignment_depths"]]
        self.structural_depths = [int(d) for d in self.recipe["structural_depths"]]
        views = self.recipe["views"]
        self.locals_per_set = int(views["locals_per_set"])
        self.groups = entropy_groups(self.recipe["entropy_view_scope"], n_global_slots=2,
                                     n_local_slots=2 * self.locals_per_set,
                                     global_weight=float(self.recipe["global_view_weight"]),
                                     local_weight=float(self.recipe["local_view_weight"]))
        self.views_per_source = 2 + 2 * self.locals_per_set
        # Entropy group of each slot (index into groups of the crop type), for row-wise gathers.
        self.slot_group = {kind: torch.tensor(slot_group_index(self.groups, kind), device=self.device)
                           for kind in KINDS}
        self.n_groups = {kind: sum(group.view_type == kind for group in self.groups) for kind in KINDS}
        self.successful_updates = 0
        marginal = self.recipe["global_marginal"]
        self.marginal_mode = marginal["mode"]
        self.marginal_momentum = float(marginal["momentum"])
        self.marginal_batch_weight = float(marginal.get("batch_weight", 0.5))
        self.marginal_exact_updates = int(marginal.get("exact_updates", 0))
        # Running marginal statistics {"S": {key: [N, K]}, "M": {key: [N]}} of past updates.
        self.running_marginal = None
        # Previous update's batch marginals {key: (mu [N, K], present [N])}, for diagnostics only.
        self._previous_marginal = None
        self.skipped_updates = 0
        # Diagnostic: re-accumulate statistics in pass 2 and report the largest
        # difference from pass 1 (drop-path/RNG replay check).
        self.verify_replay = False
        self._last_weights = (1.0, 0.0)
        self.input_dtype = (resolve_amp_dtype(runtime.get("amp_dtype", "bfloat16"))
                            if bool(runtime.get("use_amp", self.device.type == "cuda")) else torch.float32)
        self._statistics_fn = self._statistics_terms
        self._objective_fn = self._objective_terms
        if self.compile_settings is not None:
            self._compile_model()

    # ------------------------------------------------------------------ plumbing

    @property
    def model_module(self) -> JemStudentTeacher:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def resolved_recipe(self) -> dict[str, Any]:
        return {**self.recipe, "initialization": {"scheme": "scratch", "torch_initial_seed": self.init_seed},
                "source_chunk_size_resolved": self.source_chunk_size, "world_size": self.world_size}

    def regularizer_weights(self, updates: int) -> tuple[float, float]:
        warmup = int(self.recipe["regularizer_warmup_steps"])
        g = self.recipe["global_collapse_schedule"]
        s = self.recipe["structural_schedule"]
        return (global_collapse_weight(updates, warmup, start=float(g["start"]), end=float(g["end"])),
                structural_weight(updates, warmup, self.total_updates, warm=float(s["warm"]),
                                  final=float(s["final"])))

    def _rng_state(self):
        state = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _set_rng_state(self, state) -> None:
        torch.set_rng_state(state["cpu"])
        if "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def _crop_rows(self, crops: torch.Tensor, groups: int, B: int, sources: torch.Tensor) -> torch.Tensor:
        """Crop rows g * B + sources for g < groups, on the device in the input dtype.

        For pinned host storage and a contiguous source range each group is one pinned
        slice copied asynchronously; indexing first would gather into pageable memory and
        make the upload synchronous, idling the GPU.
        """
        start, n = int(sources[0]), len(sources)
        if (crops.device.type == "cpu" and self.device.type == "cuda" and crops.dtype == self.input_dtype
                and crops.is_pinned() and torch.equal(sources, torch.arange(start, start + n))):
            out = torch.empty((groups * n, *crops.shape[1:]), dtype=crops.dtype, device=self.device)
            for g in range(groups):
                out[g * n:(g + 1) * n].copy_(crops[g * B + start:g * B + start + n], non_blocking=True)
            return out
        rows = torch.cat([g * B + sources for g in range(groups)])
        return crops[rows].to(self.device, self.input_dtype, non_blocking=True)

    def _chunk(self, batch: Mapping[str, Any], sources: torch.Tensor) -> dict[str, Any]:
        """Rows of the given local source indices, with chunk-local teacher rows."""
        B = int(batch["batch_size"])
        n = len(sources)
        position = torch.full((B,), -1, dtype=torch.long)
        position[sources] = torch.arange(n)
        teacher_rows = torch.cat([s * B + sources for s in range(int(batch["n_sets"]))])
        chunk = {"n_sources": n,
                 "teacher_crops": self._crop_rows(batch["teacher_crops"], int(batch["n_sets"]), B, sources)}

        def upload(tensor):
            # Host tensors here are small temporaries; the copy is staged at once, so it need
            # not wait for the queued GPU work.
            return tensor.to(self.device, non_blocking=True)

        for kind in KINDS:
            slots = int(batch[f"n_{kind}_views"])
            rows = torch.cat([v * B + sources for v in range(slots)])
            device_rows = upload(rows)

            def gather(name):
                tensor = batch[name]
                return upload(tensor[rows if tensor.device.type == "cpu" else device_rows])

            row_sources = batch[f"{kind}_sources"][rows]
            per_set = slots // int(batch["n_sets"])
            set_index = batch[f"{kind}_slots"][rows] // per_set
            row_slots = upload(batch[f"{kind}_slots"][rows])
            entry = {
                "crops": self._crop_rows(batch[f"{kind}_crops"], slots, B, sources),
                "masks": gather(f"{kind}_masks"),
                "targets": gather(f"{kind}_targets"),
                "valid": gather(f"{kind}_valid"),
                "slots": row_slots,
                "group": self.slot_group[kind][row_slots],
                "sources": upload(row_sources),
                "teacher_rows": upload(set_index * n + position[row_sources]),
                "grid": tuple(batch[f"{kind}_grid"]),
                "mask_applied": batch[f"{kind}_mask_applied"][rows],
                "spacing": batch["spacing"][row_sources],
            }
            if f"{kind}_lattice" in batch:
                entry["lattice"] = gather(f"{kind}_lattice")
            # Alignment and structural terms read only overlapping tokens: compact them to a
            # capacity known on the host (no GPU sync); its width varies from chunk to chunk.
            capacity = overlap_capacity(int(batch[f"{kind}_overlap_count"][rows].max()), entry["targets"].shape[1])
            entry["order"] = overlap_order(entry["targets"] >= 0, capacity)
            if self.compile_settings is not None:
                torch._dynamo.maybe_mark_dynamic(entry["order"], 1)
            chunk[kind] = entry
        return chunk

    def _compile_model(self) -> None:
        """Compile the student and teacher forwards (backbone and heads), the objective and the statistics.

        The gradient pass must replay the statistics pass's drop-path masks from the saved
        RNG state. Random ops fall back to ATen instead of Inductor's per-graph Philox
        seeds, so they consume the CUDA generator exactly as eager execution does (and
        compiled and eager runs draw the same masks). In-place attribute replacement keeps
        parameters and state_dict keys unchanged.
        """
        settings = self.compile_settings
        if settings.dynamic["objective"] is False:
            raise ValueError("compile.dynamic.objective cannot be false: the compacted overlap width changes "
                             "every chunk")
        if settings.options["backend"] == "inductor":
            import torch._inductor.config as inductor_config
            inductor_config.fallback_random = True
        module = self.model_module
        module.student_forward = torch.compile(module.student_forward, **settings.for_module("student"))
        module.teacher_forward = torch.compile(module.teacher_forward, **settings.for_module("teacher"))
        # The objective and statistics need no host syncs; the objective's only varying size is
        # the compacted overlap width (marked dynamic). Schedule weights stay outside the graph.
        self._statistics_fn = torch.compile(self._statistics_terms, **settings.for_module("objective"))
        self._objective_fn = torch.compile(self._objective_terms, **settings.for_module("objective"))

    def _student(self, chunk, *, structural: bool, model=None):
        g, l = chunk["global"], chunk["local"]
        model = self.model if model is None else model
        # Backward runs outside autocast; AOTAutograd must capture backward graphs that way.
        context = (functorch_config.patch(backward_pass_autocast="off") if self.compile_settings is not None
                   else nullcontext())
        with context:
            return model(g["crops"], g["masks"].view(len(g["crops"]), -1), l["crops"],
                         l["masks"].view(len(l["crops"]), -1), structural=structural)

    # ------------------------------------------------------------------ objective

    def _log_q(self, logits):
        return F.log_softmax(logits.float() / float(self.recipe["student_temperature"]), dim=-1)

    def _statistics_terms(self, logits, entries):
        """accumulation_terms per crop type and alignment depth (detached logits)."""
        with fp32_reductions(self.device.type):
            return {kind: {depth: accumulation_terms(self._log_q(logits[kind][depth]), entries[kind]["valid"],
                                                     entries[kind]["group"], self.n_groups[kind],
                                                     entries[kind]["sources"])
                           for depth in self.alignment_depths}
                    for kind in KINDS}

    def _accumulate(self, stats: EntropyStatistics, outputs, chunk) -> None:
        logits = {kind: {depth: outputs[kind]["logits"][depth].detach() for depth in self.alignment_depths}
                  for kind in KINDS}
        entries = {kind: {key: chunk[kind][key] for key in ("valid", "group", "sources")} for kind in KINDS}
        with torch.no_grad():
            terms = self._statistics_fn(logits, entries)
        for kind in KINDS:
            for depth in self.alignment_depths:
                stats.add_terms(depth, kind, terms[kind][depth])

    def _coefficients(self, stats: EntropyStatistics, mixed: bool = False):
        """Per crop type and depth, group-stacked (coef [G, N], d [G, N, K], true coef [G, N], log mu [G, N, K]).

        coef carries the rank scaling of the surrogate; the true coefficient attributes the loss value.
        mixed: d = log(mu_hat) + 1 with mu_hat = batch_weight * mu + (1 - batch_weight) * r (running r).
        """
        surrogate_scale = float(self.world_size) if self.recipe["entropy_rank_scope"] == "logical_batch" else 1.0
        result = {}
        for kind in KINDS:
            result[kind] = {}
            for depth in self.alignment_depths:
                weight = self.depth_weights[depth]
                parts = [stats.coefficients((depth, group.name), surrogate_scale * weight * group.weight,
                                            self._running_reference((depth, group.name)) if mixed else None,
                                            self.marginal_batch_weight)
                         for group in stats.groups_for(kind)]
                true = [stats.coefficients((depth, group.name), weight * group.weight)[0]
                        for group in stats.groups_for(kind)]
                result[kind][depth] = (torch.stack([part[0] for part in parts]),
                                       torch.stack([part[1] for part in parts]),
                                       torch.stack(true), torch.stack([part[2] for part in parts]))
        return result

    def _categorical_terms(self, logits, teacher_log_p, order, index, visible, masked, n_visible, n_masked, valid,
                           coef, d):
        """Per-row alignment [rows], the entropy surrogate (scalar) and the visible/masked mean KLs.

        The surrogate covers every token; the KL covers the compacted overlap tokens: order
        [rows, P] token positions, index [rows, P] their flat teacher tokens in teacher_log_p
        [T * Nt, K], visible/masked [rows, P] (false past a row's overlap). coef [rows, N] and
        d [rows, N, K] are the surrogate coefficients of each row's entropy group.
        """
        log_q = self._log_q(logits)
        log_p = teacher_log_p[index]
        student = log_q.gather(1, order[..., None].expand(-1, -1, log_q.shape[-1]))
        kl = (log_p.exp() * (log_p - student)).sum(-1)
        visible_mean = torch.where(visible, kl, 0).sum(-1) / n_visible.clamp_min(1)
        masked_mean = torch.where(masked, kl, 0).sum(-1) / n_masked.clamp_min(1)
        alignment = alignment_from_means(visible_mean, masked_mean, visible_weight=float(self.recipe["visible_weight"]),
                                         masked_weight=float(self.recipe["masked_weight"]))
        return alignment, entropy_surrogate(log_q, valid, coef, d), visible_mean, masked_mean

    def _structural(self, a, targets, entry, kind, tile):
        """Per-row structural loss [rows] for one target depth (targets [T, Nt, C])."""
        overlap = entry["targets"] >= 0
        weighting = self.recipe["structural_pair_weighting"]
        if weighting == "uniform" and self.recipe["structural_backend"] == "contraction":
            order = entry["order"]
            index = entry["teacher_rows"][:, None] * targets.shape[1] + entry["targets"].gather(1, order).clamp_min(0)
            student = a.gather(1, order[..., None].expand(-1, -1, a.shape[-1]))
            return structural_uniform_contraction_rows(F.normalize(student.float(), dim=-1),
                                                       targets.flatten(0, 1)[index], overlap.gather(1, order))
        losses = []
        for row in range(a.shape[0]):
            selected = overlap[row].nonzero(as_tuple=True)[0]
            student = a[row, selected].float()
            student = F.normalize(student, dim=-1)
            teacher = targets[entry["teacher_rows"][row], entry["targets"][row, selected]]
            if weighting == "uniform":
                losses.append(structural_uniform_tiled(student, teacher, tile))
            else:
                coords = entry["lattice"][row, selected]
                metric, _ = distance_metric(entry["spacing"][row].tolist(), self.patch)
                key = (kind, row)
                counts = entry.setdefault("band_counts", {}).get(key)
                if counts is None:
                    counts = structural_band_counts(coords, metric, self.recipe["distance_bands"]["edges"], tile)
                    entry["band_counts"][key] = counts
                loss, counts, _ = structural_distance_stratified(
                    student, teacher, coords, metric, self.recipe["distance_bands"]["edges"],
                    self.recipe["distance_bands"]["weights"], tile, counts=counts)
                losses.append(loss)
        return torch.stack(losses)

    def _objective_terms(self, outputs, teacher, entries, coefficients):
        """Chunk loss parts before the schedule weights, and detached metrics.

        Returns (alignment, surrogate, structural, metrics): the graph-connected alignment,
        entropy surrogate and structural terms; the caller forms
        alignment + w_global * surrogate + w_struct * structural and scales the
        contrib_global_* / contrib_struct_* metrics by their weights. With the uniform
        contraction backend there are no host syncs; shapes are fixed per chunk except the
        compacted overlap width (entry["order"]).
        """
        n_views = self.views_per_source * self.local_batch_size
        n_struct = len(self.structural_depths)
        tile = int(self.recipe["structural_tile"])
        temperature = float(self.recipe["teacher_temperature"])
        metrics: dict[str, torch.Tensor] = {}

        def add(name, value):
            value = value.detach().float()
            metrics[name] = metrics[name] + value if name in metrics else value

        with fp32_reductions(self.device.type):
            teacher_targets = {depth: nonaffine_target(teacher["structural"][depth]) for depth in self.structural_depths}
            teacher_log_p = {depth: F.log_softmax(teacher["logits"][depth].float() / temperature, dim=-1)
                             for depth in self.alignment_depths}
            alignment_total = surrogate_total = structural_total = None
            for kind in KINDS:
                entry = entries[kind]
                rows = entry["targets"].shape[0]
                overlap = entry["targets"] >= 0
                visible, masked = overlap & ~entry["masks"], overlap & entry["masks"]
                n_visible, n_masked = visible.float().sum(-1), masked.float().sum(-1)
                order = entry["order"]
                compact_visible, compact_masked = visible.gather(1, order), masked.gather(1, order)
                compact_targets = entry["targets"].gather(1, order).clamp_min(0)
                group = entry["group"]
                row_alignment = torch.zeros(rows, device=overlap.device)
                row_entropy = torch.zeros(rows, device=overlap.device)
                for depth in self.alignment_depths:
                    weight = self.depth_weights[depth]
                    coef, d, true_coef, log_mu = coefficients[kind][depth]
                    log_p = teacher_log_p[depth]
                    index = entry["teacher_rows"][:, None] * log_p.shape[1] + compact_targets
                    logits = outputs[kind]["logits"][depth]
                    args = (logits, log_p.flatten(0, 1), order, index, compact_visible, compact_masked, n_visible,
                            n_masked, entry["valid"], coef[group], d[group])
                    if self.recipe["checkpoint_losses"] and torch.is_grad_enabled():
                        alignment, surrogate, visible_mean, masked_mean = checkpoint(
                            self._categorical_terms, *args, use_reentrant=False)
                    else:
                        alignment, surrogate, visible_mean, masked_mean = self._categorical_terms(*args)
                    term = weight * alignment.sum() / n_views
                    alignment_total = term if alignment_total is None else alignment_total + term
                    surrogate_total = surrogate if surrogate_total is None else surrogate_total + surrogate
                    row_alignment = row_alignment + weight * alignment.detach()
                    with torch.no_grad():
                        row_entropy = row_entropy + entropy_attribution(
                            self._log_q(logits.detach()), entry["valid"], true_coef[group], log_mu[group])
                    add(f"align_d{depth}_{kind}", alignment.sum() / n_views)
                    add(f"_kl_visible_sum_d{depth}_{kind}", (visible_mean * n_visible).sum())
                    add(f"_kl_visible_count_d{depth}_{kind}", n_visible.sum())
                    add(f"_kl_masked_sum_d{depth}_{kind}", (masked_mean * n_masked).sum())
                    add(f"_kl_masked_count_d{depth}_{kind}", n_masked.sum())
                    # Trivial baseline: the masked KL of predicting this chunk's mean teacher
                    # distribution. Masked KL below it means masked tokens are predicted from context.
                    with torch.no_grad():
                        log_mu = log_p.exp().mean((0, 1)).clamp_min(torch.finfo(torch.float32).tiny).log()
                        target = log_p.flatten(0, 1)[index]
                        baseline = (target.exp() * (target - log_mu)).sum(-1)
                        add(f"_kl_masked_marginal_sum_d{depth}_{kind}", torch.where(compact_masked, baseline, 0).sum())
                        add(f"_kl_masked_marginal_count_d{depth}_{kind}", n_masked.sum())
                row_structural = torch.zeros(rows, device=overlap.device)
                for depth, head_output in outputs[kind].get("structural", {}).items():
                    structural = self._structural(head_output, teacher_targets[depth], entry, kind, tile)
                    term = structural.sum() / (n_struct * n_views)
                    structural_total = term if structural_total is None else structural_total + term
                    row_structural = row_structural + structural.detach() / n_struct
                    add(f"struct_d{depth}_{kind}", structural.sum() / n_views)
                # Whole-view mask-status attribution with the training denominators.
                masked_view = entry["masks"].any(-1)
                for status, selector in (("masked", masked_view), ("unmasked", ~masked_view)):
                    selector = selector.float()
                    add(f"contrib_align_{status}", (row_alignment * selector).sum() / n_views)
                    add(f"contrib_struct_{status}", (row_structural * selector).sum() / n_views)
                    # Entropy shares sum to R over the logical batch; scale so the trainer's
                    # cross-rank mean reports the global sum.
                    add(f"contrib_global_{status}", (row_entropy * selector).sum() * self.world_size)
                    add(f"views_{status}_{kind}", selector.sum() / self.local_batch_size)
                # Means over this rank's rows of the type (chunks add up to the full batch).
                type_rows = self.local_batch_size * (2 if kind == "global" else 2 * self.locals_per_set)
                add(f"overlap_fraction_{kind}", overlap.float().mean(-1).sum() / type_rows)
                add(f"valid_fraction_{kind}", entry["valid"].float().mean(-1).sum() / type_rows)
                add(f"mask_fraction_{kind}", entry["masks"].float().mean(-1).sum() / type_rows)
                add(f"masked_overlap_fraction_{kind}", masked.float().mean(-1).sum() / type_rows)
            if structural_total is None:
                structural_total = alignment_total.new_zeros(())
        return alignment_total, surrogate_total, structural_total, metrics

    def _chunk_objective(self, outputs, teacher, chunk, coefficients, w_global, w_struct):
        """Graph-connected chunk loss (alignment + scheduled surrogate + structural) and detached metrics."""
        alignment, surrogate, structural, metrics = self._objective_fn(
            outputs, teacher, {kind: chunk[kind] for kind in KINDS}, coefficients)
        for status in ("masked", "unmasked"):
            metrics[f"contrib_struct_{status}"] = w_struct * metrics[f"contrib_struct_{status}"]
            metrics[f"contrib_global_{status}"] = w_global * metrics[f"contrib_global_{status}"]
        return alignment + w_global * surrogate + w_struct * structural, metrics

    def _running_target(self, batch, positions, reduce: bool) -> EntropyStatistics:
        """Statistics whose coefficients use this batch's exact counts M and the running marginal r.

        M (valid vectors per position and group) depends only on the batch's validity masks, so it
        is known before any forward; S = r * M makes marginal() return r wherever M > 0.
        """
        categories = int(self.recipe["categories"])
        target = EntropyStatistics(self.groups, self.alignment_depths, positions, categories, self.device)
        for kind in KINDS:
            valid = batch[f"{kind}_valid"].to(self.device, non_blocking=True).float()
            group = self.slot_group[kind][batch[f"{kind}_slots"].to(self.device, non_blocking=True)]
            member = (group[None, :] == torch.arange(self.n_groups[kind], device=self.device)[:, None]).float()
            counts = member @ valid  # [G, N]
            if reduce and dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts)
            for depth in self.alignment_depths:
                for index, entropy_group in enumerate(target.groups_for(kind)):
                    key = (depth, entropy_group.name)
                    S, M = self.running_marginal["S"][key], self.running_marginal["M"][key]
                    r = torch.where(M[:, None] > 0, S / M.clamp_min(torch.finfo(M.dtype).tiny)[:, None],
                                    torch.full_like(S, 1.0 / categories))
                    target.M[key] = counts[index].clone()
                    target.S[key] = r * target.M[key][:, None]
        return target

    @torch.no_grad()
    def _update_running_marginal(self, stats: EntropyStatistics) -> None:
        """Fold this update's (all-reduced) statistics into the running marginal."""
        m = self.marginal_momentum
        if self.running_marginal is None or m == 0.0:
            self.running_marginal = {"S": {key: value.clone() for key, value in stats.S.items()},
                                     "M": {key: value.clone() for key, value in stats.M.items()}}
            return
        for name in ("S", "M"):
            for key, value in getattr(stats, name).items():
                self.running_marginal[name][key].mul_(m).add_(value, alpha=1.0 - m)

    def _running_reference(self, key):
        """Running marginal r [N, K] (uniform where it has no vectors) and where it has vectors [N]."""
        S, M = self.running_marginal["S"][key], self.running_marginal["M"][key]
        present = M > 0
        r = torch.where(present[:, None], S / M.clamp_min(torch.finfo(M.dtype).tiny)[:, None],
                        torch.full_like(S, 1.0 / int(self.recipe["categories"])))
        return r, present

    @torch.no_grad()
    def _marginal_diagnostics(self, stats: EntropyStatistics) -> dict[str, torch.Tensor]:
        """Per (depth, group), means over this batch's present positions (before r is updated):

        marginal_entropy_ema: H(r), the running marginal's entropy; H(r) - marginal_entropy (the
          batch's) estimates the small-batch entropy bias that "mixed" reduces (plus drift).
        marginal_lag_kl: KL(mu || r); marginal_step_kl: KL(mu || mu of the previous update).
          Sampling noise alone gives lag about half of step (step compares two batches); a lag
          well above that means r trails the model's drift, the risk of a stale marginal.
        mixed_d_rms: sqrt(sum_k mu (log mu_hat - log mu)^2), mu_hat at the configured batch_weight:
          how far "mixed" would move the collapse gradient's d.
        """
        tiny = torch.finfo(torch.float32).tiny
        metrics, previous = {}, {}
        for key in stats.S:
            depth, name = key
            mu, present = stats.marginal(key), stats.present(key)
            previous[key] = (mu.clone(), present.clone())
            weights = present.float()
            count = weights.sum().clamp_min(1)

            def mean(values):
                return (values * weights).sum() / count

            def kl(r):
                return (torch.special.xlogy(mu, mu) - torch.special.xlogy(mu, r.clamp_min(tiny))).sum(-1)

            if self.running_marginal is not None:
                r, r_present = self._running_reference(key)
                weights = weights * r_present.float()
                count = weights.sum().clamp_min(1)
                metrics[f"marginal_entropy_ema_d{depth}_{name}"] = mean(-torch.special.xlogy(r, r).sum(-1))
                metrics[f"marginal_lag_kl_d{depth}_{name}"] = mean(kl(r))
                w = self.marginal_batch_weight
                mixed = w * mu + (1.0 - w) * r
                shift = torch.where(mu > 0, mixed.clamp_min(tiny).log() - mu.clamp_min(tiny).log(), torch.zeros_like(mu))
                metrics[f"mixed_d_rms_d{depth}_{name}"] = mean((mu * shift.square()).sum(-1).sqrt())
            if self._previous_marginal is not None and key in self._previous_marginal:
                last, last_present = self._previous_marginal[key]
                weights = present.float() * last_present.float()
                count = weights.sum().clamp_min(1)
                metrics[f"marginal_step_kl_d{depth}_{name}"] = mean(kl(last))
        self._previous_marginal = previous
        return metrics

    def _marginal_kl(self, stats: EntropyStatistics) -> torch.Tensor:
        """Mean over keys of the mean over this batch's present positions of KL(mu || r), r the running marginal."""
        values = []
        for key in stats.S:
            mu, r = stats.marginal(key), self._running_reference(key)[0]
            kl = (torch.special.xlogy(mu, mu) - torch.special.xlogy(mu, r.clamp_min(torch.finfo(r.dtype).tiny))).sum(-1)
            present = stats.present(key).float()
            values.append((kl * present).sum() / present.sum().clamp_min(1))
        return torch.stack(values).mean()

    def _teacher(self, chunk, autocast):
        with torch.no_grad(), autocast():
            return self.model_module.teacher_forward(chunk["teacher_crops"])

    def _run(self, batch, step, *, autocast, sync_context, backward, training: bool) -> StepOutput:
        if not batch.get("jem"):
            raise ValueError("JEM needs batches from the JEM view-set processor")
        B = int(batch["batch_size"])
        if B != self.local_batch_size:
            raise ValueError(f"JEM expected {self.local_batch_size} sources on this rank, got {B}")
        updates = self.successful_updates
        w_global, w_struct = self.regularizer_weights(updates)
        self._last_weights = (w_global, w_struct)
        chunks = [torch.arange(start, min(start + self.source_chunk_size, B))
                  for start in range(0, B, self.source_chunk_size)]
        positions = {kind: math.prod(batch[f"{kind}_grid"]) for kind in KINDS}
        stats = EntropyStatistics(self.groups, self.alignment_depths, positions,
                                  int(self.recipe["categories"]), self.device)
        reduce_stats = self.recipe["entropy_rank_scope"] == "logical_batch"
        single = len(chunks) == 1
        # "running": one pass per chunk with d from the running marginal, for any number of
        # chunks (a single chunk would otherwise use this batch's marginal alone). "mixed": the
        # exact scheme with d from the batch/running mixture. Both start once r exists and
        # exact_updates successful updates have run.
        adaptive = (training and self.running_marginal is not None
                    and self.successful_updates >= self.marginal_exact_updates)
        running = adaptive and self.marginal_mode == "running"
        mixing = adaptive and self.marginal_mode == "mixed"
        target = self._running_target(batch, positions, reduce_stats) if running else None
        if running:
            coefficients = self._coefficients(target)
        rng_states = []
        if not single and not running:
            # Compiled, the statistics pass runs the gradient pass's graph (gradients on,
            # structural heads included): the inference graph fuses differently, and its BF16
            # logits would differ from the replay in the last bits. Its autograd graph is
            # dropped unused, so it calls the bare module rather than DDP.
            same_graph = self.compile_settings is not None and training
            with torch.enable_grad() if same_graph else torch.no_grad():
                for sources in chunks:
                    rng_states.append(self._rng_state())
                    chunk = self._chunk(batch, sources)
                    with autocast():
                        outputs = self._student(chunk, structural=same_graph, model=self.model_module)
                    self._accumulate(stats, outputs, chunk)
                    del outputs
            if reduce_stats:
                stats.all_reduce()
            coefficients = self._coefficients(stats, mixed=mixing)

        objective = None
        metrics: dict[str, torch.Tensor] = {}
        replay = (EntropyStatistics(self.groups, self.alignment_depths, positions, int(self.recipe["categories"]),
                                    self.device) if self.verify_replay and not single else None)
        for index, sources in enumerate(chunks):
            last = index == len(chunks) - 1
            chunk = self._chunk(batch, sources)
            context = sync_context(last) if training else nullcontext()
            with context:
                teacher = self._teacher(chunk, autocast)
                if rng_states:
                    self._set_rng_state(rng_states[index])
                with autocast():
                    outputs = self._student(chunk, structural=True)
                if running:
                    self._accumulate(stats, outputs, chunk)
                elif single:
                    self._accumulate(stats, outputs, chunk)
                    if reduce_stats:
                        stats.all_reduce()
                    coefficients = self._coefficients(stats, mixed=mixing)
                elif replay is not None:
                    self._accumulate(replay, outputs, chunk)
                loss, chunk_metrics = self._chunk_objective(outputs, teacher, chunk, coefficients, w_global, w_struct)
                if training:
                    backward(loss)
            del outputs, teacher
            for name, value in chunk_metrics.items():
                metrics[name] = metrics.get(name, 0) + value
            objective = loss.detach() if objective is None else objective + loss.detach()

        if running:
            if reduce_stats:
                stats.all_reduce()
        if running or mixing:
            metrics["marginal_lag_kl"] = self._marginal_kl(stats)  # r before this update's statistics
        if training:
            # Every mode keeps r (exact mode only for these diagnostics), so the evidence for
            # switching to "mixed" is logged before any switch.
            metrics.update(self._marginal_diagnostics(stats))
            self._update_running_marginal(stats)
        if replay is not None:
            if reduce_stats:
                replay.all_reduce()
            metrics["replay_max_abs_diff"] = max(
                float((table[key] - getattr(stats, name)[key]).abs().max())
                for name in ("S", "C", "M") for table in [getattr(replay, name)] for key in table)
        for name in [name for name in metrics if name.startswith("_kl_") and "_sum_" in name]:
            count = metrics.pop(name.replace("_sum_", "_count_"))
            metrics[name[1:].replace("_sum", "")] = metrics.pop(name) / torch.as_tensor(count).clamp_min(1)
        # True objective values from the reduced statistics.
        global_loss = torch.zeros((), device=self.device)
        for depth in self.alignment_depths:
            for group in self.groups:
                values = stats.group_loss((depth, group.name))
                contribution = self.depth_weights[depth] * group.weight * values["loss"]
                global_loss = global_loss + contribution
                metrics[f"global_d{depth}_{group.name}"] = values["loss"].detach()
                metrics[f"mi_d{depth}_{group.name}"] = -values["loss"].detach()
                metrics[f"marginal_entropy_d{depth}_{group.name}"] = values["marginal_entropy"]
                metrics[f"conditional_entropy_d{depth}_{group.name}"] = values["conditional_entropy"]
                metrics[f"perplexity_d{depth}_{group.name}"] = values["perplexity"]
                if depth == self.alignment_depths[-1]:
                    metrics[f"min_vectors_{group.name}"] = values["min_vectors"]
                    metrics[f"min_sources_{group.name}"] = values["min_sources"]
        alignment = sum(metrics[f"align_d{d}_{k}"] * self.depth_weights[d] for d in self.alignment_depths
                        for k in KINDS)
        structural = sum(metrics.get(f"struct_d{d}_{k}", torch.zeros((), device=self.device))
                         for d in self.structural_depths for k in KINDS) / len(self.structural_depths)
        loss = alignment + w_global * global_loss + w_struct * structural
        metrics.update({
            "alignment_loss": alignment, "global_collapse_loss": global_loss, "structural_loss": structural,
            "alignment_weighted": alignment, "global_collapse_weighted": w_global * global_loss,
            "structural_weighted": w_struct * structural,
            "surrogate_value": objective,  # gradient carrier only; not an objective value
        })
        for name, values in batch.get("plan_stats", {}).items():
            metrics[f"plan_{name}"] = values.float().mean().to(self.device)
        metrics = {name: torch.as_tensor(value, device=self.device).float().reshape(()) for name, value in metrics.items()}
        return StepOutput(loss.detach(), metrics)

    # ------------------------------------------------------------------ TrainingMethod

    def forward(self, batch, step: int) -> StepOutput:
        raise RuntimeError("JEM backpropagates inside forward_backward (several passes per update)")

    def forward_backward(self, batch, step: int, *, autocast, sync_context, backward) -> StepOutput:
        return self._run(batch, step, autocast=autocast, sync_context=sync_context, backward=backward, training=True)

    def evaluate(self, batch, step: int) -> Mapping[str, float]:
        def autocast():
            return torch.autocast(self.device.type, dtype=self.input_dtype,
                                  enabled=self.input_dtype != torch.float32)
        output = self._run(batch, step, autocast=autocast, sync_context=lambda sync=True: nullcontext(),
                           backward=None, training=False)
        return {name: float(value) for name, value in {"loss": output.loss, **output.metrics}.items()}

    def parameter_groups(self, **kwargs):
        return self.model_module.get_params_groups(**kwargs)

    def gradient_parameters(self):
        return self.model_module.student.parameters()

    def after_optimizer_step(self, step: int) -> None:
        self.model_module.update_teacher(float(self.recipe["momentum_teacher"]))
        self.successful_updates += 1

    def optimizer_step_skipped(self, step: int) -> None:
        self.skipped_updates += 1

    def step_metrics(self, step: int) -> dict[str, float]:
        w_global, w_struct = self._last_weights
        return {"w_global": w_global, "w_struct": w_struct, "momentum_teacher": float(self.recipe["momentum_teacher"]),
                "successful_updates": float(self.successful_updates), "skipped_updates": float(self.skipped_updates)}

    def evaluation_backbone(self):
        return self.model_module.teacher.backbone

    def dataset_config(self, config, *, training: bool):
        resolved = dict(config)
        for key in ("object_points", "point_supervision", "gram_teacher_crop_size", "cross_view_token_patch_size"):
            if resolved.get(key):
                raise ValueError(f"dataset.{key} is not used by JEM (auxiliaries are disabled)")
            resolved.pop(key, None)
        views = dict(self.recipe["views"])
        resolved["jem"] = {**views, "patch": list(self.patch)}
        resolved["source_sampling_size"] = list(zyx(views["source_size"], "source_size"))
        resolved["global_crop_size"] = list(zyx(views["global_size"], "global_size"))
        resolved["local_crop_size"] = list(zyx(views["local_size"], "local_size"))
        resolved.setdefault("source_rotation", "none")
        return resolved

    def build_processor(self, dataset, device):
        from dinovol_2.dataset.jem_pipeline import JemBatchProcessor
        return JemBatchProcessor(
            dataset.config, device=device, patch=self.patch, masking=self.recipe["masking"],
            lattice=self.recipe["structural_pair_weighting"] == "distance_stratified",
            storage=self.recipe["batch_storage"], dtype=self.input_dtype)

    def monitor_subset(self, batch, count: int):
        return batch

    def save_monitor_image(self, batch, step, metrics, *, directory, config):
        from dinovol_2.training.jem_monitor import save_jem_monitor_image
        return save_jem_monitor_image(self, batch, step, metrics, directory=directory)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "student": self.model_module.student.state_dict(),
            "teacher": self.model_module.teacher.state_dict(),
            "recipe": self.resolved_recipe(),
            "successful_updates": self.successful_updates,
            "skipped_updates": self.skipped_updates,
            "running_marginal": self.running_marginal,
        }

    def load_checkpoint_state(self, state: Mapping[str, Any], step: int):
        saved = dict(state.get("recipe") or {})
        for key in ("categories", "alignment_depths", "structural_depths", "entropy_view_scope",
                    "entropy_rank_scope", "structural_pair_weighting", "distance_bands", "last_layer_type"):
            if key in saved and saved[key] != self.recipe[key]:
                raise ValueError(f"Checkpoint JEM {key}={saved[key]!r} differs from the configured {self.recipe[key]!r}")
        self.model_module.student.load_state_dict(state["student"], strict=True)
        self.model_module.teacher.load_state_dict(state["teacher"], strict=True)
        self.successful_updates = int(state.get("successful_updates", step + 1))
        self.skipped_updates = int(state.get("skipped_updates", 0))
        running = state.get("running_marginal")
        self.running_marginal = None if running is None else {
            name: {tuple(key): value.to(self.device) for key, value in table.items()} for name, table in running.items()}
        return ()
