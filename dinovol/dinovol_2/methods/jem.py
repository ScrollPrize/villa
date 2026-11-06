"""JEM: per-depth categorical alignment, logical-batch global collapse, structural collapse.

One optimizer update uses the configured global batch of independent sources. Each
rank processes its sources in chunks. The global-collapse marginal of every depth,
entropy group and position is formed over the whole logical batch:

* One chunk: forward once, reduce detached statistics across ranks, backpropagate.
* Several chunks: pass 1 runs the student without gradients on every chunk to
  collect statistics (RNG state saved per chunk); after the cross-rank reduction,
  pass 2 replays each chunk with the same RNG state (drop-path), adds the teacher,
  and backpropagates its share using the fixed marginal. Only the last chunk's
  backward synchronizes DDP gradients.

The surrogate W/M * (C + S . stop_gradient(log mu + 1)) gives, after DDP's gradient
average, the exact first-order gradient of the full-batch objective; the logged
global-collapse value is the true objective computed from the reduced statistics.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from dinovol_2.loss.jem_losses import (
    EntropyStatistics, alignment_from_means, distance_metric, entropy_attribution, entropy_groups,
    entropy_surrogate, fp32_reductions, global_collapse_weight, kl_divergence, nonaffine_target,
    structural_band_counts, structural_distance_stratified, structural_uniform_contraction,
    structural_uniform_tiled, structural_weight, subset_means,
)
from dinovol_2.model.jem import JemStudentTeacher
from dinovol_2.ops.precision import resolve_amp_dtype
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
        compile_settings = runtime.get("compile", False)
        if compile_settings is True or (isinstance(compile_settings, Mapping) and compile_settings.get("enabled")):
            raise ValueError("torch.compile is not wired up for JEM yet")

        self.patch = int(self.backbone_config["patch_size"][0])
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
        self.successful_updates = 0
        self.skipped_updates = 0
        # Diagnostic: re-accumulate statistics in pass 2 and report the largest
        # difference from pass 1 (drop-path/RNG replay check).
        self.verify_replay = False
        self._last_weights = (1.0, 0.0)
        self.input_dtype = (resolve_amp_dtype(runtime.get("amp_dtype", "bfloat16"))
                            if bool(runtime.get("use_amp", self.device.type == "cuda")) else torch.float32)

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

    def _chunk(self, batch: Mapping[str, Any], sources: torch.Tensor) -> dict[str, Any]:
        """Rows of the given local source indices, with chunk-local teacher rows."""
        B = int(batch["batch_size"])
        n = len(sources)
        position = torch.full((B,), -1, dtype=torch.long)
        position[sources] = torch.arange(n)
        teacher_rows = torch.cat([s * B + sources for s in range(int(batch["n_sets"]))])
        chunk = {"n_sources": n,
                 "teacher_crops": batch["teacher_crops"][teacher_rows].to(self.device, self.input_dtype,
                                                                          non_blocking=True)}
        for kind in KINDS:
            slots = int(batch[f"n_{kind}_views"])
            rows = torch.cat([v * B + sources for v in range(slots)])
            row_sources = batch[f"{kind}_sources"][rows]
            per_set = slots // int(batch["n_sets"])
            set_index = batch[f"{kind}_slots"][rows] // per_set
            entry = {
                "crops": batch[f"{kind}_crops"][rows].to(self.device, self.input_dtype, non_blocking=True),
                "masks": batch[f"{kind}_masks"][rows.to(batch[f"{kind}_masks"].device)].to(self.device),
                "targets": batch[f"{kind}_targets"][rows.to(batch[f"{kind}_targets"].device)].to(self.device),
                "valid": batch[f"{kind}_valid"][rows.to(batch[f"{kind}_valid"].device)].to(self.device),
                "slots": batch[f"{kind}_slots"][rows].to(self.device),
                "sources": row_sources.to(self.device),
                "teacher_rows": (set_index * n + position[row_sources]).to(self.device),
                "grid": tuple(batch[f"{kind}_grid"]),
                "mask_applied": batch[f"{kind}_mask_applied"][rows],
                "spacing": batch["spacing"][row_sources],
            }
            if f"{kind}_lattice" in batch:
                entry["lattice"] = batch[f"{kind}_lattice"][rows.to(batch[f"{kind}_lattice"].device)].to(self.device)
            chunk[kind] = entry
        return chunk

    def _student(self, chunk, *, structural: bool):
        g, l = chunk["global"], chunk["local"]
        return self.model(g["crops"], g["masks"].view(len(g["crops"]), -1), l["crops"],
                          l["masks"].view(len(l["crops"]), -1), structural=structural)

    # ------------------------------------------------------------------ objective

    def _log_q(self, logits):
        return F.log_softmax(logits.float() / float(self.recipe["student_temperature"]), dim=-1)

    def _accumulate(self, stats: EntropyStatistics, outputs, chunk) -> None:
        with fp32_reductions(self.device.type), torch.no_grad():
            for kind in KINDS:
                entry = chunk[kind]
                for depth in self.alignment_depths:
                    stats.accumulate(depth, kind, self._log_q(outputs[kind]["logits"][depth].detach()),
                                     entry["valid"], entry["slots"], entry["sources"])

    def _categorical_terms(self, logits, teacher_logits, entry, coefs):
        """Per-row alignment [rows] and the entropy surrogate (scalar) for one depth and crop type.

        coefs: list of (row selector, coefficient [N], d [N, K]) per entropy group.
        """
        log_q = self._log_q(logits)
        rows = log_q.shape[0]
        overlap = entry["targets"] >= 0
        flat_rows, flat_positions = overlap.nonzero(as_tuple=True)
        teacher_flat = entry["teacher_rows"][flat_rows] * teacher_logits.shape[1] + entry["targets"][
            flat_rows, flat_positions]
        kl = kl_divergence(teacher_logits.reshape(-1, teacher_logits.shape[-1])[teacher_flat],
                           log_q[flat_rows, flat_positions], float(self.recipe["teacher_temperature"]))
        visible, masked, visible_count, masked_count = subset_means(
            kl, flat_rows, entry["masks"][flat_rows, flat_positions], rows)
        alignment = alignment_from_means(visible, masked, visible_weight=float(self.recipe["visible_weight"]),
                                         masked_weight=float(self.recipe["masked_weight"]))
        surrogate = log_q.sum() * 0
        for selector, coef, d in coefs:
            surrogate = surrogate + entropy_surrogate(log_q[selector], entry["valid"][selector], coef, d)
        return alignment, surrogate, visible, masked, visible_count, masked_count

    def _structural(self, a, targets, entry, kind, tile):
        """Per-row structural loss [rows] for one target depth."""
        losses, extras = [], {}
        overlap = entry["targets"] >= 0
        weighting = self.recipe["structural_pair_weighting"]
        for row in range(a.shape[0]):
            selected = overlap[row].nonzero(as_tuple=True)[0]
            student = a[row, selected].float()
            student = F.normalize(student, dim=-1)
            teacher = targets[entry["teacher_rows"][row], entry["targets"][row, selected]]
            if weighting == "uniform":
                if self.recipe["structural_backend"] == "contraction":
                    losses.append(structural_uniform_contraction(student, teacher))
                else:
                    losses.append(structural_uniform_tiled(student, teacher, tile))
            else:
                coords = entry["lattice"][row, selected]
                metric, _ = distance_metric(entry["spacing"][row].tolist(), (self.patch,) * 3)
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

    def _chunk_objective(self, outputs, teacher, chunk, stats: EntropyStatistics, w_global, w_struct, training):
        """Graph-connected chunk loss (alignment + scheduled surrogate + structural) and detached metrics."""
        n_views = self.views_per_source * self.local_batch_size
        surrogate_scale = float(self.world_size) if self.recipe["entropy_rank_scope"] == "logical_batch" else 1.0
        n_struct = len(self.structural_depths)
        tile = int(self.recipe["structural_tile"])
        total = None
        metrics: dict[str, torch.Tensor] = {}

        def add(name, value):
            value = value.detach().float()
            metrics[name] = metrics.get(name, 0) + value

        with fp32_reductions(self.device.type):
            teacher_targets = {depth: nonaffine_target(teacher["structural"][depth]) for depth in self.structural_depths}
            teacher_logits = {depth: teacher["logits"][depth].float() for depth in self.alignment_depths}
            for kind in KINDS:
                entry = chunk[kind]
                rows = entry["crops"].shape[0]
                masked_view = entry["masks"].any(-1)
                row_alignment = torch.zeros(rows, device=self.device)
                row_entropy = torch.zeros(rows, device=self.device)
                for depth in self.alignment_depths:
                    weight = self.depth_weights[depth]
                    coefs, attributions = [], []
                    for group in stats.groups_for(kind):
                        selector = torch.isin(entry["slots"], entry["slots"].new_tensor(group.slots))
                        coef, d, log_mu = stats.coefficients((depth, group.name),
                                                             surrogate_scale * weight * group.weight)
                        coefs.append((selector, coef, d))
                        true_coef, _, _ = stats.coefficients((depth, group.name), weight * group.weight)
                        attributions.append((selector, true_coef, log_mu))
                    logits = outputs[kind]["logits"][depth]
                    function = self._categorical_terms
                    if self.recipe["checkpoint_losses"] and torch.is_grad_enabled():
                        alignment, surrogate, visible, masked, n_visible, n_masked = checkpoint(
                            function, logits, teacher_logits[depth], entry, coefs, use_reentrant=False)
                    else:
                        alignment, surrogate, visible, masked, n_visible, n_masked = function(
                            logits, teacher_logits[depth], entry, coefs)
                    depth_alignment = weight * alignment.sum() / n_views
                    term = depth_alignment + w_global * surrogate
                    total = term if total is None else total + term
                    row_alignment = row_alignment + weight * alignment.detach()
                    with torch.no_grad():
                        log_q = self._log_q(logits.detach())
                        for selector, true_coef, log_mu in attributions:
                            share = entropy_attribution(log_q[selector], entry["valid"][selector], true_coef, log_mu)
                            row_entropy[selector] += share
                    add(f"align_d{depth}_{kind}", alignment.detach().sum() / n_views)
                    add(f"_kl_visible_sum_d{depth}_{kind}", (visible * n_visible).sum())
                    add(f"_kl_visible_count_d{depth}_{kind}", n_visible.sum())
                    add(f"_kl_masked_sum_d{depth}_{kind}", (masked * n_masked).sum())
                    add(f"_kl_masked_count_d{depth}_{kind}", n_masked.sum())
                row_structural = torch.zeros(rows, device=self.device)
                for depth, head_output in outputs[kind].get("structural", {}).items():
                    structural = self._structural(head_output, teacher_targets[depth], entry, kind, tile)
                    total = total + w_struct * structural.sum() / (n_struct * n_views)
                    row_structural = row_structural + structural.detach() / n_struct
                    add(f"struct_d{depth}_{kind}", structural.detach().sum() / n_views)
                # Whole-view mask-status attribution with the training denominators.
                for status, selector in (("masked", masked_view), ("unmasked", ~masked_view)):
                    add(f"contrib_align_{status}", row_alignment[selector].sum() / n_views)
                    add(f"contrib_struct_{status}", w_struct * row_structural[selector].sum() / n_views)
                    # Entropy shares sum to R over the logical batch; scale so the trainer's
                    # cross-rank mean reports the global sum.
                    add(f"contrib_global_{status}", w_global * row_entropy[selector].sum() * self.world_size)
                    add(f"views_{status}_{kind}", selector.float().sum() / self.local_batch_size)
                # Means over this rank's rows of the type (chunks add up to the full batch).
                type_rows = self.local_batch_size * (2 if kind == "global" else 2 * self.locals_per_set)
                add(f"overlap_fraction_{kind}", (entry["targets"] >= 0).float().mean(-1).sum() / type_rows)
                add(f"valid_fraction_{kind}", entry["valid"].float().mean(-1).sum() / type_rows)
                add(f"mask_fraction_{kind}", entry["masks"].float().mean(-1).sum() / type_rows)
                add(f"masked_overlap_fraction_{kind}",
                    (entry["masks"] & (entry["targets"] >= 0)).float().mean(-1).sum() / type_rows)
        return total, metrics

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
        rng_states = []
        if not single:
            with torch.no_grad():
                for sources in chunks:
                    rng_states.append(self._rng_state())
                    chunk = self._chunk(batch, sources)
                    with autocast():
                        outputs = self._student(chunk, structural=False)
                    self._accumulate(stats, outputs, chunk)
                    del outputs
            if reduce_stats:
                stats.all_reduce()

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
                if not single:
                    self._set_rng_state(rng_states[index])
                with autocast():
                    outputs = self._student(chunk, structural=True)
                if single:
                    self._accumulate(stats, outputs, chunk)
                    if reduce_stats:
                        stats.all_reduce()
                elif replay is not None:
                    self._accumulate(replay, outputs, chunk)
                loss, chunk_metrics = self._chunk_objective(outputs, teacher, chunk, stats, w_global, w_struct,
                                                            training)
                if training:
                    backward(loss)
            del outputs, teacher
            for name, value in chunk_metrics.items():
                metrics[name] = metrics.get(name, 0) + value
            objective = loss.detach() if objective is None else objective + loss.detach()

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
        resolved["jem"] = {**views, "patch": self.patch}
        resolved["source_sampling_size"] = [int(views["source_size"])] * 3
        resolved["global_crop_size"] = [int(views["global_size"])] * 3
        resolved["local_crop_size"] = [int(views["local_size"])] * 3
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
        return ()
