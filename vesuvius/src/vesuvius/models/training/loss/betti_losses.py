"""
Betti Matching losses for topologically accurate segmentation.
from https://github.com/nstucki/Betti-matching
Uses the C++ implementation of Betti matching with Python bindings found here https://github.com/nstucki/Betti-Matching-3D

This implementation mirrors the example Betti matching loss in scratch/loss_function.py:
- Uses matched birth/death value differences (squared, with factor 2)
- Penalizes unmatched pairs by pushing to the diagonal (default) or other strategies
- Supports 'sublevel', 'superlevel', and 'bothlevel' filtrations

To install and use this loss, make sure you run the build_betti.py script in vesuvius/utils
"""

import importlib.util
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_betti_module() -> object:
    """Load the compiled betti_matching extension without mutating sys.path."""

    vesuvius_src_path = Path(__file__).resolve().parents[5]

    candidate_build_dirs = [
        vesuvius_src_path / "external" / "Betti-Matching-3D" / "build",
        vesuvius_src_path / "src" / "external" / "Betti-Matching-3D" / "build",
        vesuvius_src_path / "scratch" / "Betti-Matching-3D" / "build",
    ]

    betti_build_path = next((p for p in candidate_build_dirs if p.exists()), None)

    if betti_build_path is None:
        raise ImportError(
            "Betti-Matching-3D build directory not found. Checked:\n"
            + "\n".join(f"  - {p}" for p in candidate_build_dirs)
            + "\nPlease run the build_betti.py script in vesuvius/utils/. "
            "You may need to force the system libstdc++ with:\n"
            "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
        )

    candidates = []
    for pattern in ("betti_matching*.so", "betti_matching*.pyd", "betti_matching*.dll"):
        candidates.extend(betti_build_path.rglob(pattern))

    if not candidates:
        raise ImportError(
            f"betti_matching extension not found under {betti_build_path}. "
            "Please rerun build_betti.py and ensure the extension compiles."
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("betti_matching", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load betti_matching from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    bm = _load_betti_module()
except ImportError as e:
    raise ImportError(
        "Could not import betti_matching module. "
        "Please run the build_betti.py script in vesuvius/utils/.  "
        "This will clone and build Betti-Matching-3D automatically. "
        "You may need to force the system libstdc++ with:\n"
        "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6\n"
        f"Error: {e}"
    ) from e


def _grid_slices(shape: Tuple[int, ...], chunk: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
    """Tile `shape` with axis-aligned chunks of size `chunk`, allowing remainder tiles."""
    if len(shape) != len(chunk):
        raise ValueError(f"chunk dimensionality {len(chunk)} does not match input shape {len(shape)}")
    ranges = [range(0, shape[d], max(1, chunk[d])) for d in range(len(shape))]
    slices: List[Tuple[slice, ...]] = []
    for starts in product(*ranges):
        dims = []
        for dim, start in enumerate(starts):
            stop = min(start + max(1, chunk[dim]), shape[dim])
            dims.append(slice(start, stop))
        slices.append(tuple(dims))
    return slices


def _normalize_triplet(value: Optional[object], dims: int) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, int):
        return tuple([int(value)] * dims)
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(f"Expected {dims} values, got {value}")
        return tuple(int(v) for v in value)
    raise ValueError(f"Invalid chunk parameter {value!r}")


@dataclass
class _PairAggregator:
    matched: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, ...], Tuple[int, ...]]]
    unmatched_pred: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], None]
    unmatched_target: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], None]
    dim_nd: Optional[int] = None

    @staticmethod
    def create() -> "_PairAggregator":
        return _PairAggregator(matched={}, unmatched_pred={}, unmatched_target={}, dim_nd=None)

    def add_chunk_result(
        self,
        res,
        dim_nd: int,
        offset: Tuple[int, ...],
        interior_box_ext: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> None:
        if self.dim_nd is None:
            self.dim_nd = dim_nd

        def _offset_coords(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            return arr + np.asarray(offset, dtype=arr.dtype)

        def _inside(arr: np.ndarray, box: Tuple[Tuple[int, int], ...]) -> np.ndarray:
            if arr.size == 0:
                return np.zeros((0,), dtype=bool)
            keep = np.ones((arr.shape[0],), dtype=bool)
            for axis, (start, end) in enumerate(box):
                keep &= (arr[:, axis] >= start) & (arr[:, axis] < end)
            return keep

        for d in range(dim_nd):
            m1b = res.input1_matched_birth_coordinates[d]
            m1d = res.input1_matched_death_coordinates[d]
            m2b = res.input2_matched_birth_coordinates[d]
            m2d = res.input2_matched_death_coordinates[d]
            if len(m1b) == 0:
                continue

            A_b = np.ascontiguousarray(m1b)
            A_d = np.ascontiguousarray(m1d)
            B_b = np.ascontiguousarray(m2b)
            B_d = np.ascontiguousarray(m2d)

            if interior_box_ext is not None:
                keep = _inside(A_b, interior_box_ext) & _inside(A_d, interior_box_ext)
                keep &= _inside(B_b, interior_box_ext) & _inside(B_d, interior_box_ext)
                if keep.size and keep.sum() < keep.size:
                    A_b = A_b[keep]
                    A_d = A_d[keep]
                    B_b = B_b[keep]
                    B_d = B_d[keep]

            A_b = _offset_coords(A_b)
            A_d = _offset_coords(A_d)
            B_b = _offset_coords(B_b)
            B_d = _offset_coords(B_d)

            for i in range(A_b.shape[0]):
                key = (d, tuple(A_b[i].tolist()), tuple(A_d[i].tolist()))
                if key not in self.matched:
                    self.matched[key] = (tuple(B_b[i].tolist()), tuple(B_d[i].tolist()))

            if res.input1_unmatched_birth_coordinates is not None:
                u1b = res.input1_unmatched_birth_coordinates[d]
                u1d = res.input1_unmatched_death_coordinates[d]
                U_b = np.ascontiguousarray(u1b)
                U_d = np.ascontiguousarray(u1d)
                if interior_box_ext is not None:
                    keep = _inside(U_b, interior_box_ext) & _inside(U_d, interior_box_ext)
                    if keep.size and keep.sum() < keep.size:
                        U_b = U_b[keep]
                        U_d = U_d[keep]
                U_b = _offset_coords(U_b)
                U_d = _offset_coords(U_d)
                for i in range(U_b.shape[0]):
                    key = (d, tuple(U_b[i].tolist()), tuple(U_d[i].tolist()))
                    self.unmatched_pred[key] = None

            if res.input2_unmatched_birth_coordinates is not None:
                u2b = res.input2_unmatched_birth_coordinates[d]
                u2d = res.input2_unmatched_death_coordinates[d]
                V_b = np.ascontiguousarray(u2b)
                V_d = np.ascontiguousarray(u2d)
                if interior_box_ext is not None:
                    keep = _inside(V_b, interior_box_ext) & _inside(V_d, interior_box_ext)
                    if keep.size and keep.sum() < keep.size:
                        V_b = V_b[keep]
                        V_d = V_d[keep]
                V_b = _offset_coords(V_b)
                V_d = _offset_coords(V_d)
                for i in range(V_b.shape[0]):
                    key = (d, tuple(V_b[i].tolist()), tuple(V_d[i].tolist()))
                    self.unmatched_target[key] = None

    def export_numpy(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if not self.matched:
            dim = self.dim_nd or 3
            mb = md = tb = td = np.zeros((0, dim), dtype=np.int64)
        else:
            keys = list(self.matched.keys())
            dim = len(keys[0][1])
            mb = np.zeros((len(keys), dim), dtype=np.int64)
            md = np.zeros((len(keys), dim), dtype=np.int64)
            tb = np.zeros((len(keys), dim), dtype=np.int64)
            td = np.zeros((len(keys), dim), dtype=np.int64)
            for i, key in enumerate(keys):
                mb[i] = np.array(key[1], dtype=np.int64)
                md[i] = np.array(key[2], dtype=np.int64)
                tb[i] = np.array(self.matched[key][0], dtype=np.int64)
                td[i] = np.array(self.matched[key][1], dtype=np.int64)

        if not self.unmatched_pred:
            ub = ud = None
        else:
            keys = list(self.unmatched_pred.keys())
            dim = len(keys[0][1])
            ub = np.zeros((len(keys), dim), dtype=np.int64)
            ud = np.zeros((len(keys), dim), dtype=np.int64)
            for i, key in enumerate(keys):
                ub[i] = np.array(key[1], dtype=np.int64)
                ud[i] = np.array(key[2], dtype=np.int64)

        if not self.unmatched_target:
            vb = vd = None
        else:
            keys = list(self.unmatched_target.keys())
            dim = len(keys[0][1])
            vb = np.zeros((len(keys), dim), dtype=np.int64)
            vd = np.zeros((len(keys), dim), dtype=np.int64)
            for i, key in enumerate(keys):
                vb[i] = np.array(key[1], dtype=np.int64)
                vd[i] = np.array(key[2], dtype=np.int64)

        return mb, md, tb, td, ub, ud, vb, vd


def _tensor_values_at_coords(t: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    """Index values from tensor t at integer voxel coordinates coords (N, D)."""
    if coords.size == 0:
        return t.new_zeros((0,), dtype=t.dtype)
    idx = torch.as_tensor(coords, device=t.device, dtype=torch.long)
    return t[tuple(idx[:, d] for d in range(idx.shape[1]))]


def _stack_pairs(values_birth: torch.Tensor, values_death: torch.Tensor) -> torch.Tensor:
    if values_birth.numel() == 0:
        return values_birth.new_zeros((0, 2))
    return torch.stack([values_birth, values_death], dim=1)


def _loss_unmatched(pairs: torch.Tensor, push_to: str = "diagonal") -> torch.Tensor:
    """Compute unmatched loss given (N, 2) pairs of (birth, death) values.
    push_to:
      - 'diagonal': sum((birth - death)^2)
      - 'one_zero': 2 * sum((birth - 1)^2 + death^2)
      - 'death_death': 2 * sum((birth - death)^2)
    """
    if pairs.numel() == 0:
        return pairs.new_zeros(())
    if push_to == "diagonal":
        return ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    elif push_to == "one_zero":
        return 2.0 * (((pairs[:, 0] - 1.0) ** 2) + (pairs[:, 1] ** 2)).sum()
    elif push_to == "death_death":
        return 2.0 * ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()
    else:
        # default to diagonal if unknown
        return ((pairs[:, 0] - pairs[:, 1]) ** 2).sum()


def _loss_from_coords_torch(
    pred_field: torch.Tensor,
    tgt_field: torch.Tensor,
    matched_pred_birth: np.ndarray,
    matched_pred_death: np.ndarray,
    matched_tgt_birth: np.ndarray,
    matched_tgt_death: np.ndarray,
    unmatched_pred_birth: Optional[np.ndarray],
    unmatched_pred_death: Optional[np.ndarray],
    unmatched_tgt_birth: Optional[np.ndarray],
    unmatched_tgt_death: Optional[np.ndarray],
    *,
    include_unmatched_target: bool,
    push_to: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute loss directly from aggregated coordinate arrays."""
    # Matched component
    if matched_pred_birth.size == 0:
        loss_matched = pred_field.new_zeros(())
    else:
        pred_birth_vals = _tensor_values_at_coords(pred_field, matched_pred_birth)
        pred_death_vals = _tensor_values_at_coords(pred_field, matched_pred_death)
        tgt_birth_vals = _tensor_values_at_coords(tgt_field, matched_tgt_birth)
        tgt_death_vals = _tensor_values_at_coords(tgt_field, matched_tgt_death)
        pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
        tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
        loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    # Unmatched prediction component
    if unmatched_pred_birth is None or unmatched_pred_birth.size == 0:
        loss_unmatched_pred = pred_field.new_zeros(())
    else:
        pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, unmatched_pred_birth)
        pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, unmatched_pred_death)
        pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
        loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred

    loss_unmatched_tgt = pred_field.new_zeros(())
    if include_unmatched_target:
        if unmatched_tgt_birth is not None and unmatched_tgt_birth.size > 0:
            tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, unmatched_tgt_birth)
            tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, unmatched_tgt_death)
            tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
            loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
            total = total + loss_unmatched_tgt

    aux: Dict[str, torch.Tensor] = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["Betti matching loss (unmatched target)"] = loss_unmatched_tgt.reshape(1).detach()
    return total, aux


def _compute_loss_from_result(pred_field: torch.Tensor,
                              tgt_field: torch.Tensor,
                              res,
                              *,
                              include_unmatched_target: bool,
                              push_to: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mirror of scratch/_betti_matching_loss using our binding attribute names."""
    # Matched coordinates (concatenate across homology dimensions)
    def _concat(list_of_arrays):
        # Flatten potential nested lists of arrays across homology dimensions
        flat: list[np.ndarray] = []
        if list_of_arrays is not None:
            for a in list_of_arrays:
                if a is None:
                    continue
                if isinstance(a, (list, tuple)):
                    for b in a:
                        if isinstance(b, np.ndarray) and b.size > 0:
                            flat.append(b)
                elif isinstance(a, np.ndarray):
                    if a.size > 0:
                        flat.append(a)
        if len(flat) == 0:
            # infer ndim from fields
            ndim = pred_field.ndim
            return np.zeros((0, ndim), dtype=np.int64)
        return np.ascontiguousarray(np.concatenate(flat, axis=0))

    pred_birth_coords = _concat(res.input1_matched_birth_coordinates)
    pred_death_coords = _concat(res.input1_matched_death_coordinates)
    tgt_birth_coords = _concat(res.input2_matched_birth_coordinates)
    tgt_death_coords = _concat(res.input2_matched_death_coordinates)

    pred_unmatched_birth = _concat(res.input1_unmatched_birth_coordinates)
    pred_unmatched_death = _concat(res.input1_unmatched_death_coordinates)
    tgt_unmatched_birth = _concat(res.input2_unmatched_birth_coordinates)
    tgt_unmatched_death = _concat(res.input2_unmatched_death_coordinates)
    # Gather values from tensors at those coordinates
    pred_birth_vals = _tensor_values_at_coords(pred_field, pred_birth_coords)
    pred_death_vals = _tensor_values_at_coords(pred_field, pred_death_coords)
    tgt_birth_vals = _tensor_values_at_coords(tgt_field, tgt_birth_coords)
    tgt_death_vals = _tensor_values_at_coords(tgt_field, tgt_death_coords)

    pred_unmatched_birth_vals = _tensor_values_at_coords(pred_field, pred_unmatched_birth)
    pred_unmatched_death_vals = _tensor_values_at_coords(pred_field, pred_unmatched_death)

    # Matched loss: 2 * sum((birth_pred - birth_tgt)^2 + (death_pred - death_tgt)^2)
    pred_matched_pairs = _stack_pairs(pred_birth_vals, pred_death_vals)
    tgt_matched_pairs = _stack_pairs(tgt_birth_vals, tgt_death_vals)
    loss_matched = 2.0 * ((pred_matched_pairs - tgt_matched_pairs) ** 2).sum()

    # Unmatched pairs (default push to diagonal)
    pred_unmatched_pairs = _stack_pairs(pred_unmatched_birth_vals, pred_unmatched_death_vals)
    loss_unmatched_pred = _loss_unmatched(pred_unmatched_pairs, push_to=push_to)

    total = loss_matched + loss_unmatched_pred

    loss_unmatched_tgt = pred_field.new_zeros(())
    if include_unmatched_target and tgt_unmatched_birth.size > 0:
        tgt_unmatched_birth_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_birth)
        tgt_unmatched_death_vals = _tensor_values_at_coords(tgt_field, tgt_unmatched_death)
        tgt_unmatched_pairs = _stack_pairs(tgt_unmatched_birth_vals, tgt_unmatched_death_vals)
        loss_unmatched_tgt = _loss_unmatched(tgt_unmatched_pairs, push_to=push_to)
        total = total + loss_unmatched_tgt

    # Auxiliary stats (detached)
    aux = {
        "Betti matching loss (matched)": loss_matched.reshape(1).detach(),
        "Betti matching loss (unmatched prediction)": loss_unmatched_pred.reshape(1).detach(),
    }
    if include_unmatched_target:
        aux["Betti matching loss (unmatched target)"] = loss_unmatched_tgt.reshape(1).detach()
    return total.reshape(1), aux


class BettiMatchingLoss(nn.Module):
    """
    Betti matching loss for topological accuracy. See https://github.com/nstucki/Betti-matching for details

    Filtration is implemented via preprocessing (inverting values for superlevel filtration).

    Parameters:
    - filtration: 'superlevel' | 'sublevel' | 'bothlevel'
    - include_unmatched_target: whether to penalize unmatched target pairs
    - push_unmatched_to: 'diagonal' | 'one_zero' | 'death_death'
    - chunk_mode: optional sampling mode (currently supports 'grid') for tiling the volume before matching
    - chunk_size: chunk interior size per axis when chunk_mode is used
    - halo: optional halo voxels around each chunk for matching overlap
    """

    def __init__(self,
                 filtration: str = 'superlevel',
                 include_unmatched_target: bool = False,
                 push_unmatched_to: str = 'diagonal',
                 *,
                 chunk_mode: Optional[str] = None,
                 chunk_size: Optional[Tuple[int, int, int]] = None,
                 halo: Optional[Tuple[int, int, int]] = None,
                 use_chunking: bool = False):
        super().__init__()
        assert filtration in ('superlevel', 'sublevel', 'bothlevel'), "filtration must be one of: superlevel, sublevel, bothlevel"
        assert push_unmatched_to in ('diagonal', 'one_zero', 'death_death'), "push_unmatched_to must be one of: diagonal, one_zero, death_death"
        self.filtration = filtration
        self.include_unmatched_target = include_unmatched_target
        self.push_unmatched_to = push_unmatched_to
        if chunk_mode is None and use_chunking:
            chunk_mode = "grid"
        if chunk_mode is not None and chunk_mode != "grid":
            raise ValueError(f"Unsupported chunk_mode {chunk_mode!r}; only 'grid' is available.")
        if chunk_mode is not None and chunk_size is None:
            raise ValueError("chunk_size must be provided when chunk_mode is enabled.")
        self.chunk_mode = chunk_mode
        self._chunk_size_param = chunk_size
        self._chunk_halo_param = halo

    def _get_chunk_params(self, ndim: int) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        if self._chunk_size_param is None:
            raise ValueError("chunk_size must be specified when chunking is enabled.")
        chunk_size = _normalize_triplet(self._chunk_size_param, ndim)
        if any(c <= 0 for c in chunk_size):
            raise ValueError(f"chunk_size must be positive along all dimensions, got {chunk_size}")
        chunk_halo = _normalize_triplet(self._chunk_halo_param, ndim)
        if chunk_halo is not None and any(h < 0 for h in chunk_halo):
            raise ValueError(f"chunk_halo must be non-negative, got {chunk_halo}")
        return chunk_size, chunk_halo

    def _compute_chunked_loss(
        self,
        pred_field: torch.Tensor,
        tgt_field: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_field = pred_field.contiguous()
        tgt_field = tgt_field.contiguous()
        ndim = pred_field.ndim
        chunk_size, chunk_halo = self._get_chunk_params(ndim)
        shape = pred_field.shape
        slices = _grid_slices(shape, chunk_size)
        aggregator = _PairAggregator.create()

        pred_np_full = np.ascontiguousarray(pred_field.detach().cpu().numpy().astype(np.float64))
        tgt_np_full = np.ascontiguousarray(tgt_field.detach().cpu().numpy().astype(np.float64))

        sub_preds: List[np.ndarray] = []
        sub_tgts: List[np.ndarray] = []
        ext_slices: List[Tuple[slice, ...]] = []
        interior_boxes: List[Tuple[Tuple[int, int], ...]] = []

        for sl in slices:
            if chunk_halo is None:
                ext_sl = sl
                interior_box = tuple((0, sl[d].stop - sl[d].start) for d in range(ndim))
            else:
                ext_axes = []
                interior = []
                for axis, halo in enumerate(chunk_halo):
                    h = max(0, halo)
                    start = max(0, sl[axis].start - h)
                    stop = min(shape[axis], sl[axis].stop + h)
                    ext_axes.append(slice(start, stop))
                    interior.append((sl[axis].start - start, sl[axis].stop - start))
                ext_sl = tuple(ext_axes)
                interior_box = tuple(interior)
            ext_slices.append(ext_sl)
            interior_boxes.append(interior_box)
            sub_preds.append(np.ascontiguousarray(pred_np_full[ext_sl]))
            sub_tgts.append(np.ascontiguousarray(tgt_np_full[ext_sl]))

        if not sub_preds:
            raise ValueError("Chunk sampler produced no sub-volumes.")

        results = bm.compute_matching(
            sub_preds,
            sub_tgts,
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=self.include_unmatched_target,
        )

        for ext_sl, res, interior in zip(ext_slices, results, interior_boxes):
            offset = tuple(ext_sl[d].start for d in range(len(ext_sl)))
            aggregator.add_chunk_result(res, dim_nd=ndim, offset=offset, interior_box_ext=interior)

        (
            mb,
            md,
            tb,
            td,
            ub,
            ud,
            vb,
            vd,
        ) = aggregator.export_numpy()

        loss, aux = _loss_from_coords_torch(
            pred_field,
            tgt_field,
            mb,
            md,
            tb,
            td,
            ub,
            ud,
            vb,
            vd,
            include_unmatched_target=self.include_unmatched_target,
            push_to=self.push_unmatched_to,
        )
        return loss.reshape(1), aux

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute Betti matching loss matching the scratch example semantics.

        Accepts logits or probabilities for `input` with shape (B, C, ...). For C==2 uses softmax and the
        foreground channel; for C==1 applies sigmoid if outside [0,1]. Target may be 1-channel or one-hot with C==2.
        Returns either a scalar tensor or (loss, aux_dict) where aux_dict contains diagnostic components.
        """
        device = input.device
        batch_size = input.shape[0]
        num_channels = input.shape[1]

        if batch_size == 0:
            zero = input.new_tensor(0.0)
            return zero, {}

        # Select foreground channel and normalize to [0,1] if needed
        if num_channels == 2:
            probs = torch.softmax(input, dim=1)
            pred_fg = probs[:, 1:2]
            tgt_fg = target[:, 1:2] if target.shape[1] == 2 else target
        else:
            pred_fg = torch.sigmoid(input) if not (input.min() >= 0 and input.max() <= 1) else input
            tgt_fg = target

        pred_fg = pred_fg.contiguous()
        tgt_fg = tgt_fg.contiguous()

        # Build lists for batched calls to the C++ extension
        preds_fields = [pred_fg[b, 0].contiguous() for b in range(batch_size)]
        tgts_fields = [tgt_fg[b, 0].contiguous() for b in range(batch_size)]

        def _to_numpy(fields):
            return [np.ascontiguousarray(f.detach().cpu().numpy().astype(np.float64)) for f in fields]

        total_losses: List[torch.Tensor] = []
        aux_parts: List[Dict[str, torch.Tensor]] = []
        use_chunking = self.chunk_mode is not None

        if self.filtration == 'bothlevel':
            preds_sub_fields = preds_fields
            tgts_sub_fields = tgts_fields
            preds_super_fields = [1.0 - p for p in preds_fields]
            tgts_super_fields = [1.0 - t for t in tgts_fields]

            if use_chunking:
                for b in range(batch_size):
                    loss_super, aux_super = self._compute_chunked_loss(
                        preds_super_fields[b],
                        tgts_super_fields[b],
                    )
                    loss_sub, aux_sub = self._compute_chunked_loss(
                        preds_sub_fields[b],
                        tgts_sub_fields[b],
                    )
                    total_losses.append(0.5 * (loss_super + loss_sub))

                    keys = set(aux_super.keys()) | set(aux_sub.keys())
                    combined_aux: Dict[str, torch.Tensor] = {}
                    for k in keys:
                        val_super = aux_super.get(k)
                        val_sub = aux_sub.get(k)
                        if val_super is not None and val_sub is not None:
                            combined_aux[k] = 0.5 * (val_super + val_sub)
                        elif val_super is not None:
                            combined_aux[k] = 0.5 * val_super
                        elif val_sub is not None:
                            combined_aux[k] = 0.5 * val_sub
                    aux_parts.append(combined_aux)
            else:
                results_super = bm.compute_matching(
                    _to_numpy(preds_super_fields),
                    _to_numpy(tgts_super_fields),
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=self.include_unmatched_target,
                )
                results_sub = bm.compute_matching(
                    _to_numpy(preds_sub_fields),
                    _to_numpy(tgts_sub_fields),
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=self.include_unmatched_target,
                )

                for b in range(batch_size):
                    loss_super, aux_super = _compute_loss_from_result(
                        preds_super_fields[b],
                        tgts_super_fields[b],
                        results_super[b],
                        include_unmatched_target=self.include_unmatched_target,
                        push_to=self.push_unmatched_to,
                    )
                    loss_sub, aux_sub = _compute_loss_from_result(
                        preds_sub_fields[b],
                        tgts_sub_fields[b],
                        results_sub[b],
                        include_unmatched_target=self.include_unmatched_target,
                        push_to=self.push_unmatched_to,
                    )

                    total_losses.append(0.5 * (loss_super + loss_sub))

                    keys = set(aux_super.keys()) | set(aux_sub.keys())
                    combined_aux: Dict[str, torch.Tensor] = {}
                    for k in keys:
                        val_super = aux_super.get(k)
                        val_sub = aux_sub.get(k)
                        if val_super is not None and val_sub is not None:
                            combined_aux[k] = 0.5 * (val_super + val_sub)
                        elif val_super is not None:
                            combined_aux[k] = 0.5 * val_super
                        elif val_sub is not None:
                            combined_aux[k] = 0.5 * val_sub
                    aux_parts.append(combined_aux)
        else:
            if self.filtration == 'superlevel':
                preds_proc_fields = [1.0 - p for p in preds_fields]
                tgts_proc_fields = [1.0 - t for t in tgts_fields]
            else:
                preds_proc_fields = preds_fields
                tgts_proc_fields = tgts_fields

            if use_chunking:
                for b in range(batch_size):
                    loss_b, aux_b = self._compute_chunked_loss(
                        preds_proc_fields[b],
                        tgts_proc_fields[b],
                    )
                    total_losses.append(loss_b)
                    aux_parts.append(aux_b)
            else:
                results = bm.compute_matching(
                    _to_numpy(preds_proc_fields),
                    _to_numpy(tgts_proc_fields),
                    include_input1_unmatched_pairs=True,
                    include_input2_unmatched_pairs=self.include_unmatched_target,
                )

                for b in range(batch_size):
                    loss_b, aux_b = _compute_loss_from_result(
                        preds_proc_fields[b],
                        tgts_proc_fields[b],
                        results[b],
                        include_unmatched_target=self.include_unmatched_target,
                        push_to=self.push_unmatched_to,
                    )
                    total_losses.append(loss_b)
                    aux_parts.append(aux_b)

        loss = torch.mean(torch.cat(total_losses)) if len(total_losses) > 0 else torch.tensor(0.0, device=device)

        # Aggregate aux
        aux_agg: Dict[str, torch.Tensor] = {}
        if len(aux_parts) > 0:
            all_keys = set().union(*(d.keys() for d in aux_parts))
            for k in all_keys:
                values = [d[k] for d in aux_parts if k in d]
                aux_agg[k] = torch.mean(torch.cat(values))

        # Return tuple to make DeepSupervisionWrapper pick the scalar part automatically
        return loss, aux_agg
