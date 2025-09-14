"""
Benchmark: Full-resolution 3D Betti loss vs 2D full-resolution Betti loss on 8 z-slices.

- Synthetic mode: builds simple 3D volumes and compares 2D(8) vs 3D on accuracy and time.
- Label mode: load a provided TIFF label volume (any nonzero -> 1), synthesize a prediction
  by adding uniform noise, and benchmark 2D(8) vs 3D on your data.

Run (synthetic):
  python -m vesuvius.models.training.loss.scratch.benchmark_slices_vs_3d

Run (label TIFF):
  python -m vesuvius.models.training.loss.scratch.benchmark_slices_vs_3d \\
    --label-tif /path/to/label.tif --depth-axis 0 --slices 8 --noise 0.10
Note: 2D loss uses an even split of the requested slice count across axial/sagittal/coronal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

try:
    # Import production Betti matching binding and helper
    from vesuvius.models.training.loss.betti_losses import (
        _compute_loss_from_result,
        bm,  # C++ extension bindings
    )
except Exception as e:
    raise SystemExit(
        "Could not import betti loss or betti_matching bindings.\n"
        "Please build the extension via the helper script in vesuvius/utils.\n"
        f"Underlying error: {e}"
    )


# -------------------------------
# Synthetic 2D masks
# -------------------------------
def rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 1.0
    return m


def ring_mask(h: int, w: int, thickness: int = 6) -> np.ndarray:
    pad = max(4, thickness + 2)
    outer = rect_mask(h, w, pad, pad, w - pad, h - pad)
    inner = rect_mask(h, w, pad + thickness, pad + thickness, w - pad - thickness, h - pad - thickness)
    return np.clip(outer - inner, 0, 1)


def two_blobs_mask(h: int, w: int, size: int = 12, gap: int = 8) -> np.ndarray:
    cx, cy = w // 2, h // 2
    left = rect_mask(h, w, cx - gap - size, cy - size, cx - gap, cy + size)
    right = rect_mask(h, w, cx + gap, cy - size, cx + gap + size, cy + size)
    return np.clip(left + right, 0, 1)


def single_blob_mask(h: int, w: int, size: int = 18) -> np.ndarray:
    cx, cy = w // 2, h // 2
    return rect_mask(h, w, cx - size, cy - size, cx + size, cy + size)


def thin_hline_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[h // 2, :] = 1.0
    return m


def thin_vline_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[:, w // 2] = 1.0
    return m


def thin_cross_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[h // 2, :] = 1.0
    m[:, w // 2] = 1.0
    return m


def thin_diag_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    d = min(h, w)
    for i in range(d):
        m[i, i] = 1.0
    return m


def thin_ring_mask(h: int, w: int) -> np.ndarray:
    return ring_mask(h, w, thickness=1)


@dataclass
class Case:
    name: str
    mask_fn: Callable[[int, int], np.ndarray]


def make_cases() -> List[Case]:
    return [
        Case("single_blob", single_blob_mask),
        Case("two_blobs", two_blobs_mask),
        Case("ring", ring_mask),
        Case("thin_hline", thin_hline_mask),
        Case("thin_vline", thin_vline_mask),
        Case("thin_cross", thin_cross_mask),
        Case("thin_diag", thin_diag_mask),
        Case("thin_ring", thin_ring_mask),
    ]


# -------------------------------
# Betti loss computations
# -------------------------------
def betti_loss_fullres(
    pred: torch.Tensor,  # shape (B,1,H,W) or (B,1,D,H,W)
    tgt: torch.Tensor,
    filtration: str = "superlevel",
) -> torch.Tensor:
    """Compute Betti loss at full resolution for 2D or 3D fields."""
    assert pred.shape[:2] == (tgt.shape[0], 1)
    bsz = pred.shape[0]
    preds_fields = [pred[b].squeeze(0) for b in range(bsz)]
    tgts_fields = [tgt[b].squeeze(0) for b in range(bsz)]
    preds_np = [np.ascontiguousarray(p.detach().cpu().numpy().astype(np.float64)) for p in preds_fields]
    tgts_np = [(np.ascontiguousarray(t.detach().cpu().numpy()) > 0.5).astype(np.float64) for t in tgts_fields]

    def run_once(p_in: List[np.ndarray], t_in: List[np.ndarray]):
        results = bm.compute_matching(
            p_in,
            t_in,
            include_input1_unmatched_pairs=True,
            include_input2_unmatched_pairs=True,
        )
        losses: List[torch.Tensor] = []
        for b in range(bsz):
            l, _ = _compute_loss_from_result(preds_fields[b], tgts_fields[b], results[b])
            losses.append(l)
        return torch.mean(torch.cat(losses)) if losses else pred.new_tensor(0.0)

    if filtration == "bothlevel":
        preds_super = [1.0 - p for p in preds_np]
        tgts_super = [1.0 - t for t in tgts_np]
        loss_super = run_once(preds_super, tgts_super)
        loss_sub = run_once(preds_np, tgts_np)
        return 0.5 * (loss_super + loss_sub)
    else:
        if filtration == "superlevel":
            preds_np = [1.0 - p for p in preds_np]
            tgts_np = [1.0 - t for t in tgts_np]
        return run_once(preds_np, tgts_np)


def betti_loss_fullres_slices2d_axis(
    pred3d: torch.Tensor,  # (B,1,D,H,W)
    tgt3d: torch.Tensor,   # (B,1,D,H,W)
    axis: int,             # 0=D(axial), 1=H(sagittal), 2=W(coronal)
    indices: List[int],
    filtration: str = "superlevel",
) -> torch.Tensor:
    assert pred3d.ndim == 5 and tgt3d.ndim == 5
    assert axis in (0, 1, 2)
    size = pred3d.shape[2 + axis]
    indices = sorted(set(int(z) for z in indices if 0 <= int(z) < size))
    if len(indices) == 0:
        return pred3d.new_tensor(0.0)
    slice_losses: List[torch.Tensor] = []
    for idx in indices:
        if axis == 0:
            pred2d = pred3d[:, :, idx, :, :]
            tgt2d = tgt3d[:, :, idx, :, :]
        elif axis == 1:
            pred2d = pred3d[:, :, :, idx, :]
            tgt2d = tgt3d[:, :, :, idx, :]
        else:
            pred2d = pred3d[:, :, :, :, idx]
            tgt2d = tgt3d[:, :, :, :, idx]
        slice_losses.append(betti_loss_fullres(pred2d, tgt2d, filtration=filtration).reshape(1))
    return torch.mean(torch.cat(slice_losses))


def betti_loss_fullres_slices2d_multi_orient(
    pred3d: torch.Tensor,  # (B,1,D,H,W)
    tgt3d: torch.Tensor,   # (B,1,D,H,W)
    total_slices: int,
    filtration: str = "superlevel",
) -> torch.Tensor:
    """Evenly split the requested slice count across axial/sagittal/coronal and average losses."""
    if total_slices <= 0:
        return pred3d.new_tensor(0.0)
    # Distribute counts across 3 axes
    base = total_slices // 3
    rem = total_slices % 3
    counts = [base + (1 if i < rem else 0) for i in range(3)]  # [axial, sagittal, coronal]
    D, H, W = pred3d.shape[2], pred3d.shape[3], pred3d.shape[4]

    def choose(n, size):
        if n <= 0:
            return []
        if n >= size:
            return list(range(size))
        idx = np.linspace(0, size - 1, num=n)
        return np.unique(idx.astype(int)).tolist()

    idx_ax = choose(counts[0], D)
    idx_sag = choose(counts[1], H)
    idx_cor = choose(counts[2], W)

    losses: List[torch.Tensor] = []
    if idx_ax:
        losses.append(betti_loss_fullres_slices2d_axis(pred3d, tgt3d, axis=0, indices=idx_ax, filtration=filtration).reshape(1))
    if idx_sag:
        losses.append(betti_loss_fullres_slices2d_axis(pred3d, tgt3d, axis=1, indices=idx_sag, filtration=filtration).reshape(1))
    if idx_cor:
        losses.append(betti_loss_fullres_slices2d_axis(pred3d, tgt3d, axis=2, indices=idx_cor, filtration=filtration).reshape(1))
    return torch.mean(torch.cat(losses)) if losses else pred3d.new_tensor(0.0)


# -------------------------------
# 3D synthetic volumes
# -------------------------------
def make_3d_volume(mask_fn: Callable[[int, int], np.ndarray], D: int, H: int, W: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros((D, H, W), dtype=np.float32)
    dx = rng.integers(-3, 4)
    dy = rng.integers(-3, 4)
    for z in range(D):
        sx = int(round(dx * math.sin(2 * math.pi * z / max(1, D))))
        sy = int(round(dy * math.cos(2 * math.pi * z / max(1, D))))
        base = mask_fn(H, W)
        vol[z] = np.roll(np.roll(base, shift=sy, axis=0), shift=sx, axis=1)
    return vol


def to_tensor3d(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


def choose_slices(D: int, n: int = 8) -> List[int]:
    if n >= D:
        return list(range(D))
    idx = np.linspace(0, D - 1, num=n)
    idx = np.unique(idx.astype(int))
    if idx.size < n:
        extra = [i for i in range(D) if i not in idx]
        idx = np.concatenate([idx, np.array(extra[: n - idx.size], dtype=int)])
    return sorted(idx.tolist())


# -------------------------------
# Benchmark
# -------------------------------
def _pearson_corr(a: List[float], b: List[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    c = np.corrcoef(x, y)
    return float(c[0, 1])


def evaluate_slices_vs_3d(
    D: int = 32,
    H: int = 64,
    W: int = 64,
    n_trials: int = 3,
    noise_level: float = 0.1,
    filtration: str = "superlevel",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = make_cases()
    z_idx8_default = None

    print(f"Device: {device} | D,H,W=({D},{H},{W}) | trials={n_trials} | noise={noise_level} | filtration={filtration}")

    L3d_vals: List[float] = []
    L2d8_vals: List[float] = []
    t3d_vals: List[float] = []
    t2d8_vals: List[float] = []

    # Gradient samples
    grad_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for case in cases:
        for k in range(n_trials):
            seed = 10000 + hash(case.name) % 1000 + k
            vol = make_3d_volume(case.mask_fn, D, H, W, seed=seed)
            tgt3d_np = vol
            pred3d_np = (1.0 - noise_level) * tgt3d_np + noise_level * np.random.default_rng(seed + 1).random(size=tgt3d_np.shape, dtype=np.float32)
            pred3d_np = np.clip(pred3d_np, 0.0, 1.0)

            tgt3d = to_tensor3d(tgt3d_np).to(device)
            pred3d = to_tensor3d(pred3d_np).to(device)

            z_idx8 = choose_slices(D, 8)
            if z_idx8_default is None:
                z_idx8_default = z_idx8

            # Timed 3D full-res (no grad)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                L3d = betti_loss_fullres(pred3d, tgt3d, filtration=filtration).item()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()

            # Timed 2D (8 slices) full-res across orientations (no grad)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()
            with torch.no_grad():
                L2d8 = betti_loss_fullres_slices2d_multi_orient(pred3d, tgt3d, total_slices=8, filtration=filtration).item()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t3 = time.perf_counter()

            L3d_vals.append(L3d)
            L2d8_vals.append(L2d8)
            t3d_vals.append(t1 - t0)
            t2d8_vals.append(t3 - t2)

            print(f"{case.name:12s} trial={k:02d}  L3d={L3d:.5f}  L2D_8={L2d8:.5f}  time3d={t1-t0:.3f}s  time2D8={t3-t2:.3f}s")

            if len(grad_samples) < 6:
                grad_samples.append((pred3d.detach().cpu(), tgt3d.detach().cpu()))

    # Accuracy summary
    L3d_arr = np.asarray(L3d_vals, dtype=np.float64)
    L2d8_arr = np.asarray(L2d8_vals, dtype=np.float64)
    abs_err = np.abs(L2d8_arr - L3d_arr)
    rel_err = abs_err / (np.abs(L3d_arr) + 1e-8)
    corr = _pearson_corr(L3d_vals, L2d8_vals)

    # Time summary
    t3d = np.asarray(t3d_vals, dtype=np.float64)
    t2d8 = np.asarray(t2d8_vals, dtype=np.float64)
    speedup = (t3d.mean() / t2d8.mean()) if t2d8.mean() > 0 else float("inf")

    print("\n=== Summary: 2D(8 slices) vs 3D full-res ===")
    print(f"Loss abs_err: mean={abs_err.mean():.6f}  median={np.median(abs_err):.6f}")
    print(f"Loss rel_err: mean={100*rel_err.mean():.3f}%  median={100*np.median(rel_err):.3f}%  corr={corr:.3f}")
    print(f"Time per volume: 3D={t3d.mean():.3f}s  2D(8)={t2d8.mean():.3f}s  speedup≈{speedup:.1f}x")

    # Gradient proxies vs 3D
    sims = []
    deltas = []
    for pred_cpu, tgt_cpu in grad_samples:
        pred = pred_cpu
        tgt = tgt_cpu
        # Full 3D gradient
        p1 = pred.clone().detach().requires_grad_(True)
        L_full = betti_loss_fullres(p1, tgt, filtration=filtration)
        L_full.backward()
        g_full = p1.grad.detach().flatten().cpu().numpy()

        # Slices(8) gradient (multi-orientation)
        p2 = pred.clone().detach().requires_grad_(True)
        L_m = betti_loss_fullres_slices2d_multi_orient(p2, tgt, total_slices=8, filtration=filtration)
        L_m.backward()
        g_m = p2.grad.detach().flatten().cpu().numpy()
        num = float((g_full * g_m).sum())
        den = float(np.linalg.norm(g_full) * np.linalg.norm(g_m) + 1e-12)
        sims.append(num / den if den > 0 else float("nan"))

        # One-step ΔL3d
        base = betti_loss_fullres(pred, tgt, filtration=filtration).item()
        with torch.no_grad():
            new_pred = torch.clamp(p2 - 1.0 * p2.grad, 0.0, 1.0)
        new_full = betti_loss_fullres(new_pred, tgt, filtration=filtration).item()
        deltas.append(new_full - base)

    sims = [s for s in sims if np.isfinite(s)]
    deltas = [d for d in deltas if np.isfinite(d)]
    if sims:
        print(f"Gradient cosine (2D(8) vs 3D): mean={float(np.mean(sims)):.3f}  median={float(np.median(sims)):.3f}")
    if deltas:
        print(f"One-step ΔL3d with 2D(8) gradient: mean={float(np.mean(deltas)):.4f}  median={float(np.median(deltas)):.4f}  improve%={100.0*sum(1 for d in deltas if d<0)/len(deltas):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 2D(8 slices) vs 3D Betti loss")
    parser.add_argument("--label-tif", type=str, default=None, help="Path to label TIFF (2D or 3D). Any nonzero -> 1.")
    parser.add_argument("--depth-axis", type=int, default=0, help="Axis index of depth (for 3D TIFF). Default 0.")
    parser.add_argument("--slices", type=int, default=8, help="Total number of 2D slices (evenly split across axial/sagittal/coronal).")
    parser.add_argument("--noise", type=float, default=0.10, help="Uniform noise level for synthesizing prediction from label.")
    parser.add_argument("--filtration", type=str, default="superlevel", choices=["superlevel","sublevel","bothlevel"], help="Filtration type.")
    parser.add_argument("--trials", type=int, default=3, help="Trials (synthetic only)")
    parser.add_argument("--D", type=int, default=32, help="Depth for synthetic volume")
    parser.add_argument("--H", type=int, default=64, help="Height for synthetic volume")
    parser.add_argument("--W", type=int, default=64, help="Width for synthetic volume")
    parser.add_argument("--with-grad", action="store_true", help="Compute gradient proxies (label mode)")
    args = parser.parse_args()

    if args.label_tif is None:
        # Synthetic mode
        evaluate_slices_vs_3d(D=args.D, H=args.H, W=args.W, n_trials=args.trials, noise_level=args.noise, filtration=args.filtration)
    else:
        # Label mode
        # Lazy import of tif readers to avoid hard dependency
        arr = None
        err_msgs: List[str] = []
        try:
            import tifffile as tiff  # type: ignore
            arr = tiff.imread(args.label_tif)
        except Exception as e:
            err_msgs.append(f"tifffile failed: {e}")
        if arr is None:
            try:
                import imageio.v3 as iio  # type: ignore
                arr = iio.imread(args.label_tif)
            except Exception as e:
                err_msgs.append(f"imageio.v3 failed: {e}")
        if arr is None:
            try:
                from PIL import Image  # type: ignore
                img = Image.open(args.label_tif)
                frames = []
                try:
                    i = 0
                    while True:
                        img.seek(i)
                        frames.append(np.array(img))
                        i += 1
                except EOFError:
                    pass
                arr = np.stack(frames, axis=0) if len(frames) > 1 else np.array(img)
            except Exception as e:
                err_msgs.append(f"PIL failed: {e}")
        if arr is None:
            raise SystemExit("Could not read TIFF. Tried tifffile, imageio, PIL. Errors: " + " | ".join(err_msgs))

        arr = np.asarray(arr)
        # Normalize to (D,H,W)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            if args.depth_axis == 0:
                pass
            elif args.depth_axis == 1:
                arr = np.transpose(arr, (1, 0, 2))
            elif args.depth_axis == 2:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                raise SystemExit(f"Invalid --depth-axis {args.depth_axis}. Must be 0,1,2")
        else:
            raise SystemExit(f"Unsupported TIFF ndim={arr.ndim}. Expect 2D or 3D")

        # Binarize: any nonzero is label
        label3d_np = (arr != 0).astype(np.float32)
        # Synthesize prediction by blending with noise
        rng = np.random.default_rng(1234)
        pred3d_np = (1.0 - args.noise) * label3d_np + args.noise * rng.random(size=label3d_np.shape, dtype=np.float32)
        pred3d_np = np.clip(pred3d_np, 0.0, 1.0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tgt3d = torch.from_numpy(label3d_np).unsqueeze(0).unsqueeze(0).to(device)
        pred3d = torch.from_numpy(pred3d_np).unsqueeze(0).unsqueeze(0).to(device)

        print(f"Label volume shape (D,H,W)={label3d_np.shape}, total 2D slices={args.slices} (evenly split across axial/sagittal/coronal)")

        # Timing (no grad)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            L3d = betti_loss_fullres(pred3d, tgt3d, filtration=args.filtration).item()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        with torch.no_grad():
            L2d = betti_loss_fullres_slices2d_multi_orient(pred3d, tgt3d, total_slices=args.slices, filtration=args.filtration).item()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t3 = time.perf_counter()

        print("=== Label benchmark ===")
        print(f"3D loss: {L3d:.6f}  time={t1-t0:.3f}s")
        print(f"2D({args.slices} mixed-orientation slices) loss: {L2d:.6f}  time={t3-t2:.3f}s  speedup≈{(t1-t0)/(t3-t2) if (t3-t2)>0 else float('inf'):.1f}x")
        print(f"Abs err={abs(L2d-L3d):.6f}  Rel err={100*abs(L2d-L3d)/(abs(L3d)+1e-8):.3f}%")

        if args.with_grad:
            # Gradient proxies
            p1 = pred3d.clone().detach().requires_grad_(True)
            L_full = betti_loss_fullres(p1, tgt3d, filtration=args.filtration)
            L_full.backward()
            g_full = p1.grad.detach().flatten().cpu().numpy()

            p2 = pred3d.clone().detach().requires_grad_(True)
            L_slices = betti_loss_fullres_slices2d_multi_orient(p2, tgt3d, total_slices=args.slices, filtration=args.filtration)
            L_slices.backward()
            g_slices = p2.grad.detach().flatten().cpu().numpy()
            num = float((g_full * g_slices).sum())
            den = float(np.linalg.norm(g_full) * np.linalg.norm(g_slices) + 1e-12)
            cos = num / den if den > 0 else float("nan")
            base = L_full.item()
            with torch.no_grad():
                new_pred = torch.clamp(p2 - p2.grad, 0.0, 1.0)
            new_full = betti_loss_fullres(new_pred, tgt3d, filtration=args.filtration).item()
            print(f"Grad cosine (2D({args.slices}) vs 3D): {cos:.3f}")
            print(f"One-step ΔL3d with 2D({args.slices}) gradient: {new_full - base:.4f}")
