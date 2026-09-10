import argparse
import os
import os.path as osp
import tempfile
from contextlib import nullcontext
from datetime import datetime

import cv2
import numpy as np
import torch
from monai.inferers import sliding_window_inference
import tifffile
import PIL.Image
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 109951162777600

from train_resnet3d_lib.checkpointing import load_state_dict_from_checkpoint
from train_resnet3d_lib.config import (
    CFG,
    apply_metadata_hyperparameters,
    load_and_validate_base_config,
)
from train_resnet3d_lib.data_ops import ZarrSegmentVolume
from train_resnet3d_lib.model import RegressionPLModel


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone inference for train_resnet3d Vesuvius checkpoints on zarr inputs."
    )
    parser.add_argument("--metadata_json", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--segment_id", required=True, type=str)
    parser.add_argument(
        "--zarr_path",
        default=None,
        type=str,
        help="Optional explicit zarr path. If omitted, resolved from metadata dataset_root + segment_id.",
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        type=str,
        help="Optional fragment mask path (.png/.tif/.tiff). If omitted, a full-ones mask is used.",
    )
    parser.add_argument(
        "--layer_range",
        default=None,
        type=str,
        help="Optional explicit layer range as start:end (end-exclusive).",
    )
    parser.add_argument(
        "--reverse_layers",
        action="store_true",
        help="Override metadata and reverse z-layers.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Output prediction image path (png/tif). Values are probabilities in [0,255].",
    )
    parser.add_argument(
        "--output_npy",
        default=None,
        type=str,
        help="Optional output .npy path for float32 probabilities in [0,1].",
    )
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument(
        "--chunk_size",
        default=2048,
        type=int,
        help="Core chunk size for streaming inference over H/W.",
    )
    parser.add_argument(
        "--halo",
        default=None,
        type=int,
        help="Context halo around each chunk. Default: CFG.size.",
    )
    parser.add_argument(
        "--overlap",
        default=None,
        type=float,
        help="Sliding-window overlap in [0, 1). If omitted, derived from 1 - stride/size.",
    )
    parser.add_argument(
        "--allow_full_volume",
        action="store_true",
        help="Allow full-volume inference if no mask is provided/found.",
    )
    parser.add_argument("--device", default="auto", type=str, help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--amp", action="store_true", help="Use AMP autocast on CUDA.")
    return parser.parse_args()


def _read_mask(mask_path, target_hw):
    mask = None
    try:
        with PIL.Image.open(mask_path) as im:
            mask = np.array(im.convert("L"))
    except (OSError, ValueError):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file: {mask_path}")
    h, w = [int(v) for v in target_hw]
    padded = np.zeros((h, w), dtype=np.uint8)
    copy_h = min(h, int(mask.shape[0]))
    copy_w = min(w, int(mask.shape[1]))
    if copy_h > 0 and copy_w > 0:
        padded[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
    return padded


def _resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _parse_layer_range(text):
    if text is None:
        return None
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"--layer_range must be 'start:end', got {text!r}")
    start_idx = int(parts[0])
    end_idx = int(parts[1])
    if end_idx <= start_idx:
        raise ValueError(f"--layer_range must satisfy end > start, got {text!r}")
    return [start_idx, end_idx]


def _build_model_from_cfg():
    model = RegressionPLModel(
        enc="i3d",
        size=int(getattr(CFG, "size", 256)),
        norm=str(getattr(CFG, "norm", "batch")),
        group_norm_groups=int(getattr(CFG, "group_norm_groups", 32)),
        model_impl=str(getattr(CFG, "model_impl", "resnet3d_hybrid")),
        vesuvius_model_config=getattr(CFG, "vesuvius_model_config", {}),
        vesuvius_target_name=str(getattr(CFG, "vesuvius_target_name", "ink")),
        vesuvius_z_projection_mode=str(getattr(CFG, "vesuvius_z_projection_mode", "logsumexp")),
        vesuvius_z_projection_lse_tau=float(getattr(CFG, "vesuvius_z_projection_lse_tau", 1.0)),
        vesuvius_z_projection_mlp_hidden=int(getattr(CFG, "vesuvius_z_projection_mlp_hidden", 64)),
        vesuvius_z_projection_mlp_dropout=float(getattr(CFG, "vesuvius_z_projection_mlp_dropout", 0.0)),
        vesuvius_z_projection_mlp_depth=int(
            getattr(CFG, "vesuvius_z_projection_mlp_depth", None) or getattr(CFG, "in_chans", 1)
        ),
        objective=str(getattr(CFG, "objective", "erm")),
        loss_mode=str(getattr(CFG, "loss_mode", "batch")),
        total_steps=1,
        n_groups=1,
        group_names=["inference"],
    )
    return model


def _resolve_overlap(explicit_overlap):
    if explicit_overlap is not None:
        overlap = float(explicit_overlap)
    else:
        stride = float(getattr(CFG, "stride", 1))
        size = float(getattr(CFG, "size", 1))
        if size <= 0:
            raise ValueError(f"CFG.size must be > 0, got {size}")
        overlap = 1.0 - (stride / size)
    overlap = max(0.0, min(overlap, 0.99))
    return overlap


def _save_image(path, array_u8):
    ext = osp.splitext(path)[1].lower()
    if ext in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.asarray(array_u8), bigtiff=True)
        return
    ok = cv2.imwrite(path, np.asarray(array_u8))
    if not ok:
        raise RuntimeError(f"Failed to write output image: {path}")


def _auto_find_mask_path(*, segment_id, zarr_path):
    candidates = []
    sid = str(segment_id)
    dataset_root = str(getattr(CFG, "dataset_root", ""))
    for ext in (".png", ".tif", ".tiff"):
        if dataset_root:
            candidates.append(osp.join(dataset_root, sid, f"{sid}_mask{ext}"))
            candidates.append(osp.join(dataset_root, f"{sid}_mask{ext}"))

    if zarr_path:
        zp = osp.abspath(zarr_path)
        parent = osp.dirname(zp)
        if osp.basename(parent) == "0":
            parent = osp.dirname(parent)
        parent2 = osp.dirname(parent)
        zarr_base = osp.basename(parent).replace(".zarr", "")
        for ext in (".png", ".tif", ".tiff"):
            candidates.append(osp.join(parent2, f"{sid}_mask{ext}"))
            candidates.append(osp.join(parent2, f"{zarr_base}_mask{ext}"))
            candidates.append(osp.join(parent2, f"mask{ext}"))

    for p in candidates:
        if p and osp.exists(p):
            return p
    return None


def _resolve_segment_meta(base_cfg, segment_id, *, explicit_layer_range):
    segments = dict(base_cfg.get("segments") or {})
    if segment_id not in segments:
        if explicit_layer_range is None:
            raise KeyError(
                f"segment_id={segment_id!r} not found in metadata_json.segments. "
                "Provide --layer_range start:end for external zarr inputs."
            )
        return {"layer_range": explicit_layer_range, "reverse_layers": False}
    seg_meta = dict(segments[segment_id] or {})
    if "layer_range" not in seg_meta:
        raise KeyError(f"segments[{segment_id!r}] missing required key 'layer_range'")
    if "reverse_layers" not in seg_meta:
        raise KeyError(f"segments[{segment_id!r}] missing required key 'reverse_layers'")
    return seg_meta


def _prepare_dataset_root_for_explicit_zarr(zarr_path, segment_id):
    zarr_path = osp.abspath(zarr_path)
    expected_1 = osp.normpath(osp.join(osp.dirname(zarr_path), f"{segment_id}.zarr"))
    if osp.normpath(zarr_path) == expected_1:
        return osp.dirname(zarr_path), None

    tmp_root = tempfile.mkdtemp(prefix="infer_resnet3d_vesuvius_")
    link_path = osp.join(tmp_root, f"{segment_id}.zarr")
    os.symlink(zarr_path, link_path)
    return tmp_root, tmp_root


def main():
    args = _parse_args()
    if args.batch_size is not None and int(args.batch_size) < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.overlap is not None and not (0.0 <= float(args.overlap) < 1.0):
        raise ValueError(f"--overlap must be in [0,1), got {args.overlap}")
    if int(args.chunk_size) < 1:
        raise ValueError(f"--chunk_size must be >= 1, got {args.chunk_size}")
    if args.halo is not None and int(args.halo) < 0:
        raise ValueError(f"--halo must be >= 0, got {args.halo}")

    base_cfg = load_and_validate_base_config(args.metadata_json, base_dir=os.getcwd())
    apply_metadata_hyperparameters(CFG, base_cfg)

    segment_id = str(args.segment_id)
    layer_range = _parse_layer_range(args.layer_range)
    seg_meta = _resolve_segment_meta(base_cfg, segment_id, explicit_layer_range=layer_range)
    if layer_range is None:
        layer_range = seg_meta["layer_range"]
    reverse_layers = bool(args.reverse_layers or bool(seg_meta["reverse_layers"]))

    cleanup_tmp_root = None
    if args.zarr_path:
        dataset_root, cleanup_tmp_root = _prepare_dataset_root_for_explicit_zarr(args.zarr_path, segment_id)
        CFG.dataset_root = dataset_root

    volume = ZarrSegmentVolume(
        segment_id,
        seg_meta,
        layer_range=layer_range,
        reverse_layers=reverse_layers,
    )
    h, w = [int(v) for v in volume.shape[:2]]

    resolved_mask_path = args.mask_path
    if resolved_mask_path is None:
        resolved_mask_path = _auto_find_mask_path(segment_id=segment_id, zarr_path=args.zarr_path)
    if resolved_mask_path is not None:
        print(f"using mask_path={resolved_mask_path}", flush=True)
        fragment_mask = _read_mask(resolved_mask_path, (h, w))
    else:
        if not bool(args.allow_full_volume):
            raise ValueError(
                "No mask found. Provide --mask_path or pass --allow_full_volume "
                "to run over the whole volume."
            )
        print("no mask found; running on full volume", flush=True)
        fragment_mask = np.full((h, w), 255, dtype=np.uint8)

    model = _build_model_from_cfg()
    state_dict = load_state_dict_from_checkpoint(args.ckpt_path)
    incompat = model.load_state_dict(state_dict, strict=False)
    print(
        f"loaded checkpoint: missing_keys={len(incompat.missing_keys)} "
        f"unexpected_keys={len(incompat.unexpected_keys)}",
        flush=True,
    )

    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    batch_size = int(args.batch_size or getattr(CFG, "valid_batch_size", 1))
    overlap = _resolve_overlap(args.overlap)
    chunk_size = int(args.chunk_size)
    halo = int(args.halo if args.halo is not None else int(getattr(CFG, "size", 256)))
    roi_size = (int(CFG.size), int(CFG.size))

    rows_with_mask = np.any(fragment_mask != 0, axis=1)
    cols_with_mask = np.any(fragment_mask != 0, axis=0)
    if not bool(rows_with_mask.any()) or not bool(cols_with_mask.any()):
        raise ValueError("mask has no foreground pixels")
    y0 = int(np.argmax(rows_with_mask))
    y1 = int(len(rows_with_mask) - np.argmax(rows_with_mask[::-1]))
    x0 = int(np.argmax(cols_with_mask))
    x1 = int(len(cols_with_mask) - np.argmax(cols_with_mask[::-1]))

    out_dir = osp.dirname(osp.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.output_npy:
        npy_dir = osp.dirname(osp.abspath(args.output_npy))
        if npy_dir:
            os.makedirs(npy_dir, exist_ok=True)
        prob_out = np.lib.format.open_memmap(args.output_npy, mode="w+", dtype=np.float32, shape=(h, w))
    else:
        prob_out = None

    tmp_u8_path = osp.join(
        tempfile.gettempdir(),
        f"infer_u8_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.dat",
    )
    pred_u8 = np.memmap(tmp_u8_path, mode="w+", dtype=np.uint8, shape=(h, w))
    pred_u8[:] = 0

    use_amp = bool(args.amp and device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    print(
        f"streaming sliding_window roi_size={roi_size} sw_batch_size={batch_size} overlap={overlap:.4f} "
        f"chunk_size={chunk_size} halo={halo} device={device}",
        flush=True,
    )

    n_total = 0
    runnable_chunks = []
    for cy0 in range(y0, y1, chunk_size):
        cy1 = min(cy0 + chunk_size, y1)
        for cx0 in range(x0, x1, chunk_size):
            cx1 = min(cx0 + chunk_size, x1)
            n_total += 1
            if not bool(np.any(fragment_mask[cy0:cy1, cx0:cx1] != 0)):
                continue
            runnable_chunks.append((cy0, cy1, cx0, cx1))
    n_run = int(len(runnable_chunks))
    print(f"chunks total={n_total} run={n_run}", flush=True)

    with torch.inference_mode():
        for cy0, cy1, cx0, cx1 in tqdm(runnable_chunks, total=n_run, desc="infer", unit="chunk"):
            ey0 = max(0, cy0 - halo)
            ey1 = min(h, cy1 + halo)
            ex0 = max(0, cx0 - halo)
            ex1 = min(w, cx1 + halo)

            patch = volume.read_patch(ey0, ey1, ex0, ex1)
            patch = patch.astype(np.float32, copy=False) / 255.0
            patch = np.transpose(patch, (2, 0, 1))
            patch_t = torch.from_numpy(patch).unsqueeze(0).to(device, non_blocking=True)

            with amp_ctx:
                logits = sliding_window_inference(
                    inputs=patch_t,
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    predictor=model,
                    overlap=float(overlap),
                    mode="gaussian",
                    padding_mode="replicate",
                )
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            sy0 = cy0 - ey0
            sy1 = sy0 + (cy1 - cy0)
            sx0 = cx0 - ex0
            sx1 = sx0 + (cx1 - cx0)
            core = probs[sy0:sy1, sx0:sx1]

            if prob_out is not None:
                prob_out[cy0:cy1, cx0:cx1] = core
            pred_u8[cy0:cy1, cx0:cx1] = (core * 255.0).astype(np.uint8, copy=False)

    if prob_out is not None:
        prob_out.flush()
        print(f"wrote npy: {args.output_npy} shape=({h}, {w})", flush=True)

    pred_u8.flush()
    _save_image(args.output_path, pred_u8)
    print(f"wrote image: {args.output_path} shape=({h}, {w})", flush=True)
    try:
        os.remove(tmp_u8_path)
    except OSError:
        pass

    if cleanup_tmp_root is not None:
        print(f"temporary dataset_root created at {cleanup_tmp_root}", flush=True)


if __name__ == "__main__":
    main()
