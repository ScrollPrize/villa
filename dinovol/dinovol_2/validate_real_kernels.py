"""Validate kernels using checkpoint features from fully cached real volume crops."""
from __future__ import annotations

from dinovol_2.training.config import as_legacy_config
from dinovol_2.training.checkpoint import method_state

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from dinovol_2.dataset.volume_reader import open_volume

from dinovol_2.dataset.cuda_pipeline import normalize_batch
from dinovol_2.loss.ibot_patch_loss import iBOTPatchLoss
from dinovol_2.model import build_backbone
from dinovol_2.model.model import DINOHead
from dinovol_2.ops.point_embeddings import sample_grid_points, sample_selected_grid_points


def prepare(checkpoint, volume, start):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config, state = as_legacy_config(saved["config"], legacy_checkpoint=True), method_state(saved, expected_method="dino")["teacher"]
    crop_size = tuple(config["model"]["global_crops_size"])
    if crop_size != (192, 192, 192):
        raise ValueError("This validation workload expects the 192^3 checkpoint recipe")
    meta = json.loads((volume / ".zarray").read_text())
    array = open_volume(volume)
    # Two sample pairs with overlapping, non-token-aligned crop origins, in view-major order.
    starts = [
        tuple(start),
        tuple(a + 128 for a in start),
        tuple(a + b for a, b in zip(start, (32, 17, 23))),
        tuple(a + b for a, b in zip(start, (147, 157, 139))),
    ]
    crops = []
    for origin in starts:
        if any(a < 0 or a + n > length for a, n, length in zip(origin, crop_size, array.shape)):
            raise ValueError(f"Crop outside array bounds: {origin}")
        ranges = [range(a // c, (a + n - 1) // c + 1)
                  for a, n, c in zip(origin, crop_size, meta["chunks"])]
        for chunk in itertools.product(*ranges):
            key = meta.get("dimension_separator", ".").join(map(str, chunk))
            if not (volume / key).is_file():
                raise FileNotFoundError(f"Crop includes uncached chunk {volume / key}; refusing fill-value data")
        raw = np.asarray(array[tuple(slice(a, a + n) for a, n in zip(origin, crop_size))])
        fraction = float(np.count_nonzero(raw) / raw.size)
        print(json.dumps({"origin_zyx": origin, "shape": raw.shape, "nonzero_fraction": fraction,
                          "raw_min": int(raw.min()), "raw_max": int(raw.max())}), flush=True)
        if fraction < .1:
            raise ValueError("Choose a tissue-containing origin; this crop is mostly empty")
        crops.append(normalize_batch(torch.from_numpy(raw.copy())[None, None],
                                     config["dataset"].get("normalization_scheme", "robust")))
    array.close()

    backbone = build_backbone({**config["model"], "grad_checkpointing": False})
    backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")})
    backbone = backbone.cuda().eval()
    head = DINOHead(
        config["model"]["embed_dim"], 131072, hidden_dim=2048, bottleneck_dim=256,
    )
    head.load_state_dict({k.removeprefix("ibot_head."): v for k, v in state.items() if k.startswith("ibot_head.")})
    head = head.cuda().eval()
    patch = tuple(backbone.patch_size)
    token_shape = tuple(n // p for n, p in zip(crop_size, patch))
    axes = [torch.arange(n, device="cuda") * p + (p - 1) / 2 for n, p in zip(token_shape, patch)]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    source_rows = torch.tensor([2, 3, 0, 1], device="cuda")
    origins = torch.tensor(starts, device="cuda")
    coordinates = centers[None] + (origins - origins[source_rows])[:, None]
    masks = torch.rand(4, centers.shape[0], device="cuda") < .3
    masks[2:] = False
    valid = ((coordinates >= -.5) & (coordinates <= 191.5)).all(-1)
    indices = (masks & valid).flatten().nonzero().flatten()
    counts = (masks & valid).sum(-1).clamp_min(1)
    masks_weight = (1. / counts)[indices // centers.shape[0]]
    teacher_tokens, student_tokens = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row, crop in enumerate(crops):
            x = crop.cuda()
            teacher_tokens.append(backbone(x)["x_norm_patchtokens"].cpu())
            student_tokens.append(backbone(x, masks=masks[row:row + 1])["x_norm_patchtokens"].cpu())
            print(f"Encoded real crop {row + 1}/4", flush=True)
        teacher_tokens = torch.cat(teacher_tokens).cuda()
        student_tokens = torch.cat(student_tokens).cuda()
        grid = teacher_tokens.reshape(4, *token_shape, -1)
        # Reference teacher features are intentionally generated with the existing sampler.
        selected_teacher = sample_grid_points(grid[source_rows], coordinates, crop_size).to(grid.dtype).flatten(0, 1)[indices]
        selected_student = student_tokens.flatten(0, 1)[indices]
        # The trainer casts normalized bottlenecks to the AMP logit dtype before loss.
        features = head(selected_student, apply_last_layer=False).to(torch.bfloat16)
        teacher_features = head(selected_teacher, apply_last_layer=False).to(torch.bfloat16)
        weight = head.last_layer.weight.to(features.dtype)
    result = {
        "metadata": {"checkpoint": str(checkpoint), "volume": str(volume), "origins_zyx": starts,
                     "crop_size": crop_size, "amp_dtype": "bfloat16", "note": "Teacher checkpoint initializes both masked and unmasked paths"},
        "grid": grid, "coordinates": coordinates, "source_rows": source_rows, "indices": indices,
        "student_features": features, "teacher_features": teacher_features,
        "student_weight": weight, "teacher_weight": weight, "masks_weight": masks_weight,
    }
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}


def validate(fixture):
    f = {k: v.cuda() if torch.is_tensor(v) else v for k, v in fixture.items()}
    shape = tuple(f["metadata"]["crop_size"])
    with torch.no_grad():
        reference = sample_grid_points(f["grid"][f["source_rows"]], f["coordinates"], shape)
        reference = reference.to(f["grid"].dtype).flatten(0, 1)[f["indices"]]
        selected = sample_selected_grid_points(f["grid"], f["coordinates"], shape, f["indices"], f["source_rows"])
    torch.testing.assert_close(selected, reference, rtol=1e-5, atol=2e-6)
    report = {"selected_rows": f["indices"].numel(),
              "interpolation_max_abs": float((selected - reference).abs().max())}
    results = []
    m, k = f["student_features"].shape[0], f["student_weight"].shape[0]
    for backend in ("torch", "triton"):
        loss_fn = iBOTPatchLoss(k, masked_loss_chunk_size=4096, kernel_backend=backend).cuda()
        if backend == "torch":
            loss_fn.compile_kernels(backend="inductor", mode="default")
        s = f["student_features"].clone().requires_grad_()
        w = f["student_weight"].clone().requires_grad_()
        with torch.autocast("cuda", dtype=s.dtype):
            scales = loss_fn.sinkhorn_knopp_prototype_scales(
                f["teacher_features"], f["teacher_weight"], .07, torch.tensor([m], device="cuda")
            )
            value = loss_fn.forward_masked_from_features(
                s, w, f["teacher_features"], f["teacher_weight"], scales, .07,
                torch.empty(4, 13824, device="cuda", dtype=torch.bool), f["masks_weight"],
            )
        value.backward()
        results.append((scales.detach().cpu(), value.detach().cpu(), s.grad.cpu(), w.grad.cpu()))
        print(f"Validated real-data {backend} forward/backward", flush=True)
    for name, actual, expected in zip(("scales", "loss", "features_grad", "weight_grad"), results[1], results[0]):
        a, e = actual.float(), expected.float()
        error = (a - e).norm() / e.norm().clamp_min(1e-12)
        report[name] = {"max_abs": float((a - e).abs().max()), "relative_l2": float(error)}
        if name in ("scales", "loss"):
            torch.testing.assert_close(a, e, rtol=1e-5, atol=3e-6)
        else:
            if not bool(a.isfinite().all()) or float(error) > 5e-4:
                raise AssertionError(f"{name}: {report[name]}")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--volume", type=Path, help="Local Zarr array directory, including its pyramid level")
    parser.add_argument("--start", type=int, nargs=3, default=(10240, 3840, 3840))
    parser.add_argument("--features", type=Path, required=True, help="Write/read a reusable feature fixture")
    parser.add_argument("--reuse", action="store_true", help="Validate the saved fixture without rereading the volume")
    args = parser.parse_args()
    torch.manual_seed(5)
    if args.reuse:
        fixture = torch.load(args.features, map_location="cpu", weights_only=True)
    else:
        if args.checkpoint is None or args.volume is None:
            parser.error("--checkpoint and --volume are required without --reuse")
        fixture = prepare(args.checkpoint, args.volume, args.start)
        torch.save(fixture, args.features)
        print(f"Saved real-data features to {args.features}", flush=True)
    validate(fixture)


if __name__ == "__main__":
    main()
