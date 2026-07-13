#!/usr/bin/env python3
"""CPU-only dense exemplar search adapter for official DINOv3 ViT-S/16."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ALLOWED_MODELS = {"dinov3_vits16", "dinov3_vits16plus"}
PATCH_SIZE = 16
MAX_SIDE = 2048
MAX_PIXELS = 4_194_304


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image_path = Path(request["image_path"]).resolve()
    repo_path = Path(request["repository_path"]).resolve()
    weights_path = Path(request["weights_path"]).resolve()
    model_name = request.get("model", "dinov3_vits16")
    repository_commit = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if repository_commit != request["repository_commit"]:
        raise ValueError(f"DINOv3 repository commit mismatch: {repository_commit}")
    weights_hash = sha256(weights_path)
    if weights_hash != request["weights_sha256"]:
        raise ValueError("DINOv3 weights SHA-256 mismatch")
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {sorted(ALLOWED_MODELS)}")

    from PIL import Image
    image = np.asarray(Image.open(image_path).convert("L"))
    bbox = request.get("search_bbox", {"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]})
    x, y = int(bbox["x"]), int(bbox["y"])
    width, height = int(bbox["width"]), int(bbox["height"])
    if x < 0 or y < 0 or width < PATCH_SIZE or height < PATCH_SIZE or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("search_bbox is outside the image or too small")
    if width > MAX_SIDE or height > MAX_SIDE or width * height > MAX_PIXELS:
        raise ValueError("search_bbox exceeds the DINOv3 CPU limit")
    crop = image[y:y + height, x:x + width]
    padded_h = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    padded_w = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    crop = np.pad(crop, ((0, padded_h - height), (0, padded_w - width)), mode="edge")

    import torch
    torch.set_num_threads(max(1, int(request.get("cpu_threads", 4))))
    model = torch.hub.load(str(repo_path), model_name, source="local", weights=str(weights_path)).eval().to("cpu")
    tensor = torch.from_numpy(crop.copy()).float().div_(255.0)
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0)
    with torch.inference_mode():
        features = model.forward_features(tensor)["x_norm_patchtokens"][0]
        features = torch.nn.functional.normalize(features.float(), dim=-1)
    grid_h, grid_w = padded_h // PATCH_SIZE, padded_w // PATCH_SIZE
    features = features.reshape(grid_h, grid_w, -1)

    def exemplar_mean(points: list[dict]) -> torch.Tensor:
        maps = []
        for point in points:
            px, py = float(point["x"]) - x, float(point["y"]) - y
            if px < 0 or py < 0 or px >= width or py >= height:
                raise ValueError("exemplar point is outside search_bbox")
            reference = features[min(grid_h - 1, int(py) // PATCH_SIZE), min(grid_w - 1, int(px) // PATCH_SIZE)]
            maps.append(torch.einsum("hwd,d->hw", features, reference))
        return torch.stack(maps).mean(0) if maps else torch.zeros((grid_h, grid_w))

    positive = exemplar_mean(request["positive_examples"])
    negative = exemplar_mean(request.get("negative_examples", []))
    combined = positive - negative
    top_k = min(int(request.get("top_k", 100)), combined.numel())
    values, indices = torch.topk(combined.flatten(), top_k)
    matches = []
    for value, index in zip(values.tolist(), indices.tolist()):
        gy, gx = divmod(index, grid_w)
        matches.append({"x": x + gx * PATCH_SIZE + PATCH_SIZE / 2,
                        "y": y + gy * PATCH_SIZE + PATCH_SIZE / 2,
                        "exemplar_similarity": value})

    for name, value in (("positive-similarity", positive), ("negative-similarity", negative), ("exemplar-similarity", combined)):
        array = value.cpu().numpy().astype(np.float32)
        np.save(output / f"{name}.npy", array)
        lo, hi = np.percentile(array[np.isfinite(array)], (1, 99))
        preview = np.zeros(array.shape, np.uint8) if hi <= lo else np.clip((array - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
        Image.fromarray(preview).resize((padded_w, padded_h), resample=Image.Resampling.NEAREST).crop((0, 0, width, height)).save(output / f"{name}.png")

    manifest = {
        "kind": "dinov3_exemplar_cpu_v1",
        "backend": "cpu",
        "model": model_name,
        "repository": str(repo_path),
        "repository_commit": repository_commit,
        "weights": str(weights_path),
        "weights_sha256": weights_hash,
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "search_bbox": bbox,
        "patch_size": PATCH_SIZE,
        "positive_examples": request["positive_examples"],
        "negative_examples": request.get("negative_examples", []),
        "score_semantics": "positive_minus_negative_cosine_similarity_not_ink_probability",
        "top_matches": matches,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
