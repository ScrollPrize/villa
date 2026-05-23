import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from batchgeneratorsv2.transforms.noise.extranoisetransforms import DecohesionTransform


IGNORE_LABEL = 2


def _default_dataset_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[7]
    return workspace_root / "data" / "smoke" / "surface-detection" / "kaggle-mini-fixture"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_volume(path: Path) -> torch.Tensor:
    from tifffile import imread

    arr = imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF volume at {path}, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _read_label(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from tifffile import imread

    arr = imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF label volume at {path}, got shape {arr.shape}")
    label = torch.from_numpy((arr == 1).astype(np.float32)).unsqueeze(0)
    valid_mask = torch.from_numpy((arr != IGNORE_LABEL).astype(np.float32)).unsqueeze(0)
    return label, valid_mask


class Tiny3DUNet(nn.Module):
    def __init__(self, features: int = 6):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, features, 3, padding=1),
            nn.InstanceNorm3d(features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.MaxPool3d(2)
        self.mid = nn.Sequential(
            nn.Conv3d(features, features * 2, 3, padding=1),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features * 2, features * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up = nn.ConvTranspose3d(features * 2, features, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv3d(features * 2, features, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.out = nn.Conv3d(features, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.enc1(x)
        mid = self.mid(self.down(enc))
        up = self.up(mid)
        return self.out(self.dec(torch.cat([up, enc], dim=1)))


def _masked_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    probs = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask
    intersection = (probs * target).sum()
    dice = (2 * intersection + 1e-6) / (probs.sum() + target.sum() + 1e-6)
    return bce + (1 - dice)


def _dice_from_logits(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float() * valid_mask
    target = target * valid_mask
    intersection = (pred * target).sum()
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return float(dice.item())


def _make_decohesion_transform() -> DecohesionTransform:
    return DecohesionTransform(
        shift=((-2, 2), (-2, 2)),
        alpha=(0.15, 0.45),
        num_prev_slices=(1, 2),
        smear_axis=1,
        p_per_channel=1.0,
    )


def _augment(image: torch.Tensor, transform: DecohesionTransform, seed: int) -> torch.Tensor:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return transform(image=image.clone())["image"]


def _train_variant(
    name: str,
    train_image: torch.Tensor,
    train_label: torch.Tensor,
    train_mask: torch.Tensor,
    val_image: torch.Tensor,
    val_label: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    use_decohesion: bool,
) -> dict:
    _seed_everything(seed)
    model = Tiny3DUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    transform = _make_decohesion_transform()

    train_image = train_image.to(device)
    train_label = train_label.to(device)
    train_mask = train_mask.to(device)
    val_image = val_image.to(device)
    val_label = val_label.to(device)
    val_mask = val_mask.to(device)

    rows = []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        image = train_image
        if use_decohesion:
            image = _augment(image[0], transform, seed + epoch).unsqueeze(0)
        logits = model(image)
        loss = _masked_bce_dice_loss(logits, train_label, train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            clean_val_logits = model(val_image)
            clean_val_dice = _dice_from_logits(clean_val_logits, val_label, val_mask)
            corrupted_val = _augment(val_image[0], transform, seed + 10000 + epoch).unsqueeze(0)
            corrupted_val_logits = model(corrupted_val)
            corrupted_val_dice = _dice_from_logits(corrupted_val_logits, val_label, val_mask)
        rows.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "clean_val_dice": clean_val_dice,
            "decohesion_val_dice": corrupted_val_dice,
        })

    elapsed = perf_counter() - start
    return {
        "name": name,
        "use_decohesion": use_decohesion,
        "seconds": elapsed,
        "final": rows[-1],
        "epochs": rows,
    }


def _write_tsv(rows: list[dict], path: Path) -> None:
    fieldnames = ["seed", "variant", "epoch", "train_loss", "clean_val_dice", "decohesion_val_dice"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _aggregate_runs(runs: list[dict]) -> dict:
    by_variant = {}
    for run in runs:
        by_variant.setdefault(run["name"], []).append(run["final"])

    aggregates = {}
    for name, finals in by_variant.items():
        aggregates[name] = {
            "train_loss_mean": _mean([row["train_loss"] for row in finals]),
            "clean_val_dice_mean": _mean([row["clean_val_dice"] for row in finals]),
            "decohesion_val_dice_mean": _mean([row["decohesion_val_dice"] for row in finals]),
        }

    if {"baseline", "decohesion"}.issubset(aggregates):
        aggregates["delta_decohesion_minus_baseline"] = {
            "train_loss_mean": aggregates["decohesion"]["train_loss_mean"] - aggregates["baseline"]["train_loss_mean"],
            "clean_val_dice_mean": aggregates["decohesion"]["clean_val_dice_mean"] -
                                   aggregates["baseline"]["clean_val_dice_mean"],
            "decohesion_val_dice_mean": aggregates["decohesion"]["decohesion_val_dice_mean"] -
                                        aggregates["baseline"]["decohesion_val_dice_mean"],
        }
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny train/eval ablation for DecohesionTransform.")
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--out-dir", type=Path, default=Path("decohesion-ablation-out"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", default="201,202,203")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_image = _read_volume(args.dataset_root / "train_images" / "synthetic_surface_000.tif").unsqueeze(0)
    train_label, train_mask = _read_label(args.dataset_root / "train_labels" / "synthetic_surface_000.tif")
    train_label, train_mask = train_label.unsqueeze(0), train_mask.unsqueeze(0)
    val_image = _read_volume(args.dataset_root / "train_images" / "synthetic_surface_001.tif").unsqueeze(0)
    val_label, val_mask = _read_label(args.dataset_root / "train_labels" / "synthetic_surface_001.tif")
    val_label, val_mask = val_label.unsqueeze(0), val_mask.unsqueeze(0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    runs = []
    for seed in seeds:
        baseline = _train_variant("baseline", train_image, train_label, train_mask, val_image, val_label, val_mask,
                                  args.epochs, args.lr, seed, device, False)
        baseline["seed"] = seed
        runs.append(baseline)
        decohesion = _train_variant("decohesion", train_image, train_label, train_mask, val_image, val_label, val_mask,
                                    args.epochs, args.lr, seed, device, True)
        decohesion["seed"] = seed
        runs.append(decohesion)

    epoch_rows = []
    for variant in runs:
        for row in variant["epochs"]:
            epoch_rows.append({"seed": variant["seed"], "variant": variant["name"], **row})
    _write_tsv(epoch_rows, args.out_dir / "decohesion_ablation_epochs.tsv")

    aggregate = _aggregate_runs(runs)
    report = {
        "summary": "Tiny smoke-scale model-performance ablation for DecohesionTransform.",
        "claim_safety": (
            "Uses a two-case synthetic surface-detection fixture. This is real train/eval evidence for the "
            "augmentation path, but not a production nnU-Net scroll-dataset ablation."
        ),
        "dataset_root": str(args.dataset_root),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "seeds": seeds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "runs": runs,
        "aggregate": aggregate,
    }
    (args.out_dir / "decohesion_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "aggregate": aggregate,
        "report": str(args.out_dir / "decohesion_ablation_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
