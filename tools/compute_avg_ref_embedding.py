"""Compute the reference DINO embedding from recorded patch tokens.

Per-row L2-normalize -> mean across rows -> L2-normalize the mean.
"""
import argparse
from pathlib import Path

import numpy as np


def compute_avg_ref(in_paths, out_path):
    arrs = []
    for p in in_paths:
        a = np.load(p).astype(np.float32)
        if a.ndim != 2:
            raise ValueError(f"{p}: expected 2D (N, D), got {a.shape}")
        arrs.append(a)
    embs = np.concatenate(arrs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    embs = embs / norms
    mean = embs.mean(axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mean.astype(np.float32))
    print(f"in_count={len(arrs)} total_rows={embs.shape[0]} dim={embs.shape[1]}")
    print(f"out={out_path} norm={np.linalg.norm(mean):.6f} dtype={mean.dtype}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    compute_avg_ref(args.inputs, args.output)


if __name__ == "__main__":
    main()
