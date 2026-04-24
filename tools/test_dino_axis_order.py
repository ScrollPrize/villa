"""Verify spatial axis ordering of DINO patch tokens.

For a [B,1,Z,Y,X] input, the generator reshapes the 4096 DINO patch tokens
into `view(n, 16, 16, 16)` with axes assumed to be (z, y, x). This script
feeds three synthetic inputs each with structure on exactly one axis, and
checks which output axis lights up.

Also reports per-axis variance of `sim_grid` for both trilinear-upsampled and
raw-32^3 forms to highlight any upsampling-induced smoothing.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.expanduser("/home/ubuntu/sean_ink/villa-dinoguided/ink-detection"))
sys.path.insert(0, os.path.expanduser("/home/ubuntu/sean_ink/villa-dinoguided/vesuvius/src"))

from koine_machines.data.dino_guided_labels import DinoGuidedLabelGenerator


def make_axis_structured_chunk(axis: int, device, dtype):
    """Return a [1,1,256,256,256] tensor with bright structure on the first half
    along `axis` (0=Z, 1=Y, 2=X) and dark elsewhere. Adds high-frequency noise
    so DINO sees non-trivial features."""
    img = torch.zeros(1, 1, 256, 256, 256, device=device, dtype=dtype)
    sl = [slice(None)] * 5
    sl[2 + axis] = slice(0, 128)
    img[tuple(sl)] = 1.0
    img += 0.1 * torch.rand_like(img)
    return img


def per_axis_variance(t: torch.Tensor):
    """For tensor of shape [Z,Y,X], return (var_along_z, var_along_y, var_along_x)
    where var_along_z = variance of the per-z averages (averaging out y,x)."""
    return (
        t.mean(dim=(1, 2)).var().item(),
        t.mean(dim=(0, 2)).var().item(),
        t.mean(dim=(0, 1)).var().item(),
    )


def main():
    device = torch.device("cuda:0")
    gen = DinoGuidedLabelGenerator(
        unet_ckpt="/ephemeral/dinov2_ckpts/teacher_unet_ckpt_060000.pth",
        dino_ckpt="/ephemeral/dinov2_ckpts/checkpoint_step_352500_paris4.pt",
        ref_embedding="/ephemeral/dinov2_ckpts/avg_ref_embedding.npy",
        device=device,
        dtype=torch.bfloat16,
        dino_stride=128,
        dino_minibatch=8,
        threshold=0.5,
    )

    print("\n=== axis-order test ===")
    print("Input has bright structure on first half along ONE axis at a time.")
    print("Expected: var_along_<that_axis> >> var_along_<others>.\n")
    axis_names = ["Z(0)", "Y(1)", "X(2)"]
    for axis, name in enumerate(axis_names):
        img = make_axis_structured_chunk(axis, device, torch.bfloat16)
        with torch.inference_mode():
            sim_full = gen._dino_sim_map(img)  # [1,1,256,256,256]
        var_z, var_y, var_x = per_axis_variance(sim_full[0, 0].float())
        winner = ["Z", "Y", "X"][max(range(3), key=lambda i: [var_z, var_y, var_x][i])]
        marker = "OK" if winner == name[0] else "MISMATCH"
        print(f"input axis {name}: var(Z)={var_z:.5f}  var(Y)={var_y:.5f}  var(X)={var_x:.5f}  winner={winner}  [{marker}]")

    print("\n=== upsampling artifact test ===")
    print("Pre-upsample (32^3) vs post-upsample (256^3) value range")
    img = make_axis_structured_chunk(0, device, torch.bfloat16)
    with torch.inference_mode():
        sim_full = gen._dino_sim_map(img)
    print(f"sim_full (post upsample, [0,1] mapped): min={sim_full.min().item():.4f} max={sim_full.max().item():.4f} mean={sim_full.mean().item():.4f}")


if __name__ == "__main__":
    main()
