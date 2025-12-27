"""
Modify a checkpoint for extended training.
Resets current_epoch to 0 while preserving weights and optimizer momentum.

Usage:
    python extend_training.py /path/to/checkpoint_final.pth
    python extend_training.py /path/to/checkpoint_final.pth -o /path/to/output.pth
"""
import torch
import argparse
from pathlib import Path


def extend_checkpoint(checkpoint_path: str, output_path: str = None):
    """Reset current_epoch to 0 for fresh LR schedule, keep weights + momentum."""
    checkpoint_path = Path(checkpoint_path)

    if output_path is None:
        output_path = checkpoint_path.parent / "checkpoint_extended.pth"
    else:
        output_path = Path(output_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Original current_epoch: {checkpoint['current_epoch']}")
    print(f"Original best_ema: {checkpoint['_best_ema']}")

    # Reset epoch to 0 for fresh LR schedule
    checkpoint['current_epoch'] = 0

    # Optionally reset best_ema so new training can track improvement
    checkpoint['_best_ema'] = -1

    # Save modified checkpoint
    torch.save(checkpoint, output_path)
    print(f"Saved extended checkpoint to: {output_path}")
    print("Preserved: network_weights, optimizer_state (momentum)")
    print("Reset: current_epoch=0, _best_ema=-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify a checkpoint for extended training. "
                    "Resets current_epoch to 0 for fresh LR schedule while preserving weights and optimizer momentum."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint_final.pth")
    parser.add_argument("-o", "--output", help="Output path (default: checkpoint_extended.pth in same directory)")
    args = parser.parse_args()
    extend_checkpoint(args.checkpoint, args.output)
