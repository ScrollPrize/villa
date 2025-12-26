import json
import click
from pathlib import Path

from vesuvius.neural_tracing import trace as trace_module


def _parse_point(point):
    """Return (x, y, z) as a tuple of ints from a list/tuple-like object."""
    coords = point
    if len(coords) != 3:
        raise ValueError(f"Point must have 3 coordinates, got {coords}")
    x, y, z = map(int, coords)
    return x, y, z


@click.command()
@click.option("--points_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to JSON file listing evaluation volumes and points")
@click.option("--prefix", type=str, required=True, help="String added to patch UUIDs, e.g. nt-eval_PREFIX_...")
@click.option("--checkpoint_path", "checkpoint_paths", type=click.Path(exists=True), multiple=True, required=True, help="Path to checkpoint file or directory of checkpoints (can be given multiple times)")
@click.option("--steps_per_crop", type=int, default=1, help="Number of steps to take before sampling a new crop")
@click.option("--max_size", type=int, default=60, show_default=True, help="Maximum patch side length (in vertices) for trace_patch_v4")
def main(points_path, prefix, checkpoint_paths, steps_per_crop, max_size):
    """
    Run `trace.trace` for a collection of start points grouped by volume.
    """
    with open(points_path, "rt") as fp:
        volumes = json.load(fp)

    for vol_idx, volume in enumerate(volumes):
        name = volume.get("name", f"volume_{vol_idx:03}")
        volume_zarr = volume.get("volume_zarr")
        if not volume_zarr:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'volume_zarr'")

        if "volume_scale" not in volume:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'volume_scale'")
        vol_scale = int(volume["volume_scale"])

        paths_dir = volume.get("paths_dir")
        if not paths_dir:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'paths_dir'")

        volume_points = volume["point_xyzs"]

        for pt_idx, point in enumerate(volume_points):
            x, y, z = _parse_point(point)

            for checkpoint_path in checkpoint_paths:
                ckpt = Path(checkpoint_path)
                ckpt_dir = ckpt.parent.name if ckpt.is_file() else ckpt.name
                uuid = f"nt-eval_{prefix}_{x}-{y}-{z}_{ckpt_dir}"
                click.echo(
                    f"Tracing volume={name} "
                    f"point_index={pt_idx} start_xyz=({x}, {y}, {z}) "
                    f"checkpoint={checkpoint_path} "
                    f"into {paths_dir}"
                )

                trace_module.trace.main(
                    args=[
                        "--checkpoint_path",
                        checkpoint_path,
                        "--out_path",
                        paths_dir,
                        "--start_xyz",
                        str(x),
                        str(y),
                        str(z),
                        "--volume_zarr",
                        volume_zarr,
                        "--volume_scale",
                        str(vol_scale),
                        "--uuid",
                        uuid,
                        "--steps_per_crop",
                        str(steps_per_crop),
                        "--max_size",
                        str(max_size),
                    ],
                    standalone_mode=False,
                )


if __name__ == "__main__":
    main()

