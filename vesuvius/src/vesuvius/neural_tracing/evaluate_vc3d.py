import json
import subprocess
import sys
import time
from pathlib import Path

import click


def _parse_point(point):
    """Return (x, y, z) as a tuple of ints from a list/tuple-like object."""
    coords = point
    if len(coords) != 3:
        raise ValueError(f"Point must have 3 coordinates, got {coords}")
    x, y, z = map(int, coords)
    return x, y, z


def _default_grow_seg_bin():
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "build" / "bin" / "vc_grow_seg_from_seed"


def _wait_for_socket(socket_path, process, timeout_s=60.0):
    start = time.time()
    socket_path = Path(socket_path)
    while time.time() - start < timeout_s:
        if socket_path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError("trace_service exited before creating socket")
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for trace_service socket at {socket_path}")


def _start_trace_service(checkpoint_path, volume_zarr, volume_scale, socket_path):
    cmd = [
        sys.executable,
        "-m",
        "vesuvius.neural_tracing.trace_service",
        "--checkpoint_path",
        str(checkpoint_path),
        "--volume_zarr",
        volume_zarr,
        "--volume_scale",
        str(volume_scale),
        "--socket_path",
        str(socket_path),
    ]
    return subprocess.Popen(cmd)


@click.command()
@click.option(
    "--points_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to JSON file listing evaluation volumes and points",
)
@click.option(
    "--prefix",
    type=str,
    required=True,
    help="String added to patch UUIDs, e.g. nt-eval_(x-y-z)_PREFIX_...",
)
@click.option(
    "--checkpoint_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to checkpoint file used by trace_service",
)
@click.option(
    "--params_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="JSON parameters file passed to vc_grow_seg_from_seed",
)
@click.option(
    "--grow_seg_bin",
    type=click.Path(exists=True, dir_okay=False),
    default=lambda: str(_default_grow_seg_bin()),
    show_default=True,
    help="Path to the vc_grow_seg_from_seed binary",
)
def main(points_path, prefix, checkpoint_path, params_path, grow_seg_bin):
    """
    Run vc_grow_seg_from_seed for a collection of start points grouped by volume.
    """
    with open(points_path, "rt") as fp:
        volumes = json.load(fp)

    params_path = Path(params_path)
    with open(params_path, "rt") as fp:
        params = json.load(fp)
    socket_path = params.get("neural_socket")
    if not socket_path:
        raise click.UsageError(
            "Params JSON must include a non-empty 'neural_socket' for trace_service"
        )

    checkpoint_path = Path(checkpoint_path)
    ckpt_dir = checkpoint_path.parent.name if checkpoint_path.is_file() else checkpoint_path.name
    grow_seg_bin = Path(grow_seg_bin)
    if not grow_seg_bin.exists():
        raise click.UsageError(f"vc_grow_seg_from_seed not found at {grow_seg_bin}")

    for vol_idx, volume in enumerate(volumes):
        name = volume.get("name", f"volume_{vol_idx:03}")
        volume_zarr = volume.get("volume_zarr")
        if not volume_zarr:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'volume_zarr'")

        if "volume_scale" not in volume:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'volume_scale'")
        volume_scale = int(volume["volume_scale"])

        paths_dir = volume.get("paths_dir")
        if not paths_dir:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'paths_dir'")

        volume_points = volume.get("point_xyzs")
        if volume_points is None:
            raise click.UsageError(f"Volume entry {name!r} is missing required key 'point_xyzs'")

        socket_file = Path(socket_path)
        if socket_file.exists():
            click.echo(f"Warning: removing stale trace_service socket at {socket_file}")
            socket_file.unlink()
        click.echo(f"Starting trace_service for volume={name} socket={socket_path}")
        service = _start_trace_service(
            checkpoint_path=checkpoint_path,
            volume_zarr=volume_zarr,
            volume_scale=volume_scale,
            socket_path=socket_path,
        )
        try:
            _wait_for_socket(socket_path, service)
            for pt_idx, point in enumerate(volume_points):
                x, y, z = _parse_point(point)
                uuid = f"nt-eval_{x}-{y}-{z}_{prefix}_{ckpt_dir}"
                out_dir = Path(paths_dir) / uuid
                out_dir.mkdir(parents=True, exist_ok=True)

                click.echo(
                    f"Tracing volume={name} "
                    f"point_index={pt_idx} start_xyz=({x}, {y}, {z}) "
                    f"params={params_path} "
                    f"into {out_dir}"
                )

                cmd = [
                    str(grow_seg_bin),
                    "--volume",
                    volume_zarr,
                    "--target-dir",
                    str(out_dir),
                    "--params",
                    str(params_path),
                    "--seed",
                    str(x),
                    str(y),
                    str(z),
                    "--segment-name",
                    uuid,
                ]
                subprocess.run(cmd, check=True)
        finally:
            service.terminate()
            try:
                service.wait(timeout=10)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()


if __name__ == "__main__":
    main()
