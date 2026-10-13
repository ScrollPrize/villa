"""Exercise the built Python binding through the production SFTP transport.

The existing C++ test executable acts as ssh on PATH and execs the installed
OpenSSH sftp-server. No network, credentials, Qt application, or Python SSH
implementation is needed. Authentication itself remains a live/manual test.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def expected(dtype):
    import numpy as np

    data = np.arange(64, dtype=dtype).reshape(4, 4, 4)
    data += 1 if dtype == "uint8" else 1000
    data[2:, 2:, 2:] = 0  # Deliberately absent sparse chunk.
    return data


def create_volume(root, dtype):
    import numpy as np

    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"scan": {"voxelsize": 1.0}}))
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4",
        "axes": [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
        "datasets": [{"path": str(level), "coordinateTransformations": [
            {"type": "scale", "scale": [2**level] * 3}
        ]} for level in range(2)],
    }]}))
    for level, data in enumerate((expected(dtype), np.full((2, 2, 2), 17, dtype=dtype))):
        directory = root / str(level)
        directory.mkdir()
        (directory / ".zarray").write_text(json.dumps({
            "zarr_format": 2, "shape": list(data.shape), "chunks": [2, 2, 2],
            "dtype": "|u1" if dtype == "uint8" else "<u2", "compressor": None,
            "fill_value": 0, "order": "C", "filters": None,
        }))
        for z, y, x in itertools.product(range(data.shape[0] // 2), repeat=3):
            if level == 0 and (z, y, x) == (1, 1, 1):
                continue
            chunk = data[z*2:z*2+2, y*2:y*2+2, x*2:x*2+2]
            (directory / f"{z}.{y}.{x}").write_bytes(
                chunk.astype("u1" if dtype == "uint8" else "<u2").tobytes())


def worker(root, phase):
    import numpy as np
    import vc

    vc.set_chunk_cache_io_threads(2)
    vc.set_chunk_cache_budget(8 * 1024 * 1024)
    for dtype in ("uint8", "uint16"):
        path = root / f"{dtype} space.zarr"
        url = "sftp://binding-test" + path.as_uri().removeprefix("file://")
        for scheme in ("sftp", "ssh"):
            volume = vc.Volume.open_url(url.replace("sftp:", scheme + ":", 1))
            assert volume.is_remote
            assert volume.remote_url == url
            assert volume.remote_locator == url
            assert tuple(volume.shape) == (4, 4, 4)
            assert volume.num_scales == 2
            assert Path(volume.remote_cache_root) == root / "cache"
            volume.prefetch_zyx((0, 0, 0), (2, 2, 2), wait=True)
            np.testing.assert_array_equal(volume.read_chunk(0, (0, 0, 0)), expected(dtype)[:2, :2, :2])
            np.testing.assert_array_equal(
                volume.read_zyx((0, 0, 0), (4, 4, 4)), expected(dtype))
            np.testing.assert_array_equal(
                volume.read_zyx((0, 0, 0), (2, 2, 2), level=1),
                np.full((2, 2, 2), 17, dtype=dtype))
            if dtype == "uint8":
                # The surface sampler reserves the all-zero coordinate as a sentinel.
                coords = np.array([[[1, 0, 0], [1, 1, 1]]], dtype=np.float32)
                image, valid, *_ = volume.sample_coords(
                    coords, np.ones((1, 2), dtype=bool), sampling="nearest")
                np.testing.assert_array_equal(image, [[[2], [22]]])
                assert np.asarray(valid).all()
            assert any(p.is_file() for p in Path(volume.remote_cache_path).rglob("*"))
        rebased = vc.Volume.open_url(url + "#vc-base-scale=1")
        assert rebased.base_scale_level == 1
        assert tuple(rebased.shape) == (2, 2, 2)
        np.testing.assert_array_equal(
            rebased.read_zyx((0, 0, 0), (2, 2, 2)), np.full((2, 2, 2), 17, dtype=dtype))

    for bad_url in ("sftp://binding-test" + str(root / "missing.zarr"),
                    "sftp://user:password@binding-test/data.zarr"):
        try:
            vc.Volume.open_url(bad_url)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid/missing remote volume was accepted")
    print(f"Python SFTP {phase}: reads, sampling, prefetch, sparse data, scales and cache passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-fixture", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--phase", default="cold")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.phase)
        return 0
    server = next((p for p in ("/usr/lib/ssh/sftp-server", "/usr/lib/openssh/sftp-server",
                              "/usr/libexec/sftp-server") if Path(p).is_file()), None)
    if os.name == "nt" or server is None:
        print("SKIP: Python integration fixture requires an installed POSIX OpenSSH sftp-server")
        return 77
    assert args.ssh_fixture.is_file(), "Build test_sftp_fetch first"
    with tempfile.TemporaryDirectory(prefix="python-sftp-", dir=args.work_dir) as temporary:
        root = Path(temporary)
        (root / "ssh").symlink_to(args.ssh_fixture.resolve())
        config = root / "config"
        config.mkdir()
        (config / "VC3D.ini").write_text(f"[viewer]\nremote_cache_dir={root / 'cache'}\n")
        for dtype in ("uint8", "uint16"):
            create_volume(root / f"{dtype} space.zarr", dtype)
        env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                   VC3D_CONFIG_DIR=str(config), VC_SFTP_REAL_SERVER=server)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(root)]
        subprocess.run(command, env=env, check=True, timeout=90)
        # Retain source metadata, but remove its voxel payloads. A fresh process
        # must read the persistent disk cache, not RAM or the original files.
        for chunk in root.glob("*.zarr/*/*"):
            if not chunk.name.startswith("."):
                chunk.unlink()
        subprocess.run(command + ["--phase", "warm-disk"], env=env, check=True, timeout=90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
