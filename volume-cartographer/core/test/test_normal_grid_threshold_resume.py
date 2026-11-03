"""Exercise threshold changes and resume on a real PHerc0172 level-5 crop.

Usage: python test_normal_grid_threshold_resume.py /path/to/vc_gen_normalgrids
Optionally pass --volume /path/to/a/local/real/u8/zarr to avoid downloading.
Only Python's standard library is required.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request


def stage_crop(root):
    # Preserve the compressed bytes of z=256..383, y/x=0..127. This is the
    # same public scan used by test_normal_grid_live, limited to one chunk.
    base = (
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0172/volumes/"
        "20241024131838-7.910um-53keV-masked.zarr/5/"
    )
    with urllib.request.urlopen(base + ".zarray", timeout=30) as response:
        metadata = json.load(response)
    with urllib.request.urlopen(base + "2/0/0", timeout=30) as response:
        chunk = response.read()
    assert metadata["chunks"] == [128, 128, 128]
    assert metadata["dtype"] == "|u1"
    metadata["shape"] = [128, 128, 128]
    (root / "0/0/0").mkdir(parents=True)
    (root / "0/.zarray").write_text(json.dumps(metadata))
    (root / "0/0/0/0").write_bytes(chunk)
    (root / ".zgroup").write_text('{"zarr_format":2}')
    (root / ".zattrs").write_text(json.dumps({
        "multiscales": [{
            "axes": [{"name": axis, "type": "space"} for axis in "zyx"],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1]},
            ]}],
        }],
    }))
    return root


def snapshot(root):
    # Include directories too: a rejected shard must not create output dirs.
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for path in root.rglob("*")
    }


def check_resume(binary, volume, root):
    env = dict(os.environ, OMP_NUM_THREADS="2")

    def generate(output, threshold=None, extra=(), rejected=False, direction="xy"):
        command = [
            str(binary), "generate", "-i", str(volume), "-o", str(output),
            "--direction", direction, "--sparse-volume", "16", "--preview-every", "0",
            "--io-threads", "1",
        ]
        if threshold is not None:
            command += ["--threshold", str(threshold)]
        result = subprocess.run(
            command + list(extra), env=env, capture_output=True, text=True, timeout=120,
        )
        log = result.stdout + result.stderr
        if rejected:
            assert result.returncode != 0, log
            assert "Use a fresh output directory" in log, log
        else:
            assert result.returncode == 0, log

    # An omitted threshold field is the legacy threshold 1. Also exercise
    # explicit non-default metadata and changes between two non-default cuts.
    for previous in (None, 51):
        output = root / f"threshold-{previous}"
        generate(output, previous)
        grids = sorted((output / "xy").glob("*.grid"))
        assert grids, "the real crop must produce grids for this test"
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata.get("threshold", 1) == (previous or 1)
        if previous is None:
            assert "threshold" not in metadata
        original = snapshot(output)
        generate(output, previous)
        assert snapshot(output) == original, "same-threshold resume changed output"
        # Shard 1 may read metadata while shard 0 republishes it.
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = [pool.submit(generate, output, previous, (
                "--num-parts", "2", "--part-id", str(part),
            )) for part in (0, 1)]
            for run in runs:
                run.result()
        assert snapshot(output) == original, "concurrent resume changed output"

        # Simulate an interrupted run. A different threshold must neither
        # replace the metadata nor fill in the missing slice, on any shard.
        grids[0].unlink()
        interrupted = snapshot(output)
        for requested in ((51,) if previous is None else (1, 128)):
            for part in (0, 1):
                generate(output, requested, (
                    "--num-parts", "2", "--part-id", str(part),
                ), rejected=True)
                assert snapshot(output) == interrupted, "rejection changed output"
            # Check the guard runs before even creating a new direction dir.
            generate(output, requested, rejected=True, direction="xz")
            assert snapshot(output) == interrupted, "rejection created output dirs"

        generate(output, previous)
        assert snapshot(output) == original, "resume did not restore the missing grid"
    print("PASS: threshold mismatches rejected without changes; same-threshold resume restored grids")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--volume", type=Path)
    args = parser.parse_args()
    # The CLI currently reports runtime errors via an uncaught exception.
    # Expected rejections must not leave core dumps in the caller's directory.
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryDirectory(prefix="vc_ng_threshold_resume_") as directory:
        root = Path(directory)
        try:
            volume = args.volume.resolve() if args.volume else stage_crop(root / "crop")
        except (urllib.error.URLError, TimeoutError) as error:
            if os.environ.get("VC_TEST_REQUIRE_NETWORK", "0") not in ("", "0"):
                raise
            print(f"SKIP: could not download PHerc0172 crop: {error}")
            return 77
        check_resume(args.binary.resolve(), volume, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
