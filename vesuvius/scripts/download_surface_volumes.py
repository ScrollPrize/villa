"""Download verified level-zero public surface Zarrs with bounded rclone concurrency."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["root"])
    flags = ["--config", "/dev/null", "--s3-provider", "AWS", "--s3-region", "us-east-1",
             "--s3-env-auth=false", "--s3-no-check-bucket", "--transfers", "512",
             "--checkers", "128", "--buffer-size", "2M", "--size-only", "--fast-list",
             "--retries", "6", "--low-level-retries", "20", "--stats", "30s"]
    for item in plan["datasets"]:
        dest = Path(item["input"])
        dest.mkdir(parents=True, exist_ok=True)
        ready = root / (item["id"] + "-download.json")
        if ready.exists() and json.loads(ready.read_text()).get("complete"):
            continue
        remote = item["source_s3"].rstrip("/")
        public = remote.replace("s3://vesuvius-challenge-open-data/",
                                "https://vesuvius-challenge-open-data.s3.amazonaws.com/")
        originals = {}
        for name in (".zgroup", ".zattrs", "0/.zarray"):
            with urllib.request.urlopen(public + "/" + name, timeout=60) as stream:
                originals[name] = json.load(stream)
        assert originals["0/.zarray"]["dtype"] == "|u1"
        assert originals["0/.zarray"]["shape"] == item["shape"]
        write_json(dest / "source-metadata.json", originals)
        attrs = deepcopy(originals[".zattrs"])
        for scale in attrs.get("multiscales", []):
            scale["datasets"] = [d for d in scale["datasets"] if d["path"] == "0"]
        attrs["download_source"] = remote
        write_json(dest / ".zgroup", originals[".zgroup"])
        write_json(dest / ".zattrs", attrs)
        started = time.time()
        for operation in ("copy", "check"):
            command = ["rclone", operation, ":s3:" + remote.removeprefix("s3://") + "/0",
                       str(dest / "0"), *flags]
            if operation == "check":
                command += ["--one-way"]
            write_json(ready, {"complete": False, "operation": operation,
                              "started_unix": started, "command": command})
            with (root / (item["id"] + "-download.log")).open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if shutil.disk_usage(root).free < plan.get("min_free_gb", 500) * 1e9:
                        process.terminate()
                        process.wait()
                        raise RuntimeError("Download stopped at the disk reserve")
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f"rclone {operation} failed for {item['id']}")
        elapsed = time.time() - started
        write_json(ready, {"complete": True, "source": remote, "input": str(dest),
                          "seconds": elapsed, "finished_unix": time.time(),
                          "verified": "rclone one-way size check; transfer checksum verification enabled"})
        print(json.dumps({"download_complete": item["id"], "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
