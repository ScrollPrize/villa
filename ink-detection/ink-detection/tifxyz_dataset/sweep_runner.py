import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def _parse_override_value(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _set_nested(config, dotted_key, value):
    parts = [part for part in dotted_key.split("__") if part]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    cursor = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        elif not isinstance(next_value, dict):
            raise TypeError(
                f"Cannot set nested override for {dotted_key!r}: "
                f"intermediate key {part!r} is a {type(next_value).__name__}"
            )
        cursor = next_value

    cursor[parts[-1]] = value


def _parse_overrides(argv):
    overrides = {}
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if not arg.startswith("--"):
            raise ValueError(f"Unexpected argument format: {arg!r}")

        if "=" in arg:
            key, raw_value = arg[2:].split("=", 1)
            idx += 1
        else:
            if idx + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg!r}")
            key = arg[2:]
            raw_value = argv[idx + 1]
            idx += 2

        overrides[key] = _parse_override_value(raw_value)

    return overrides


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python sweep_runner.py /path/to/base_config.json [--key=value ...]")

    base_config_path = Path(sys.argv[1]).resolve()
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    overrides = _parse_overrides(sys.argv[2:])
    for key, value in overrides.items():
        _set_nested(config, key, value)

    if "out_dir" in config and config["out_dir"] not in (None, ""):
        config["out_dir"] = os.path.join(str(config["out_dir"]), f"sweep-{uuid.uuid4().hex[:12]}")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(config, tmp, indent=2)
        tmp_path = tmp.name

    try:
        subprocess.run(
            [sys.executable, str(base_config_path.parent / "train.py"), tmp_path],
            check=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
