from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .config import atomic_write_text


Command = tuple[str, ...]
COMMAND_REGISTRY: tuple[tuple[Command, str], ...] = (
    (("config", "init"), "initialize global configuration"),
    (("config", "show"), "show resolved configuration"),
    (("fetch",), "refresh the open-data catalog"),
    (("snapshot", "ls"), "list configured snapshots"),
    (("volume", "ls"), "list open-data volumes"),
    (("volume", "prefetch"), "download one volume scale"),
    (("inference", "ls"), "list durable inference records"),
    (("inference", "run"), "launch inference in tmux"),
    (("run", "ls"), "list live manager runs"),
    (("tmux", "attach"), "attach or link a run window"),
    (("open-data", "validate"), "validate an inference bundle for Atlas"),
    (("open-data", "upload"), "stage and ingest an inference bundle"),
    (("completion", "bash"), "emit Bash completion setup"),
    (("completion", "zsh"), "emit Zsh completion setup"),
    (("completion", "install"), "install path-aware user completion"),
)
COMMANDS: tuple[Command, ...] = tuple(command for command, _description in COMMAND_REGISTRY)


_OPTIONS: dict[Command, dict[str, tuple[str, ...] | None]] = {
    ("config", "init"): {"--force": None},
    ("snapshot", "ls"): {"--backend": ("fiber3d", "lasagna")},
    ("volume", "ls"): {"--sample": ("@samples",), "--format": ("@formats",), "--json": None},
    ("volume", "prefetch"): {"--workers": (), "--no-remote-inventory": None},
    ("inference", "run"): {
        "--backend": ("fiber3d", "lasagna"), "--download-workers": (),
        "--no-prefetch": None, "--legacy-config": (),
    },
    ("open-data", "validate"): {},
    ("open-data", "upload"): {},
}


def _unique(token: str, choices: Sequence[str]) -> str | None:
    if token in choices:
        return token
    matches = [choice for choice in choices if choice.startswith(token)]
    return matches[0] if len(matches) == 1 else None


def _candidate(value: str, description: str = "") -> tuple[str, str]:
    return value, description


def _cached_catalog(config):
    if config is None:
        return None
    try:
        from .catalog import get_catalog

        return get_catalog(config, allow_network=False)
    except (FileNotFoundError, RuntimeError, ValueError, OSError):
        return None


def _volume_records(config) -> list[Any]:
    cache = _cached_catalog(config)
    if cache is None:
        return []
    from .catalog import index_volumes

    return index_volumes(cache)


def _scale_candidates(config, selector: str) -> list[tuple[str, str]]:
    if config is None or not selector:
        return []
    from .catalog import resolve_volume
    from .prefetch import volume_cache_root

    try:
        volume = resolve_volume(_volume_records(config), selector)
        root = volume_cache_root(config, volume)
    except (FileNotFoundError, ValueError, OSError):
        return []
    levels: set[int] = set()
    attrs_path = root / ".zattrs"
    try:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8")) if attrs_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        attrs = {}
    for multiscale in attrs.get("multiscales", ()) if isinstance(attrs, dict) else ():
        if not isinstance(multiscale, dict):
            continue
        for dataset in multiscale.get("datasets", ()) or ():
            path = dataset.get("path") if isinstance(dataset, dict) else None
            if isinstance(path, str) and path.isdigit():
                levels.add(int(path))
    if root.is_dir():
        for child in root.iterdir():
            if child.name.isdigit() and (child / ".zarray").is_file():
                levels.add(int(child.name))
    return [_candidate(str(level), "cached OME-Zarr level") for level in sorted(levels)]


def _dynamic_values(config, marker: str) -> list[tuple[str, str]]:
    if marker in {"@samples", "@formats", "@models"}:
        cache = _cached_catalog(config)
        if cache is None:
            return []
        if marker == "@models":
            models = cache.document.get("models", {})
            values = models.keys() if isinstance(models, dict) else ()
            return [_candidate(str(value), "catalog model") for value in sorted(values)]
        records = _volume_records(config)
        values = {record.sample_id for record in records} if marker == "@samples" else {
            record.data_format for record in records if record.data_format
        }
        return [_candidate(str(value), "catalog value") for value in sorted(values)]
    return []


def contextual_candidates(config, words: Sequence[str]) -> list[tuple[str, str]]:
    """Return completion candidates for argv words excluding the program name."""
    argv = list(words) or [""]
    current = argv[-1]
    prior = argv[:-1]
    roots = sorted({command[0] for command in COMMANDS})
    if not prior:
        return [_candidate(root, "command") for root in roots if root.startswith(current)]
    root = _unique(prior[0], roots)
    if root is None:
        return []
    children = sorted({command[1] for command in COMMANDS if len(command) > 1 and command[0] == root})
    if children and len(prior) == 1:
        return [_candidate(child, "subcommand") for child in children if child.startswith(current)]
    child = _unique(prior[1], children) if children and len(prior) > 1 else None
    command: Command = (root, child) if child is not None else (root,)
    if command not in COMMANDS:
        return []
    args = prior[len(command):]
    options = _OPTIONS.get(command, {})
    if "--" in args:
        return []
    if current.startswith("--") and "=" in current:
        option, value_prefix = current.split("=", 1)
        declared = options.get(option)
        if declared is None:
            return []
        candidates: list[tuple[str, str]] = []
        for value in declared:
            candidates.extend(
                _dynamic_values(config, value)
                if value.startswith("@") else [_candidate(value, "option value")]
            )
        return [
            _candidate(f"{option}={value}", description)
            for value, description in candidates if value.startswith(value_prefix)
        ]
    if args and args[-1] in options and options[args[-1]] is not None:
        values = options[args[-1]] or ()
        candidates: list[tuple[str, str]] = []
        for value in values:
            candidates.extend(_dynamic_values(config, value) if value.startswith("@") else [_candidate(value, "option value")])
        return [item for item in candidates if item[0].startswith(current)]
    if current.startswith("-"):
        return [_candidate(option, "option") for option in sorted(options) if option.startswith(current)]
    positionals: list[str] = []
    skip_value = False
    for value in args:
        if skip_value:
            skip_value = False
        elif value in options:
            skip_value = options[value] is not None
        elif not value.startswith("-"):
            positionals.append(value)
    position = len(positionals)
    candidates = []
    if command == ("volume", "prefetch"):
        if position == 0:
            candidates = [
                _candidate(record.selector, "catalog volume") for record in _volume_records(config)
            ]
        elif position == 1:
            candidates = _scale_candidates(config, positionals[0])
    elif command == ("inference", "run"):
        if position == 0 and config is not None:
            try:
                from .snapshots import index_snapshots
                candidates = [_candidate(record.selector, "snapshot") for record in index_snapshots(config, cached_only=True)]
            except (FileNotFoundError, RuntimeError, ValueError, OSError):
                candidates = []
        elif position == 1:
            candidates = [_candidate(record.selector, "catalog volume") for record in _volume_records(config)]
        elif position == 2:
            candidates = _scale_candidates(config, positionals[1])
    elif command in {("tmux", "attach"), ("open-data", "validate"), ("open-data", "upload")} and position == 0 and config is not None:
        try:
            from .runs import read_runs
            from .tmux import Tmux
            records = read_runs(config)
            if command == ("tmux", "attach"):
                client = Tmux()
                records = [item for item in records if item[1].get("tmux_session") and client.has_session(item[1]["tmux_session"])]
            candidates = [
                _candidate(record.get("run_name", path.name), str(record.get("status", "-")))
                for path, record in records
            ]
        except (FileNotFoundError, RuntimeError, ValueError, OSError):
            candidates = []
    elif command == ("completion", "install") and position == 0:
        candidates = [_candidate("bash", "shell")]
    return [item for item in candidates if item[0].startswith(current)]


def canonical_executable(argv0: str) -> Path:
    """Return the stable external command identity used by install and dispatch."""
    candidate = Path(argv0).expanduser()
    if not candidate.is_absolute():
        found = shutil.which(argv0)
        if found is None:
            raise ValueError(f"cannot resolve executable {argv0!r} from PATH")
        candidate = Path(found)
    if candidate.name != "las_manager":
        raise ValueError(
            "completion installation requires the installed las_manager console script; "
            "invoke las_manager directly instead of python -m"
        )
    resolved = candidate.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(
            "completion installation requires the installed las_manager console script; "
            "invoke las_manager directly instead of python -m"
        )
    return resolved


def provider_id(executable: Path) -> str:
    return hashlib.sha256(str(executable.resolve()).encode("utf-8")).hexdigest()[:20]


def _data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()
    return (Path.home() / ".local" / "share").resolve()


def _load_registry(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid completion provider registry {path}: {error}") from error
    if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
        raise ValueError(f"invalid completion provider registry {path}: expected string mapping")
    return dict(raw)


def _dispatcher_script(providers: Mapping[str, str], provider_dir: Path) -> str:
    lines = [
        "# Generated by las_manager completion install; do not edit.",
    ]
    for identity in sorted(providers):
        provider_path = provider_dir / f"{identity}.bash"
        lines.append(f"[[ -r {shlex_quote(str(provider_path))} ]] && source {shlex_quote(str(provider_path))}")
    lines.extend([
        "_las_manager_completion_dispatch() {",
        "  local executable identity",
        "  executable=\"$(type -P las_manager 2>/dev/null)\" || return 0",
        "  [[ -n \"$executable\" && -f \"$executable\" && -x \"$executable\" ]] || return 0",
        "  identity=\"$(\"$executable\" _completion-provider-id 2>/dev/null)\" || return 0",
        "  case \"$identity\" in",
    ])
    for identity in sorted(providers):
        lines.append(f"    {identity}) _las_manager_complete_{identity} \"$@\" ;;")
    lines.extend([
        "  esac",
        "}",
        "complete -F _las_manager_completion_dispatch las_manager",
        "",
    ])
    return "\n".join(lines)


def shlex_quote(value: str) -> str:
    # Kept local so generated shell text has one auditable quoting boundary.
    import shlex

    return shlex.quote(value)


def install_bash_completion(executable: Path, provider_script: str) -> Path:
    executable = executable.resolve()
    identity = provider_id(executable)
    data_home = _data_home()
    root = data_home / "las_manager" / "completions" / "bash"
    provider_dir = root / "providers"
    registry_path = root / "providers.json"
    loader_path = data_home / "bash-completion" / "completions" / "las_manager"

    providers = _load_registry(registry_path)
    providers[identity] = str(executable)
    atomic_write_text(provider_dir / f"{identity}.bash", provider_script.rstrip() + "\n")
    atomic_write_text(registry_path, json.dumps(providers, indent=2, sort_keys=True) + "\n")
    atomic_write_text(loader_path, _dispatcher_script(providers, provider_dir))
    return loader_path
