from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Callable, Sequence

from .catalog import get_catalog, index_volumes, resolve_volume
from .completion import (
    COMMAND_REGISTRY,
    COMMANDS,
    canonical_executable,
    contextual_candidates,
    install_bash_completion,
    provider_id,
    shlex_quote,
)
from .config import display_config, initialize_config, load_config
from .prefetch import prefetch_volume
from .open_data import upload_inference, validate_atlas_inference
from .runs import launch_inference, read_runs, reconcile_runs, resolve_run
from .snapshots import index_snapshots, resolve_snapshot
from .tmux import Tmux


def _resolve_token(token: str, choices: Sequence[str]) -> str:
    if token in choices:
        return token
    matches = sorted(choice for choice in choices if choice.startswith(token))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"unknown command {token!r}; choices: {', '.join(sorted(choices))}")
    raise ValueError(f"ambiguous command {token!r}; matches: {', '.join(matches)}")


def _expand_command(argv: list[str]) -> list[str]:
    if not argv or argv[0].startswith(("-", "_")):
        return argv
    first_choices = sorted({command[0] for command in COMMANDS})
    argv[0] = _resolve_token(argv[0], first_choices)
    children = sorted({command[1] for command in COMMANDS if len(command) > 1 and command[0] == argv[0]})
    if children and len(argv) > 1 and not argv[1].startswith("-"):
        argv[1] = _resolve_token(argv[1], children)
    return argv


def _rewrite_contextual_help(argv: list[str]) -> list[str]:
    if not argv or argv[-1] != "help" or "--" in argv[:-1]:
        return argv
    if len(argv) == 1:
        return ["--help"]
    roots = sorted({command[0] for command in COMMANDS})
    root = _resolve_token(argv[0], roots)
    prefix = [root]
    children = sorted({command[1] for command in COMMANDS if len(command) > 1 and command[0] == root})
    if children and len(argv) > 2 and not argv[1].startswith("-"):
        try:
            prefix.append(_resolve_token(argv[1], children))
        except ValueError:
            pass
    return prefix + ["--help"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="las_manager", description="Manage Lasagna and Fiber inference data")
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    init = config_sub.add_parser("init", help="initialize the global configuration")
    init.add_argument("--force", action="store_true")
    config_sub.add_parser("show", help="show resolved configuration")
    sub.add_parser("fetch", help="refresh the open-data catalog")
    snapshot = sub.add_parser("snapshot")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_ls = snapshot_sub.add_parser("ls", help="list configured snapshots")
    snapshot_ls.add_argument("--backend", choices=("fiber3d", "lasagna"), default=None)
    volume = sub.add_parser("volume")
    volume_sub = volume.add_subparsers(dest="volume_command", required=True)
    volume_ls = volume_sub.add_parser("ls", help="list open-data volumes")
    volume_ls.add_argument("--sample")
    volume_ls.add_argument("--format", dest="data_format")
    volume_ls.add_argument("--json", action="store_true")
    prefetch = volume_sub.add_parser("prefetch", help="download one OME-Zarr group into the configured cache")
    prefetch.add_argument("volume")
    prefetch.add_argument("scale", type=int)
    prefetch.add_argument("--workers", type=int, default=64)
    prefetch.add_argument("--no-remote-inventory", action="store_true")
    inference = sub.add_parser("inference")
    inference_sub = inference.add_subparsers(dest="inference_command", required=True)
    inference_sub.add_parser("ls", help="list durable inference records")
    inference_run = inference_sub.add_parser("run", help="prefetch and launch inference in tmux")
    inference_run.add_argument("snapshot")
    inference_run.add_argument("volume")
    inference_run.add_argument("scale", type=int)
    inference_run.add_argument(
        "--backend", choices=("fiber3d", "lasagna"), default=None,
        help="Restrict snapshot resolution; normally inferred from the snapshot selector.",
    )
    inference_run.add_argument("--download-workers", type=int, default=64)
    inference_run.add_argument("--no-prefetch", action="store_true")
    inference_run.add_argument(
        "--legacy-config", default=None,
        help="Explicit config JSON for a legacy checkpoint without embedded config.",
    )
    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_sub.add_parser("ls", help="list live manager tmux runs")
    tmux = sub.add_parser("tmux")
    tmux_sub = tmux.add_subparsers(dest="tmux_command", required=True)
    attach = tmux_sub.add_parser("attach", help="attach or link a run's tmux window")
    attach.add_argument("run")
    open_data = sub.add_parser("open-data")
    open_data_sub = open_data.add_subparsers(dest="open_data_command", required=True)
    validate = open_data_sub.add_parser("validate", help="validate a completed portable bundle")
    validate.add_argument("inference")
    validate.add_argument("--model-id")
    validate.add_argument("--register-model", action="store_true")
    upload = open_data_sub.add_parser("upload", help="atomically stage and ingest a portable bundle")
    upload.add_argument("inference")
    upload.add_argument("--model-id")
    upload.add_argument(
        "--register-model", action="store_true",
        help="register the explicit/checkpoint Atlas model after presenting its metadata",
    )
    completion = sub.add_parser("completion", help="emit or install shell completion")
    completion.add_argument("completion_action", choices=("bash", "zsh", "install"))
    completion.add_argument("shell", nargs="?", choices=("bash",))
    hidden = sub.add_parser("_complete", help=argparse.SUPPRESS)
    hidden.add_argument("kind", choices=("volume", "snapshot", "inference", "run"))
    hidden.add_argument("prefix", nargs="?", default="")
    contextual = sub.add_parser("_complete-argv", help=argparse.SUPPRESS)
    contextual.add_argument("words", nargs=argparse.REMAINDER)
    sub.add_parser("_completion-provider-id", help=argparse.SUPPRESS)
    return parser


def _completion_script(
    shell: str,
    *,
    command: str = "las_manager",
    function_name: str | None = None,
    register: bool = True,
) -> str:
    roots = " ".join(sorted({command[0] for command in COMMANDS}))
    children = {
        root: " ".join(sorted(command[1] for command in COMMANDS if len(command) > 1 and command[0] == root))
        for root in sorted({command[0] for command in COMMANDS})
    }
    if shell == "bash":
        function_name = function_name or "_las_manager_complete"
        script = r'''@FUNCTION@() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  local lines value description
  lines="$( @COMMAND@ _complete-argv "${COMP_WORDS[@]:1}" 2>/dev/null)"
  COMPREPLY=()
  while IFS=$'\t' read -r value description; do
    [[ -n "$value" && "$value" == "$cur"* ]] && COMPREPLY+=("$value")
  done <<< "$lines"
}
@REGISTER@'''
    else:
        script = r'''#compdef las_manager
_las_manager_dynamic() {
  local line
  local -a lines values
  lines=("${(@f)$(@COMMAND@ _complete-argv "${words[@]:2}" 2>/dev/null)}")
  for line in $lines; do
    values+=("${line%%$'\t'*}")
  done
  compadd -- $values
}
_las_manager() {
  _las_manager_dynamic
}
compdef _las_manager las_manager'''
    replacements = {
        "ROOTS": roots,
        "FUNCTION": function_name or "_las_manager",
        "COMMAND": shlex_quote(command),
        "REGISTER": f"complete -F {function_name or '_las_manager_complete'} las_manager" if register else "",
        **{root.upper(): value for root, value in children.items()},
    }
    for name, value in replacements.items():
        script = script.replace(f"@{name}@", value)
    return script


def _complete(config, kind: str, prefix: str) -> None:
    if kind == "volume":
        records = index_volumes(get_catalog(config, allow_network=False))
        for record in records:
            if record.selector.startswith(prefix):
                shape = "x".join(str(v) for v in record.shape)
                print(f"{record.selector}\t{shape} {record.pixel_size_um or '-'}um")
    elif kind == "snapshot":
        for record in index_snapshots(config, cached_only=True):
            if record.selector.startswith(prefix):
                print(f"{record.selector}\tstep {record.step if record.step is not None else '-'}")
    else:
        client = Tmux()
        for _path, record in read_runs(config):
            selector = record.get("run_name", "")
            session = record.get("tmux_session", "")
            if selector.startswith(prefix) and (kind != "run" or (session and client.has_session(session))):
                print(f"{selector}\t{record.get('status', '-')}")


def _age(created_at: str | None) -> str:
    if not created_at:
        return "-"
    try:
        then = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        seconds = max(0, int((datetime.now(timezone.utc) - then).total_seconds()))
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"
    except ValueError:
        return "-"


def _print_run(path, record) -> None:
    source = record.get("source", {})
    volume = source.get("volume", {})
    snapshot = record.get("snapshot", {})
    print(
        f"{record.get('run_name', path.name)}\tstatus={record.get('status', '-')}"
        f"\tage={_age(record.get('created_at'))}\tvolume={volume.get('selector', '-')}"
        f"\tsnapshot={snapshot.get('selector', '-')}\tscale={source.get('scale', '-')}"
        f"\tsession={record.get('tmux_session', '-')}\tlog={path / record.get('log_path', 'run.log')}"
    )


def _print_volume(record) -> None:
    shape = "x".join(str(v) for v in record.shape) or "-"
    voxel = f"{record.pixel_size_um:g}um" if record.pixel_size_um is not None else "-"
    origins = sorted({root.get("type", "?") for origin in record.origins for root in origin.get("access_roots", ())})
    print(f"{record.selector}\tid={record.volume_id}\tshape={shape}\tvoxel={voxel}\tformat={record.data_format or '-'}\torigins={','.join(origins) or '-'}")


def _print_snapshot(index: int, record) -> None:
    metric = "-" if record.metric_value is None else f"{record.metric_value:.6g}"
    patch = "-" if record.patch_shape is None else "x".join(str(v) for v in record.patch_shape)
    atlas = record.atlas_model_id or "unresolved"
    print(f"{index}\t{record.selector}\tstep={record.step if record.step is not None else '-'}\tmetric={record.metric_name or '-'}:{metric}\tpatch={patch}\tprecision={record.precision_policy or '-'}\tatlas_model={atlas}\tsha256={record.sha256[:12]}")


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        inference_args: list[str] = []
        parse_list = list(args_list)
        if "--" in parse_list:
            separator = parse_list.index("--")
            inference_args = parse_list[separator + 1:]
            parse_list = parse_list[:separator]
        parse_list = _rewrite_contextual_help(parse_list)
        args = _parser().parse_args(_expand_command(parse_list))
        if args.command == "config" and args.config_command == "init":
            print(initialize_config(force=args.force))
            return 0
        if args.command == "_completion-provider-id":
            print(provider_id(canonical_executable(sys.argv[0])))
            return 0
        if args.command == "_complete-argv":
            try:
                config = load_config()
            except (FileNotFoundError, ValueError, OSError):
                config = None
            for value, description in contextual_candidates(config, args.words):
                print(f"{value}\t{description}")
            return 0
        if args.command == "completion":
            if args.completion_action in {"bash", "zsh"}:
                print(_completion_script(args.completion_action))
                return 0
            shell = args.shell or "bash"
            executable = canonical_executable(sys.argv[0])
            identity = provider_id(executable)
            provider_script = _completion_script(
                shell,
                command=str(executable),
                function_name=f"_las_manager_complete_{identity}",
                register=False,
            )
            installed = install_bash_completion(executable, provider_script)
            print(installed)
            print("Open a new shell, or source this file to enable completion now.")
            return 0
        config = load_config()
        if args.command == "config":
            print(json.dumps(display_config(config), indent=2, sort_keys=True))
        elif args.command == "fetch":
            cache = get_catalog(config, force_refresh=True)
            if cache.warning:
                print(f"warning: {cache.warning}", file=sys.stderr)
            print(f"catalog sha256={cache.metadata['sha256']} fetched_at={cache.metadata.get('fetched_at')} samples={len(cache.document.get('samples', {}))}")
        elif args.command == "volume":
            cache = get_catalog(config)
            if cache.warning:
                print(f"warning: {cache.warning}", file=sys.stderr)
            records = index_volumes(cache)
            if args.volume_command == "prefetch":
                volume = resolve_volume(records, args.volume)
                path = prefetch_volume(
                    config, volume, args.scale, workers=args.workers,
                    remote_inventory=not args.no_remote_inventory,
                )
                print(path)
                return 0
            if args.sample:
                records = [record for record in records if record.sample_id == args.sample]
            if args.data_format:
                records = [record for record in records if record.data_format == args.data_format]
            if args.json:
                print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
            else:
                for record in records:
                    _print_volume(record)
        elif args.command == "snapshot":
            records = index_snapshots(config)
            if args.backend:
                records = [record for record in records if record.backend == args.backend]
            for index, record in enumerate(records, 1):
                _print_snapshot(index, record)
        elif args.command == "inference":
            if args.inference_command == "ls":
                for path, record in reconcile_runs(config):
                    _print_run(path, record)
            else:
                cache = get_catalog(config)
                volume = resolve_volume(index_volumes(cache), args.volume)
                snapshots = index_snapshots(config)
                if args.backend:
                    snapshots = [record for record in snapshots if record.backend == args.backend]
                snapshot = resolve_snapshot(snapshots, args.snapshot)
                if not args.no_prefetch:
                    prefetch_volume(config, volume, args.scale, workers=args.download_workers)
                run_dir = launch_inference(
                    config, snapshot, volume, args.scale,
                    original_argv=args_list, extra_args=inference_args,
                    legacy_config=args.legacy_config,
                )
                print(run_dir)
        elif args.command == "run":
            client = Tmux()
            for path, record in read_runs(config):
                if record.get("tmux_session") and client.has_session(record["tmux_session"]):
                    _print_run(path, record)
        elif args.command == "tmux":
            _path, record = resolve_run(config, args.run)
            session = record.get("tmux_session")
            client = Tmux()
            if not session or not client.has_session(session):
                raise ValueError(f"run {record.get('run_name')!r} has no live tmux session")
            client.attach(session)
        elif args.command == "open-data":
            if args.open_data_command == "validate":
                plan, atlas = validate_atlas_inference(
                    config, args.inference, model_id=args.model_id,
                    register_model=args.register_model,
                )
                print(json.dumps({
                    "run_uuid": plan.provenance["run_uuid"],
                    "artifact_kind": plan.provenance["artifact_kind"],
                    "model_id": plan.model_id,
                    "staging_url": f"s3://{plan.bucket}/{plan.prefix}/",
                    "files": len(plan.files),
                    "bundle_digest": plan.bundle_digest,
                    "atlas": atlas,
                }, indent=2, sort_keys=True))
            else:
                record = upload_inference(
                    config, args.inference, model_id=args.model_id,
                    register_model=args.register_model,
                )
                print(json.dumps({
                    "run_uuid": record["run_uuid"],
                    "staging_upload": record["lifecycle"]["staging_upload"],
                    "atlas_ingest": record["lifecycle"]["atlas_ingest"],
                    "atlas_publication": record["lifecycle"]["atlas_publication"],
                    "staging_url": record["upload"]["staging_url"],
                }, indent=2, sort_keys=True))
        elif args.command == "_complete":
            _complete(config, args.kind, args.prefix)
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"las_manager: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
