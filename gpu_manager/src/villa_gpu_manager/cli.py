"""Service and administrative commands use the same public protocol."""

import argparse
import json
import os
from pathlib import Path

from .client import Client
from .paths import default_socket_path
from .processes import NvidiaMonitor, nvidia_inventory
from .server import Server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--gpus", nargs="+", required=True, help="explicit managed NVIDIA GPU UUIDs")
    serve.add_argument("--cpu-cores", type=int, required=True)
    serve.add_argument("--reserve-bytes", type=int, default=1024**3)
    serve.add_argument("--allow-uid", type=int, action="append", default=[])
    serve.add_argument("--controller-uid", type=int, action="append", default=[])
    serve.add_argument("--shared-group", action="store_true")
    sub.add_parser("status")
    for name in ("pause", "resume"):
        sub.add_parser(name).add_argument("task")
    request = sub.add_parser("request")
    request.add_argument("task")
    request.add_argument("json_file", type=Path)
    overrides = sub.add_parser("override").add_subparsers(dest="action", required=True)
    setting = overrides.add_parser("set")
    setting.add_argument("json_file", type=Path)
    setting.add_argument("--until-task")
    overrides.add_parser("clear")
    args = parser.parse_args(argv)
    if args.socket is None:
        args.socket = default_socket_path()
    if args.command == "serve":
        inventory = nvidia_inventory(args.cpu_cores, args.gpus, args.reserve_bytes)
        with Server(args.socket, inventory, allowed_uids=[os.getuid(), *args.allow_uid],
                    controller_uids=[os.getuid(), *args.controller_uid], shared_group=args.shared_group,
                    monitor=NvidiaMonitor(inventory, args.reserve_bytes)) as server:
            while True:
                try:
                    server.step()
                    if server.engine.draining and all(not t["granted"]["gpus"] and not t["granted"]["cpu_cores"]
                                                       for t in server.engine.tasks.values()):
                        return 0
                except KeyboardInterrupt:
                    if server.engine.draining:
                        print("Forced shutdown: reservations retained in journal; clients must unload.", flush=True)
                        server.close(force=True)
                        return 130
                    server.drain()
                    print("Draining: waiting for clients to release. Ctrl-C again forces shutdown.", flush=True)
    else:
        with Client(args.socket) as client:
            status = client.status()
            if args.command == "status":
                result = status
            elif args.command in ("pause", "resume"):
                result = client.call("set_paused", {"task_id": args.task, "scope": "controller", "paused": args.command == "pause"})
            elif args.command == "request":
                task = next((t for t in status["tasks"] if t["task_id"] == args.task), None)
                if task is None:
                    parser.error("unknown task")
                result = client.call("set_request", {"task_id": args.task, "request": json.loads(args.json_file.read_text()),
                                                     "expected_request_revision": task["request_revision"]})
            else:
                body = {"expected_policy_revision": status["policy_revision"]}
                if args.action == "set":
                    body.update(allocations=json.loads(args.json_file.read_text()), bound_task_id=args.until_task)
                result = client.call("set_override" if args.action == "set" else "clear_override", body)
            print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
