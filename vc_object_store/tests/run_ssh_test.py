#!/usr/bin/env python3
"""Exercise hosted-store SSH transport with disposable keys and a local sshd.

Requires installed OpenSSH tools; installs nothing. Optional --store is a closed,
current-schema real store copied for testing. --gui also checks VC3D attachment
and reopening, including a host-trust callback dispatched on the GUI thread.
"""
import argparse
import getpass
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_port(port, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with status {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("Service did not start listening")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--catalog-module", type=Path, help="Also test a freshly built vc_reader _catalog Python module")
    args = parser.parse_args()
    binaries = args.build.resolve() / "bin"
    ssh = shutil.which("ssh")
    sshd = shutil.which("sshd") or "/usr/sbin/sshd"
    if not ssh or not Path(sshd).is_file() or not shutil.which("ssh-keygen"):
        parser.error("Installed ssh, sshd and ssh-keygen are required")
    if (args.gui or args.catalog_module) and not args.store:
        parser.error("--gui and --catalog-module require --store")
    with tempfile.TemporaryDirectory(prefix="vc-ssh-test-") as temporary:
        root = Path(temporary)
        processes = []
        try:
            for name in ("host", "client"):
                subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)], check=True)
            ssh_port = free_port()
            server_config = root / "sshd_config"
            server_config.write_text(
                f"Port {ssh_port}\nListenAddress 127.0.0.1\nHostKey {root / 'host'}\n"
                f"PidFile {root / 'sshd.pid'}\nAuthorizedKeysFile {root / 'client.pub'}\n"
                "StrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n"
                "UsePAM no\nAllowTcpForwarding local\nPermitRootLogin prohibit-password\n"
                "Subsystem sftp internal-sftp\n"
            )
            config = root / "ssh_config"
            config.write_text(
                f"Host vc-store-test\n HostName 127.0.0.1\n Port {ssh_port}\n User {getpass.getuser()}\n"
                f" IdentityFile {root / 'client'}\n IdentitiesOnly yes\n UserKnownHostsFile {root / 'known_hosts'}\n"
                " GlobalKnownHostsFile /dev/null\n"
                f"Host vc-store-refused\n HostName 127.0.0.1\n Port {ssh_port}\n User vc-store-refused-user\n"
                f" UserKnownHostsFile {root / 'known_hosts'}\n GlobalKnownHostsFile /dev/null\n IdentitiesOnly yes\n"
            )
            wrapper = root / "bin" / "ssh"
            wrapper.parent.mkdir()
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(ssh)} -F {shlex.quote(str(config))} "$@"\n')
            wrapper.chmod(0o700)
            env = dict(os.environ, PATH=str(wrapper.parent) + os.pathsep + os.environ.get("PATH", ""),
                       VC_SSH_TEST_HOST="vc-store-test")
            if args.store:
                env["VC_SSH_TEST_STORE"] = str(args.store.resolve())
            processes.append(subprocess.Popen([sshd, "-D", "-e", "-f", str(server_config)]))
            wait_port(ssh_port, processes[-1])
            subprocess.run([str(binaries / "test_object_store_ssh")], env=env, check=True, timeout=120)
            if args.gui or args.catalog_module:
                store = root / "gui.vcobj"
                shutil.copy2(args.store, store)
                artifacts = Path(str(args.store) + ".artifacts")
                if artifacts.exists():
                    shutil.copytree(artifacts, Path(str(store) + ".artifacts"))
                service_port = free_port()
                processes.append(subprocess.Popen([str(binaries / "vc_object_server"), str(store), "--port", str(service_port)]))
                wait_port(service_port, processes[-1])
                if args.catalog_module:
                    code = '''import importlib.util, json, sys, urllib.request
spec = importlib.util.spec_from_file_location("_catalog", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for address in ("ssh://host", "sftp://host:8750/file.vcobj"):
    try: module.SshForward(address)
    except ValueError: pass
    else: raise AssertionError("invalid address accepted")
forward = module.SshForward(sys.argv[2])
with urllib.request.urlopen(forward.endpoint + "/v1/status", timeout=5) as response:
    assert json.load(response)
del forward
print("Python SSH forward passed")
'''
                    subprocess.run([sys.executable, "-c", code, str(args.catalog_module.resolve()),
                                    f"ssh://vc-store-test:{service_port}"], env=env, check=True, timeout=30)
                if args.gui:
                    (root / "known_hosts").unlink()  # exercise the GUI-thread trust prompt
                    env.update(QT_QPA_PLATFORM="offscreen", VC3D_CONFIG_DIR=str(root / "config"),
                               XDG_CONFIG_HOME=str(root / "config"),
                               VC3D_SSH_STORE_TEST=f"ssh://vc-store-test:{service_port}")
                    subprocess.run([str(binaries / "VC3D"), "--test-object-store"], env=env, check=True, timeout=120)
        finally:
            for process in reversed(processes):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
