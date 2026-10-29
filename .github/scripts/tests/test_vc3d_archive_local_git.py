"""Release-archiving integration checks: real local Git, simulated release API.

Run from the repository root (Python 3.9+, Git, Bash; no third-party packages):
    python3 -m unittest discover -s .github/scripts/tests -p 'test_vc3d_archive*.py' -v

Both production helpers run unchanged as subprocesses. Git can use ONLY the
file protocol, all repositories/configuration are disposable, and `gh` is a
strict local fake. No credentials, network, Actions, VC3D binaries or scroll
voxels are used. Release creation below is a fixture operation, NOT the full
publishing YAML. These checks do not establish GitHub permissions/API semantics.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
CHANNELS = ("latest", "stable")
SUFFIXES = ("win64.exe", "win64.zip", "macos-arm64.dmg", "macos-arm64.zip",
            "linux-x86_64.AppImage", "linux-x86_64-debug.tar.zst")


def fake_gh(args):
    """A deliberately small API model; never delegates to a real gh or network."""
    state_path = Path(os.environ["VC3D_TEST_STATE"])
    state = json.loads(state_path.read_text())
    prefix = "repos/offline/fixture/releases/"
    if len(args) < 2 or args[0] != "api" or not args[1].startswith(prefix[:-1]):
        raise SystemExit("unsupported fake gh command")
    endpoint, options = args[1][len(prefix) - 1:], args[2:]
    if endpoint == "?per_page=100" and options == ["--paginate", "--slurp"]:
        state["calls"].append(["GET", "releases?per_page=100"])
        if state.get("deny_listing"):
            state_path.write_text(json.dumps(state, sort_keys=True))
            print("gh: Forbidden (HTTP 403)", file=sys.stderr)
            raise SystemExit(1)
        # Two slurped pages deliberately exercise the real CLI result shape.
        pages = [[], list(state["releases"].values())]
        state_path.write_text(json.dumps(state, sort_keys=True))
        print(json.dumps(pages))
        raise SystemExit(0)
    if not endpoint.startswith("/"):
        raise SystemExit("unsupported fake gh endpoint")
    endpoint = endpoint[1:]
    method = "PATCH" if options[:2] == ["-X", "PATCH"] else "GET"
    state["calls"].append([method, endpoint])

    def reply(value=None, error=None):
        state_path.write_text(json.dumps(state, sort_keys=True))
        if error:
            print(error, file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(value))
        raise SystemExit(0)

    releases = state["releases"]
    if endpoint.startswith("tags/"):
        # The real REST endpoint returns published releases by tag only.
        obj = next((r for r in releases.values()
                    if r["tag_name"] == endpoint[5:] and not r.get("draft", False)), None)
    else:
        obj = releases.get(endpoint)
    if obj is None:
        reply(error="gh: Not Found (HTTP 404)")
    if method == "PATCH":
        if len(options) != 4 or options[2] != "-f" or not options[3].startswith("tag_name="):
            raise SystemExit("unsupported fake PATCH")
        if state.get("deny_patch"):
            reply(error="gh: Forbidden (HTTP 403)")
        obj["tag_name"] = options[3].split("=", 1)[1]
    elif options:
        raise SystemExit("unsupported fake GET options")
    reply(obj)


def release(channel, commit, rid):
    """Six synthetic records; digests describe tiny fixture strings, not VC3D."""
    result = {"id": rid, "tag_name": channel, "target_commitish": "main",
              "name": "VC3D Latest" if channel == "latest" else "VC3D Stable",
              "body": (f"Automatically packaged from main at commit {commit}."
                       if channel == "latest" else
                       f"Promoted from tested package artifacts built at commit {commit}."),
              "assets": []}
    for n, suffix in enumerate(SUFFIXES):
        payload = f"synthetic:{channel}:{commit}:{suffix}".encode()
        result["assets"].append({"id": rid * 10 + n,
            "name": f"VC3D-{commit[:7]}-2026-10-29-{suffix}", "size": len(payload),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest()})
    return result


class LocalFixture:
    def __init__(self, channel):
        self.tmp = tempfile.TemporaryDirectory(prefix="vc3d-archive-test-")
        self.root = Path(self.tmp.name)
        self.work, self.remote = self.root / "work", self.root / "remote.git"
        self.work.mkdir()
        self.channel = channel
        bindir = self.root / "bin"
        bindir.mkdir()
        self.state_path = self.root / "api.json"
        self.env = {
            "PATH": str(bindir) + os.pathsep + os.path.dirname(sys.executable) + os.pathsep + os.defpath,
            "HOME": str(self.root), "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_ALLOW_PROTOCOL": "file",
            "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_DATE": "2026-10-29T12:00:00Z",
            "GIT_COMMITTER_DATE": "2026-10-29T12:00:00Z",
            "GITHUB_REPOSITORY": "offline/fixture", "GITHUB_RUN_ID": "1001",
            "VC3D_TEST_STATE": str(self.state_path),
        }
        gh = bindir / "gh"
        gh.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                      + shlex.quote(str(Path(__file__).resolve())) + ' --fake-gh "$@"\n')
        gh.chmod(0o700)
        self.run("git", "init", "--bare", str(self.remote))
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Offline fixture")
        self.git("config", "user.email", "offline@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")
        self.git("remote", "add", "origin", str(self.remote))
        self.commits = []
        for label in ("old", "next", "third"):
            self.git("commit", "--allow-empty", "-m", label)
            self.commits.append(self.git("rev-parse", "HEAD"))
        self.old, self.new, self.third = self.commits
        self.git("push", "origin", "main")
        self.git("tag", "-a", channel, self.old, "-m", f"VC3D {channel} release at {self.old}")
        self.old_oid = self.git("rev-parse", "refs/tags/" + channel)
        self.archive = f"vc3d-archive-{channel}-{self.old[:12]}-41"
        self.git("update-ref", "refs/tags/" + self.archive, self.old_oid)
        self.git("push", "origin", "--tags")
        self.save({"releases": {"41": release(channel, self.old, 41)}, "calls": []})

    def close(self):
        self.tmp.cleanup()

    def run(self, *args, check=True):
        p = subprocess.run(args, cwd=self.work, env=self.env,
                           capture_output=True, text=True, timeout=20)
        if check and p.returncode:
            raise AssertionError(f"{args!r}: {p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def git(self, *args):
        return self.run("git", *args).stdout.strip()

    def load(self):
        return json.loads(self.state_path.read_text())

    def save(self, state):
        self.state_path.write_text(json.dumps(state, sort_keys=True))

    def refs(self):
        return self.git("ls-remote", "--refs", "origin")

    def patch_calls(self):
        return [c for c in self.load()["calls"] if c[0] == "PATCH"]

    def helper(self, sha, probe=False, check=True):
        args = [sys.executable, str(SCRIPTS / "vc3d_archive_zero_copy.py"), self.channel, sha]
        if probe:
            args.append("--probe-only")
        return self.run(*args, check=check)

    def receipt(self, sha, **kwargs):
        return json.loads(self.helper(sha, **kwargs).stdout)

    def prestage(self, receipt, sha, run_id=1001, check=True):
        self.env["GITHUB_RUN_ID"] = str(run_id)
        return self.run("bash", str(SCRIPTS / "vc3d_atomic_prestage.sh"),
                        self.channel, sha, receipt["old_tag_object"], receipt["archive"],
                        str(receipt["release_id"]), ",".join(map(str, receipt["asset_ids"])),
                        check=check)

    def create_release(self, sha, rid):
        # Fixture operation: uploads/release creation and workflow YAML are not executed.
        state = self.load()
        assert all(r["tag_name"] != self.channel for r in state["releases"].values())
        state["releases"][str(rid)] = release(self.channel, sha, rid)
        self.save(state)

    def complete(self, sha=None, rid=42, run_id=1001):
        sha = sha or self.new
        receipt = self.receipt(sha)
        self.prestage(receipt, sha, run_id=run_id)
        self.create_release(sha, rid)
        return receipt


@unittest.skipUnless(os.name == "posix" and shutil.which("git") and shutil.which("bash"),
                     "requires POSIX, Git and Bash")
class ArchiveLocalGit(unittest.TestCase):
    def fixture(self, channel):
        fixture = LocalFixture(channel)
        self.addCleanup(fixture.close)
        return fixture

    def test_two_promotions_preserve_original_ids_and_exact_tag_objects(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                original = f.load()["releases"]["41"]["assets"]
                first = f.complete()
                second_assets = f.load()["releases"]["42"]["assets"]
                second = f.complete(f.third, rid=43, run_id=1002)
                state = f.load()["releases"]
                self.assertEqual(state["41"]["assets"], original)
                self.assertEqual(state["42"]["assets"], second_assets)
                self.assertEqual(len({a["id"] for r in state.values() for a in r["assets"]}), 18)
                for receipt in (first, second):
                    remote = f.git("ls-remote", "--refs", "origin", "refs/tags/" + receipt["archive"])
                    self.assertEqual(remote.split()[0], receipt["old_tag_object"])
                refs, writes = f.refs(), len(f.patch_calls())
                self.assertEqual(f.receipt(f.third)["status"], "ALREADY_PUBLISHED")
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_crash_after_patch_without_receipt_requires_manual_recovery(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.receipt(f.new)
                refs, writes = f.refs(), len(f.patch_calls())
                before = f.load()["releases"]
                result = f.helper(f.new, probe=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("maintainer recovery required", result.stderr)
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_unreceipted_recovery_rejects_replacement_asset_ids(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.receipt(f.new)  # PATCH succeeded, but the rolling tag has not moved.
                state = f.load()
                state["releases"]["41"]["assets"][0]["id"] += 9999
                f.save(state)
                refs, writes = f.refs(), len(f.patch_calls())
                result = f.helper(f.new, probe=True, check=False)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("maintainer recovery required", result.stderr)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_same_commit_detects_missing_current_prestaged_alias(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.complete()
                stage = f"vc3d-archive-{channel}-{f.new[:12]}-1001"
                f.git("push", "origin", ":refs/tags/" + stage)
                before = f.load()["releases"]
                refs, writes = f.refs(), len(f.patch_calls())
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("staged archive receipt/remote alias mismatch", result.stderr)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_draft_after_six_uploads_does_not_authorize_duplicate_release(self):
        # gh release create drafts, uploads, THEN publishes (not atomically).
        # Previously the 404 by tag was misdiagnosed as a missing release.
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.complete()
                state = f.load()
                state["releases"]["42"]["draft"] = True
                f.save(state)
                before, refs, writes = f.load()["releases"], f.refs(), len(f.patch_calls())
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("interrupted draft", result.stderr)
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)
                # An inability to read the list also must fail closed.
                state = f.load()
                state["deny_listing"] = True
                f.save(state)
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("HTTP 403", result.stderr)

    def test_crash_after_tag_push_allows_release_only_repair(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                receipt = f.receipt(f.new)
                f.prestage(receipt, f.new)
                refs, writes = f.refs(), len(f.patch_calls())
                self.assertEqual(f.receipt(f.new)["status"], "REPAIR_RELEASE_ONLY")
                f.create_release(f.new, 42)
                self.assertEqual(f.receipt(f.new)["status"], "ALREADY_PUBLISHED")
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_old_asset_replaced_is_detected_despite_complete_current_release(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.complete()
                state = f.load()
                state["releases"]["41"]["assets"][0]["id"] += 9999
                f.save(state)
                refs, writes = f.refs(), len(f.patch_calls())
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("historical six asset IDs differ", result.stderr)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)

    def test_same_commit_different_tag_object_is_rejected(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.complete()
                f.git("tag", "-a", "replacement", f.old, "-m", "different annotation")
                oid = f.git("rev-parse", "refs/tags/replacement")
                self.assertNotEqual(oid, f.old_oid)
                f.git("push", "--force", "origin", "refs/tags/replacement:refs/tags/" + f.archive)
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("historical annotated tag object changed", result.stderr)

    def test_atomic_push_rejection_leaves_both_refs_unchanged(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                receipt = f.receipt(f.new)
                stage = f"refs/tags/vc3d-archive-{channel}-{f.new[:12]}-1001"
                hook = f.remote / "hooks" / "update"
                hook.write_text(f'#!/bin/sh\n[ "$1" != "{stage}" ]\n')
                hook.chmod(0o700)
                refs = f.refs()
                result = f.prestage(receipt, f.new, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.load()["releases"]["41"]["assets"]), 6)

    def test_denied_patch_does_not_move_refs_or_change_release(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                state = f.load()
                state["deny_patch"] = True
                f.save(state)
                before, refs = state["releases"], f.refs()
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("HTTP 403", result.stderr)
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(f.refs(), refs)

    def test_missing_bootstrap_and_incomplete_upload_fail_without_mutation(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                f = self.fixture(channel)
                f.git("push", "origin", ":refs/tags/" + f.archive)
                before, refs = f.load()["releases"], f.refs()
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("maintainer bootstrap needed", result.stderr)
                self.assertFalse(f.patch_calls())
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(f.refs(), refs)
                # Restore only the local fixture alias, then simulate a partial new upload.
                f.git("push", "origin", "refs/tags/" + f.archive)
                f.complete()
                state = f.load()
                state["releases"]["42"]["assets"].pop()
                f.save(state)
                before, refs, writes = state["releases"], f.refs(), len(f.patch_calls())
                result = f.helper(f.new, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("5 assets, expected six", result.stderr)
                self.assertEqual(f.load()["releases"], before)
                self.assertEqual(f.refs(), refs)
                self.assertEqual(len(f.patch_calls()), writes)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fake-gh"]:
        fake_gh(sys.argv[2:])
    else:
        unittest.main()
