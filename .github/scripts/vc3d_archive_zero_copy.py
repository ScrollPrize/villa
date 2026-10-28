#!/usr/bin/env python3
"""Archive a VC3D rolling release without deleting/reuploading its six assets.

Intended for the *two* official publishing workflows, before a rolling tag moves.
Only safe after explicit upstream permission review and maintainer bootstrap.
"""
import argparse
import json
import os
import re
import subprocess
import sys


class ArchiveError(Exception):
    pass


def cmd(*argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True)
    if check and p.returncode:
        raise ArchiveError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''}: {p.stderr.strip()[:400]}")
    return p


def git(*argv):
    return cmd("git", *argv).stdout.strip()


def gh(repo, path, *args, allow_404=False):
    result = cmd("gh", "api", f"repos/{repo}/{path}", *args, check=False)
    if result.returncode:
        if allow_404 and "HTTP 404" in result.stderr:
            return None
        raise ArchiveError(f"GitHub API request failed: {result.stderr.strip()[:300]}")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise ArchiveError("invalid GitHub JSON response") from exc


def assets(release):
    items = release.get("assets", [])
    if len(items) != 6:
        raise ArchiveError(f"release {release.get('id')} has {len(items)} assets, expected six")
    unique = set()
    names = set()
    record = {}
    for item in items:
        aid, name, size = item.get("id"), item.get("name"), item.get("size")
        if not isinstance(aid, int) or aid <= 0 or not isinstance(size, int) or size <= 0 or not name:
            raise ArchiveError("asset identity invalid")
        if aid in unique or name in names:
            raise ArchiveError("duplicate asset ID or name")
        unique.add(aid); names.add(name)
        record[aid] = (name, size, item.get("digest"))
    return record


def verify_release_origin(release, channel, source_sha):
    """Bind six original release assets to the *tag's* full source commit.

    Merely keeping asset IDs is not provenance: a tag can already have moved
    without its release being updated (or a malformed release may be present).
    This contract matches the current two upstream VC3D publishing workflows.
    """
    expected_title = "VC3D Latest" if channel == "latest" else "VC3D Stable"
    if release.get("name") != expected_title:
        raise ArchiveError("historical release has unexpected title")
    expected_note = (f"Automatically packaged from main at commit {source_sha}."
                     if channel == "latest" else
                     f"Promoted from tested package artifacts built at commit {source_sha}.")
    if release.get("body", "").strip() != expected_note:
        raise ArchiveError("historical release body contradicts tag commit")
    record = assets(release)
    suffixes = (
        r"win64\.exe", r"win64\.zip", r"macos-[A-Za-z0-9_-]+\.dmg",
        r"macos-[A-Za-z0-9_-]+\.zip", r"linux-[A-Za-z0-9_-]+\.AppImage",
        r"linux-[A-Za-z0-9_-]+-debug\.tar\.zst",
    )
    prefix = re.escape("VC3D-" + source_sha[:7] + "-") + r"[0-9]{4}-[0-9]{2}-[0-9]{2}-"
    names = [v[0] for v in record.values()]
    if not all(sum(bool(re.fullmatch(prefix + suffix, name)) for name in names) == 1
               for suffix in suffixes):
        raise ArchiveError("historical asset platforms/commit mismatch with rolling tag")
    if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", digest or "")
               for _name, _size, digest in record.values()):
        raise ArchiveError("historical asset checksum absent or malformed")
    return record


def remote_object(ref):
    p = cmd("git", "ls-remote", "--refs", "origin", f"refs/tags/{ref}")
    fields = p.stdout.split()
    return fields[0] if len(fields) == 2 else None


def annotated(oid, expected_commit):
    if git("cat-file", "-t", oid) != "tag":
        raise ArchiveError("tag must be annotated")
    if git("rev-parse", f"{oid}^{{commit}}") != expected_commit:
        raise ArchiveError("annotated tag has wrong commit")


def staged_archive(channel, source_sha, old_oid):
    """Find an alias PRE-STAGED while the tagged commit was eligible to publish.

    The source's annotated tag object itself must be retained: the archive
    points to exactly old_oid, not to a newly-created, later tag object.
    On a first maintainer bootstrap, older tags may lack the receipt; only
    one exact-object alias is acceptable, otherwise fail without mutation.
    """
    prefix = f"vc3d-archive-{channel}-{source_sha[:12]}-"
    candidates = cmd("git", "ls-remote", "--refs", "origin",
                     f"refs/tags/{prefix}*").stdout.splitlines()
    matches = []
    for row in candidates:
        parts = row.split()
        if len(parts) == 2 and parts[1].startswith("refs/tags/" + prefix):
            name = parts[1].removeprefix("refs/tags/")
            if re.fullmatch(re.escape(prefix) + r"[1-9][0-9]*", name) and parts[0] == old_oid:
                matches.append(name)
    annotation = git("cat-file", "-p", old_oid)
    hints = re.findall(r"^Staged archive: (\S+)$", annotation, flags=re.MULTILINE)
    if hints:
        if len(hints) != 1 or matches != [hints[0]]:
            raise ArchiveError("staged archive receipt/remote alias mismatch")
        return hints[0]
    if len(matches) != 1:
        raise ArchiveError("missing or ambiguous pre-staged alias; maintainer bootstrap needed")
    return matches[0]


def verify_retained_history(repo, channel, rolling_oid, rolling_sha):
    """Read-only proof that ALL earlier releases in the tag receipt chain survive.

    Checking only the immediately previous release permits a later promotion to
    conceal the disappearance of an older six-asset archive. No API mutations.
    """
    seen = set()
    oid, sha = rolling_oid, rolling_sha
    depth = 0
    while True:
        if oid in seen:
            raise ArchiveError("cycle in historical release tag chain")
        seen.add(oid)
        annotated(oid, sha)
        message = git("cat-file", "-p", oid)
        header, separator, body = message.partition("\n\n")
        if not separator or not re.search(rf"^tag {re.escape(channel)}$", header, re.MULTILINE):
            raise ArchiveError("historical tag has unexpected channel annotation")
        older_refs = re.findall(rf"^Previous archive: (vc3d-archive-{re.escape(channel)}-[0-9a-f]{{12}}-[1-9][0-9]*)$",
                                body, re.MULTILINE)
        older_ids = re.findall(r"^Previous archive release ID: ([1-9][0-9]*)$", body, re.MULTILINE)
        expected_assets = re.findall(r"^Previous archive asset IDs: ([1-9][0-9]*(?:,[1-9][0-9]*){5})$", body, re.MULTILINE)
        expected_oids = re.findall(r"^Previous archive tag object: ([0-9a-f]{40})$", body, re.MULTILINE)
        if not older_refs and not older_ids and not expected_assets and not expected_oids:
            if body.strip() != f"VC3D {channel} release at {sha}":
                raise ArchiveError("historical tag is not a genuine original VC3D annotation")
            return depth
        if (len(older_refs) != 1 or len(older_ids) != 1
                or len(expected_assets) != 1 or len(expected_oids) != 1):
            raise ArchiveError("ambiguous or incomplete historical ID receipt")
        recorded_asset_ids = [int(v) for v in expected_assets[0].split(",")]
        if len(set(recorded_asset_ids)) != 6 or sorted(recorded_asset_ids) != recorded_asset_ids:
            raise ArchiveError("unsorted or duplicate historical asset IDs")
        archive_ref, archive_id = older_refs[0], int(older_ids[0])
        old_oid = remote_object(archive_ref)
        if old_oid is None:
            raise ArchiveError("earlier archive in receipt chain is missing")
        if old_oid != expected_oids[0]:
            raise ArchiveError("historical annotated tag object changed from immutable receipt")
        git("fetch", "--no-tags", "origin", f"refs/tags/{archive_ref}")
        if git("rev-parse", "FETCH_HEAD") != old_oid:
            raise ArchiveError("earlier archive tag moved during historical fetch")
        old_sha = git("rev-parse", "FETCH_HEAD^{commit}")
        if archive_ref.split("-")[-2] != old_sha[:12]:
            raise ArchiveError("earlier archive tag name disagrees with source commit")
        if old_oid == oid:
            raise ArchiveError("self-reference in historical archive chain")
        archived = gh(repo, f"releases/{archive_id}", allow_404=True)
        if (archived is None or archived.get("id") != archive_id
                or archived.get("tag_name") != archive_ref
                or archived.get("target_commitish") != "main"):
            raise ArchiveError("earlier historical release ID/tag is missing or changed")
        actual_asset_ids = sorted(verify_release_origin(archived, channel, old_sha))
        if actual_asset_ids != recorded_asset_ids:
            raise ArchiveError("historical six asset IDs differ from immutable tag receipt")
        if remote_object(archive_ref) != old_oid:
            raise ArchiveError("earlier archive ref moved during historical readback")
        depth += 1
        if depth > 2000:
            raise ArchiveError("archive history exceeds explicit audit bound; manual review")
        oid, sha = old_oid, old_sha


def same_commit_replay(repo, channel, expected_sha, rolling_oid):
    """Read-only replay of a previously archived rolling release.

    A complete release is idempotent. An absent release may be repaired
    *only* if the new rolling tag itself records the exact pre-existing archive
    and GitHub confirms that archival release still contains its six assets.
    No PATCH/push happens in this function.
    """
    def emit(status, **extra):
        if remote_object(channel) != rolling_oid:
            raise ArchiveError("rolling ref raced during replay probe")
        print(json.dumps(dict(status=status, channel=channel, commit=expected_sha, **extra), sort_keys=True))

    # A missing release is not proof of a safe retry. The *new* stable tag must
    # carry a provenance receipt written only after the old release was archived.
    annotation = git("cat-file", "-p", rolling_oid)
    # First adoption: the original upstream rolling tag has NO prior-archive
    # receipt yet. A same-commit workflow_run must be a read-only no-op,
    # rather than failing or falsely claiming a previous release was archived.
    # Accept ONLY a genuine legacy tag with an exact-object bootstrap alias and
    # an intact six-asset rolling release. A later tag with even a partial
    # archive receipt is not eligible for this exception.
    header, separator, body = annotation.partition("\n\n")
    legacy_body = f"VC3D {channel} release at {expected_sha}"
    if (separator and re.search(rf"^tag {re.escape(channel)}$", header, flags=re.MULTILINE)
            and body.strip() == legacy_body):
        archive = staged_archive(channel, expected_sha, rolling_oid)
        release = gh(repo, f"releases/tags/{channel}", allow_404=True)
        if (release is None or release.get("tag_name") != channel
                or release.get("target_commitish") != "main"
                or not isinstance(release.get("id"), int) or release["id"] <= 0):
            raise ArchiveError("bootstrap same-commit: existing rolling release is absent/unsafe")
        # Initial aliases must carry the preserved Release ID, not an
        # arbitrary run ID. Future staged aliases intentionally use run IDs.
        if archive != f"vc3d-archive-{channel}-{expected_sha[:12]}-{release['id']}":
            raise ArchiveError("bootstrap alias suffix does not match rolling Release ID")
        six = verify_release_origin(release, channel, expected_sha)
        if remote_object(archive) != rolling_oid:
            raise ArchiveError("bootstrap archive alias moved during no-op check")
        emit("BOOTSTRAPPED_UNCHANGED", archive=archive,
             release_id=release["id"], asset_ids=sorted(six), tag_object=rolling_oid)
        return
    lines = re.findall(rf"^Previous archive: (vc3d-archive-{re.escape(channel)}-([0-9a-f]{{12}})-[1-9][0-9]*)$",
                       annotation, flags=re.MULTILINE)
    release_ids = re.findall(r"^Previous archive release ID: ([1-9][0-9]*)$",
                             annotation, flags=re.MULTILINE)
    if len(lines) != 1 or len(release_ids) != 1:
        raise ArchiveError("no unique prior-archive + release-ID receipt; cannot repair")
    archive, expected_old_prefix = lines[0]
    rid_text = release_ids[0]
    archive_oid = remote_object(archive)
    if not archive_oid:
        raise ArchiveError("receipt references missing archive tag")
    git("fetch", "--no-tags", "origin", f"refs/tags/{archive}")
    if git("rev-parse", "FETCH_HEAD") != archive_oid:
        raise ArchiveError("archive ref moved during replay fetch")
    archive_sha = git("rev-parse", "FETCH_HEAD^{commit}")
    annotated(archive_oid, archive_sha)
    if archive_sha[:12] != expected_old_prefix or archive_sha == expected_sha:
        raise ArchiveError("receipt archive commit mismatch")
    archived = gh(repo, f"releases/{rid_text}", allow_404=True)
    if (archived is None or archived.get("id") != int(rid_text)
            or archived.get("tag_name") != archive
            or archived.get("target_commitish") != "main"):
        raise ArchiveError("receipt archived release missing/mismatch/unsafe target_commitish")
    six = verify_release_origin(archived, channel, archive_sha)
    if remote_object(archive) != archive_oid:
        raise ArchiveError("archive tag raced during replay")
    # Even a *complete* newly published stable release is NOT evidence that
    # its predecessor survived. Do not report success before validating the
    # same archive receipt and six original IDs required for crash recovery.
    release = gh(repo, f"releases/tags/{channel}", allow_404=True)
    if release is not None:
        fresh = assets(release)
        prefix = f"VC3D-{expected_sha[:7]}-"
        platforms = (
            r".+-win64\.exe", r".+-win64\.zip",
            r".+-macos-[^.]+\.dmg", r".+-macos-[^.]+\.zip",
            r".+-linux-[^.]+\.AppImage", r".+-linux-[^.]+-debug\.tar\.zst",
        )
        names = [name for name, _size, _digest in fresh.values()]
        valid_platforms = all(sum(bool(re.fullmatch(prefix + pat, name))
                                  for name in names) == 1 for pat in platforms)
        valid_digests = all(re.fullmatch(r"sha256:[0-9a-f]{64}", digest or "")
                            for _name, _size, digest in fresh.values())
        if (release.get("tag_name") != channel or not isinstance(release.get("id"), int)
                or release.get("target_commitish") != "main"
                or not valid_platforms or not valid_digests):
            raise ArchiveError("existing rolling release is not the requested build; manual review")
        # Both channels must carry the exact source-commit note and official
        # release title; asset filenames alone cannot authenticate a replay.
        verify_release_origin(release, channel, expected_sha)
        emit("ALREADY_PUBLISHED", release_id=release["id"], asset_ids=sorted(fresh),
             previous_archive=archive, archived_release_id=int(rid_text),
             archived_asset_ids=sorted(six))
        return

    emit("REPAIR_RELEASE_ONLY", archive=archive, archived_release_id=int(rid_text),
         archived_asset_ids=sorted(six))


def main(channel, new_sha, repo, probe_only=False):
    if channel not in ("latest", "stable") or not re.fullmatch(r"[0-9a-f]{40}", new_sha):
        raise ArchiveError("bad channel or candidate SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ArchiveError("invalid GITHUB_REPOSITORY")
    # A publication that has never had a release must be bootstrapped by a maintainer.
    old_oid = remote_object(channel)
    if not old_oid:
        raise ArchiveError(f"rolling tag {channel} missing; bootstrap manually")
    git("fetch", "--no-tags", "origin", f"refs/tags/{channel}")
    # The identity of the original annotated tag object matters, not only the
    # commit it peels to. Fail if the rolling tag moved during the fetch.
    if git("rev-parse", "FETCH_HEAD") != old_oid:
        raise ArchiveError("rolling tag changed during initial fetch")
    old_sha = git("rev-parse", "FETCH_HEAD^{commit}")
    annotated(old_oid, old_sha)
    # Before any new release PATCH or no-op success, verify the complete
    # retained history, not just the latest link in its receipt chain.
    retained_count = verify_retained_history(repo, channel, old_oid, old_sha)

    if old_sha == new_sha:
        same_commit_replay(repo, channel, new_sha, old_oid)
        return
    release = gh(repo, f"releases/tags/{channel}", allow_404=True)
    if release is None:
        # A crash after PATCH but before rolling-tag movement is resumable: the
        # archived release still has its *original* ID and six original assets.
        archive_ref = staged_archive(channel, old_sha, old_oid)
        archived = gh(repo, f"releases/tags/{archive_ref}", allow_404=True)
        if (archived is None or archived.get("tag_name") != archive_ref
                or not isinstance(archived.get("id"), int)
                or archived.get("target_commitish") != "main"):
            raise ArchiveError("no matching archived release with safe main target; manual recovery")
        rid_text = str(archived["id"])
        six = verify_release_origin(archived, channel, old_sha)
        git("fetch", "--no-tags", "origin", "refs/tags/" + archive_ref)
        if git("rev-parse", "FETCH_HEAD") != old_oid:
            raise ArchiveError("archive lost exact original annotated tag object")
        annotated(old_oid, old_sha)
        if remote_object(channel) != old_oid:
            raise ArchiveError("rolling ref raced during archive recovery")
        print(json.dumps({"status":"ALREADY_ARCHIVED_ZERO_COPY", "archive":archive_ref,
                          "channel":channel, "release_id":int(rid_text), "asset_ids":sorted(six),
                          "old_tag_object":old_oid},sort_keys=True))
        return
    # A preflight may probe recovery, but MUST NEVER PATCH if the rolling
    # release reappears between the initial 404 and this independent read.
    # The ordinary publication path retains its existing explicit PATCH.
    if probe_only:
        raise ArchiveError("read-only recovery probe refused: rolling release now exists")
    rid = release.get("id")
    if (not isinstance(rid, int) or rid <= 0 or release.get("tag_name") != channel
            or release.get("target_commitish") != "main"):
        raise ArchiveError("rolling release metadata mismatch or target_commitish is not main")
    before = verify_release_origin(release, channel, old_sha)
    # Ensure a real new candidate, not a downgrade or arbitrary unrelated SHA.
    git("cat-file", "-e", f"{new_sha}^{{commit}}")
    if channel == "latest" and cmd("git", "merge-base", "--is-ancestor", old_sha, new_sha, check=False).returncode:
        raise ArchiveError("latest cannot move backward")

    archive = staged_archive(channel, old_sha, old_oid)
    expected_obj = remote_object(archive)
    if expected_obj != old_oid:
        raise ArchiveError("pre-staged tag ref is missing or not original annotated tag object")
    remote_archive = remote_object(archive)
    if remote_archive != old_oid:
        raise ArchiveError("archive tag ref does not match old tag object")
    git("fetch", "--no-tags", "origin", f"refs/tags/{archive}")
    if git("rev-parse", "FETCH_HEAD") != old_oid:
        raise ArchiveError("fetched archive tag object changed")
    annotated(old_oid, old_sha)
    # Detect a race which moved the rolling ref before any release API mutation.
    if remote_object(channel) != old_oid:
        raise ArchiveError("rolling tag changed; old release must not be retargeted")
    after = gh(repo, f"releases/{rid}", "-X", "PATCH", "-f", f"tag_name={archive}")
    if after.get("id") != rid or after.get("tag_name") != archive or assets(after) != before:
        raise ArchiveError("archive PATCH response identity mismatch; stop before rolling tag mutation")
    confirm = gh(repo, f"releases/{rid}")
    if (confirm.get("id") != rid or confirm.get("tag_name") != archive
            or confirm.get("target_commitish") != "main" or assets(confirm) != before):
        raise ArchiveError("archive readback identity mismatch; stop")
    # A concurrent writer can move the *archive* alias without moving the
    # rolling channel. Checking only latest/stable would silently lose the
    # exact historical tag object after the release PATCH succeeded.
    verify_release_origin(confirm, channel, old_sha)
    if remote_object(archive) != old_oid:
        raise ArchiveError("archive alias moved after PATCH; stop for manual recovery")
    if remote_object(channel) != old_oid:
        raise ArchiveError("rolling tag raced after PATCH; stop for manual recovery")
    print(json.dumps({"status": "ARCHIVED_ZERO_COPY", "channel": channel, "archive": archive,
                      "old_commit": old_sha, "old_tag_object": old_oid,
                      "release_id": rid, "asset_ids": sorted(before)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", choices=("latest", "stable"))
    parser.add_argument("new_commit")
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--probe-only", action="store_true",
                        help="read-only crash recovery validation; refuse PATCH even after races")
    args = parser.parse_args()
    try:
        main(args.channel, args.new_commit.lower(), args.repo,
             probe_only=args.probe_only)
    except ArchiveError as ex:
        print(f"::error::{ex}", file=sys.stderr)
        sys.exit(2)
