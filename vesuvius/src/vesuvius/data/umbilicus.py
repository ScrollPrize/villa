"""Published umbilicus files from the Hugging Face dataset ``SharpMeow/vesuvius-center-lines``.

The dataset gathers public umbilici for the open-data scans in one place: the organizers' lines from the
open-data bucket, community lines (Alexey Drobkov, Iyan Dopico, Abd Elilah) and model lines, each copied
unchanged in VC3D's umbilicus JSON format (``control_points`` of ``{x, y, z}`` in level-0 voxels of the
scan the line names). Its ``index.json`` lists every file with its scan, volume, source, kind (``person``
or ``model``), credit, license, upstream link and sha256.

``fetch_umbilicus`` downloads one line, checks its sha256 against the index and returns the local path,
so the file can be used wherever villa already reads ``umbilicus.json`` (VC3D, spiral-fitting, lasagna,
the sheet_strip tracer). Points are not rescaled: a tool working on a downsampled level has to scale them
(for example ``"umbilicus": {"coordinate_scale": 0.25}`` in spiral-fitting's ``spiral-scroll.json`` for
level 2). Lines with ``kind == "model"`` are model output; each row's ``note`` says what was checked.

Mirror. ``mirror_umbilicus(dest)`` copies one dataset revision from Hugging Face into ``dest`` with every
sha256 checked, so a second host (for example the Vesuvius Challenge data server) can keep a copy in step
with the dataset. ``fetch_umbilicus(..., mirror=URL)``, or the ``VESUVIUS_UMBILICUS_MIRROR`` environment
variable, then reads from that copy instead of Hugging Face, with the same checks. The layout is
``<mirror>/<commit>/index.json`` and ``<mirror>/<commit>/<file>``, one folder per dataset commit, plus
``<mirror>/refs/<name>`` holding the commit of a named revision such as ``main``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

REPO_ID = "SharpMeow/vesuvius-center-lines"
# v2.6.1 of the dataset (25 Oct 2026). Pass revision="main" for the newest version.
DEFAULT_REVISION = "5160338283e3ede64684e7f68734ad5eb2de7415"
# sha256 of index.json at DEFAULT_REVISION, so the pinned index is the same from Hugging Face or a mirror.
DEFAULT_INDEX_SHA256 = "7eff02c7e044bdfdb4dac8a2bf277e5579b318254a5abea026524bc03b5a77c0"
MIRROR_ENV = "VESUVIUS_UMBILICUS_MIRROR"
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "Fetching published umbilicus files from Hugging Face needs the huggingface_hub package. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    return huggingface_hub


def _download(filename, revision, cache_dir):
    """Local path of ``filename`` in the dataset at ``revision``, from the Hugging Face cache."""
    return Path(_hub().hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir))


def _http_get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _write_atomic(target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _mirror_base(mirror):
    base = mirror if mirror is not None else os.environ.get(MIRROR_ENV)
    return base.rstrip("/") if base else None


def _mirror_download(base, filename, revision, cache_dir):
    """Local path of ``filename`` at ``revision`` read from a mirror in the layout ``mirror_umbilicus`` writes."""
    if not _COMMIT.fullmatch(revision):
        name = revision
        revision = _http_get(f"{base}/refs/{name}").decode().strip()
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{base}/refs/{name} does not hold a commit id")
    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vesuvius" / "umbilicus-mirror"
    target = root / revision / filename
    if not target.exists():
        _write_atomic(target, _http_get(f"{base}/{revision}/{filename}"))
    return target


def _fetch(filename, revision, cache_dir, mirror):
    base = _mirror_base(mirror)
    if base:
        return _mirror_download(base, filename, revision, cache_dir)
    return _download(filename, revision, cache_dir)


def _check(path, expected, name):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{name}: sha256 {digest} does not match the index ({expected})")


def _index(revision, cache_dir, mirror=None):
    path = _fetch("index.json", revision, cache_dir, mirror)
    if revision == DEFAULT_REVISION:
        _check(path, DEFAULT_INDEX_SHA256, "index.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows(index):
    return list(index.get("files", [])) + list(index.get("other_scans", []))


def list_umbilicus_lines(scroll=None, *, revision=DEFAULT_REVISION, cache_dir=None, mirror=None):
    """Rows of the dataset index (``files`` then ``other_scans``), for one scroll if ``scroll`` is given.

    Each row is a dict with at least ``scroll``, ``scan``, ``file``, ``source``, ``kind``, ``credit``,
    ``licence``, ``sha256``, ``points``, ``z`` and ``volume`` (None where the source file names none).
    Rows that have a newer line carry ``superseded_by``.
    """
    rows = _rows(_index(revision, cache_dir, mirror))
    if scroll is not None:
        rows = [row for row in rows if row["scroll"] == scroll]
    return rows


def _select(index, scroll, scan, source):
    scrolls = index.get("scrolls", {})
    if scroll not in scrolls:
        raise ValueError(f"no umbilicus for scroll {scroll!r}; known scrolls: {', '.join(sorted(scrolls))}")
    if scan is None:
        scan = scrolls[scroll]["scan"]
        rows = [row for row in index.get("files", []) if row["scroll"] == scroll and row["scan"] == scan]
    else:
        rows = [row for row in _rows(index) if row["scroll"] == scroll and row["scan"] == scan]
        if not rows:
            scans = sorted({row["scan"] for row in _rows(index) if row["scroll"] == scroll})
            raise ValueError(f"no umbilicus for {scroll} scan {scan!r}; scans with one: {', '.join(scans)}")
    if source is not None:
        picked = [row for row in rows if row["source"] == source]
        if not picked:
            sources = sorted({row["source"] for row in rows})
            raise ValueError(f"no {source!r} umbilicus for {scroll} scan {scan}; sources: {', '.join(sources)}")
    else:
        picked = [row for row in rows if not row.get("superseded_by")]
        if len(picked) > 1:
            people = [row for row in picked if row["kind"] == "person"]
            picked = people or picked
    if len(picked) != 1:
        sources = ", ".join(row["source"] for row in picked)
        raise ValueError(f"{len(picked)} umbilici for {scroll} scan {scan} ({sources}); pass source= to pick one")
    return picked[0]


def fetch_umbilicus(
    scroll, *, scan=None, source=None, into=None, revision=DEFAULT_REVISION, cache_dir=None, mirror=None
):
    """Download one published umbilicus and return ``(path, row)``.

    ``scan`` defaults to the scan the index names for ``scroll``; another scan of the same scroll is looked
    up in ``other_scans``. Without ``source``, lines with a newer version are skipped and a line made by a
    person is taken over a model line; when that still leaves more than one, a ValueError lists the
    sources so the caller can choose. The file's sha256 must match the index.

    With ``into`` (a folder such as a volume package), the file is also copied there as ``umbilicus.json``.
    That refuses to replace an existing ``umbilicus.json`` and refuses a folder that holds
    ``estimated_umbilicus.json``, since VC3D will not choose between two umbilicus files.

    ``mirror`` (or the ``VESUVIUS_UMBILICUS_MIRROR`` environment variable) is the base URL of a copy written
    by ``mirror_umbilicus``; files then come from there instead of Hugging Face, with the same checks.

    ``row`` is the line's index entry; keep its ``credit`` and ``licence``, which differ between sources.
    """
    row = _select(_index(revision, cache_dir, mirror), scroll, scan, source)
    path = _fetch(row["file"], revision, cache_dir, mirror)
    _check(path, row["sha256"], row["file"])
    if into is not None:
        folder = Path(into)
        target = folder / "umbilicus.json"
        if target.exists():
            raise FileExistsError(f"{target} exists; remove it first to replace it")
        if (folder / "estimated_umbilicus.json").exists():
            raise FileExistsError(f"{folder} holds estimated_umbilicus.json; VC3D would refuse two umbilicus files")
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".umbilicus.json.{os.getpid()}.tmp"
        shutil.copyfile(path, tmp)
        os.replace(tmp, target)
        path = target
    return path, row


def _resolve(revision):
    """Commit id of a dataset revision (branch, tag or commit) on Hugging Face."""
    if _COMMIT.fullmatch(revision):
        return revision
    return _hub().HfApi().dataset_info(REPO_ID, revision=revision).sha


def mirror_umbilicus(dest, *, revision="main", cache_dir=None):
    """Copy one dataset revision from Hugging Face into ``dest`` and return its folder.

    Writes ``<dest>/<commit>/index.json`` and every file the index lists, each checked against its
    sha256, into a staging folder that is renamed into place only when complete. A commit already present
    is left as it is, since a commit's files never change. When ``revision`` is a name such as ``main``,
    ``<dest>/refs/<name>`` is updated to the commit, so running this on a schedule keeps the mirror in step
    with the dataset while older commits stay readable for pinned calls.
    """
    commit = _resolve(revision)
    dest = Path(dest)
    folder = dest / commit
    if not folder.exists():
        index_path = _download("index.json", commit, cache_dir)
        if commit == DEFAULT_REVISION:
            _check(index_path, DEFAULT_INDEX_SHA256, "index.json")
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        staging = dest / f".{commit}.{os.getpid()}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copyfile(index_path, staging / "index.json")
        for row in _rows(index):
            path = _download(row["file"], commit, cache_dir)
            _check(path, row["sha256"], row["file"])
            (staging / row["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, staging / row["file"])
        os.replace(staging, folder)
    if revision != commit:
        _write_atomic(dest / "refs" / revision, f"{commit}\n".encode())
    return folder


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m vesuvius.data.umbilicus")
    sub = parser.add_subparsers(dest="command", required=True)
    mirror = sub.add_parser("mirror", help="copy a dataset revision from Hugging Face into a mirror folder")
    mirror.add_argument("dest")
    mirror.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    print(mirror_umbilicus(args.dest, revision=args.revision))


if __name__ == "__main__":
    main()
