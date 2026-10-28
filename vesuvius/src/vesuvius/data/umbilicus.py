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
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ID = "SharpMeow/vesuvius-center-lines"
# v2.6.1 of the dataset (25 Oct 2026). Pass revision="main" for the newest version.
DEFAULT_REVISION = "5160338283e3ede64684e7f68734ad5eb2de7415"


def _download(filename, revision, cache_dir):
    """Local path of ``filename`` in the dataset at ``revision``, from the Hugging Face cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Fetching published umbilicus files needs the huggingface_hub package. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir))


def _index(revision, cache_dir):
    with open(_download("index.json", revision, cache_dir), encoding="utf-8") as f:
        return json.load(f)


def _rows(index):
    return list(index.get("files", [])) + list(index.get("other_scans", []))


def list_umbilicus_lines(scroll=None, *, revision=DEFAULT_REVISION, cache_dir=None):
    """Rows of the dataset index (``files`` then ``other_scans``), for one scroll if ``scroll`` is given.

    Each row is a dict with at least ``scroll``, ``scan``, ``file``, ``source``, ``kind``, ``credit``,
    ``licence``, ``sha256``, ``points``, ``z`` and ``volume`` (None where the source file names none).
    Rows that have a newer line carry ``superseded_by``.
    """
    rows = _rows(_index(revision, cache_dir))
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


def fetch_umbilicus(scroll, *, scan=None, source=None, into=None, revision=DEFAULT_REVISION, cache_dir=None):
    """Download one published umbilicus and return ``(path, row)``.

    ``scan`` defaults to the scan the index names for ``scroll``; another scan of the same scroll is looked
    up in ``other_scans``. Without ``source``, lines with a newer version are skipped and a line made by a
    person is taken over a model line; when that still leaves more than one, a ValueError lists the
    sources so the caller can choose. The file's sha256 must match the index.

    With ``into`` (a folder such as a volume package), the file is also copied there as ``umbilicus.json``.
    That refuses to replace an existing ``umbilicus.json`` and refuses a folder that holds
    ``estimated_umbilicus.json``, since VC3D will not choose between two umbilicus files.

    ``row`` is the line's index entry; keep its ``credit`` and ``licence``, which differ between sources.
    """
    row = _select(_index(revision, cache_dir), scroll, scan, source)
    path = _download(row["file"], revision, cache_dir)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"{row['file']}: sha256 {digest} does not match the index ({row['sha256']})")
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
