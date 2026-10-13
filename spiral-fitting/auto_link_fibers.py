#!/usr/bin/env python3
"""Write ``explicit_link:<patch id>`` tags into vc3d_fiber JSON files.

fit_spiral.py reads a fiber's document-level ``tags`` list and, under
``pcl_fiber_explicit_links_only``, attaches the fiber only to the patches
(tifxyz folder names) its ``explicit_link:`` tags name. This tool writes those
tags from one of two sources:

* ``--links``: a ``vc_fiber_patch_links`` JSON written by
  an external linker or an earlier VC3D fiber review. Every entry of a fiber's
  ``linked`` list names a patch.
* ``--review``: a ``vc3d_fiber_review`` workspace saved by VC3D's fiber review
  (version 1 or 2). Every decision with ``status == "accepted"`` (version 1:
  ``accepted == true``) names a patch; the review's ``patch:`` id prefix is
  stripped. Such workspaces can be several GB: the document is parsed with the
  standard library and pruned to fiber paths and decision statuses as it is
  decoded, so only that much is retained.

Fibers are found by their file basename under ``--fibers``. By default the new
tags are merged into the existing list; ``--replace`` first drops every
existing ``explicit_link:`` tag so fibers that lost all their links end up
untagged. Everything else in the documents is preserved.

By default this is a dry run that prints the planned changes. Pass
``--apply`` to write; every touched file is copied to ``--backup-dir`` first.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

import click

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spiral_helpers import EXPLICIT_LINK_TAG_PREFIX, explicit_patch_links_from_tags  # noqa: E402

_PATCH_ID_PREFIX = 'patch:'


def _strip_patch_prefix(patch_id: str) -> str:
    patch_id = str(patch_id)
    if patch_id.startswith(_PATCH_ID_PREFIX):
        return patch_id[len(_PATCH_ID_PREFIX):]
    return patch_id


# ---------------------------------------------------------------------------
# Input formats: fiber basename -> set of patch ids
# ---------------------------------------------------------------------------

def read_patch_links(path: str) -> dict[str, set[str]]:
    """``vc_fiber_patch_links`` (version 1): the ``linked`` patches per fiber."""
    with open(path, 'rt') as fp:
        document = json.load(fp)
    if document.get('type') != 'vc_fiber_patch_links':
        raise click.ClickException(f'{path}: not a vc_fiber_patch_links document')
    if document.get('version') != 1:
        raise click.ClickException(
            f'{path}: unsupported vc_fiber_patch_links version {document.get("version")!r}')
    links: dict[str, set[str]] = {}
    for name, row in (document.get('fibers') or {}).items():
        basename = os.path.basename(str(name))
        patches = links.setdefault(basename, set())
        for entry in (row or {}).get('linked') or []:
            patch_id = entry.get('patch') if isinstance(entry, dict) else None
            if patch_id:
                patches.add(_strip_patch_prefix(patch_id))
    return links


def _prune_review_object(pairs):
    """object_pairs_hook keeping only what read_fiber_review needs.

    A fiber entry (``path`` + ``direction`` + ``decisions``) keeps its path
    and decisions; a decision (``status``/``accepted`` + ``order``/
    ``alignment``) keeps its status. Everything else decodes as usual, so the
    heavy geometry (sources, base graphs, proposals, alignments) is released
    as soon as its enclosing object is built rather than retained.
    """
    keys = [key for key, _ in pairs]
    if 'path' in keys and 'direction' in keys and 'decisions' in keys:
        return {key: value for key, value in pairs if key in ('path', 'decisions')}
    if ('status' in keys or 'accepted' in keys) and ('alignment' in keys or 'order' in keys):
        return {key: value for key, value in pairs if key in ('status', 'accepted')}
    return dict(pairs)


def read_fiber_review(path: str) -> dict[str, set[str]]:
    """``vc3d_fiber_review`` (version 1 or 2): the accepted patches per fiber."""
    with open(path, 'rt') as fp:
        document = json.load(fp, object_pairs_hook=_prune_review_object)
    if document.get('type') != 'vc3d_fiber_review':
        raise click.ClickException(f'{path}: not a vc3d_fiber_review workspace')
    version = document.get('version')
    if version not in (1, 2):
        raise click.ClickException(f'{path}: unsupported vc3d_fiber_review version {version!r}')
    fibers = document.get('fibers') or {}
    del document
    links: dict[str, set[str]] = {}
    for fiber_id, entry in fibers.items():
        fiber_path = (entry or {}).get('path')
        if not fiber_path:
            print(f'WARNING: review fiber {fiber_id!r} has no path; skipping')
            continue
        patches = links.setdefault(os.path.basename(str(fiber_path)), set())
        for patch_id, decision in ((entry or {}).get('decisions') or {}).items():
            if not isinstance(decision, dict):
                continue
            if version == 1:
                accepted = bool(decision.get('accepted', False))
            else:
                accepted = decision.get('status') == 'accepted'
            if accepted:
                patches.add(_strip_patch_prefix(patch_id))
    return links


# ---------------------------------------------------------------------------
# Planning and writing
# ---------------------------------------------------------------------------

@dataclass
class FiberPlan:
    basename: str
    path: str
    existing: list[str]            # explicit links already in the file
    wanted: set[str]               # links from the input
    replace: bool
    data: dict | None = None
    tags: list[str] = field(default_factory=list)  # the new tags list

    @property
    def added(self) -> list[str]:
        return sorted(self.wanted - set(self.existing))

    @property
    def removed(self) -> list[str]:
        return sorted(set(self.existing) - self.wanted) if self.replace else []

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def _load_fiber_document(path: str) -> dict | None:
    with open(path, 'rt') as fp:
        data = json.load(fp)
    if not isinstance(data, dict) or data.get('type') != 'vc3d_fiber':
        return None
    return data


def _merged_tags(tags, wanted: set[str], *, replace: bool) -> list[str]:
    """The document's tags with the explicit links set to ``wanted`` (merged
    into the existing ones unless ``replace``), other tags kept in order."""
    tags = list(tags) if isinstance(tags, (list, tuple)) else []
    if replace:
        tags = [tag for tag in tags
                if not (isinstance(tag, str) and tag.startswith(EXPLICIT_LINK_TAG_PREFIX))]
    present = set(explicit_patch_links_from_tags(tags))
    tags.extend(f'{EXPLICIT_LINK_TAG_PREFIX}{patch_id}' for patch_id in sorted(wanted - present))
    return tags


def plan_fibers(links: dict[str, set[str]], fibers_dir: str, *, replace: bool,
                include_untouched: bool) -> tuple[list[FiberPlan], list[str]]:
    """One FiberPlan per fiber file under ``fibers_dir``; input fibers without
    a file are returned separately. ``include_untouched`` also plans fibers
    absent from the input (for ``--replace``, which strips their tags)."""
    names = set(links)
    if include_untouched:
        names |= {name for name in os.listdir(fibers_dir) if name.endswith('.json')}
    plans, missing = [], []
    for basename in sorted(names):
        path = os.path.join(fibers_dir, basename)
        if not os.path.isfile(path):
            missing.append(basename)
            continue
        data = _load_fiber_document(path)
        if data is None:
            if basename in links:
                print(f'WARNING: {basename}: not a vc3d_fiber document; skipping')
            continue
        existing = explicit_patch_links_from_tags(data.get('tags'))
        wanted = set(links.get(basename, ()))
        plan = FiberPlan(basename=basename, path=path, existing=existing,
                         wanted=wanted, replace=replace, data=data)
        plan.tags = _merged_tags(data.get('tags'), wanted, replace=replace)
        plans.append(plan)
    return plans, missing


def _detect_indent(path: str) -> int | None:
    """Indent width of an existing JSON file (None when written on one line)."""
    with open(path, 'rt') as fp:
        head = fp.read(4096)
    newline = head.find('\n')
    while newline != -1:
        rest = head[newline + 1:]
        stripped = rest.lstrip(' ')
        if stripped and not stripped.startswith('\n'):
            return (len(rest) - len(stripped)) or None
        newline = head.find('\n', newline + 1)
    return None


def write_plans(plans: list[FiberPlan], fibers_dir: str, *, apply: bool,
                backup_dir: str | None) -> int:
    changed = [plan for plan in plans if plan.changed]
    if not apply:
        print('dry run: nothing written (pass --apply to write)')
        return 0
    if not changed:
        print('nothing to write')
        return 0
    if backup_dir is None:
        stamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_dir = os.path.join(os.path.dirname(os.path.abspath(fibers_dir)),
                                  f'fibers_backup_{stamp}')
    os.makedirs(backup_dir, exist_ok=True)
    for plan in changed:
        shutil.copy2(plan.path, os.path.join(backup_dir, plan.basename))
    print(f'backed up {len(changed)} files to {backup_dir}')
    for plan in changed:
        indent = _detect_indent(plan.path)
        document = dict(plan.data)
        document['tags'] = plan.tags
        tmp = plan.path + '.tmp'
        with open(tmp, 'wt') as fp:
            json.dump(document, fp, indent=indent)
        os.replace(tmp, plan.path)
    print(f'wrote {len(changed)} files')
    return len(changed)


def _report(plans: list[FiberPlan], missing: list[str], links: dict[str, set[str]]) -> None:
    for plan in plans:
        if not plan.changed and plan.basename not in links:
            continue
        state = 'changed' if plan.changed else 'unchanged'
        detail = f'{len(plan.wanted)} linked, {len(plan.existing)} tagged before'
        if plan.added:
            detail += f', +{len(plan.added)}'
        if plan.removed:
            detail += f', -{len(plan.removed)}'
        print(f'{plan.basename}: {state} ({detail}) -> '
              f'{len(explicit_patch_links_from_tags(plan.tags))} explicit link(s)')
    for basename in missing:
        print(f'WARNING: {basename}: named by the input but not found under --fibers')
    total_links = sum(len(patches) for patches in links.values())
    print(f'{len(links)} fiber(s) in input with {total_links} patch link(s); '
          f'{len(plans)} fiber file(s) planned, {sum(1 for p in plans if p.changed)} to change, '
          f'{len(missing)} missing')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(help=__doc__)
@click.option('--links', 'links_path', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Existing vc_fiber_patch_links JSON')
@click.option('--review', 'review_path', type=click.Path(exists=True, dir_okay=False), default=None,
              help='vc3d_fiber_review workspace JSON (VC3D fiber review)')
@click.option('--fibers', 'fibers_dir', type=click.Path(exists=True, file_okay=False), required=True,
              help='folder holding the vc3d_fiber JSON files, found by basename')
@click.option('--replace', is_flag=True,
              help='drop every existing explicit_link: tag first (also on fibers absent '
                   'from the input); default merges the new links into the existing tags')
@click.option('--apply', is_flag=True, help='write the files; default is a dry run')
@click.option('--backup-dir', type=click.Path(file_okay=False), default=None,
              help='where touched files are copied before writing; '
                   'defaults to <fibers>/../fibers_backup_<timestamp>')
def main(links_path, review_path, fibers_dir, replace, apply, backup_dir):
    if (links_path is None) == (review_path is None):
        raise click.UsageError('pass exactly one of --links or --review')
    links = (read_patch_links(links_path) if links_path is not None
             else read_fiber_review(review_path))
    plans, missing = plan_fibers(links, fibers_dir, replace=replace, include_untouched=replace)
    _report(plans, missing, links)
    write_plans(plans, fibers_dir, apply=apply, backup_dir=backup_dir)


if __name__ == '__main__':
    main()
