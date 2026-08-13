#!/usr/bin/env python3
"""One-time cleanup of stale fiber review tags.

VC3D used to run a specialized review workflow on the machine-managed
reserved tag `interp_unreviewed` (present = prediction-traced geometry
still needs a human look). That mechanism is gone: review state is now
just the ordinary free-form `reviewed` tag, edited through the generic
tag UI, and consumed by `vc_sync.py hfsync` as its publish gate.

Two leftovers need clearing once, per fiber directory:

  * `interp_unreviewed` on any fiber — the tag no longer means anything
    and would otherwise show up as a mystery checkbox forever.
  * `reviewed` on fibers whose stored geometry has NO prediction-traced
    span. Under the old workflow those verdicts were never expressible
    through the review actions (they rejected untraced fibers), so a
    `reviewed` tag there is either hand-applied noise or a leftover from
    a fiber that has since been re-fit away from traces. Traced fibers
    keep their `reviewed` tag untouched.

Nothing else changes: tag order is otherwise preserved, geometry is
never touched, unknown top-level keys survive verbatim, and `generation`
is bumped by one on modified files (matching VC3D's save behavior, so a
three-way sync sees the rewrite as the newer revision rather than a
conflict).

The target directory is normally s3sync-tracked; the mtime and content
changes mean every modified file re-uploads to S3 on the next sync. That
is expected.

Usage:
    fiber_strip_stale_review_tags.py <dir> [--dry-run]

<dir> is one project's fiber directory (e.g.
fibers/PHercParis4.volpkg.json/); *.json files are processed
non-recursively, dotfiles (.hfsync.json, .s3sync.db) and non-vc3d_fiber
documents are skipped. Re-running is a no-op.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fiber_merge

REVIEWED_TAG = 'reviewed'
# ^ the human review verdict; also vc_sync.py hfsync's default publish
#   gate and kReviewedTag in apps/VC3D/LineAnnotationFiberSegments.hpp.

STALE_REVIEW_TAG = 'interp_unreviewed'
# ^ the retired reserved tag of the removed trace-review workflow. It no
#   longer exists anywhere in VC3D or fiber_merge, so this literal is
#   deliberately local to this cleanup script.


def _atomic_write(path, doc):
    """Byte-compatible with VC3D / vc_sync fiber output: two-space indent,
    no NaN, trailing newline, fsynced temp file swapped in with rename."""
    tmp = path + '.review-tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(doc, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def strip_doc_tags(doc):
    """Returns (new_tags, stripped_reviewed, stripped_stale) for `doc`
    without modifying it. Order of the surviving tags is preserved."""
    tags = list(doc.get('tags') or [])
    stripped_stale = tags.count(STALE_REVIEW_TAG)
    tags = [tag for tag in tags if tag != STALE_REVIEW_TAG]
    stripped_reviewed = False
    if REVIEWED_TAG in tags and not fiber_merge._has_trace_span(doc):
        tags = [tag for tag in tags if tag != REVIEWED_TAG]
        stripped_reviewed = True
    return tags, stripped_reviewed, bool(stripped_stale)


def strip_directory(directory, dry_run=False, log=print):
    """Strips stale review tags from every fiber in `directory`. Returns a
    dict of counters; 'failed' > 0 means at least one file could not be
    processed."""
    counts = {'fibers': 0, 'reviewed_stripped': 0, 'stale_tag_stripped': 0,
              'unchanged': 0, 'skipped_non_fiber': 0, 'failed': 0,
              'end_reviewed': 0, 'end_stale_tag': 0}
    for name in sorted(os.listdir(directory)):
        if name.startswith('.') or not name.endswith('.json'):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (ValueError, OSError) as ex:
            log(f"  FAILED {name}: unreadable JSON ({ex})")
            counts['failed'] += 1
            continue
        if not fiber_merge.is_fiber_doc(doc):
            if isinstance(doc, dict) and doc.get('type') == 'vc3d_fiber':
                log(f"  FAILED {name}: invalid vc3d_fiber document")
                counts['failed'] += 1
            else:
                counts['skipped_non_fiber'] += 1
            continue

        counts['fibers'] += 1
        tags, stripped_reviewed, stripped_stale = strip_doc_tags(doc)
        if not stripped_reviewed and not stripped_stale:
            counts['unchanged'] += 1
        else:
            doc['tags'] = tags
            doc['generation'] = max(1, int(doc.get('generation') or 1)) + 1
            if not dry_run:
                try:
                    _atomic_write(path, doc)
                except (ValueError, OSError) as ex:
                    log(f"  FAILED {name}: could not write ({ex})")
                    counts['failed'] += 1
                    continue
            if stripped_reviewed:
                counts['reviewed_stripped'] += 1
            if stripped_stale:
                counts['stale_tag_stripped'] += 1
        if REVIEWED_TAG in tags:
            counts['end_reviewed'] += 1
        if STALE_REVIEW_TAG in tags:
            counts['end_stale_tag'] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0])
    parser.add_argument('directory',
                        help="one project's fiber directory")
    parser.add_argument('--dry-run', action='store_true',
                        help='report without writing')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory}")

    counts = strip_directory(args.directory, dry_run=args.dry_run)
    action = 'would strip' if args.dry_run else 'stripped'
    print(f"fibers: {counts['fibers']}  "
          f"{action} {REVIEWED_TAG}: {counts['reviewed_stripped']}  "
          f"{action} {STALE_REVIEW_TAG}: {counts['stale_tag_stripped']}  "
          f"unchanged: {counts['unchanged']}  "
          f"skipped (not fibers): {counts['skipped_non_fiber']}  "
          f"failed: {counts['failed']}")
    print(f"end state: {REVIEWED_TAG}: {counts['end_reviewed']}  "
          f"{STALE_REVIEW_TAG}: {counts['end_stale_tag']}")
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
