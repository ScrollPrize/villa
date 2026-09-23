"""Move only the patch directories listed in a reviewed fiber-merge move plan.

The plan retains any patch also assigned to an unresolved merge fiber. Moves
are atomic renames between sibling directories and can resume after interruption.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def move_plan(plan_path: Path, *, execute: bool) -> dict:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    source = Path(plan['source_root'])
    destination = Path(plan['destination_root'])
    moving = plan['move_patches']
    keeping = plan['keep_in_original']
    if (not source.is_dir() or destination.parent != source.parent or
            destination == source or source.is_symlink() or destination.is_symlink()):
        raise ValueError('Expected distinct, nonsymlink sibling patch directories')
    if (len(moving) != plan['move_count'] or len(keeping) != plan['keep_count'] or
            len(set(moving + keeping)) != len(moving) + len(keeping) or
            moving != sorted(moving) or keeping != sorted(keeping) or
            any(Path(name).name != name or name in ('.', '..') for name in moving + keeping)):
        raise ValueError('Invalid or overlapping patch names in move plan')
    status_path = plan_path.with_name('fiber_merge_patch_move_status.json')
    digest = hashlib.sha256(plan_bytes).hexdigest()
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous and previous['plan_sha256'] != digest:
        raise ValueError('Move plan changed after execution began')
    already = []
    pending = []
    for name in moving:
        old, new = source / name, destination / name
        if old.is_dir() and not old.is_symlink() and not new.exists():
            pending.append(name)
        elif not old.exists() and new.is_dir() and not new.is_symlink() and previous:
            already.append(name)
        else:
            raise ValueError(f'Patch is missing, duplicated, or has an unexpected type: {name}')
    for name in keeping:
        if not (source / name).is_dir() or (source / name).is_symlink():
            raise ValueError(f'Unresolved fiber patch is missing from original folder: {name}')
    result = {'plan_sha256': digest, 'total': len(moving), 'already_moved': len(already),
              'pending': len(pending), 'kept_in_original': len(keeping), 'complete': not pending}
    if not execute:
        return result
    destination.mkdir(exist_ok=True)
    write_json(status_path, result)
    for index, name in enumerate(pending, 1):
        (source / name).rename(destination / name)
        if index % 100 == 0:
            write_json(status_path, dict(result, already_moved=len(already) + index,
                                         pending=len(pending) - index, complete=False))
    for name in moving:
        if (source / name).exists() or not (destination / name).is_dir():
            raise ValueError(f'Move verification failed: {name}')
    for name in keeping:
        if not (source / name).is_dir():
            raise ValueError(f'Kept patch verification failed: {name}')
    result.update(already_moved=len(moving), pending=0, complete=True)
    write_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    lock_path = args.plan.with_name('fiber_merge_patch_move.lock')
    if args.execute:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(move_plan(args.plan, execute=True), indent=2))
    else:
        print(json.dumps(move_plan(args.plan, execute=False), indent=2))


if __name__ == '__main__':
    main()
