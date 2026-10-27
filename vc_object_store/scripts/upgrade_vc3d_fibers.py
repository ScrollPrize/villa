#!/usr/bin/env python3
"""Copy a folder of VC3D fiber files, giving each version 4 link its target_uuid.

VC3D saved fibers as version 4 before the object store, naming each link's other
fiber only by `branch_file`. Store ingest reads version 4 as the transfer format
and requires `target_uuid`, so those folders (the published annotations among
them) are refused. This script makes the copies ingest exactly as version 3
files do: each link gets `target_uuid`, the peer file's `uuid` or, for a file
without one, the name-based UUID ingest gives it; and as VC3D declared each
link from both ends with separately computed directions, the second
declaration is checked against the first (1e-9) and dropped, as
FiberInput.cpp does for version 3. A level-0 `vc_open_data_coordinate_space`
becomes the `coordinates` binding, as FiberJson.cpp does for version 3. Links in
version 1 and 3 files take part too, since a version 4 file can link to one of
them. Nothing else in a document changes, version numbers included.

  python3 scripts/upgrade_vc3d_fibers.py INPUT_DIR OUTPUT_DIR
  vc_object_ingest --db fibers.vcobj --author NAME --fiber OUTPUT_DIR

Standard library only. The source stays read-only; a link whose `branch_file`
is not in INPUT_DIR is an error and nothing is written, as are two declarations
of one link that disagree. Links that already have `target_uuid` are copied
unchanged. OUTPUT_DIR must
not exist; a private sibling staging directory keeps failures from publishing a
partial result.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# util::nameUuid: RFC 4122 version 5 in the store's fixed namespace.
NAMESPACE = uuid.UUID('e14f6968-9145-494d-8b06-ab2f2fac7f3b')
LINK_KEYS = ('branches', 'adjacent_branches')


def file_uuid(document, path):
    """The UUID ingest gives a fiber file (FiberInput.cpp fileUuid)."""
    if 'uuid' in document:
        return str(uuid.UUID(document['uuid']))
    return str(uuid.uuid5(NAMESPACE, 'vc3d_fiber/' + path.name))


def same_direction(left, right):
    """FiberInput.cpp sameDirection: both absent, or every axis within 1e-9."""
    if left is None or right is None:
        return left is None and right is None
    return all(abs(a - b) <= 1e-9 for a, b in zip(left, right))


def version3_coordinates(document, name):
    """FiberJson.cpp version3Coordinates: a level-0 open-data frame becomes the binding."""
    space = document.get('vc_open_data_coordinate_space')
    if space is not None:
        slash, level = space.find('/'), space.rfind('@L')
        if slash < 0 or level <= slash + 1:
            raise ValueError(f'{name}: invalid coordinate space {space}')
        if space[level:] != '@L0' or document.get('vc_open_data_source_coordinate_scale_factor', 1) != 1:
            raise ValueError(f'{name}: coordinates are supported only at level 0: {space}')
        document['coordinates'] = dict(volume_uuid=space[slash + 1:level], scale_factor=1)
    for key in [key for key in document if key.startswith('vc_open_data_')]:
        del document[key]


def upgrade(source, target):
    documents = {}
    for path in sorted(source.glob('*.json')):
        document = json.loads(path.read_text())
        if isinstance(document, dict) and document.get('type') == 'vc3d_fiber':
            documents[path.name] = (path, document)
    uuids = {name: file_uuid(document, path) for name, (path, document) in documents.items()}
    filled, dropped, problems, seen = 0, 0, [], {}
    # All versions in one pass: a version 4 file can link to a version 3 one, and
    # each link must reach ingest from one end only.
    for name, (path, document) in documents.items():
        own_uuid = uuids[name]
        if document.get('version') == 4 and 'coordinates' not in document:
            try:
                version3_coordinates(document, name)
            except ValueError as error:
                problems.append(str(error))
        for key in LINK_KEYS:
            kept = []
            for link in document.get(key) or []:
                if 'target_uuid' in link or 'branch_file' not in link:
                    kept.append(link)
                    continue
                peer = Path(link['branch_file']).name
                if peer not in uuids:
                    problems.append(f'{name}: {key} names {peer}, which is not in {source}')
                    continue
                peer_uuid = uuids[peer]
                own, other = link['control_point_index'], link['branch_control_point_index']
                first = seen.get((peer_uuid, other, own_uuid, own))
                if first is not None:
                    if not (same_direction(first.get('control_point_direction'), link.get('branch_control_point_direction'))
                            and same_direction(first.get('branch_control_point_direction'), link.get('control_point_direction'))):
                        problems.append(f'{name}: {key} to {peer} disagrees with the link declared from {peer}')
                    dropped += 1
                    continue
                link['target_uuid'] = peer_uuid
                seen[(own_uuid, own, peer_uuid, other)] = link
                kept.append(link)
                filled += 1
            if key in document:
                document[key] = kept
    if problems:
        raise SystemExit('nothing written:\n' + '\n'.join(problems))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', dir=target.parent))
    try:
        for name, (path, document) in documents.items():
            (staging / name).write_text(json.dumps(document))
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(documents), filled, dropped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input', type=Path, help='folder of VC3D fiber JSON files')
    parser.add_argument('output', type=Path, help='new folder for the copies (must not exist)')
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f'{args.input} is not a folder')
    if args.output.exists():
        parser.error(f'{args.output} already exists')
    files, links, dropped = upgrade(args.input, args.output)
    print(f'{files} fiber files copied to {args.output}; {links} links given target_uuid, '
          f'{dropped} second declarations of a link dropped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
