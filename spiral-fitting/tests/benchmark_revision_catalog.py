"""Measure full catalog snapshot construction without modifying dataset inputs.

Run with the project's Python environment. All snapshots are temporary and
removed on exit; the JSON report is the only retained output. This measures
catalog startup, not optimizer throughput or a production fit configuration.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_session import resolve_dataset_root
from service_editing import EditingWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    sources = resolve_dataset_root(args.dataset).to_dict()
    with TemporaryDirectory(prefix='spiral-catalog-') as output:
        workspace = EditingWorkspace(args.dataset, output, sources, lambda: None)
        started = time.perf_counter()
        # No mutation or dataset lease is needed: this is a read-only source
        # snapshot benchmark, not an editing session or an acceptance shortcut.
        workspace._seed()
        elapsed = time.perf_counter() - started
        entries = workspace.catalog.entries()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot_bytes = sum((Path(directory) / name).stat().st_size
                             for directory, _, files in os.walk(workspace.root)
                             for name in files)
        report = {'dataset': str(args.dataset), 'seconds': elapsed,
                  'entries': len(entries), 'kinds': dict(Counter(e.identity.kind for e in entries)),
                  'snapshot_bytes': snapshot_bytes,
                  'peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        workspace.close()


if __name__ == '__main__':
    main()
