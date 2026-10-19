"""Validate the trainer geometry adapter against its own target implementations."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from collections import OrderedDict

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vc-site-packages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.append(str(args.vc_site_packages))
    importlib.import_module('_editable_skbc_vc_reader')
    import fiber_geometry as adapter
    from vc_reader.geometry import VoxelSet
    from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, arclength, frame_from_heading
    from vesuvius.neural_tracing.tracers.fibers.data.afv_neighbors import AFVBank
    from vesuvius.neural_tracing.tracers.fibers.data.observations import ComponentRule
    from vesuvius.neural_tracing.tracers.shared.compile import compile_options

    gpu = adapter.install(validate=True, placement='main')
    rng = np.random.default_rng(42)
    for _ in range(100):
        p = np.cumsum(rng.normal(size=(100, 3)), axis=0)+10000
        s = arclength(p)
        frame = frame_from_heading(rng.normal(size=3))
        adapter.plane_targets(p, s, s[30], s[-1], p[30], frame, np.linspace(.1, 12, 137))
        adapter.plane_targets(p[::-1], s[-1]-s[::-1], s[30], s[-1], p[30], frame, np.linspace(.1, 12, 137))

    crop = CropSpec(depth=24, width=32, behind=12, spacing=.5)
    cfg = SimpleNamespace(fine=crop, future_step=1., n_future=8)
    bank = AFVBank.__new__(AFVBank)
    bank.root, bank.exclusion, bank._trees = Path('/synthetic'), 1.5, OrderedDict()
    fibers, records, items = [], [], []
    for i in range(8):
        frame = frame_from_heading(rng.normal(size=3))
        pos = np.array([13000., 14000., 15000.])
        z = np.linspace(-40, 40, 321)
        identity = np.column_stack((.1*np.sin(z), .1*np.cos(z), z))
        parent = identity @ frame.T+pos
        fibers.append(SimpleNamespace(points=parent))
        local = rng.uniform([-12, -12, -10], [12, 12, 10], (4000, 3))
        local = np.concatenate((local, local[:50]))
        records.append([dict(samples=local @ frame.T+pos)] if i != 0 else [])
        items.append(dict(fiber_ref=(i, 0., False), pos=pos, frame=frame,
                          identity_curve=identity, location_source=0))
    bank.fibers = fibers
    bank.spatial_records = lambda fi, world: records[fi]
    builder = SimpleNamespace(cfg=cfg, neighbors=bank, fibers=None,
                              sampling=SimpleNamespace(rule=ComponentRule()), lateral=[])
    batch = adapter.bank_targets(builder, items)
    expected = batch['_vc_geometry']['expected']['foreign'].cuda()
    gpu.materialize(batch)

    # Exercise compiled lookup, including exact and adjacent half-cell ties,
    # nonfinite queries and cells just outside the crop.
    shape = (crop.depth, crop.width, crop.width)
    cells = rng.uniform(-1, np.array(shape)+1, (8, 10000, 3)).astype(np.float32)
    cells[:, :100] = np.round(cells[:, :100])+.5
    cells[:, 100:200] = np.nextafter(cells[:, :100], np.float32(np.inf))
    cells[:, 200:300] = np.nextafter(cells[:, :100], np.float32(-np.inf))
    cells[:, 300], cells[:, 301], cells[:, 302] = np.nan, np.inf, -np.inf
    local = np.stack(((cells[..., 2]-(crop.width-1)/2)*crop.spacing,
                      (cells[..., 1]-(crop.width-1)/2)*crop.spacing,
                      (cells[..., 0]-crop.behind)*crop.spacing), -1)
    points = torch.tensor(local, device='cuda')
    reference = adapter.ORIGINAL_HITS(points, dict(foreign=expected), cfg)
    eager = adapter.foreign_hits(points, batch, cfg)
    torch.testing.assert_close(eager, reference)
    compiled = torch.compile(adapter.foreign_hits, **compile_options(fullgraph=True))
    actual = compiled(points, dict(foreign=batch['foreign']), cfg)
    compiled_reference = torch.compile(adapter.ORIGINAL_HITS, **compile_options(fullgraph=True))
    expected_compiled = compiled_reference(points, dict(foreign=expected), cfg)
    torch.testing.assert_close(actual, expected_compiled)
    # The same sparse representation is used by CPU decision diagnostics.
    torch.testing.assert_close(adapter.foreign_hits(points.cpu(), dict(foreign=batch['foreign'].cpu()), cfg), reference.cpu())
    empty = VoxelSet(torch.empty((0, 3), device='cuda'), shape, batches=8)
    keys = torch.empty_like(batch['foreign'])
    empty.sorted_keys(capacity=keys.numel(), out=keys.view(-1))
    assert not compiled(points, dict(foreign=keys), cfg).any()
    report = dict(plane_cases=200, lookup_queries=points.shape[0]*points.shape[1],
                  compiled_lookup='exact agreement with compiled grid_sample',
                  compiled_vs_eager_boundary_differences=int((expected_compiled != reference).sum()),
                  geometry=gpu.stats())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
