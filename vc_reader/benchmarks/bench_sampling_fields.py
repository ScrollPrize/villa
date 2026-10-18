"""Aligned multichannel sampling of the bundled real tifxyz coordinate values.

The fixture is re-encoded as CZYX TIFF in a temporary directory. Its UV pixel
lattice is an explicit test reference grid; this is not surface projection.
XYZ values retain their PHercParis4 level-2 frame (9.6 um voxels).
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import tifffile
from bench_sampling import timed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path(__file__).parent/'fixtures/fiber_132487')
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from bench_pipeline import load_extensions
    load_extensions(args.native_dir)
    from vc_reader import Affine, Source, Storage, ValueFrame, VolumeRegistration
    from vc_reader.sampling import open_sample
    hashes = {name: hashlib.sha256((args.fixture/(name+'.tif')).read_bytes()).hexdigest() for name in ('mask', 'x', 'y', 'z')}
    coordinates = np.stack([tifffile.imread(args.fixture/(name+'.tif')) for name in ('x', 'y', 'z')]).astype('float32')[:, None]
    mask = tifffile.imread(args.fixture/'mask.tif')[None]
    requests = [((0, j, 300+j*600), (1, 16, 512)) for j in range(6)]
    report = dict(command=sys.argv, fixture=str(args.fixture), hashes=hashes, platform=platform.platform(), requests=requests,
                  reference='UV lattice with singleton Z; no inferred CT placement',
                  value_frame='PHercParis4 level-2 voxels, 9.6 micrometers', runs={})
    with tempfile.TemporaryDirectory(prefix='vc-aligned-fields-') as directory:
        root = Path(directory)
        tifffile.imwrite(root/'reference.tif', mask, metadata={'axes': 'ZYX'}, photometric='minisblack')
        tifffile.imwrite(root/'field.tif', coordinates, metadata={'axes': 'CZYX'}, photometric='minisblack')
        reg = VolumeRegistration('fixture', 'uv-grid', affine=Affine.identity())
        reference = Source('real-mask', Storage(str(root/'reference.tif'), 'tiff'), role='image', registration=reg)
        pull = Affine.from_rows(((1, .125, 0, .25), (0, 1, 0, .25), (0, 0, 1, 0)))
        field = Source('real-xyz', Storage(str(root/'field.tif'), 'tiff'), role='coordinate',
                       registration=replace(reg, affine=pull.inverse()), component_order=('x', 'y', 'z'),
                       value_frame=ValueFrame(kind='external', name='PHercParis4-level2', units='9.6um voxels'))
        expected = None
        for backend in ('cpu', 'cuda'):
            now = time.perf_counter()
            with open_sample(reference, {'coordinates': field}, backend=backend) as reader:
                opened = (time.perf_counter()-now)*1000
                cold = []
                results = []
                for request in requests:
                    now = time.perf_counter(); result = reader.read(*request); cold.append((time.perf_counter()-now)*1000)
                    results.append((result.data['coordinates'].numpy(), result.valid['coordinates'].numpy()) if backend == 'cuda'
                                   else (result.data['coordinates'], result.valid['coordinates']))
                if expected is None:
                    expected = results
                else:
                    for (values, valid), (cpu, cpu_valid) in zip(results, expected):
                        np.testing.assert_allclose(values, cpu, rtol=2e-6, atol=.002)
                        np.testing.assert_array_equal(valid, cpu_valid)
                run = timed(reader.read, requests)
                run.update(open_ms=opened, first_reads_ms=cold, stats=reader.stats(),
                           output_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a, _ in results])
                report['runs'][backend] = run
        # Direct TIFF region reads are a useful I/O lower bound, not an affine
        # interpolation baseline. Values/validity use the independent unit oracle.
        with field.open() as raw:
            report['direct_io_lower_bound'] = timed(raw.read, requests)
    report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v['summary'] for k, v in report['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
