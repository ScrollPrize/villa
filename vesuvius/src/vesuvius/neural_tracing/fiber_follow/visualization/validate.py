"""Reconcile saved atlas measurements, current patch projection and local assets."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..regression.patch_encoder import pad_to_patch_grid
from ..regression.survival_confidence import survival_predictions
from ..shared.policy import commit_prefix
from .capture import validate_attention


def validate(directory):
    dest = Path(directory)
    m = json.loads((dest / 'sample_provenance.json').read_text())
    torch.set_num_threads(m['threads'])
    r = json.loads((dest / 'analysis_summary.json').read_text())
    manifest = json.loads((dest / 'figure_manifest.json').read_text())
    with np.load(dest / 'sample_activations.npz') as data:
        a = dict(data)
    with np.load(dest / 'analysis_arrays.npz') as data:
        b = dict(data)
    for collection in (a, b):
        for key, value in collection.items():
            assert np.isfinite(value).all(), key
    for key, value in a.items():
        assert list(value.shape) == m['shapes'][key], key
    for key in ('points', 'confidence', 'refinement_points', 'refinement_confidence'):
        np.testing.assert_array_equal(b['baseline_'+key], a[key])
    for name, metric in r['metrics'].items():
        points, confidence = b[name+'_points'], b[name+'_confidence']
        shift = np.linalg.norm(points-a['points'], axis=-1)
        np.testing.assert_allclose([shift.mean(), shift.max()], [metric['mean_path_shift'], metric['max_path_shift']], rtol=2e-5, atol=1e-7)
        counts, allowed = commit_prefix(torch.from_numpy(points[None]), torch.from_numpy(confidence[None]),
                                       m['confidence_threshold'], m['n_commit'], m['model_cfg']['max_recovery_distance'])
        assert int(counts[0]) == metric['commit'] and bool(allowed[0]) == metric['connection_allowed']
        _, expected = survival_predictions(torch.from_numpy(b[name+'_hazard_logits']))
        np.testing.assert_allclose(confidence, expected.numpy(), rtol=2e-6, atol=1e-7)
        assert (np.diff(confidence) <= 0).all()
        np.testing.assert_allclose(confidence[-1], metric['last_confidence'])
        np.testing.assert_allclose(b[name+'_fixed_curve_confidence'][-1], metric['fixed_curve_last_confidence'])
    validate_attention(a)
    for i, stats in enumerate(r['encoder_blocks']):
        before = a['encoder_input' if i == 0 else f'axial_{i-1}'].astype(np.float64)
        after = a[f'axial_{i}'].astype(np.float64)
        relative = np.linalg.norm(after-before, axis=-1) / np.linalg.norm(before, axis=-1).clip(1e-12)
        np.testing.assert_allclose(relative, b[f'block_{i+1}_relative_change'], rtol=2e-6)
        np.testing.assert_allclose(np.median(relative), stats['median_relative_change'], rtol=2e-6)
    shape = a['token_xyz'].shape[:3]
    nimage = int(np.prod(shape))
    for key, weights in a.items():
        if '_attention_' not in key or '_history_' in key:
            continue
        volume = weights[:, :nimage].reshape(-1, *shape)
        for axis in (2, 3):
            np.testing.assert_allclose(volume.sum(axis).sum((1, 2)), weights[:, :nimage].sum(-1), rtol=2e-6)
    valid = a['input_history_valid'].astype(bool)
    assert int(valid.sum()) == m['observations']
    np.testing.assert_array_equal(a['history_padding'], np.repeat(~valid, 162))
    assert (a['history_tokens'][a['history_padding']] == 0).all()
    cp = Path(m['checkpoint'])
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == m['checkpoint_sha256']
    ema = torch.load(cp, map_location='cpu', weights_only=False)['ema']
    with torch.inference_mode():
        # CPU load_checkpoint copies CUDA channels-last weights into contiguous
        # parameters. Match that layout to preserve its reduction order exactly.
        embedded = F.conv3d(pad_to_patch_grid(torch.from_numpy(a['input'][None])),
                            ema['encoder.patch_projection.weight'].contiguous(),
                            ema['encoder.patch_projection.bias'], stride=4, padding=1)
    np.testing.assert_array_equal(embedded[0].permute(1, 2, 3, 0).numpy(), a['patch'])
    assert len(manifest['pages']) == 7
    for pair in manifest['spatial_pairs']:
        assert pair['views'] == ['u-forward', 'v-forward']
    for page in manifest['pages']:
        for suffix in ('', '_preview'):
            with Image.open(dest / (page['name']+suffix+'.png')) as image:
                assert min(image.size) >= 1000
                image.verify()
        assert (dest / (page['name']+'.pdf')).read_bytes().startswith(b'%PDF-')
    for suffix in ('', '_preview'):
        with Image.open(dest / ('input_channels'+suffix+'.png')) as image:
            image.verify()
    assert (dest / 'model_interpretation.pdf').read_bytes().startswith(b'%PDF-')

    class Assets(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name in ('src', 'href'):
                    assert not value.startswith(('http:', 'https:', '//'))
                    if not value.startswith('#') and value != 'validation.json':
                        assert (dest / value).is_file(), value
    Assets().feed((dest / 'index.html').read_text())
    result = dict(status='PASS', pages=7, paired_spatial_groups=len(manifest['spatial_pairs']),
                  checks=['Instrumented / ordinary baseline equality', 'Restored intervention baseline',
                          'Finite arrays and shapes', 'Commit policy and monotone survival',
                          'Image and history attention masking and normalization',
                          'Probability marginals preserve mass', 'History validity masks',
                          'Full-vector encoder update statistics',
                          'Full overlapping Conv3d patch projection', 'Checkpoint SHA256',
                          'Readable PNG/PDF and local HTML assets'])
    (dest / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    validate(args.directory)
