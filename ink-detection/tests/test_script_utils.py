import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_resnet3d_lib.script_utils import (  # noqa: E402
    build_tile_cache_dir,
    build_tile_cache_key,
    resolve_fragment_base_path,
)


class ScriptUtilsTest(unittest.TestCase):
    def _cache_context(self):
        return {
            'cache_format_version': 1,
            'valid_id': 'fold_a',
            'size': 256,
            'tile_size': 512,
            'stride': 128,
            'in_chans': 62,
            'fragments': [
                {
                    'segment_id': 'fold_a',
                    'layer_range': [1, 63],
                    'reverse_layers': False,
                    'base_path': '2um_dataset/0139_2um',
                    'tile_size': 512,
                },
            ],
        }

    def test_cache_key_changes_with_validation_split(self):
        context = self._cache_context()
        other_context = dict(context, valid_id='fold_b')

        self.assertNotEqual(build_tile_cache_key(context), build_tile_cache_key(other_context))

    def test_cache_key_changes_with_fragment_spec(self):
        context = self._cache_context()
        other_context = self._cache_context()
        other_context['fragments'][0] = dict(other_context['fragments'][0], base_path='2um_dataset/1451_2um')

        self.assertNotEqual(build_tile_cache_key(context), build_tile_cache_key(other_context))

    def test_build_tile_cache_dir_writes_metadata(self):
        context = self._cache_context()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(build_tile_cache_dir(tmpdir, context))
            metadata = json.loads((cache_dir / 'cache_context.json').read_text(encoding='utf-8'))

        self.assertEqual(metadata, context)

    def test_resolve_fragment_base_path_ignores_missing_optional_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / '1451_2um'
            fallback = root / '2um_dataset'
            candidate.mkdir()
            fallback.mkdir()
            (candidate / 'fragment-a').mkdir()

            resolved = resolve_fragment_base_path(
                'fragment-a',
                [str(root / 'missing_root'), str(candidate)],
                str(fallback),
            )

        self.assertEqual(resolved, str(candidate))

    def test_resolve_fragment_base_path_falls_back_when_fragment_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / '1451_2um'
            fallback = root / '2um_dataset'
            candidate.mkdir()
            fallback.mkdir()

            resolved = resolve_fragment_base_path(
                'fragment-a',
                [str(candidate)],
                str(fallback),
            )

        self.assertEqual(resolved, str(fallback))


if __name__ == '__main__':
    unittest.main()
