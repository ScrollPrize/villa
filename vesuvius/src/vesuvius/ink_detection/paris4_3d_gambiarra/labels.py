"""VC3D labeled scribbles and the unlabelled CT parent pool. Every stage uses all scribble samples."""
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'assets'
SCRIBBLES_BUCKET = 'scrollprize/datasets'
SCRIBBLES_FILE = 'ink_paris4_3d_4.8um_gambiarra/paris4_ink_scribbles.vc3d.json'
SCRIBBLES_SHA256 = '1bf04b8ebf457286c88e627b575a30f8bb35b3a5953874b52f70ffdd185b7be1'
POOL_CSV = DATA / 'unlabelled_pool.csv'
CT_SHAPE = (75784, 32693, 32693)  # PHercParis4 level 0, ZYX


def resolve_scribbles(path=None):
    """Use a local JSON, or download the released labels once into the Hugging Face assets cache."""
    if path is not None:
        return Path(path)
    from filelock import FileLock
    from huggingface_hub import cached_assets_path, download_bucket_files
    cache = cached_assets_path('vesuvius', namespace='paris4_3d_gambiarra', subfolder=SCRIBBLES_SHA256)
    target = cache / Path(SCRIBBLES_FILE).name
    with FileLock(str(target) + '.lock'):
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SCRIBBLES_SHA256:
            return target
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            downloaded = Path(tmp) / target.name
            download_bucket_files(SCRIBBLES_BUCKET, [(SCRIBBLES_FILE, downloaded)], raise_on_missing_files=True)
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != SCRIBBLES_SHA256:
                raise ValueError('downloaded scribbles differ from the released dataset (sha256 mismatch)')
            downloaded.replace(target)
    return target


class Points:
    """q: level-0 ZYX float64 [N, 3]; y: labels [N]; parents: origins floor(q) - 128.

    Numeric VC3D point IDs determine the stable sample order across collections. All samples train;
    heldout is empty, retained for the geometry helpers. Viewer links do not affect training targets.
    """
    def __init__(self, path=None):
        doc = json.loads(resolve_scribbles(path).read_text())
        if not isinstance(doc, dict) or str(doc.get('vc_pointcollections_json_version')) != '1':
            raise ValueError('expected a version-1 keyed VC3D point collection')
        if doc.get('vc_open_data_source_coordinate_level') != 0 or doc.get('vc_open_data_source_coordinate_scale_factor') != 1:
            raise ValueError('scribbles must use level-0 coordinates with scale factor 1')
        if doc.get('vc_open_data_coordinate_space') != 'PHercParis4/20260411134726@L0':
            raise ValueError('scribbles must identify the PHercParis4/20260411134726 level-0 coordinate space')
        collections = doc.get('collections')
        if not isinstance(collections, dict):
            raise ValueError('expected keyed ink and noink collections')
        rows, names, ids = [], set(), set()
        for collection in collections.values():
            if not isinstance(collection, dict):
                raise ValueError('each VC3D collection must be an object')
            name = collection.get('name')
            if name not in ('ink', 'noink') or name in names:
                raise ValueError('expected exactly one ink and one noink collection')
            names.add(name)
            points = collection.get('points')
            if not isinstance(points, dict) or len(points) < 4:
                raise ValueError('each collection must contain at least four keyed points for anchor sampling')
            for key, point in points.items():
                try:
                    identifier = int(key)
                except (TypeError, ValueError):
                    raise ValueError('VC3D point IDs must be positive integers') from None
                if identifier <= 0 or identifier in ids:
                    raise ValueError('VC3D point IDs must be positive and globally unique')
                ids.add(identifier)
                xyz = point.get('p') if isinstance(point, dict) else None
                if (not isinstance(xyz, list) or len(xyz) != 3 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in xyz) or
                        not np.isfinite(xyz).all()):
                    raise ValueError(f'point {key} must contain a finite numeric XYZ triple in p')
                rows.append((identifier, int(name == 'ink'), xyz))
        if names != {'ink', 'noink'}:
            raise ValueError('expected exactly one ink and one noink collection')
        rows.sort(key=lambda row: row[0])
        self.q = np.array([r[2][::-1] for r in rows], dtype=np.float64)
        self.y = np.array([r[1] for r in rows], dtype=np.int64)
        if len(np.unique(self.q, axis=0)) != len(rows):
            raise ValueError('scribbles contain duplicate or conflicting sample coordinates')
        self.parents = np.floor(self.q).astype(np.int64) - 128
        self.train, self.heldout = np.arange(len(rows), dtype=np.int64), np.empty(0, np.int64)
        # Every parent keeps a 32-voxel margin inside the CT for the 320^3 teacher context.
        if not (np.all(self.parents >= 32) and np.all(self.parents + 288 <= np.array(CT_SHAPE))):
            raise ValueError('scribble parents must keep a 32-voxel context margin inside PHercParis4')


def read_pool(path=POOL_CSV):
    """Unlabelled parent256 pool rows (dicts with index, origin [z, y, x], low, high, split, raw256_sha256).

    Row index i is the pool index used in unlabelled plans. Training views use the rows whose split is 'train'."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        assert int(r['index']) == i
        out.append(dict(index=i, origin=[int(r['z']), int(r['y']), int(r['x'])], low=float(r['low']), high=float(r['high']),
                        split=r['split'], raw256_sha256=r['raw256_sha256']))
    return out
