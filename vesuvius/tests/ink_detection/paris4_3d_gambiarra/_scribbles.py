"""Small synthetic labels for offline tests; full-data checks accept an external release JSON."""
import os
from pathlib import Path

RELEASE_JSON = Path(os.environ['PARIS4_INK_SCRIBBLES']) if os.environ.get('PARIS4_INK_SCRIBBLES') else None


def synthetic_scribbles():
    return dict(vc_pointcollections_json_version='1', vc_open_data_source_coordinate_level=0,
                vc_open_data_source_coordinate_scale_factor=1,
                vc_open_data_coordinate_space='PHercParis4/20260411134726@L0',
                collections={str(label): dict(name=name, points={str(1 + label * 4 + i):
                    dict(p=[1000.25 + label * 100 + i * 3, 2000.5, 3000.75]) for i in range(4)})
                    for label, name in enumerate(('ink', 'noink'))})
