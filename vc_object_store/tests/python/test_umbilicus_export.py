"""Umbilicus exports restate the frame a legacy import recorded, so frame-checking readers accept them."""
import json

from vc_object_store import Umbilicus
from object_helpers import export


def test_umbilicus_export_restates_legacy_frame_metadata(db, tmp_path):
    legacy = dict(volume='vol-2um.zarr', volume_width=300, volume_height=300, volume_slices=40, voxelsize_um=2.)
    uid = db.upsert_object(dict(kind='umbilicus', name='umbilicus', meta={
        'coordinates': {'volume_uuid': 'vol', 'scale_factor': 1},
        'source': {'provenance': {'tool': 'legacy-umbilicus-import', 'legacy_metadata': legacy}}}))
    db.put_geometry(uid, Umbilicus([[1, 2, 3], [4, 5, 6]]))
    output = tmp_path / 'umbilicus.vc_umbilicus.json'
    result = export(db, uid, output)
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text())
    assert document['vc_umbilicus_json_version'] == '1'
    assert document['metadata'] == legacy
    assert document['provenance']['legacy_metadata'] == legacy
    assert [(p['x'], p['y'], p['z']) for p in document['points']] == [(1, 2, 3), (4, 5, 6)]


def test_umbilicus_export_without_legacy_metadata_states_no_frame(db, tmp_path):
    uid = db.upsert_object(dict(kind='umbilicus', meta={'coordinates': {'volume_uuid': 'vol', 'scale_factor': 1}}))
    db.put_geometry(uid, Umbilicus([[1, 2, 3], [4, 5, 6]]))
    output = tmp_path / 'plain.json'
    assert export(db, uid, output).returncode == 0
    assert 'metadata' not in json.loads(output.read_text())
