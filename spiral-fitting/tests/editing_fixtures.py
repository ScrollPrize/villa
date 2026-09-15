"""Shared editing residents, uploads and service state without HTTP sockets."""

import copy
import hashlib
import io
import json
from uuid import uuid4
import pytest
from spiral_service import ServiceState
from service_fixtures import _attach_fake_session


class Resident:
    def __init__(self):
        self.calls = []
        self.members = {}
        self.fail = False

    def apply_input_changes(self, command, records, influence_config=None):
        self.calls.append((command, copy.deepcopy(records)))
        if self.fail:
            return {'applied': False, 'errors': {'rank0': 'invalid input'}}
        self.members.update({r['id']: r for r in records})
        return {'applied': True}


def upload(workspace, name, kind='pcl', input_id=None, branches=None):
    doc = ({'vc_pointcollections_json_version': '1', 'collections': {'0': {'name': name, 'points': {str(i): {'p': [i, 2, 3], 'creation_time': i} for i in range(2)}}}} if kind == 'pcl' else
           {'type': 'vc3d_fiber', 'version': 1, 'points': []})
    if branches is not None:
        doc['branches'] = branches
    data = json.dumps(doc).encode()
    request = {'upload_id': uuid4().hex, 'id': input_id or str(uuid4()), 'kind': kind,
               'files': [{'name': 'input.json', 'size': len(data),
                          'sha256': hashlib.sha256(data).hexdigest()}]}
    if kind == 'pcl':
        request['role'] = 'same_winding'
    transfer = workspace.uploads.begin(request)['upload_id']
    workspace.uploads.receive(transfer, 'input.json', io.BytesIO(data), len(data))
    workspace.uploads.finalize(transfer)
    return transfer


@pytest.fixture
def editing_state(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    state = ServiceState(dataset_root=dataset)
    session = _attach_fake_session(state, output, dataset)
    resident = Resident()
    session.apply_input_changes = resident.apply_input_changes
    state.editing().claim('owner', 'claim')
    yield state
    state.close()
