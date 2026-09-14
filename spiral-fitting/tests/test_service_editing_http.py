"""Exercise workspace recovery through the real authenticated HTTP routes."""
import hashlib
import json
import urllib.request
import urllib.error
from uuid import uuid4

from test_spiral_service_v2 import HttpServiceFixture, _attach_fake_session
from test_service_editing import Resident


class EditingHttpTests(HttpServiceFixture):
    def setUp(self):
        super().setUp()
        dataset = self.root / 'dataset'
        dataset.mkdir()
        output = self.root / 'output'
        output.mkdir()
        self.state.dataset_root = str(dataset)
        session = _attach_fake_session(self.state, output, dataset)
        self.resident = Resident()
        session.apply_input_changes = self.resident.apply_input_changes
        self.owner = {'X-Spiral-Workspace-Token': 'owner'}
        self.assertEqual(self.request('POST', '/session/editing/claim',
            headers=self.owner, body={'command_id': 'claim'})[0], 200)

    def tearDown(self):
        self.state.close()
        super().tearDown()

    def put(self, upload_id, data, offset):
        request = urllib.request.Request(
            self.base + f'/session/inputs/{upload_id}/files/fiber.json?offset={offset}',
            data=data, method='PUT', headers={**self.owner, 'Authorization': 'Bearer secret-key'})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_lost_responses_reconcile_one_logical_input_and_exact_commit(self):
        data = b'{"type":"vc3d_fiber","version":1,"points":[]}'
        input_id, upload_id = str(uuid4()), uuid4().hex
        manifest = {'upload_id': upload_id, 'id': input_id, 'kind': 'fiber',
            'files': [{'name': 'fiber.json', 'size': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()}]}
        for _ in range(2):
            self.assertEqual(self.request('POST', '/session/inputs', headers=self.owner,
                                          body=manifest)[0], 200)
        self.assertEqual(self.put(upload_id, data[:8], 0)[0], 200)
        code, result = self.put(upload_id, data[:8], 0)
        self.assertEqual((code, result['offset']), (409, 8))
        _, payload, _ = self.request('GET', f'/session/inputs/{upload_id}')
        self.assertEqual(json.loads(payload)['files'][0]['offset'], 8)
        self.assertEqual(self.put(upload_id, data[8:], 8)[0], 200)
        for _ in range(2):
            self.assertEqual(self.request('POST', f'/session/inputs/{upload_id}/finalize',
                                          headers=self.owner, body={})[0], 200)
        self.assertEqual(self.state.editing().catalog.entries(), ())
        command = {'command_id': 'apply', 'changes': [{'id': input_id, 'kind': 'fiber',
                    'expected_revision': 0, 'upload_id': upload_id}]}
        for _ in range(2):
            code, payload, _ = self.request('POST', '/session/input-changes',
                                            headers=self.owner, body=command)
            self.assertEqual(code, 200, payload)
            self.assertTrue(json.loads(payload)['applied'])
        self.assertEqual(len(self.resident.calls), 1)
        self.assertFalse((self.root / 'dataset' / 'fibers' / f'{input_id}.json').exists())
        # Transport reconnect does not allocate a new workspace or reset uploads.
        self.request('POST', '/session/editing/claim', headers=self.owner,
                     body={'command_id': 'reconnect'})
        code, payload, _ = self.request('GET', '/session/input-commands/apply')
        self.assertEqual(json.loads(payload)['state'], 'completed')
        code, payload, _ = self.request('GET', '/session/input-catalog')
        self.assertEqual(len(json.loads(payload)['inputs']), 1)
        for _ in range(2):
            code, payload, _ = self.request('POST', '/session/commit-inputs', headers=self.owner,
                body={'command_id': 'commit', 'revisions': [{'id': input_id, 'revision': 1}]})
            self.assertEqual(code, 200, payload)
        self.assertEqual((self.root / 'dataset' / 'fibers' / f'{input_id}.json').read_bytes(), data)
        self.assertEqual(len(self.resident.calls), 1)
        command['changes'][0]['deleted'] = True
        self.assertEqual(self.request('POST', '/session/input-changes', headers=self.owner,
                                     body=command)[0], 409)

    def test_ownership_and_removed_routes(self):
        for path, body in [('/session/run', {'command_id': 'run'}),
                           ('/session/commit-inputs', {'command_id': 'commit'}),
                           ('/session/inputs', {}),
                           ('/session/rebuild', {'command_id': 'rebuild'})]:
            self.assertEqual(self.request('POST', path, body=body)[0], 403)
        self.assertEqual(self.request('GET', '/session/status')[0], 200)
        self.assertEqual(self.request('DELETE', '/session/ephemeral-inputs/fiber/test',
                                      headers=self.owner)[0], 404)

    def test_release_retries_are_idempotent_over_http(self):
        root = self.state.editing().root
        for _ in range(2):
            code, payload, _ = self.request('POST', '/session/editing/release', headers=self.owner,
                body={'command_id': 'release'})
            self.assertEqual(code, 200, payload)
        self.assertFalse(root.exists())
        self.request('GET', '/session/status')
        self.request('GET', '/session/input-catalog')
        self.assertIsNone(self.state.editing_workspace)
        self.request('POST', '/session/editing/claim', headers=self.owner,
                     body={'command_id': 'new-claim'})
        fresh = self.state.editing_workspace
        self.assertNotEqual(root, fresh.root)
        self.request('POST', '/session/editing/release', headers=self.owner,
                     body={'command_id': 'release'})
        self.assertTrue(fresh.root.exists())

    def test_fiber_editor_artifact_contains_immutable_desired_peers(self):
        from test_service_editing import upload
        workspace = self.state.editing()
        ids = [str(uuid4()), str(uuid4())]
        changes = [{'id': input_id, 'kind': 'fiber', 'name': name,
                    'expected_revision': 0, 'upload_id': upload(workspace, name, 'fiber')}
                   for input_id, name in zip(ids, ['first', 'peer'])]
        workspace.change('owner', {'command_id': 'fibers', 'changes': changes})
        first = self.state.input_content_artifact(ids[0], 1)
        manifest = self.state.artifacts.manifest(first['artifact']['id'])
        self.assertEqual({f['name'] for f in manifest['files']}, {'first.json', 'peer.json'})
        workspace.change('owner', {'command_id': 'remove-peer', 'changes': [
            {'id': ids[1], 'expected_revision': 1, 'deleted': True}]})
        second = self.state.input_content_artifact(ids[0], 1)
        self.assertNotEqual(first['artifact']['id'], second['artifact']['id'])
        self.assertEqual(len(self.state.artifacts.manifest(first['artifact']['id'])['files']), 2)
        self.assertEqual(len(self.state.artifacts.manifest(second['artifact']['id'])['files']), 1)

    def test_rebuild_retry_does_not_restart_resident_construction(self):
        workspace = self.state.editing()
        original = workspace.replay_resident
        builds, replays = [], []
        def build(request):
            builds.append(request)
            return {'accepted': True}
        def replay(generation):
            replays.append(generation)
            if len(replays) == 1:
                raise TimeoutError('lost resident replay outcome')
            return original(generation)
        workspace.replay_resident = replay
        request = {'command_id': 'rebuild-retained'}
        with self.assertRaises(TimeoutError):
            self.state.editing_lifecycle('owner', 'session_rebuild', request, build)
        result = self.state.editing_lifecycle('owner', 'session_rebuild', request, build)
        self.assertTrue(result['accepted'])
        self.assertEqual(len(builds), 1)
        self.assertEqual(len(replays), 2)
