"""Private HTTP fixture for the Qt/Python revision workflow integration test."""
import json
import os
from pathlib import Path
import sys
import time
import socket
import signal
import threading
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spiral_service import ServiceState, SpiralServer, SpiralHandler, resolve_dataset_root
from test_spiral_service_v2 import _attach_fake_session

root = Path(sys.argv[1])
dataset = root / 'dataset'
native = bool(os.environ.get('SPIRAL_REVISION_CLIENT_LIVE'))
protocol_stdout = sys.stdout
if native:
    sys.stdout = sys.stderr
    from test_revisioned_live_fit import make_real_revision_session
    import torch
    source_patch, source_digest, dataset, baseline, replacement, patch, config, actual = make_real_revision_session(root)
    deadline = time.monotonic() + 180
    while actual.status()['state'] == 'Loading' and time.monotonic() < deadline:
        time.sleep(.05)
    if actual.status()['state'] != 'Idle':
        raise RuntimeError(actual.status())
    vertices = patch.zyxs[patch.valid_vertex_mask]
    endpoints = vertices[[0, -1]][:, [2, 1, 0]].tolist()
else:
    dataset.mkdir()
    endpoints = [[100, 200, 300], [110, 210, 310]]
state = ServiceState(dataset_root=dataset, dataset_resolution=resolve_dataset_root(dataset))
session = _attach_fake_session(state, root / 'output', dataset)
if native:
    state.session = session = actual
(root / 'fiber-template.json').write_text(json.dumps({'type': 'vc3d_fiber', 'version': 1,
    'points': [], 'line_points': [], 'control_points': [[v * 4 for v in xyz] for xyz in endpoints]}))
(root / 'pcl-template.json').write_text(json.dumps({'vc_pointcollections_json_version': '1',
    'collections': {'0': {'name': 'client-pcl', 'points': {str(i): {'p': xyz, 'creation_time': i}
                                                        for i, xyz in enumerate(endpoints)}}}}))
original_apply = session.apply_input_changes if native else None
boundaries = []
state._refresh_pcl_artifacts = lambda: None
state.export_preview = lambda _: {**state.status(), "accepted": True}
resident = {}


def apply(command, records, influence_config=None):
    (root / 'applying').write_text(command)
    while (root / 'hold-apply').exists():
        time.sleep(0.01)
    if native:
        if session.status()['state'] == 'Idle':
            session.run(16, autosave_on_pause=False)
        context = session._context
        model, optimizer = context.spiral_and_transform, context.optimiser
        torch.cuda.reset_peak_memory_stats()
        result = original_apply(command, records, influence_config=influence_config)
        assert context.spiral_and_transform is model and context.optimiser is optimizer
        boundaries.append({**result, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'kinds': [r['kind'] for r in records]})
        (root / 'boundaries.json').write_text(json.dumps(boundaries))
        report = os.environ.get('SPIRAL_REVISION_CLIENT_REPORT')
        if report:
            Path(report).write_text(json.dumps({'source': str(source_patch), 'config': config,
                'device': torch.cuda.get_device_name(), 'boundaries': boundaries}, indent=2))
        if not result.get('applied'):
            return result
    for record in records:
        resident[record['id']] = {**record, 'document': None if record.get('deleted') else
                                 (json.loads(Path(record['path']).read_text()) if record['kind'] != 'patch' else 'patch')}
    (root / 'resident.json').write_text(json.dumps(resident))
    return result if native else {'applied': True}


session.apply_input_changes = apply
dropped = set()
class FaultHandler(SpiralHandler):
    def _send(self, status, payload, **kwargs):
        path = urlparse(self.path).path
        family = None
        if self.command == 'PUT' and '/files/' in path: family = 'bytes'
        elif self.command == 'POST':
            if path == '/session/inputs': family = 'create'
            elif path.endswith('/finalize'): family = 'finalize'
            elif path == '/session/input-changes': family = 'apply'
            elif path == '/session/commit-inputs': family = 'commit'
        if os.environ.get('SPIRAL_REVISION_DROP_REPLIES') and int(status) == 200 and family and family not in dropped:
            dropped.add(family)
            (root / 'dropped.json').write_text(json.dumps(sorted(dropped)))
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        return super()._send(status, payload, **kwargs)

if os.environ.get('SPIRAL_REVISION_FAIL_PUBLICATION'):
    from input_publication import _Publication
    original_publish = _Publication.publish
    failed_publication = False
    def interrupted(publication):
        global failed_publication
        original_publish(publication)
        if not failed_publication:
            failed_publication = True
            raise OSError('injected lost publication acknowledgement')
    _Publication.publish = interrupted

server = SpiralServer(('127.0.0.1', 0), ['test-key'], state)
server.RequestHandlerClass = FaultHandler
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
print(server.server_port, flush=True, file=protocol_stdout)
try:
    server.serve_forever()
finally:
    state.close()
    server.server_close()
    if native:
        # Release fixture-held bound methods/session references before native
        # modules unload, so their destructors can run in a valid interpreter.
        import gc
        original_apply = None
        state.session = None
        session = actual = None
        gc.collect()
