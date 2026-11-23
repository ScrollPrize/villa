#!/usr/bin/env python3
"""A Spiral service for VC3D's protocol tests: the real HTTP service bound to
an object store, with the resident runtime faked.

    python protocol_client_service.py --store STORE.vcobj --root DIR --port-file FILE

The store file is shared with the client on this host. Builds link and pin
positions of it as a real service's do.
The faked session freezes inputs the way a real one does: it reads the
store's default selection (at the pinned position) read-only, writes the run directory's
object_store_inputs.json and reports its digest, summary and manifest path,
so the service publishes the input manifest artifact a client resolves pins
from. A preview export publishes a fixed flattened generation of those inputs
(published_generation) in place of the Lasagna flatten, so a client can
display and commit it. Bearer credential: "key".
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiral_runtime  # noqa: E402
import spiral_service  # noqa: E402
from config import Config  # noqa: E402
from fit_session import SessionState  # noqa: E402
from lasagna_publish import PublishedPreview  # noqa: E402
from published_generation import MODEL_STATE, publish  # noqa: E402

NO_DENSE = {'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}


class FakeRuntime:
    """Just enough of InteractiveFitSession for the service's lifecycle."""

    def __init__(self, *args, **kwargs):
        self.state = SessionState.Idle
        self.completed = 0
        self.target = 0
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            return {
                "state": self.state, "phase": str(self.state),
                "current_iteration": self.completed,
                "target_iteration": self.target, "latest_metrics": {},
                "warnings": [], "error": None, "preview_manifest_path": None,
                "preview_generation": 0, "progress": None,
                "applied_config": Config().as_dict(),
                "run_config": {}, "default_advanced_config": {},
            }

    def run(self, count, **kwargs):
        with self.lock:
            self.state = SessionState.Running
            self.target = self.completed + count

        def finish():
            time.sleep(0.05)
            with self.lock:
                self.completed = self.target
                self.state = SessionState.Idle
        threading.Thread(target=finish, daemon=True).start()
        return self.target

    def stop(self, barrier=None):
        with self.lock:
            self.state = SessionState.Idle
            self.target = self.completed

    def save_checkpoint(self, path, timeout=120.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"PK\x03\x04checkpoint")
        return path

    def load_checkpoint(self, path, timeout=600.0):
        return {"completed_iterations": self.completed, "config_revision": 1}

    def export_preview(self, timeout=600.0, diagnostics=False):
        return {}

    def disable_automatic_previews(self, error=None):
        pass

    def close(self, timeout=15.0):
        with self.lock:
            self.state = SessionState.Closing


class FrozenInputsRuntime(FakeRuntime):
    def __init__(self, paths, run, preview, scroll, status_callback=None, **kwargs):
        super().__init__()
        self.paths, self.status_callback = paths, status_callback
        self.applied = Config(dict(run.config)).as_dict()
        self.input_generation = 0
        self.preview_generation = 0
        self.preview_manifest = None
        self._freeze(self.applied['object_store_input_position'])

    def _freeze(self, position):
        import object_store_inputs as osi
        config = self.applied
        with osi.open_store(config, read_only=True) as store:
            request = osi.request_from_config(config)
            if position is not None:
                request = dataclasses.replace(request, pins=osi.InputPins.from_provenance(position))
            snapshot = osi.read_snapshot(store, request)
        manifest = snapshot.manifest()
        manifest['link_refresh'] = {'enabled': False}
        self.position = manifest['position']
        self.digest = osi.manifest_digest(manifest)
        run_directory = Path(self.paths.output_directory) / 'run'
        run_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = run_directory / 'object_store_inputs.json'
        self.manifest_path.write_text(json.dumps({'digest': self.digest, 'manifest': manifest}))
        self.summary = {
            'digest': self.digest, 'store_uuid': snapshot.store_uuid,
            'volume_uuid': snapshot.volume_uuid,
            'selection': {kind: manifest['selection'][kind]['count'] for kind in osi.KINDS},
            'relationships': {}, 'excluded_relationships': {}, 'constraints': {},
            'link_refresh': False}
        if self.status_callback is not None:
            # A real session reports each status change through this; the
            # service publishes the input manifest artifact from it.
            threading.Timer(0.2, lambda: self.status_callback(self.status())).start()

    def update_inputs(self, position, timeout=1800.0):
        """A resident input update: the inputs of the position (the pins'
        provenance), as pinned."""
        self._freeze(position)
        self.applied = {**self.applied, 'object_store_input_position': self.position}
        self.input_generation += 1
        return {'digest': self.digest, 'input_generation': self.input_generation}

    def export_preview(self, timeout=600.0, diagnostics=False):
        """A raw generation of the current model and inputs, which the
        service then publishes (publish_flattened)."""
        self.preview_generation += 1
        raw = Path(self.paths.output_directory) / 'raw' / f'generation-{self.preview_generation}'
        raw.mkdir(parents=True, exist_ok=True)
        self.preview_manifest = raw / 'manifest.json'
        self.preview_manifest.write_text(json.dumps({
            'model_state_sha256': MODEL_STATE, 'input_digest': self.digest,
            'input_position': self.position, 'source_fit_iteration': self.completed,
            'applied_config': self.applied}))
        if self.status_callback is not None:
            self.status_callback(self.status())
        return {}

    def status(self):
        status = super().status()
        status.update(preview_generation=self.preview_generation,
                      preview_manifest_path=self.preview_manifest and str(self.preview_manifest),
                      preview_source_iteration=self.completed)
        status.update(input_digest=self.digest, input_summary=self.summary,
                      input_manifest_path=str(self.manifest_path),
                      input_manifest={'object_store_digest': self.digest},
                      input_generation=self.input_generation,
                      applied_config=self.applied)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--store', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--port-file', required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    # The client's project gives the scroll and umbilicus; the service binds
    # only its host paths.
    output = root / 'output'
    spiral_runtime.create_session = FrozenInputsRuntime
    state = spiral_service.ServiceState(
        bindings=spiral_service.bind_service_inputs(output, root / 'cache'),
        object_store=spiral_service.bind_object_store(args.store, 'protocol-test'),
        startup_run={'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)})

    def publish_flattened(session_id, generation, raw_manifest_path):
        # The fixed generation in place of a Lasagna flatten of the raw one.
        raw = json.loads(Path(raw_manifest_path).read_text())
        root = publish(output / '.spiral-published' / session_id / f'generation-{generation}',
                       input_digest=raw['input_digest'], input_position=raw['input_position'],
                       model_state_sha256=raw['model_state_sha256'],
                       source_fit_iteration=raw['source_fit_iteration'],
                       applied_config=raw['applied_config'])
        return SimpleNamespace(), PublishedPreview(
            manifest_path=root / 'manifest.json', surface_id='preview-lasagna.tifxyz',
            generation=generation, raw_manifest=raw, raw_manifest_path=Path(raw_manifest_path),
            publish_parent=root.parent, correspondence=None, flattened_valid=None,
            source_fit_iteration=raw['source_fit_iteration'])
    state._publish_flattened_preview = publish_flattened
    server = spiral_service.SpiralServer(('127.0.0.1', 0), ['key'], state)
    # Written whole, then renamed, so a reader never sees it half written.
    incoming = Path(args.port_file).with_suffix('.incoming')
    incoming.write_text(str(server.server_port))
    incoming.replace(args.port_file)
    try:
        server.serve_forever()
    finally:
        state.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
