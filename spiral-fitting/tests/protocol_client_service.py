#!/usr/bin/env python3
"""A Spiral service for VC3D's protocol tests: the real HTTP service bound to
an object store, with the resident runtime faked.

    python protocol_client_service.py --store STORE.vcobj [--hub HUB] --root DIR --port-file FILE

With --hub the store is the service's replica of that hub (cloned when
absent); without, the store is shared in place. Builds link and pin
positions (the hub's, or the store's own) as a real service's do.
The faked session freezes inputs the way a real one does: it reads the
store's default selection (at the pinned position) read-only, writes the run directory's
object_store_inputs.json and reports its digest, summary and manifest path,
so the service publishes the input manifest artifact a client resolves pins
from. Bearer credential: "key".
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiral_runtime  # noqa: E402
import spiral_service  # noqa: E402
from config import Config  # noqa: E402
from fit_session import resolve_dataset_root  # noqa: E402
from service_lifecycle_transcript import FakeRuntime  # noqa: E402

NO_DENSE = {'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}


class FrozenInputsRuntime(FakeRuntime):
    def __init__(self, paths, run, preview, scroll, status_callback=None, **kwargs):
        super().__init__()
        self.paths, self.status_callback = paths, status_callback
        self.applied = Config(dict(run.config)).as_dict()
        self.input_generation = 0
        self._freeze(self.applied['object_store_input_position'])

    def _freeze(self, position):
        import object_store_inputs as osi
        from vc_object_store import ObjectStore
        config = self.applied
        with ObjectStore.open(config['object_store_path'], read_only=True) as store:
            request = osi.request_from_config(config)
            if position is not None:
                request = dataclasses.replace(request, position=osi.InputPosition(
                    osi.position_owner(store), position))
            snapshot = osi.read_snapshot(store, request)
        manifest = snapshot.manifest()
        manifest['link_refresh'] = {'enabled': False}
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
        """A resident input update: the inputs of the position, as pinned."""
        self.applied = {**self.applied, 'object_store_input_position': position}
        self._freeze(position)
        self.input_generation += 1
        return {'digest': self.digest, 'input_generation': self.input_generation}

    def status(self):
        status = super().status()
        status.update(input_digest=self.digest, input_summary=self.summary,
                      input_manifest_path=str(self.manifest_path),
                      input_manifest={'object_store_digest': self.digest},
                      input_generation=self.input_generation,
                      applied_config=self.applied)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--store', required=True)
    parser.add_argument('--hub', default=None)
    parser.add_argument('--root', required=True)
    parser.add_argument('--port-file', required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    dataset = root / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / 'spiral-scroll.json').write_text(json.dumps({
        'schema_version': 1, 'name': 'protocol', 'voxel_size_um': 9.6,
        'spiral_outward_sense': 'CW', 'base_shape_zyx': [64, 64, 64]}))
    (dataset / 'umbilicus.json').write_text('{}')
    output = root / 'output'
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(dataset), output, root / 'cache')
    spiral_runtime.create_session = FrozenInputsRuntime
    state = spiral_service.ServiceState(
        dataset_root=str(dataset), dataset_resolution=resolution,
        object_store=spiral_service.bind_object_store(args.store, 'protocol-test', args.hub),
        startup_run={'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)})
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
