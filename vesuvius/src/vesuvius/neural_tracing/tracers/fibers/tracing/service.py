"""Fiber-follow tracing service for VC3D line annotation: the model of a checkpoint, behind a unix socket.

One request per connection: a JSON line in, a JSON line out (VC3D's socket client). Requests are answered one at a
time. Coordinates are VC3D fiber base voxels (L0 of VC3D's volume, ``base_um`` per voxel, given with each request);
the service maps them to the checkpoint's base grid (``--base-um``; by default parsed from its CT location, e.g.
"2.400um") and its trace grid, reads CT through the checkpoint's own volume, and returns base-voxel points.

Requests:
- {"op": "hello"} -> model, grid scale, operating policy.
- {"op": "extrapolate", "history": [[x,y,z], ...], "direction": [dx,dy,dz], "distance": d, "base_um": u,
   "family": "H"|"V"|null}: continue the line ``history`` (travel order, ending at the start point) outward along
   ``direction`` for up to ``distance`` base voxels. With at least 12 trace voxels of history the trace resumes as
   the tracer would at that head (the whole history as its observed path, heading fit, learned frame); otherwise
   it starts fresh from the CT seed heading of its family, signed along ``direction`` (``direction`` itself where
   the CT shows no clear sheet).
   -> {"points": [...], "reason": stop reason, "reached_length": bool, "decisions": n}
- {"op": "segment", "history": [...], "target": [x,y,z], "direction": [...] (optional), "base_um": u,
   "family": ...}: trace from the end of ``history`` towards ``target``; the trace ends where it passes the
   target's plane (normal: start -> target) or after 1.5 times the chord plus 16 trace voxels. Accepted when its
   closest approach to the target is within ``tolerance`` base voxels (default 3 trace voxels): the points up to
   that approach, then the target. -> {"points": [...], "accepted": bool, "meeting_error": base voxels, "reason"}
Family, when not given: V when |direction_z| >= cos 45 degrees, else H.

  python -m vesuvius.neural_tracing.tracers.fibers.tracing.service --checkpoint CKPT --socket PATH [--parent-pid P]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.tracers.fibers.frames.ct import ct_seed_heading, trace_heading
from vesuvius.neural_tracing.tracers.fibers.frames.learned import predict_frames, trace_family_kwargs
from vesuvius.neural_tracing.tracers.fibers.frames.ct import ct_frame
from vesuvius.neural_tracing.tracers.fibers.tracing.volume import build_tracer
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, normalize, resample_polyline
from vesuvius.neural_tracing.tracers.shared.ct_structure import SeedHeadingError
from vesuvius.neural_tracing.tracers.shared.reference import trace_history

HEADING_FIT = 12.  # trace voxels of history for a resumed trace (the tracer's heading fit span)


def model_base_um(spec, given=None):
    if given:
        return float(given)
    match = re.search(r'(\d+(?:\.\d+)?)um', str(spec.ct_zarr))
    if not match:
        raise SystemExit('Cannot infer the checkpoint CT base voxel size; pass --base-um')
    return float(match.group(1))


class FiberFollow:
    def __init__(self, args):
        options = SimpleNamespace(
            checkpoint=args.checkpoint, device=args.device, precision=args.precision, confidence=args.confidence,
            n_commit=None, gate=None, cache_dir=args.cache_dir, out=args.state_dir, max_len=8000.,
            forward_chunk=64, sampling_seed=0, crops='gpu' if args.device.startswith('cuda') else 'cpu',
            gpu_chunks=args.gpu_chunks, axis_workers=0)
        Path(args.state_dir).mkdir(parents=True, exist_ok=True)
        self.tracer, self.vol, self.spec, self.policy = build_tracer(options)
        self.base_um = model_base_um(self.spec, args.base_um)
        self.g = float(self.spec.grid_scale)
        self.lock = threading.Lock()

    # -- coordinates: VC3D base <-> trace grid
    def to_grid(self, points, base_um):
        return np.asarray(points, np.float64).reshape(-1, 3)*(float(base_um)/self.base_um)/self.g

    def from_grid(self, points, base_um):
        return np.asarray(points, np.float64).reshape(-1, 3)*self.g*(self.base_um/float(base_um))

    @staticmethod
    def family(request, direction):
        family = request.get('family')
        if family in ('H', 'V'):
            return family
        return 'V' if abs(direction[2]) >= np.cos(np.radians(45)) else 'H'

    def start_state(self, path, direction, family):
        """(head, heading, initial state or None) of a trace continuing ``path`` (trace grid, travel order)."""
        pos = path[-1]
        if len(path) > 1 and arclength(path)[-1] >= HEADING_FIT:
            path = resample_polyline(path, 1.)
            path[-1] = pos
            heading = trace_heading(path, 0, direction)
            heading = heading if heading @ direction >= 0 else -heading
            predictor = self.tracer.frame_predictor
            if predictor is not None:
                frame = predict_frames(predictor, self.vol, [pos], [heading], [path], [family])[0]
            else:
                frame = ct_frame(self.vol, pos, heading)
            hist, hmask = trace_history(list(path), self.tracer.n_history)
            first = path[min(len(path)-1, 12)]-path[0]
            state = dict(pos=pos, frame=np.asarray(frame), hist=hist, hmask=hmask, heading_start=0, observed_path=path,
                         frame_policy=getattr(self.tracer, 'frame_policy'), fiber_family=family,
                         ct_frame_diagnostics={}, seed_pos=path[0].copy(), seed_tangent=normalize(first),
                         seed_age=float(arclength(path)[-1]), seed_valid=True)
            return pos, heading, state
        try:
            heading = ct_seed_heading(self.vol, pos, family)
        except SeedHeadingError:  # no clear sheet at the seed: VC3D's direction instead
            heading = direction
        return pos, (heading if heading @ direction >= 0 else -heading), None

    def run(self, path, direction, family, max_len, on_decision=None):
        pos, heading, state = self.start_state(path, direction, family)
        self.tracer.p.max_len = float(max_len)
        decisions = [0]

        def watch(i, s):
            decisions[0] += 1
            return on_decision(s) if on_decision else None
        paths, reasons = self.tracer.trace(pos[None], heading[None], on_decision=watch,
                                           initial_states=[state] if state is not None else None,
                                           **trace_family_kwargs(self.tracer, [family]))
        return np.asarray(paths[0]), reasons[0], decisions[0]

    # -- requests
    def hello(self, request):
        return dict(checkpoint=os.path.abspath(ARGS.checkpoint), grid_scale=self.g, base_um=self.base_um,
                    ct=str(self.spec.ct_zarr),
                    operating_policy=self.policy.to_dict())

    def extrapolate(self, request):
        u = request['base_um']
        history = self.to_grid(request['history'], u)
        direction = normalize(np.asarray(request['direction'], np.float64))
        family = self.family(request, direction)
        distance = float(request['distance'])*(float(u)/self.base_um)/self.g
        path, reason, n = self.run(history, direction, family, distance)
        travelled = arclength(path)[-1] if len(path) > 1 else 0.
        points = self.from_grid(path, u)
        points[0] = np.asarray(request['history'][-1], np.float64)
        return dict(points=points.tolist(), reason=reason, reached_length=bool(travelled >= distance-1.),
                    decisions=n, family=family)

    def segment(self, request):
        u = request['base_um']
        history = self.to_grid(request['history'], u)
        target = self.to_grid(request['target'], u)[0]
        start = history[-1]
        chord = target-start
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            return dict(points=[request['history'][-1], request['target']], accepted=False, meeting_error=0.,
                        reason='degenerate span')
        axis = chord/length
        direction = normalize(np.asarray(request.get('direction') or axis, np.float64))
        direction = direction if direction @ axis > 0 else axis
        family = self.family(request, axis)
        tolerance = float(request.get('tolerance', 3.*self.g*self.base_um/float(u)))*(float(u)/self.base_um)/self.g

        def past_target(state):
            return 'target_plane' if (np.asarray(state['pos'])-target) @ axis > 0 else None
        path, reason, n = self.run(history, direction, family, 1.5*length+16., on_decision=past_target)
        d = np.linalg.norm(path-target, axis=1)
        k = int(np.argmin(d))
        meeting = float(d[k])*self.g*self.base_um/float(u)
        accepted = d[k] <= tolerance
        points = self.from_grid(np.concatenate([path[:k+1], target[None]]) if accepted else path, u)
        points[0] = np.asarray(request['history'][-1], np.float64)
        if accepted:
            points[-1] = np.asarray(request['target'], np.float64)
        return dict(points=points.tolist(), accepted=bool(accepted), meeting_error=meeting,
                    reason=reason if accepted else f'missed the target by {meeting:.1f} base voxels ({reason})',
                    decisions=n, family=family)

    def handle(self, request):
        op = request.get('op')
        if op not in ('hello', 'extrapolate', 'segment'):
            raise ValueError(f'unknown op {op!r}')
        with self.lock:
            return getattr(self, op)(request)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        t = time.time()
        try:
            request = json.loads(line)
            reply = dict(ok=True, **SERVICE.handle(request))
        except Exception as error:  # reported to VC3D, which falls back like for a native trace failure
            traceback.print_exc()
            reply = dict(ok=False, error=f'{type(error).__name__}: {error}')
        reply['seconds'] = round(time.time()-t, 3)
        self.wfile.write((json.dumps(reply)+'\n').encode())
        self.wfile.flush()


def watch_parent(pid, server):
    while True:
        time.sleep(2.)
        try:
            os.kill(pid, 0)
        except OSError:
            server.shutdown()
            return


def main(argv=None):
    global SERVICE, ARGS
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--socket', required=True)
    ap.add_argument('--parent-pid', type=int, help='exit when this process ends (VC3D)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    ap.add_argument('--confidence', type=float, default=0.5)
    ap.add_argument('--base-um', type=float, help='base voxel size of the checkpoint CT (default: from its location)')
    ap.add_argument('--gpu-chunks', type=int, default=512)
    ap.add_argument('--cache-dir', help="CT chunk cache (default: the checkpoint's)")
    ap.add_argument('--state-dir', default=str(Path.home()/'.cache'/'fiber_follow_service'),
                    help='CT normalization record')
    ARGS = ap.parse_args(argv)
    SERVICE = FiberFollow(ARGS)
    path = Path(ARGS.socket)
    path.unlink(missing_ok=True)
    server = socketserver.UnixStreamServer(str(path), Handler)
    if ARGS.parent_pid:
        threading.Thread(target=watch_parent, args=(ARGS.parent_pid, server), daemon=True).start()
    print(f'fiber_follow service listening on {path} (grid scale {SERVICE.g:g}, base {SERVICE.base_um:g} um, '
          f'confidence {SERVICE.policy.confidence:g})', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        path.unlink(missing_ok=True)
        SERVICE.tracer.close()


if __name__ == '__main__':
    main()
