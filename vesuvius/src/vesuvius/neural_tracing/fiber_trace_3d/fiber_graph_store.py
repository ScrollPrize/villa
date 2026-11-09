"""Shared vcobj boundary for measured fiber graphs and derived solve results.

UUIDs identify fibers/relations. Dense integer indices exist only in solver
tensors. Plain, packed and sharded stores use exactly the same native reader.
"""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from vc_object_store import CollectionReader, fiber_graph_solve_table, fiber_graph_solve_row, write_extension_store, geometry_scale_to_volume

PREFIX = 'ext_fiber_graph_solve_'
CACHE_FORMAT = 1


def workflow_volume(config):
    """The measurement frame, or an explicit frame for already-prepared inputs.

    Legacy format-1 measurements use the source geometry's volume. Format 2
    names its frame explicitly; a registered source volume is not equivalent.
    Only metadata is read here, not the constraint or geometry payloads.
    """
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume
    volume=config.get('working_volume')
    if config.get('database'):
        with CollectionReader(str(config['database'])) as store:
            table=next((t for t in store.extension_tables() if t['name']==PREFIX+'constraints'),None)
        if table is None:raise ValueError('Store has no fiber constraint measurements')
        frame=table['provisioning'].get('coordinate_space')
        if table['format_version']==2 and not frame:
            raise ValueError('Format-2 measurements must name their coordinate space')
        measured=frame.rsplit('/',1)[-1] if frame else fiber_volume(config['database'])
        if volume is not None and volume!=measured:
            raise ValueError('Working volume differs from constraint measurement frame')
        volume=measured
    return volume if volume is not None else fiber_volume(config['fibers'])


def store_digest(path):
    with CollectionReader(str(path)) as store:
        source = store.source
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


class FiberGraph:
    """One immutable graph snapshot, joined by UUID in C++, decoded once."""
    def __init__(self, path):
        with CollectionReader(str(path)) as store:
            self.source = store.source
            records = store.objects('fiber')
            tables = store.extension_tables()
            self.tables = {t['name']: t for t in tables}
            self.measurement_info = self.tables[PREFIX+'constraints']
            data = store.fiber_constraints()
            runs = (store.extension_rows(PREFIX+'runs') if PREFIX+'runs' in self.tables else [])
            extraction = [r[0] for r in runs if r[1] == 'constraint_extraction']
            if len(extraction) > 1:
                raise ValueError('Select one extraction graph, not multiple independent graphs')
            if extraction:
                rows = store.extension_rows(PREFIX+'fibers')
                selected = {str(uuid.UUID(bytes=r[0])) for r in rows if r[1] == extraction[0]}
                records = [r for r in records if r['uuid'] in selected]
        self.ids = np.asarray([r['uuid'] for r in records], dtype='U36')
        volumes = {(r['meta'].get('coordinates') or {}).get('volume_uuid') for r in records}
        frame=self.measurement_info['provisioning'].get('coordinate_space')
        if frame:
            self.volume=frame.rsplit('/',1)[-1]
        else:
            if len(volumes) != 1 or None in volumes:
                raise ValueError('Extracted graph must name its measurement volume')
            self.volume=volumes.pop()
        scales=[geometry_scale_to_volume(r['meta'],self.volume) for r in records]
        self.lengths=np.zeros(len(records),np.float64)
        mapped=[]
        for i,(record,scale) in enumerate(zip(records,scales)):
            if scale is None:mapped.append(i)
            else:self.lengths[i]=record['fiber']['length']*scale
        if mapped:
            from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import iter_polylines
            curves=iter_polylines(path,self.ids[mapped],volume=self.volume)
            for i,(row,_) in zip(mapped,curves):
                if row['id']!=self.ids[i]:raise ValueError('Mapped fiber order differs')
                self.lengths[i]=row['length']
        self.event_ids = np.frombuffer(data['ids'], dtype='=i8')
        self.edge_uuid = np.asarray(data['edge_uuid'], dtype='U36')
        self.fiber_a = np.asarray(data['fiber_a'], dtype='U36')
        self.fiber_b = np.asarray(data['fiber_b'], dtype='U36')
        values = np.frombuffer(data['values'], dtype='=f8').reshape(-1, 21)
        self.arc_a, self.arc_b = values[:, 0], values[:, 1]
        self.point_a, self.point_b = values[:, 2:5], values[:, 5:8]
        self.parallel_score, self.perpendicular_score = values[:, 8], values[:, 9]
        self.parallel = self.parallel_score > self.perpendicular_score
        self.distance, self.parallel_distance, self.evidence_weight = values[:, 10], values[:, 11], values[:, 12]
        self.normal_a, self.normal_b = values[:, 13:16], values[:, 16:19]
        self.valid_a, self.valid_b = values[:, 19].astype(bool), values[:, 20].astype(bool)
        self.i, self.j = self.indices(self.fiber_a), self.indices(self.fiber_b)

    def save_cache(self, root):
        """Decode once; worker processes share read-only memory-mapped columns."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        columns = []
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                np.save(root/(name+'.npy'), value, allow_pickle=False)
                columns.append(name)
        manifest = dict(format_version=CACHE_FORMAT,source=self.source, volume=self.volume, columns=columns)
        (root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))

    @classmethod
    def cached(cls, path, root):
        root = Path(root)
        manifest = root/'manifest.json'
        with CollectionReader(str(path)) as store:
            source = store.source
        if not manifest.exists():
            cls(path).save_cache(root)
        info = json.loads(manifest.read_text())
        if info.get('format_version') != CACHE_FORMAT:
            raise ValueError('Constraint cache format changed; rebuild the graph cache')
        required={'ids','lengths','event_ids','edge_uuid','fiber_a','fiber_b','arc_a','arc_b',
                  'point_a','point_b','parallel_score','perpendicular_score','parallel','distance',
                  'parallel_distance','evidence_weight','normal_a','normal_b','valid_a','valid_b','i','j'}
        if set(info['columns']) != required:
            raise ValueError('Constraint cache columns differ; rebuild the graph cache')
        if info['source'] != source:
            raise ValueError('Constraint snapshot changed; use a new graph cache')
        result = object.__new__(cls)
        result.source, result.volume = info['source'], info['volume']
        for name in info['columns']:
            setattr(result, name, np.load(root/(name+'.npy'), mmap_mode='r', allow_pickle=False))
        return result

    def indices(self, ids):
        index = np.searchsorted(self.ids, ids)
        if np.any(index >= len(self.ids)) or not np.array_equal(self.ids[index], ids):
            raise ValueError('Constraint refers to a fiber outside the extracted graph')
        return index

    def endpoint_rows(self):
        """Source-ordered endpoint records for small diagnostic traversals.

        Large preparation jobs use the numeric columns/batches directly.
        Normal records are decoded values, never serialized JSON.
        """
        for k, eid in enumerate(self.event_ids):
            yield (int(eid), self.fiber_a[k], self.fiber_b[k], float(self.arc_a[k]), float(self.arc_b[k]),
                   *self.point_a[k].tolist(), *self.point_b[k].tolist(),
                   float(self.parallel_score[k]), float(self.perpendicular_score[k]),
                   float(self.distance[k]), float(self.parallel_distance[k]),
                   dict(valid=bool(self.valid_a[k]), xyz=self.normal_a[k].tolist()),
                   dict(valid=bool(self.valid_b[k]), xyz=self.normal_b[k].tolist()))

    def batches(self, size=65536, first=None, last=None):
        lo = 0 if first is None else np.searchsorted(self.event_ids, first)
        hi = len(self.event_ids) if last is None else np.searchsorted(self.event_ids, last)
        for begin in range(lo, hi, size):
            yield slice(begin, min(begin+size, hi))


def write_results(path, dependencies, stage, method, tables, *, settings=None, run_uuid=None):
    """Write only typed result rows in a sealed overlay; input geometry is shared.

    Each supplied row is a dictionary without run_uuid. Settings are explicit
    scalar algorithm settings, never a dumped configuration or input report.
    """
    run = uuid.UUID(run_uuid) if run_uuid else uuid.uuid4()
    if any(not isinstance(value, (str, int, float, bool, type(None))) for value in (settings or {}).values()):
        raise ValueError('Run settings must contain scalar metadata, not data arrays or configurations')
    definitions, payload = [], {}
    sources = []
    for source in dependencies:
        with CollectionReader(str(source)) as store:
            sources.append(store.source)
    rows_by_role = dict(tables)
    rows_by_role['runs'] = [dict(stage=stage, method=method, settings=json.dumps(settings or {}, sort_keys=True))]
    rows_by_role['inputs'] = [dict(role=f'input_{i}', store_uuid=uuid.UUID(s['store_uuid']).bytes,
                                  generation_uuid=uuid.UUID(s['generation']).bytes,
                                  commit_sequence=s['commit_sequence']) for i, s in enumerate(sources)]
    for role, rows in rows_by_role.items():
        definition = fiber_graph_solve_table(role)
        definition['based_on'] = sources
        columns = [c['name'] for c in definition['columns']]
        materialized = []
        for row in rows:
            values = dict(row, run_uuid=run.bytes)
            if set(values)-set(columns):
                raise ValueError(f'Unknown {role} result fields: {set(values)-set(columns)}')
            for key in ('fiber_uuid', 'edge_uuid', 'input_run_uuid'):
                if isinstance(values.get(key), str):
                    values[key] = uuid.UUID(values[key]).bytes
            materialized.append(fiber_graph_solve_row(role, values)['values'])
        definitions.append(definition)
        payload[definition['name']] = materialized
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_extension_store(str(path), list(map(str, dependencies)), definitions, payload)
    return str(run)


def write_winding_result(path, fibers, ids, winding, *, method, family=None, component=None, continuous=None):
    """Persist existing solver readouts without re-rounding or changing labels."""
    ids, winding = np.asarray(ids), np.asarray(winding)
    if ids.shape != winding.shape or not np.isfinite(winding).all():
        raise ValueError('Invalid winding result shape/values')
    if continuous is not None and np.shape(continuous) != ids.shape:
        raise ValueError('Continuous winding shape differs')
    rows=[]
    for i, fid in enumerate(ids):
        value=float(winding[i])
        row=dict(fiber_uuid=str(fid),enabled=1,winding_mean=float(continuous[i]) if continuous is not None else value)
        if value.is_integer():row['selected_winding']=int(value)
        if family is not None:row['family']=int(family[i])
        if component is not None:row['component']=int(component[i])
        rows.append(row)
    return write_results(path,[fibers],'winding',method,{'fibers':rows})
