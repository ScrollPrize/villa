"""Shared vcobj boundary for measured fiber graphs and derived solve results.

UUIDs identify fibers/relations. Dense integer indices exist only in solver
tensors. Plain, packed and sharded stores use exactly the same native reader.
"""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from vc_object_store import CollectionReader, fiber_graph_solve_table, fiber_graph_solve_row, write_extension_store, store_position_from_json

PREFIX = 'ext_fiber_graph_solve_'
CACHE_FORMAT = 3
CACHE_COLUMNS = ('ids','uuid_bytes','lengths','event_ids','edge_uuid_bytes','i','j','arc_a','arc_b',
                 'normal_a','normal_b','connector','distance','parallel','valid_a','valid_b','valid_connector')


def workflow_volume(config):
    """The measurement frame, or an explicit frame for already-prepared inputs.

    Compact measurements name their frame explicitly; a registered source
    volume is not equivalent.
    Only metadata is read here, not the constraint or geometry payloads.
    """
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_volume
    volume=config.get('working_volume')
    if config.get('database'):
        with CollectionReader(str(config['database'])) as store:
            table=next((t for t in store.extension_tables() if t['name']==PREFIX+'constraints'),None)
        if table is None:raise ValueError('Store has no fiber constraint measurements')
        frame=table['provisioning'].get('coordinate_space')
        if table['format_version']!=4 or not frame:
            raise ValueError('Compact format-4 measurements must name their coordinate space; convert older inputs explicitly')
        measured=frame.rsplit('/',1)[-1]
        if volume is not None and volume!=measured:
            raise ValueError('Working volume differs from constraint measurement frame')
        volume=measured
    return volume if volume is not None else fiber_volume(config['fibers'])


def store_digest(path):
    with CollectionReader(str(path)) as store:
        source = store.source
    return snapshot_digest(source)


def snapshot_digest(source):
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def original_fiber_source(database):
    """Resolve the recorded extraction source through the native pinned scope."""
    with CollectionReader(str(database)) as store:
        table=next(t for t in store.extension_tables() if t['name']==PREFIX+'constraints')
        pin=store_position_from_json(json.dumps(table['provisioning'].get('fiber_source',{})))
        return Path(store.resolve_store_position(pin)),pin


class _IndexedFiberIds:
    """Diagnostic view: selecting rows never expands the whole endpoint column."""
    dtype=np.dtype('U36')
    ndim=1
    def __init__(self,ids,index):self.ids,self.index=ids,index
    @property
    def shape(self):return self.index.shape
    def __len__(self):return len(self.index)
    def __getitem__(self,key):return self.ids[self.index[key]]
    def __array__(self,dtype=None,copy=None):
        values=self.ids[self.index]
        return values.astype(dtype,copy=bool(copy)) if dtype is not None else values.copy() if copy else values
    def tolist(self):return np.asarray(self).tolist()
    def __iter__(self):return map(self.ids.__getitem__,self.index)
    def __eq__(self,other):
        if isinstance(other,(str,np.str_)):
            at=np.searchsorted(self.ids,other)
            return self.index==at if at<len(self.ids) and self.ids[at]==other else np.zeros(len(self.index),bool)
        return np.asarray(self)==other
    def __ne__(self,other):return ~self.__eq__(other)


class _BinaryUuids:
    """Legacy text access over compact binary keys, formatting selected rows only."""
    dtype=np.dtype('U36')
    ndim=1
    def __init__(self,values):self.values=values
    @property
    def shape(self):return self.values.shape[:-1]
    def __len__(self):return len(self.values)
    def __getitem__(self,key):
        from vc_object_store import format_uuid_ascii
        values=self.values[key]
        if values.ndim==1:return str(uuid.UUID(bytes=values.tobytes()))
        text=format_uuid_ascii(np.ascontiguousarray(values).reshape(-1,16))
        return text.view('S36').reshape(values.shape[:-1]).astype('U36')
    def __array__(self,dtype=None,copy=None):
        values=self[:]
        return values.astype(dtype,copy=bool(copy)) if dtype is not None else values.copy() if copy else values
    def tolist(self):return np.asarray(self).tolist()
    def __iter__(self):return map(self.__getitem__,range(len(self)))
    def __eq__(self,other):
        if isinstance(other,(str,np.str_)):
            key=uuid.UUID(str(other)).bytes
            return self.values.view('V16').reshape(-1)==np.void(key)
        return np.asarray(self)==other
    def __ne__(self,other):return ~self.__eq__(other)


class FiberGraph:
    """One immutable graph snapshot, joined by UUID in C++, decoded once."""
    def __init__(self, path, workers=32, progress=False):
        with CollectionReader(str(path)) as store:
            self.source = store.source
            tables = store.extension_tables()
            self.tables = {t['name']: t for t in tables}
            self.measurement_info = self.tables[PREFIX+'constraints']
            data = store.fiber_graph_columns(workers,progress)
        # Constant column adaptation only. Native code resolves selected nodes,
        # maps lengths, decodes measurements and joins binary UUID endpoints.
        self.ids=data['fiber_uuid_ascii'].view('S36').reshape(-1).astype('U36')
        self.uuid_bytes=data['fiber_uuid'];self.volume=data['volume'];self.lengths=data['lengths']
        self.event_ids=data['event_ids'];self.edge_uuid_bytes=data['edge_uuid']
        self.i,self.j=data['endpoints'][:,0],data['endpoints'][:,1]
        self.arc_a,self.arc_b=data['arcs'][:,0],data['arcs'][:,1]
        self.normal_a,self.normal_b,self.connector=data['normal_a'],data['normal_b'],data['connector']
        self.distance=data['distance'];self.parallel=data['parallel'].view(bool)
        self.valid_a=data['valid_a'].view(bool);self.valid_b=data['valid_b'].view(bool)
        self.valid_connector=data['valid_connector'].view(bool)

    @property
    def fiber_a(self):return _IndexedFiberIds(self.ids,self.i)

    @property
    def fiber_b(self):return _IndexedFiberIds(self.ids,self.j)

    @property
    def edge_uuid(self):return _BinaryUuids(self.edge_uuid_bytes)

    @property
    def point_a(self):
        raise ValueError('Compact winding constraints do not store endpoint XYZ; request the source fiber geometry explicitly')

    @property
    def point_b(self):
        raise ValueError('Compact winding constraints do not store endpoint XYZ; request the source fiber geometry explicitly')

    def save_cache(self, root):
        """Decode once; worker processes share read-only memory-mapped columns."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name in CACHE_COLUMNS:
            np.save(root/(name+'.npy'),getattr(self,name),allow_pickle=False)
        manifest = dict(format_version=CACHE_FORMAT,source=self.source, volume=self.volume,
                        columns=list(CACHE_COLUMNS),measurement_info=self.measurement_info)
        (root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))

    @classmethod
    def cached(cls, path, root, workers=32, progress=True):
        root = Path(root)
        manifest = root/'manifest.json'
        with CollectionReader(str(path)) as store:
            source = store.source
        if not manifest.exists():
            cls(path,workers=workers,progress=progress).save_cache(root)
        info = json.loads(manifest.read_text())
        if info.get('format_version') != CACHE_FORMAT:
            raise ValueError('Constraint cache format changed; rebuild the graph cache')
        required=set(CACHE_COLUMNS)
        if set(info['columns']) != required:
            raise ValueError('Constraint cache columns differ; rebuild the graph cache')
        if info['source'] != source:
            raise ValueError('Constraint snapshot changed; use a new graph cache')
        result = object.__new__(cls)
        result.source, result.volume = info['source'], info['volume']
        result.measurement_info=info['measurement_info'];result.tables={PREFIX+'constraints':result.measurement_info}
        for name in info['columns']:
            setattr(result, name, np.load(root/(name+'.npy'), mmap_mode='r', allow_pickle=False))
        return result

    def indices(self, ids):
        index = np.searchsorted(self.ids, ids)
        if np.any(index >= len(self.ids)) or not np.array_equal(self.ids[index], ids):
            raise ValueError('Constraint refers to a fiber outside the extracted graph')
        return index

    def endpoint_rows(self):
        """Canonical endpoint records for small diagnostic traversals.

        Large preparation jobs use the numeric columns/batches directly.
        Normal records are decoded values, never serialized JSON.
        """
        for k, eid in enumerate(self.event_ids):
            yield dict(event_id=int(eid),edge_uuid=self.edge_uuid[k],fiber_a=self.fiber_a[k],fiber_b=self.fiber_b[k],
                       arc_a=float(self.arc_a[k]),arc_b=float(self.arc_b[k]),parallel=bool(self.parallel[k]),
                       distance=float(self.distance[k]),connector=self.connector[k].tolist(),
                       valid_connector=bool(self.valid_connector[k]),
                       normal_a=dict(valid=bool(self.valid_a[k]),xyz=self.normal_a[k].tolist()),
                       normal_b=dict(valid=bool(self.valid_b[k]),xyz=self.normal_b[k].tolist()))

    def batches(self, size=65536, first=None, last=None):
        # Bound before casting: a final exclusive endpoint may be INT64_MAX+1.
        # Passing that Python int directly to NumPy can promote comparisons to
        # float64 and lose the low source-ID bits near the integer limit.
        def position(value, default):
            if value is None:return default
            if not len(self.event_ids) or value<=int(self.event_ids[0]):return 0
            if value>int(self.event_ids[-1]):return len(self.event_ids)
            return int(np.searchsorted(self.event_ids,np.int64(value)))
        lo = position(first,0)
        hi = position(last,len(self.event_ids))
        for begin in range(lo, hi, size):
            yield slice(begin, min(begin+size, hi))


def write_results(path, dependencies, stage, method, tables, *, settings=None, run_uuid=None, input_runs=None):
    """Write only typed result rows in a sealed overlay; input geometry is shared.

    Each supplied row is a dictionary without run_uuid. Settings are explicit
    scalar algorithm settings, never a dumped configuration or input report.
    """
    run = uuid.UUID(run_uuid) if run_uuid else uuid.uuid4()
    if any(not isinstance(value, (str, int, float, bool, type(None))) for value in (settings or {}).values()):
        raise ValueError('Run settings must contain scalar metadata, not data arrays or configurations')
    definitions, payload = [], {}
    sources = []
    if input_runs is not None and len(input_runs)!=len(dependencies):
        raise ValueError('Expected one selected input run per dependency')
    for source in dependencies:
        with CollectionReader(str(source)) as store:
            sources.append(store.source)
    rows_by_role = dict(tables)
    rows_by_role['runs'] = [dict(stage=stage, method=method, settings=json.dumps(settings or {}, sort_keys=True))]
    rows_by_role['inputs'] = [dict(role=f'input_{i}', store_uuid=uuid.UUID(s['store_uuid']).bytes,
                                  generation_uuid=uuid.UUID(s['generation']).bytes,
                                  commit_sequence=s['commit_sequence']) for i, s in enumerate(sources)]
    if input_runs is not None:
        for row, selected in zip(rows_by_role['inputs'],input_runs):
            if selected is not None:row['input_run_uuid']=str(selected)
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
