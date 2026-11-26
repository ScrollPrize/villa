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


def configure_solver_sqlite(config):
    """Bulk-worker startup only: avoid SQLite's global allocator counters.

    This must run before opening a connection, not inside a live read API.
    Ordinary library users retain their existing SQLite runtime defaults.
    """
    from vc_object_store import configure_sqlite_runtime
    configure_sqlite_runtime(memory_statistics=bool(config.get('sqlite_memory_statistics',False)),
        pagecache_bulk_bytes=int(config.get('sqlite_pagecache_bulk_bytes',0)))


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


HANDEDNESS_META_KEY='winding_handedness'
HANDEDNESS_SOURCE_META_KEY='winding_handedness_source'  # who states it: 'manual' or 'catalog <sample>/<volume>'


def catalog_handedness(volume):
    """The open-data catalog's winding sense for ``volume``, or None when it does not orient it.

    Same reading as VC3D's fiber map: the catalog describes one spiral seen
    from the top, so the sense is -1 when exactly one of
    ``z_direction_is_top_to_bottom`` and ``left_handed_coordinates`` holds."""
    try:
        from vc_reader import Catalog
        catalog=Catalog.load()
        record=catalog.volumes.get(str(volume)) if hasattr(catalog.volumes,'get') else None
        if record is None:
            matches=[v for v in catalog.volumes if getattr(v,'id',None)==str(volume)]
            record=matches[0] if len(matches)==1 else None
    except Exception as error:  # offline or unknown volume: the catalog states nothing
        print(json.dumps(dict(stage='catalog_handedness_unavailable',volume=str(volume),error=str(error)[:200])),flush=True)
        return None
    if record is None:return None
    top=record.z_direction_is_top_to_bottom;left=record.left_handed_coordinates
    if top is None or left is None:return None
    return -1 if bool(top)!=bool(left) else 1


def store_handedness(path, volume):
    """``winding_handedness`` stated on the store's umbilicus object for ``volume``, or None."""
    with CollectionReader(str(path)) as store:
        objects=[r for r in store.objects('umbilicus')
                 if volume is None or ((r.get('meta') or {}).get('coordinates') or {}).get('volume_uuid') in (None,str(volume))]
    values={int((r.get('meta') or {})[HANDEDNESS_META_KEY]) for r in objects if HANDEDNESS_META_KEY in (r.get('meta') or {})}
    if len(values)>1:raise ValueError('Store umbilicus objects state different winding handedness values')
    return values.pop() if values else None


def tag_store_handedness(path, value):
    """Record the scroll's winding direction on the store's umbilicus object (a writable, unsealed store)."""
    if value not in (1,-1):raise ValueError('handedness must be +1 or -1')
    from vc_object_store import ObjectStore
    with CollectionReader(str(path)) as reader:objects=reader.objects('umbilicus')
    if len(objects)!=1:raise ValueError('Store must contain exactly one umbilicus object to tag, found %d'%len(objects))
    with ObjectStore.open(str(path),author='vc_wgraph') as store:
        store.set_meta(objects[0]['uuid'],HANDEDNESS_META_KEY,int(value))
        store.set_meta(objects[0]['uuid'],HANDEDNESS_SOURCE_META_KEY,'manual')


def winding_handedness(config, path, volume):
    """The scroll's winding direction (+1/-1) for seam transport; never inferred.

    Precedence: explicit ``handedness`` setting, the store umbilicus object's
    ``winding_handedness`` meta, the open-data catalog orientation. A missing
    value is an error so that mirrored solves cannot happen silently."""
    explicit=config.get('handedness')
    if explicit is not None:
        if explicit not in (1,-1):raise ValueError('handedness must be +1 or -1')
        return int(explicit)
    stated=store_handedness(path,volume)
    if stated is not None:
        if stated not in (1,-1):raise ValueError('Store winding_handedness must be +1 or -1')
        return stated
    catalog=catalog_handedness(volume)
    if catalog is not None:return catalog
    raise ValueError(f'Winding handedness unknown for volume {volume}: set handedness in the stage settings, '
                     f'tag the store umbilicus with {HANDEDNESS_META_KEY} (vc_wgraph handedness --set), '
                     'or orient the volume in the open-data catalog')


def fiber_families_from_record(path, ids, mode='any'):
    """H(0)/V(1) families from the fibers' stored annotation tags, -1 otherwise.

    ``mode`` selects ``any`` tag, only ``manual`` tags or only ``automatic``
    tags (the store's ``manual_family`` flag). Returns the family column
    aligned with ``ids`` and tag-source counts.
    """
    if mode not in ('any','manual','automatic'):raise ValueError('tags must be any, manual or automatic')
    with CollectionReader(str(path)) as store:
        records={record['uuid']:record.get('fiber') or {} for record in store.objects('fiber')}
    ids=np.asarray(ids,dtype=str)
    family=np.full(len(ids),-1,np.int64);counts=dict(manual=0,automatic=0,untagged=0,unknown=0,filtered=0)
    for i,fiber in enumerate(ids):
        detail=records.get(str(fiber))
        if detail is None:counts['unknown']+=1;continue
        label=detail.get('family') or '';manual=bool(detail.get('manual_family',False))
        if label not in ('H','V'):counts['untagged']+=1;continue
        if (mode=='manual' and not manual) or (mode=='automatic' and manual):counts['filtered']+=1;continue
        counts['manual' if manual else 'automatic']+=1
        family[i]=0 if label=='H' else 1
    return family,counts


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
        # What the fibers' normal surfaces found about each crossing, when the extraction
        # recorded it with the constraints (vc_object_store geometry/FiberCrossingFilter.hpp).
        self.has_crossing_findings=bool(data['has_crossing_findings'])
        self.crossing_findings=data['crossing_findings'];self.crossing_alternative=data['crossing_alternative']

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


def write_fiber_result_columns(path,dependency,expected_source,stage,method,ids,columns,settings=None,
                               input_run=None,metrics=(),workers=32,run_uuid=None,timings_path=None):
    """Write a result overlay from numeric columns, never Python corpus rows.

    Only the fixed schema's column names and the small run-level metrics are
    adapted in Python. UUID parsing, routing, validation and bounded row
    construction execute in the native parallel initial-import path.
    """
    from vc_object_store import export_fiber_result_columns,fiber_uuid_bytes
    import time
    started=time.monotonic()
    with CollectionReader(str(dependency)) as source:
        if source.source!=expected_source:raise ValueError('Result export input snapshot changed')
    ids=np.asarray(ids)
    if ids.ndim==2 and ids.shape[1]==16 and ids.dtype==np.uint8:
        binary=np.ascontiguousarray(ids)
    else:
        if ids.ndim!=1 or ids.dtype.kind not in 'US':raise ValueError('Expected a UUID string or binary [N,16] column')
        text=np.char.encode(ids,'ascii') if ids.dtype.kind=='U' else ids
        if text.dtype.itemsize!=36:raise ValueError('Expected canonical-width UUID strings')
        binary=fiber_uuid_bytes(np.ascontiguousarray(text).view(np.uint8).reshape(-1,36),workers)
    schema=fiber_graph_solve_table('fibers')
    types={column['name']:column['type'] for column in schema['columns']}
    typed={}
    for name,values in columns.items():
        if name not in types or types[name] not in ('integer','real'):
            raise ValueError('Unknown or reserved numeric fiber result field: '+name)
        array=np.asarray(values)
        if array.ndim!=1:raise ValueError('Fiber result columns must be one-dimensional: '+name)
        if types[name]=='integer' and (array.dtype.kind not in 'biu' or not np.can_cast(array.dtype,np.int64,casting='safe')):
            raise ValueError('Fiber result field requires a safely representable integer column: '+name)
        dtype=np.int64 if types[name]=='integer' else np.float64
        typed[name]=np.ascontiguousarray(values,dtype=dtype)
    result=export_fiber_result_columns(str(path),str(dependency),expected_source,
        str(run_uuid or uuid.uuid4()),None if input_run is None else str(input_run),stage,method,
        settings or {},binary,typed,metrics,workers)
    if timings_path is not None:
        from .fiber_min_flow_experiment import write
        result['wall_seconds']=time.monotonic()-started
        result['bytes']=Path(result['path']).stat().st_size
        write(Path(timings_path),result)
    return result['run_uuid']


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
