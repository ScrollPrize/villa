"""Language-only adapters for the native parallel fixed-winding UV importer."""
import numpy as np


def _select(data, *, graph_to_catalog=None, valid_a=None, valid_b=None,
            valid_connector=None, node_mask=None, workers=32):
    from vc_object_store import prepare_uv_event_columns
    return prepare_uv_event_columns(
        np.ascontiguousarray(data['i'],dtype=np.int64),np.ascontiguousarray(data['j'],dtype=np.int64),
        graph_to_catalog,np.ascontiguousarray(data['family'],dtype=np.int64),
        np.asarray(data['ds_a'],dtype=np.float64),np.asarray(data['ds_b'],dtype=np.float64),
        np.ascontiguousarray(data['parallel'],dtype=bool),valid_a,valid_b,valid_connector,node_mask,workers)


def _adapt(result, ids, family, lengths):
    selected=result.pop('selected_indices')
    # These bounded-count column gathers are NumPy-native, not Python object
    # iteration. All event-sized filtering/copying is owned by the C++ pass.
    result.update(source_id=ids[selected],family=family[selected],end=lengths[selected])
    return result,selected


def prepare_events(paths, graph, workers=32, node_mask=None):
    """Consume native owned graph buffers without Python event/batch iteration."""
    if paths.graph_index is None:raise ValueError('Bind the graph to packed geometry before preparation')
    result=_select(dict(i=graph.i,j=graph.j,family=paths.catalog['family'],
                        ds_a=graph.arc_a,ds_b=graph.arc_b,parallel=graph.parallel),
        graph_to_catalog=np.asarray(paths.graph_index,dtype=np.int64),
        valid_a=graph.valid_a,valid_b=graph.valid_b,valid_connector=graph.valid_connector,
        node_mask=node_mask,workers=workers)
    return _adapt(result,paths.ids,paths.catalog['family'],paths.keys['arc'][paths.offset[1:]-1])


def select_inputs(data, workers=32, node_mask=None):
    """The same native selector for an already-decoded unsigned input graph."""
    result=_select(data,node_mask=node_mask,workers=workers)
    return _adapt(result,np.asarray(data['source_id']),np.asarray(data['family']),np.asarray(data['end']))


def write_uv_results(config, data, q, x, v, widths, *, optimizer, global_integer_shift):
    """Native column export; no millions-of-dictionaries Python result path."""
    import uuid
    from vc_object_store import CollectionReader,export_fixed_uv_columns,fiber_uuid_bytes
    from .fiber_min_flow_experiment import write
    from pathlib import Path
    import time
    dependency=config['input_store'];expected=config['input_source']
    with CollectionReader(str(dependency)) as source:
        if source.source!=expected:raise ValueError('UV export input snapshot changed')
    workers=int(config.get('workers',32));started=time.monotonic()
    text=np.char.encode(data['source_id'],'ascii').view(np.uint8).reshape(-1,36)
    binary=fiber_uuid_bytes(text,workers)
    result=export_fixed_uv_columns(config['result_store'],str(dependency),expected,
        str(uuid.uuid4()),config.get('input_run_uuid'),optimizer,
        dict(global_integer_shift=int(global_integer_shift),batch_size=int(config.get('batch_size',32768)),assignments_fixed=True),
        binary,np.ascontiguousarray(data['family'],dtype=np.int64),
        np.ascontiguousarray(data['fixed_winding'],dtype=np.float64),
        np.ascontiguousarray(data['direction'],dtype=np.float64),
        np.ascontiguousarray(data['sample_owner'],dtype=np.int64),
        np.ascontiguousarray(data['sample_ds'],dtype=np.float64),
        np.ascontiguousarray(q,dtype=np.float64),np.ascontiguousarray(x,dtype=np.float64),np.ascontiguousarray(v,dtype=np.float64),
        np.ascontiguousarray(widths,dtype=np.float64),workers)
    result['wall_seconds']=time.monotonic()-started
    result['bytes']=Path(result['path']).stat().st_size
    write(Path(config['output_root'])/'uv-export-timings.json',result)
    return result['run_uuid']
