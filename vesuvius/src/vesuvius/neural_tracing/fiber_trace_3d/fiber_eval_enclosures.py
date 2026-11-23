"""Import annotated fibers, transfer normal signs and export enclosure graphs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin
from urllib.request import urlopen
import uuid
import numpy as np
from lasagna.inference_provenance import sha256_file
from .fiber_min_flow_experiment import write


def download(config, root):
    source = root/'source'; source.mkdir(exist_ok=True)
    index = source/'index.html'
    if not index.exists():
        with urlopen(config['url'], timeout=60) as stream: index.write_bytes(stream.read())
    names = sorted(set(re.findall(r'href="([^"/]+\.json)"', index.read_text())))
    if not names: raise ValueError('No fiber JSONs in listing')
    def fetch(name):
        if Path(name).name != name: raise ValueError('Unsafe source filename')
        p = source/name
        if not p.exists():
            with urlopen(urljoin(config['url'], name), timeout=120) as stream:
                payload = stream.read()
            json.loads(payload)
            tmp = p.with_suffix('.partial'); tmp.write_bytes(payload); tmp.replace(p)
        return name, sha256_file(p)
    with ThreadPoolExecutor(max_workers=16) as pool: hashes = dict(pool.map(fetch, names))
    write(root/'download.json', dict(url=config['url'], files=hashes))
    return [source/n for n in names]


def prepare(config, root):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, fiber_volume
    files = download(config, root); curves = []; index = []; families = []; lengths = []
    volume=config.get('volume') or fiber_volume(config['frame_template'])
    expected=np.asarray(config['base_shape_zyx'])
    coordinate_space=config['coordinate_space']
    if coordinate_space.split('/')[-1]!=volume:raise ValueError('Target volume/coordinate space differs')
    frame=dict(vc_open_data_coordinate_space=coordinate_space,
               vc_open_data_source_coordinate_level=0,vc_open_data_source_coordinate_scale_factor=1)
    shape_notes = {}; excluded = []
    for file in files:
        d = json.loads(file.read_text()); xyz = np.asarray(d.get('line_points', []), float)
        tag = d.get('hv_classification', {})
        family = tag.get('manual_tag') or tag.get('automatic_tag')
        if family not in ('H','V'):
            excluded.append(dict(file=file.name, reason='missing_family')); continue
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz)<2 or not np.isfinite(xyz).all():
            raise ValueError('Invalid line_points: '+file.name)
        if 'vc_open_data_coordinate_space' in d:
            for key, value in frame.items():
                actual = d.get(key)
                if key == 'vc_open_data_coordinate_space': actual = str(actual).removesuffix('@L0')
                if actual != value: raise ValueError('Coordinate frame mismatch: '+file.name)
        elif not any(frame['vc_open_data_coordinate_space'].replace('/', '-') in
                     p.get('segment_to_next', {}).get('fiber_manifest', '') for p in d.get('control_points', [])):
            raise ValueError('No matching coordinate provenance: '+file.name)
        shape = np.asarray(d.get('coordinate_base_shape_zyx', expected))
        if not np.array_equal(shape, expected):
            allowed = config.get('allowed_source_shape_zyx')
            if allowed is None or not np.array_equal(shape, allowed): raise ValueError('Unapproved shape discrepancy')
            shape_notes[file.name] = shape.tolist()
        if np.any(xyz<0) or np.any(xyz>=expected[::-1]): raise ValueError('Points outside target coordinate frame')
        # Frame discrepancy is metadata-only: never rescale, shift or clip points.
        fid = str(uuid.UUID(d['uuid'])) if d.get('uuid') else str(uuid.uuid5(uuid.NAMESPACE_URL,config['url']+file.name)); length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        curves.append((dict(id=fid, name=file.name, family=family, meta={k:v for k,v in d.items() if k not in ('line_points','control_points')}), xyz))
        index.append(dict(source_id=fid, name=file.name, family=family, length=length))
        families.append(0 if family=='H' else 1); lengths.append(length)
    # All solver arrays follow the same global UUID order as store readers.
    order=np.argsort([row['id'] for row,_ in curves],kind='stable')
    curves=[curves[i] for i in order];families=np.asarray(families)[order];lengths=np.asarray(lengths)[order]
    write_initial_fibers(root/'fibers.vcobj',curves,volume=volume,packed=True)
    n=len(curves)
    np.savez(root/'fiber_catalog.npz',source_id=np.asarray([row['id'] for row,_ in curves]),family=families,
             start=np.zeros(n),end=lengths,node_ds=np.column_stack((np.zeros(n),lengths)),links=np.empty((0,4)))
    write(root/'import.json', dict(fibers=index, excluded=excluded, source_shape_discrepancies=shape_notes,
                                 coordinate_action='none; original native base XYZ preserved'))
    print(f'imported {n} complete fibers; excluded {len(excluded)} without H/V', flush=True)


def run(config):
    from .fiber_native_workflow import run as native_run
    from .fiber_reference_signs import validate_reference
    from .fiber_min_flow_selection import EdgeStore
    from .fiber_min_flow_benchmark import Paths, graph
    from .fiber_enclosure_graph import extract_relations, graph_output, components
    started = time.monotonic(); root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=True)
    if (root/'request.json').exists() and json.loads((root/'request.json').read_text()) != config:
        raise ValueError('Changed request; choose new output root')
    write(root/'request.json', config)
    if config.get('existing_extraction'):
        prior=Path(config['existing_extraction']).resolve()
        for name in ('import.json','fibers.vcobj','fiber_catalog.npz','extraction.json','orientation.json','raw.vcobj','oriented.vcobj'):
            if not (root/name).exists():(root/name).symlink_to(prior/name)
    if not (root/'import.json').exists(): prepare(config, root)
    for stage, template, done in [('extract','extract_template','extraction.json'), ('orient','orient_template','orientation.json')]:
        if (root/done).exists(): continue
        cfg = json.loads(Path(config[template]).read_text())
        cfg.update(fibers=str(root/'fibers.vcobj'), full_volume=True, output=str(root/done),
                   database=str(root/('raw.vcobj' if stage=='extract' else 'oriented.vcobj')),
                   progress_file=str(root/(stage+'_progress.jsonl')))
        if stage=='orient': cfg['orient_database'] = str(root/'raw.vcobj')
        path = root/(stage+'_config.json'); write(path, cfg)
        native_run(dict(output_root=str(root/'native'), native_binary=config['native_binary'],
                        input_files=[str(path)], commands=[dict(name=stage,binary='vc_fiber_constraints',args=[str(path)])]))
    with np.load(root/'fiber_catalog.npz') as z: d = dict(z)
    validate_reference(config['reference_signs'], root/'fibers.vcobj')
    if not (root/'sign_alignment.json').exists():
        if config.get('sign_alignment') != 'reference_constraints': raise ValueError('Use native reference-constraint alignment')
        from .fiber_reference_constraint_signs import run as align
        settings=dict(config['reference_constraint_alignment'],fibers=str((root/'fibers.vcobj').resolve()),database=str((root/'oriented.vcobj').resolve()))
        if (Path(settings['output_root'])/'result.json').exists():
            records=json.loads((Path(settings['output_root'])/'result.json').read_text())['fibers']
        else:records=align(settings)
        write(root/'sign_alignment.json', dict(reference=config['reference_signs'], fibers=records))
    records = json.loads((root/'sign_alignment.json').read_text())['fibers']
    signs = {r['source_id']:r['normal_sign'] for r in records}
    ids = d['source_id']; known = [i for i in ids if signs[str(i)]]
    store = EdgeStore(dict(database=str(root/'oriented.vcobj')), known, dict(zip(ids,d['family'])), signs)
    store.load(known); edges = store.edges()
    for col,key in enumerate(('i','j','ds_a','ds_b','distance','sign','parallel')):
        v = edges[:,col]
        d[key] = np.searchsorted(ids, v.astype('U36')) if col<2 else v.astype(bool if key=='parallel' else float)
    np.savez(root/'ordinary_tensors.npz', **d)
    paths = Paths(dict(fibers=str(root/'fibers.vcobj'),umbilicus=config['umbilicus']), d)
    relations, rejected = extract_relations(d, paths)
    ordinary, _ = graph(config, d, paths)
    output = graph_output(ordinary, relations, config.get('constraint_mode','only'))
    np.savez(root/'graph.npz', **output)
    labels, potential, groups, conflicts = components(len(ids), relations)
    np.savez(root/'enclosure_components.npz', source_id=ids, component=labels, relative_winding=potential)
    sizes = sorted((len(g) for g in groups), reverse=True)
    detail = [dict(component=c, fibers=len(g), h=int((d['family'][g]==0).sum()),
                   v=int((d['family'][g]==1).sum()), source_ids=ids[g].tolist(),
                   relations=int(sum(labels[r['h']]==c for r in relations)),
                   conflicting_relations=int(sum(labels[relations[e]['h']]==c for e in conflicts))) for c,g in enumerate(groups)]
    result = dict(fibers=len(ids), h=int((d['family']==0).sum()), v=int((d['family']==1).sum()),
                  signs_supported=len(known), signs_unresolved=len(ids)-len(known),
                  ordinary_events=len(edges), enclosure_relations=len(relations),
                  components=len(groups), nontrivial_components=sum(s>1 for s in sizes),
                  isolated_fibers=sum(s==1 for s in sizes), component_sizes=sizes,
                  contradictory_cycles=len(conflicts), conflicting_relation_indices=conflicts,
                  rejected_pairs={r:sum(x['reason']==r for x in rejected) for r in sorted(set(x['reason'] for x in rejected))},
                  seconds=time.monotonic()-started)
    write(root/'enclosures.json', dict(relations=relations,rejected=rejected,components=detail))
    write(root/'results.json', result); print(json.dumps(result), flush=True)
    return result
