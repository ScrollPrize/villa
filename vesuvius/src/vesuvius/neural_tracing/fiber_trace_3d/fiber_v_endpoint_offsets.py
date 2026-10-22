"""Whole-source-fiber endpoint V and optional U offsets for same-sheet matching."""
import numpy as np
import torch


def fixed_geometry(d, state, widths, config, *, sigma=None, maximum_sigma=None):
    """Instantiate MAP geometry once; offsets never change membership."""
    from . import fiber_implicit_winding as imp
    with torch.no_grad():
        h, lp = imp.hypotheses(state, config['winding_sigma'] if sigma is None else sigma,
                              maximum_sigma)
        winner = lp.argmax(1)
        mapped = h[torch.arange(len(state), device=state.device), winner]
        confidence = lp.exp().amax(1)*state[:, 3].sigmoid()
        retained = confidence > .5
        i = torch.as_tensor(d['i'], device=state.device)
        j = torch.as_tensor(d['j'], device=state.device)
        family = torch.as_tensor(d['family'], device=state.device)
        qa, pa, va = imp.soft.points(mapped[i], family[i], torch.as_tensor(d['ds_a'], device=state.device), widths)
        qb, pb, vb = imp.soft.points(mapped[j], family[j], torch.as_tensor(d['ds_b'], device=state.device), widths)
        separation = mapped[j,0]-mapped[i,0]+torch.floor(pb-pa+.5)
        mask = retained[i] & retained[j] & (family[i]!=family[j]) & (separation==0)
        mask &= ~torch.as_tensor(d['parallel'], device=state.device).bool()
        boundaries = torch.cat((widths.new_zeros(1), widths.cumsum(0)))
        du = imp.soft.uv.q_to_u(qb,widths,boundaries)-imp.soft.uv.q_to_u(qa,widths,boundaries)
        return retained, mapped[:,0], mask, du[mask], (vb-va)[mask]


def exact_deformation(top, delta, smooth_h, smooth_v, axis=None, transverse_anchor=0.,
                      fixed_nodes=None, fixed_values=None):
    """Eliminate meeting equalities, then solve a gauge-fixed sparse quadratic."""
    from scipy.sparse import coo_matrix, vstack
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    if min(smooth_h,smooth_v)<=0: raise ValueError('Exact deformation stiffness must be positive')
    if not np.isfinite(transverse_anchor) or transverse_anchor<0:raise ValueError('Invalid transverse anchor')
    if transverse_anchor and axis not in ('u','v'):raise ValueError('Anchor requires U/V axis')
    n=len(top['v_offset_arc']);a=top['v_offset_a'];b=top['v_offset_b']
    adjacency=[[] for _ in range(n)]
    for x,y,d in zip(a,b,np.asarray(delta,dtype=np.float64)):
        adjacency[x].append((y,-d));adjacency[y].append((x,d))
    group=np.full(n,-1,dtype=np.int64);relative=np.zeros(n);count=0
    for root in range(n):
        if group[root]>=0:continue
        group[root]=count;stack=[root]
        while stack:
            x=stack.pop()
            for y,shift in adjacency[x]:
                expected=relative[x]+shift
                if group[y]<0:
                    group[y]=count;relative[y]=expected;stack.append(y)
                elif abs(relative[y]-expected)>1e-8:
                    raise ValueError('Inconsistent exact-location meeting cycle')
        count+=1
    l,r=top['v_offset_left'],top['v_offset_right']
    gap=top['v_offset_arc'][r]-top['v_offset_arc'][l]
    if np.any(gap<=0):raise ValueError('Nonpositive deformation arc interval')
    # Integral of squared deformation rate; no absolute-position penalty.
    weights=np.where(top['v_offset_family'][l]==0,smooth_h,smooth_v)/gap
    row=np.arange(len(l));scale=np.sqrt(weights)
    B=coo_matrix((np.r_[-scale,scale],(np.r_[row,row],np.r_[group[l],group[r]])),shape=(len(l),count)).tocsr()
    B.sum_duplicates();B.eliminate_zeros()
    rhs=-scale*(relative[r]-relative[l])
    graph=coo_matrix((np.ones(2*len(l)),(np.r_[group[l],group[r]],np.r_[group[r],group[l]])),shape=(count,count)).tocsr()
    components,labels=connected_components(graph,directed=False)
    node_component=labels[group]
    support=np.zeros(n)
    np.add.at(support,l,gap/2);np.add.at(support,r,gap/2)
    support[support==0]=1. # isolated event: one voxel integration support
    anchor_weights=transverse_anchor*support/1000**2*(top['v_offset_family']==(1 if axis=='u' else 0))
    anchored=np.bincount(node_component,weights=anchor_weights,minlength=components)>0
    hard = {}
    if fixed_nodes is not None:
        fixed_nodes=np.asarray(fixed_nodes,dtype=np.int64);fixed_values=np.asarray(fixed_values,dtype=np.float64)
        if fixed_nodes.shape!=fixed_values.shape or fixed_nodes.ndim!=1 or np.any((fixed_nodes<0)|(fixed_nodes>=n)):
            raise ValueError('Invalid hard reference targets')
        for node, value in zip(fixed_nodes, fixed_values):
            g = int(group[node]); target = float(value-relative[node])
            if not np.isfinite(target): raise ValueError('Nonfinite hard target')
            if g in hard and abs(hard[g]-target)>1e-8:
                raise ValueError('Inconsistent fixed reference targets')
            hard[g] = target
        if hard: anchored[labels[list(hard)]] = True
    fixed=np.unique(labels,return_index=True)[1][~anchored]
    anchor_nodes=np.flatnonzero(anchor_weights)
    if len(anchor_nodes):
        scales=np.sqrt(anchor_weights[anchor_nodes])
        A=coo_matrix((scales,(np.arange(len(anchor_nodes)),group[anchor_nodes])),shape=(len(anchor_nodes),count)).tocsr()
        B=vstack((B,A),format='csr');rhs=np.r_[rhs,-scales*relative[anchor_nodes]]
    free=np.ones(count,dtype=bool);free[fixed]=False
    translation=np.zeros(count)
    if hard:
        hard_ids=np.array(list(hard));translation[hard_ids]=list(hard.values());free[hard_ids]=False
        rhs=rhs-B[:,hard_ids]@translation[hard_ids]
    matrix=B[:,free];normal=(matrix.T@matrix).tocsc();target=matrix.T@rhs
    if free.any():translation[free]=spsolve(normal,target)
    offsets=relative+translation[group]
    # Mean-zero per connected geometry component is a gauge choice, not a cost.
    means=np.bincount(node_component,weights=offsets)/np.bincount(node_component)
    offsets-=np.where(anchored[node_component],0.,means[node_component])
    equality=np.asarray(delta)+offsets[b]-offsets[a]
    if fixed_nodes is not None and np.max(np.abs(offsets[fixed_nodes]-fixed_values),initial=0)>1e-7:
        raise RuntimeError('Fixed reference moved during deformation')
    stationarity=normal@translation[free]-target
    residual=float(np.max(np.abs(stationarity),initial=0)/max(1.,np.max(np.abs(target),initial=0)))
    if not np.isfinite(offsets).all() or np.max(np.abs(equality),initial=0)>1e-7 or residual>1e-7:
        raise RuntimeError('Exact deformation solve failed residual checks')
    return offsets,dict(components=int(components),free_translations=int(free.sum()),
        relative_stationarity=residual,max_equality_error=float(np.max(np.abs(equality),initial=0)),
        deformation_energy=float(np.sum(weights*(offsets[r]-offsets[l])**2)),
        transverse_anchor_energy=float(np.sum(anchor_weights*offsets**2)),anchored_components=int(anchored.sum()))


def run_exact(config):
    """CPU-only exact postprocess; no GPU reservation or optimization of windings."""
    import time
    import json
    from pathlib import Path
    from lasagna.inference_provenance import atomic_write, sha256_file
    from .fiber_implicit_winding import saved_sigma
    config=dict(config)
    validate_existing=config.pop('validate_existing',False)
    config['initial_solution_sha256']=sha256_file(config['initial_solution'])
    if config.get('fixed_reference_events'):
        digest=sha256_file(config['fixed_reference_events'])
        if config.get('fixed_reference_sha256',digest)!=digest:raise ValueError('Fixed reference file changed')
        config['fixed_reference_sha256']=digest
    start=time.monotonic();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():
        if not validate_existing:raise FileExistsError('Completed output exists')
        result=json.loads((root/'summary.json').read_text())
        previous=json.loads((root/'config.json').read_text())
        for p in [config['initial_solution'],config['tensor_cache']]+([config['fixed_reference_events']] if config.get('fixed_reference_events') else []):
            if result['input_hashes'][str(p)]!=sha256_file(p):raise ValueError('Postpass input changed')
        previous['initial_solution_sha256']=config['initial_solution_sha256']
        if previous!=config:raise ValueError('Postpass configuration changed')
        with np.load(config['initial_solution']) as parent,np.load(root/'solution.npz') as saved:
            if any(not np.array_equal(parent[k],saved[k]) for k in parent.files):
                raise ValueError('Postpass changed frozen parent arrays')
        # Repair only copied provenance; numerical output is never overwritten.
        result['config']=config
        atomic_write(root/'config.json',config,durable=False)
        atomic_write(root/'summary.json',result,durable=False)
        return result
    with np.load(config['tensor_cache']) as f:d={k:f[k] for k in f.files}
    with np.load(config['initial_solution']) as f:arrays={k:f[k] for k in f.files}
    for k in ('source_id','start','end','family'):
        if not np.array_equal(arrays[k],d[k]):raise ValueError('Input identity mismatch: '+k)
    state=torch.as_tensor(arrays['state']);widths=torch.as_tensor(arrays['widths'])
    sigma,bound=saved_sigma(state,arrays,config)
    sigma64,bound64=saved_sigma(state.double(),arrays,config)
    retained,winding,mask,_,_=fixed_geometry(d,state,widths,config,sigma=sigma,maximum_sigma=bound)
    _,_,mask64,du,dv=fixed_geometry(d,state.double(),widths.double(),config,sigma=sigma64,maximum_sigma=bound64)
    if not torch.equal(mask,mask64):raise ValueError('Float64 changed fixed meeting membership')
    graph=dict(d);selected=mask.numpy()
    for k in ('i','j','ds_a','ds_b','sign','distance','parallel'):graph[k]=d[k][selected]
    references=None;reference_selection=np.array([],dtype=np.int64)
    if config.get('fixed_reference_events'):
        from . import fiber_implicit_winding as imp
        with np.load(config['fixed_reference_events']) as f: references={k:f[k] for k in f.files}
        ri=torch.as_tensor(references['owner']);family=torch.as_tensor(d['family'])
        mapped=state.double().clone();mapped[:,0]=winding.double()
        qa,pa,va=imp.soft.points(mapped[ri],family[ri],torch.as_tensor(references['ds']),widths.double())
        rs=torch.as_tensor(references['state'],dtype=torch.float64)
        separation=rs[:,0]-mapped[ri,0]+torch.floor(rs[:,1]-pa+.5)
        rm=retained[ri] & (family[ri]!=torch.as_tensor(references['family'])) & (separation==0)
        rm &= ~torch.as_tensor(references['parallel'])
        reference_selection=np.flatnonzero(rm.numpy())
        bounds=torch.cat((widths.new_zeros(1),widths.double().cumsum(0)))
        reference_delta=dict(u=references['u'][reference_selection]-imp.soft.uv.q_to_u(qa,widths.double(),bounds)[rm].numpy(),
                             v=references['state'][reference_selection,2]-va[rm].numpy())
        # Self-edges add reference endpoint nodes to the common topology only;
        # actual equalities below remain test-test meetings plus hard targets.
        for k,values in [('i',references['owner']),('j',references['owner']),
                         ('ds_a',references['ds']),('ds_b',references['ds']),
                         ('sign',references['sign']),('distance',references['distance']),
                         ('parallel',references['parallel'])]:
            graph[k]=np.r_[graph[k],values[reference_selection]]
    if not len(graph['i']):raise ValueError('No fixed meetings')
    top=topology(graph,merge_distance=0.)
    reference_nodes=top['v_offset_a'][int(mask.sum()):].copy()
    for k in ('v_offset_a','v_offset_b'):top[k]=top[k][:int(mask.sum())]
    offsets={};checks={};before={};after={};deformation={}
    def stats(x):
        x=np.abs(x)
        if not len(x):return dict(mean=None,median=None,p95=None,max=None)
        return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))
    for axis,delta in [('u',du.numpy()),('v',dv.numpy())]:
        o,check=exact_deformation(top,delta,config[axis+'_offset_smooth_h'],config[axis+'_offset_smooth_v'],
            axis=axis,transverse_anchor=config.get('exact_transverse_anchor',0.),
            fixed_nodes=reference_nodes if len(reference_nodes) else None,
            fixed_values=reference_delta[axis] if len(reference_nodes) else None)
        offsets[axis+'_offsets']=o;checks[axis]=check;before[axis]=stats(delta)
        after[axis]=stats(delta+o[top['v_offset_b']]-o[top['v_offset_a']])
        l,r=top['v_offset_left'],top['v_offset_right'];gap=top['v_offset_arc'][r]-top['v_offset_arc'][l]
        deformation[axis]=dict(offset=stats(o),rate=stats((o[r]-o[l])/gap))
    arrays.update(top,**offsets,retained=retained.numpy(),fixed_winding=winding.numpy(),fixed_event_indices=np.flatnonzero(selected))
    arrays.update(fixed_reference_event_indices=reference_selection,fixed_reference_nodes=reference_nodes)
    np.savez(root/'solution.npz',**arrays)
    with np.load(root/'solution.npz') as f:
        for axis,delta in [('u',du.numpy()),('v',dv.numpy())]:
            error=delta+f[axis+'_offsets'][top['v_offset_b']]-f[axis+'_offsets'][top['v_offset_a']]
            if np.max(np.abs(error),initial=0)>1e-7:raise RuntimeError('Saved equality failed')
    result=dict(model='exact_fixed_deformation_v1',config=config,before=before,after=after,
        checks=checks,deformation=deformation,retained_pieces=int(retained.sum()),same_sheet_pairs=int(mask.sum()),
        nodes=len(top['v_offset_arc']),seconds=time.monotonic()-start,
        fixed_reference_meetings=len(reference_nodes),
        reference_after={axis:stats(offsets[axis+'_offsets'][reference_nodes]-reference_delta[axis])
                         for axis in ('u','v')} if len(reference_nodes) else {},
        gauge='weak transverse alignment where configured; mean-zero gauge only for unanchored components',
        input_hashes={str(p):sha256_file(p) for p in [config['initial_solution'],config['tensor_cache']]+
                      ([config['fixed_reference_events']] if config.get('fixed_reference_events') else [])},
        sigma_mode='cavity' if config.get('cavity_winding_sigma') else 'learned' if config.get('learned_winding_sigma') else 'fixed')
    atomic_write(root/'config.json',config,durable=False);atomic_write(root/'summary.json',result,durable=False)
    return result


def refine_fixed(d, state, widths, config):
    """Postprocess fixed confident fibers, training only shared endpoint offsets."""
    import json
    import time
    from pathlib import Path
    from lasagna.inference_provenance import atomic_write
    retained, winding, mask, du, dv = fixed_geometry(d,state,widths,config)
    if not mask.any(): raise ValueError('No confident same-sheet perpendicular pairs')
    selected = mask.cpu().numpy()
    graph = dict(d)
    for key in ('i','j','ds_a','ds_b','sign','distance','parallel'):
        graph[key] = d[key][selected]
    ext = EndpointOffsets(graph,config,state.device)
    optimizer = torch.optim.Adam(ext.parameters(), lr=config['v_offset_lr'])
    root = Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    def errors():
        return du+ext.u_values[ext.b]-ext.u_values[ext.a], dv+ext.values[ext.b]-ext.values[ext.a]
    def stats(x):
        x=x.detach().abs()
        return dict(mean=float(x.mean()),median=float(x.median()),p95=float(torch.quantile(x,.95)),max=float(x.max()))
    def objective():
        u,v=errors()
        terms=ext.regularization()
        terms['u_match']=config['u_offset_match_weight']*(u/1000).square().mean()
        terms['v_match']=config['v_offset_match_weight']*(v/1000).square().mean()
        return sum(terms.values()),terms
    before=dict(u=stats(du),v=stats(dv))
    history=[]; start=time.monotonic()
    for iteration in range(config['steps']+1):
        if iteration%config.get('log_every',100)==0 or iteration==config['steps']:
            with torch.no_grad():
                loss,terms=objective();u,v=errors()
                record=dict(iteration=iteration,loss=float(loss),u=stats(u),v=stats(v),seconds=time.monotonic()-start,
                    terms={k:float(x) for k,x in terms.items()})
            history.append(record);print(json.dumps(record),flush=True)
            atomic_write(root/'history.json',history,durable=False)
        if iteration==config['steps']:break
        optimizer.zero_grad(set_to_none=True)
        loss,_=objective()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fixed offset loss')
        loss.backward();optimizer.step()
    # Preserve the regular solution arrays exactly, including original width bytes.
    with np.load(config['initial_solution']) as saved:
        arrays={k:saved[k] for k in saved.files}
    arrays.update(ext.arrays(),retained=retained.cpu().numpy(),fixed_winding=winding.cpu().numpy(),
        fixed_event_indices=np.flatnonzero(selected))
    np.savez(root/'solution.npz',**arrays)
    summary=dict(model='fixed_assignment_offset_refinement_v1',config=config,before=before,final=history[-1],
        retained_pieces=int(retained.sum()),total_pieces=len(state),same_sheet_pairs=int(mask.sum()),
        offset_diagnostics=ext.diagnostics(),frozen_state_and_widths=True,solve_wall_seconds=time.monotonic()-start)
    atomic_write(root/'summary.json',summary,durable=False)
    return summary


def topology(d, merge_distance=4., *, event_mask=None, piece_local=False):
    """Build nodes only for selected events; excluded endpoint maps are -1.

    Full event indexing is preserved so callers can retain their original
    constraint arrays without manufacturing geometry for excluded events.
    """
    if not np.isfinite(merge_distance) or merge_distance < 0:
        raise ValueError('v_offset_merge_distance must be finite and nonnegative')
    owners = np.r_[d['i'], d['j']]
    direction = np.sign(d['node_ds'][:, -1])
    if not np.allclose(d['node_ds'][:,0],0) or not np.allclose(
            np.abs(d['node_ds'][:,-1]),d['end']-d['start'],rtol=1e-5,atol=.01):
        raise ValueError('V offsets require start-anchored signed arc-length geometry')
    if not np.isin(direction, [-1, 1]).all():
        raise ValueError('V offsets require resolved signed arc directions')
    arc = d['start'][owners] + direction[owners] * np.r_[d['ds_a'], d['ds_b']]
    source = owners if piece_local else d['source_id'][owners]
    selected=np.arange(len(owners))
    if event_mask is not None:
        event_mask=np.asarray(event_mask,dtype=bool)
        if event_mask.shape!=(len(d['i']),):raise ValueError('Invalid topology event mask')
        selected=selected[np.tile(event_mask,2)]
    order = selected[np.lexsort((arc[selected], source[selected]))]
    mapping = np.full(len(owners), -1, dtype=np.int64)
    sources, positions, families, left, right = [], [], [], [], []
    cursor = 0
    while cursor < len(order):
        first = order[cursor]; stop = cursor + 1
        while stop < len(order) and source[order[stop]] == source[first] and arc[order[stop]]-arc[first] <= merge_distance:
            stop += 1
        group = order[cursor:stop]; family = d['family'][owners[group]]
        if not np.all(family == family[0]):
            raise ValueError('One original fiber has inconsistent H/V families')
        node = len(sources); mapping[group] = node
        if sources and sources[-1] == source[first]:
            if families[-1] != family[0]:
                raise ValueError('One original fiber has inconsistent H/V families')
            left.append(node-1); right.append(node)
        sources.append(source[first]); positions.append(arc[group].mean()); families.append(family[0])
        cursor = stop
    return dict(v_offset_source=np.asarray(sources, dtype=np.int64),
        v_offset_arc=np.asarray(positions), v_offset_family=np.asarray(families, dtype=np.int64),
        v_offset_left=np.asarray(left, dtype=np.int64), v_offset_right=np.asarray(right, dtype=np.int64),
        v_offset_a=mapping[:len(d['i'])], v_offset_b=mapping[len(d['i']):])


class EndpointOffsets(torch.nn.Module):
    def __init__(self, d, config, device):
        super().__init__()
        self.config = config
        for key in ('v_offset_match_weight','v_offset_smooth_h','v_offset_smooth_v','v_offset_anchor_weight','v_offset_lr'):
            value = config[key]
            if not np.isfinite(value) or value < 0 or (key == 'v_offset_lr' and value == 0):
                raise ValueError(key+' must be finite and nonnegative (LR positive)')
        self.saved = topology(d, config.get('v_offset_merge_distance', 4.))
        self.values = torch.nn.Parameter(torch.zeros(len(self.saved['v_offset_arc']), device=device))
        self.u_values = None
        if config.get('u_endpoint_offsets',False):
            for key in ('u_offset_match_weight','u_offset_smooth_h','u_offset_smooth_v','u_offset_anchor_weight'):
                if not np.isfinite(config[key]) or config[key] < 0:
                    raise ValueError(key+' must be finite and nonnegative')
            self.u_values = torch.nn.Parameter(torch.zeros_like(self.values))
        for key, val in self.saved.items():
            self.register_buffer(key, torch.as_tensor(val, device=device))
        from .fiber_soft_winding import constraint_family_compatible
        keep = constraint_family_compatible(d) if config.get('filter_constraint_families') else np.ones(len(d['i']), dtype=bool)
        self.register_buffer('a', self.v_offset_a[keep])
        self.register_buffer('b', self.v_offset_b[keep])

    def edge_offsets(self,edges,side):
        ids=getattr(self,side)[edges]
        return (self.u_values[ids] if self.u_values is not None else torch.zeros_like(self.values[ids]),self.values[ids])

    def prepare_fields(self,d,samples):
        """Precompute linear whole-source interpolation for non-event queries."""
        links=d['links']
        queries=dict(unary=(samples['owner'],samples['ds']),v_unary=(samples['v_owner'],samples['v_ds']),
            link_a=(links[:,0].astype(np.int64),links[:,2]),link_b=(links[:,1].astype(np.int64),links[:,3]))
        sources=self.saved['v_offset_source'];arc=self.saved['v_offset_arc']
        self.fields={}
        for name,(owners,ds) in queries.items():
            query_source=d['source_id'][owners]
            query_arc=d['start'][owners]+np.sign(d['node_ds'][owners,-1])*ds
            left=np.zeros(len(owners),dtype=np.int64);right=left.copy()
            alpha=np.zeros(len(owners));valid=np.zeros(len(owners),dtype=bool)
            for source in np.unique(query_source):
                q=np.flatnonzero(query_source==source)
                lo,hi=np.searchsorted(sources,source,side='left'),np.searchsorted(sources,source,side='right')
                if hi==lo:continue
                pos=np.searchsorted(arc[lo:hi],query_arc[q])+lo
                right[q]=np.clip(pos,lo,hi-1);left[q]=np.clip(pos-1,lo,hi-1)
                gap=arc[right[q]]-arc[left[q]]
                alpha[q]=np.divide(query_arc[q]-arc[left[q]],gap,out=np.zeros(len(q)),where=gap>0)
                alpha[q]=np.clip(alpha[q],0,1);valid[q]=True
            self.fields[name]=tuple(torch.as_tensor(x,device=self.values.device) for x in (left,right,alpha.astype(np.float32),valid))

    def field_offsets(self,name):
        left,right,alpha,valid=self.fields[name]
        def interpolate(values):return ((1-alpha)*values[left]+alpha*values[right])*valid
        return (interpolate(self.u_values) if self.u_values is not None else torch.zeros_like(alpha),interpolate(self.values))

    def axis_regularization(self, values, axis):
        left, right = self.v_offset_left, self.v_offset_right
        gap = (self.v_offset_arc[right]-self.v_offset_arc[left]).to(self.values.dtype)
        change = values[right]-values[left]
        terms = {}
        for family, name in [(0,'h'),(1,'v')]:
            mask = self.v_offset_family[left] == family
            terms[axis+'_offset_smooth_'+name] = self.config[axis+'_offset_smooth_'+name] * (
                (change[mask].square()/gap[mask]).sum()/gap[mask].sum().clamp_min(1))
        terms[axis+'_offset_anchor'] = self.config[axis+'_offset_anchor_weight']*(values/1000).square().mean()
        return terms

    def regularization(self):
        terms = self.axis_regularization(self.values,'v')
        if self.u_values is not None: terms.update(self.axis_regularization(self.u_values,'u'))
        return terms

    def terms(self, state, widths, data, edges):
        from . import fiber_implicit_winding as imp
        i, j = data['i'][edges], data['j'][edges]
        fa, fb = data['family'][i], data['family'][j]
        a, la = imp.hypotheses(state[i], self.config['winding_sigma'])
        b, lb = imp.hypotheses(state[j], self.config['winding_sigma'])
        qa, ua, va = imp.hypothesis_points(a, fa, data['ds_a'][edges], widths)
        qb, ub, vb = imp.hypothesis_points(b, fb, data['ds_b'][edges], widths)
        same = (b[:,None,:,0]-a[:,:,None,0]+torch.floor(ub[:,None,:]-ua[:,:,None]+.5) == 0)
        same = same & ((~data['parallel'][edges]) & (fa != fb))[:,None,None]
        probability = (la[:,:,None]+lb[:,None,:]).exp()*same
        delta = vb[:,None,:]-va[:,:,None] + (self.values[self.b[edges]]-self.values[self.a[edges]])[:,None,None]
        gate = data['class_weight'][edges]*state[i,3].sigmoid()*state[j,3].sigmoid()
        terms = self.regularization()
        terms['v_offset_match'] = self.config['v_offset_match_weight'] * (gate*(probability*(delta/1000).square()).sum((1,2))).mean()
        if self.u_values is not None:
            boundaries=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
            xa=imp.soft.uv.q_to_u(qa,widths,boundaries)
            xb=imp.soft.uv.q_to_u(qb,widths,boundaries)
            du=xb[:,None,:]-xa[:,:,None]+(self.u_values[self.b[edges]]-self.u_values[self.a[edges]])[:,None,None]
            terms['u_offset_match']=self.config['u_offset_match_weight']*(gate*(probability*(du/1000).square()).sum((1,2))).mean()
        return terms

    def arrays(self):
        return dict(self.saved, v_offsets=self.values.detach().cpu().numpy(),
            **({'u_offsets':self.u_values.detach().cpu().numpy()} if self.u_values is not None else {}))

    def axis_diagnostics(self, values, axis):
        absolute = values.detach().abs()
        left, right = self.v_offset_left, self.v_offset_right
        slope = (values[right]-values[left]).abs()/(self.v_offset_arc[right]-self.v_offset_arc[left])
        result = dict(v_offset_nodes=len(absolute),v_offset_mean_abs=float(absolute.mean()),
            v_offset_p95_abs=float(torch.quantile(absolute,.95)),v_offset_max_abs=float(absolute.max()))
        for family,name in [(0,'h'),(1,'v')]:
            x=slope[self.v_offset_family[left]==family]
            result['v_offset_'+name+'_p95_slope']=float(torch.quantile(x,.95)) if len(x) else None
        return {axis+key[1:]:value for key,value in result.items()}

    def diagnostics(self):
        result=self.axis_diagnostics(self.values,'v')
        if self.u_values is not None:result.update(self.axis_diagnostics(self.u_values,'u'))
        return result
