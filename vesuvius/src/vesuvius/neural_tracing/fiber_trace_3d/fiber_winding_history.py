"""Read-only, per-update bucket probability history for diagnostic replays."""
import numpy as np
import torch


def attachment_support(history, d, carry, config, window=100):
    """Mean effective meeting activation over time, then incident perp edges.

    Only edges with CURRENT joint same-sheet mass >0.5 enter the denominator.
    Their entire history window counts, including early inactive updates.
    No error magnitude or class/loss weight enters this confidence measure.
    """
    from .fiber_geometry_confidence import confidence_gate
    if window < 1:
        raise ValueError('history window must be positive')
    its=np.asarray(history['iteration'])
    selected=np.flatnonzero(its>its[-1]-window)
    if len(selected)!=window or not np.array_equal(its[selected],np.arange(its[-1]-window+1,its[-1]+1)):
        raise ValueError('Output support requires a complete consecutive history window')
    i,j=d['i'],d['j'];n=len(d['family'])
    edges=np.flatnonzero(~d['parallel'].astype(bool) & (d['family'][i]!=d['family'][j]))
    a,b=i[edges],j[edges]
    labels=history['labels'];joint=history['joint']
    mass=np.zeros(len(edges),dtype=np.float64)
    for k in range(joint.shape[2]):
        matching=labels[-1,b,:]==(labels[-1,a,k]-carry[edges])[:,None]
        mass+=joint[-1,a,k]*(joint[-1,b]*matching).sum(1)
    edges=edges[mass>.5];a,b=i[edges],j[edges]
    total=np.zeros(len(edges),dtype=np.float64)
    for t in selected:
        p=torch.as_tensor(joint[t])
        gated=p*confidence_gate(p,config) if config.get('geometry_confidence_gated',False) else p
        g=gated.numpy()
        # Explicit equality also handles zero-mass padded history columns.
        for k in range(p.shape[1]):
            matching=labels[t,b,:]==(labels[t,a,k]-carry[edges])[:,None]
            total+=g[a,k]*(g[b]*matching).sum(1)
    edge_mean=total/window
    count=np.bincount(np.r_[a,b],minlength=n)
    sums=np.bincount(np.r_[a,b],weights=np.r_[edge_mean,edge_mean],minlength=n)
    score=np.divide(sums,count,out=np.zeros(n),where=count>0)
    return score,edges,edge_mean,count


class WindingHistory:
    def __init__(self, transfer_interval=100):
        if transfer_interval < 1:
            raise ValueError('History transfer interval must be positive')
        self.rows=[]
        self.pending=[]
        self.transfer_interval=transfer_interval

    @torch.no_grad()
    def capture(self, iteration, state, extension, widths):
        h, lp=extension.local_hypotheses(state)
        # Snapshot on the producing stream; never read CUDA tensors on the CPU
        # per update. Clone widths because callers may update their storage.
        self.pending.append((iteration,h[:,:,0].to(torch.int32).clone(),
                             lp.exp()*state[:,3].sigmoid()[:,None],widths.clone()))
        if state.device.type != 'cuda' or len(self.pending)>=self.transfer_interval:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        # Windows may change size between updates. Flatten before batching and
        # restore their original shapes; padding happens only during export.
        arrays=[torch.cat([r[k].reshape(-1) for r in self.pending]).cpu().numpy()
                for k in (1,2,3)]
        offsets=[0,0,0]
        for row in self.pending:
            restored=[]
            for k,values in enumerate(arrays):
                tensor=row[k+1];end=offsets[k]+tensor.numel()
                restored.append(values[offsets[k]:end].reshape(tuple(tensor.shape)))
                offsets[k]=end
            self.rows.append((row[0],*restored))
        self.pending.clear()

    def save(self,path,source_id):
        self.flush()
        count=max(r[1].shape[1] for r in self.rows)
        labels=np.zeros((len(self.rows),len(source_id),count),dtype=np.int32)
        joint=np.zeros(labels.shape,dtype=np.float32)
        for t,r in enumerate(self.rows):
            labels[t,:,:r[1].shape[1]]=r[1]
            joint[t,:,:r[2].shape[1]]=r[2]
        np.savez_compressed(path,iteration=np.array([r[0] for r in self.rows]),
            labels=labels,joint=joint,
            widths=np.stack([r[3] for r in self.rows]),source_id=source_id)


def summarize(history, final_labels):
    labels=history['labels'];joint=history['joint'];its=history['iteration']
    probability=np.where(labels==np.asarray(final_labels)[None,:,None],joint,0).sum(2)
    map_labels=np.take_along_axis(labels,joint.argmax(2)[...,None],2)[...,0]
    visible=(map_labels==final_labels[None,:]) & (probability>=.5)
    result=[]
    for i in range(len(final_labels)):
        absent=np.flatnonzero(~visible[:,i])
        entry=int(absent[-1]+1) if len(absent) else 0
        entry_it=int(its[entry]) if entry<len(its) else None
        result.append(dict(entry=entry_it,age=int(its[-1])-entry_it if entry_it is not None else None,
            mean100=float(probability[-100:,i].mean()),min100=float(probability[-100:,i].min()),
            visible100=int(visible[-100:,i].sum()),
            switches100=int(np.count_nonzero(np.diff(visible[-100:,i]))),
            map_switches100=int(np.count_nonzero(np.diff(map_labels[-100:,i])))))
    return result
