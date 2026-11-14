"""Fixed-length UV least squares: exact linear V, matrix-free GN/LM for U.

The graph CG implementation is shared with wrapped IRLS. No discrete labels,
loss weights or targets are changed. Width lookup intervals are re-evaluated
at every trial; acceptance tests the actual continuous objective, not its local
quadratic. Physical anchor X is a bijective reparameterization of k+shared-U.
"""
import time
import json
import torch

from . import fiber_soft_winding as soft
from .fiber_geometry_cache import width_boundaries
from .fiber_matrix_free_cg import solve as cg
from .fiber_wrapped_irls import quadratic_step


class StraightSystem:
    def __init__(self, objective):
        self.objective=objective;self.config=objective.config
        d=objective.data;self.n=len(objective.state)
        self.tensor=objective.tensor
        self.i=self.tensor(d['i'],torch.long);self.j=self.tensor(d['j'],torch.long)
        family=self.tensor(d['family'],torch.long)
        self.dua=self.tensor(d['ds_a'])*(family[self.i]==0)
        self.dub=self.tensor(d['ds_b'])*(family[self.j]==0)
        self.dva=self.tensor(d['ds_a'])*(family[self.i]==1)
        self.dvb=self.tensor(d['ds_b'])*(family[self.j]==1)
        self.target=self.tensor(d['fixed_target'])
        self.owner=self.tensor(d['sample_owner'],torch.long)
        self.du=self.tensor(d['sample_ds'])*(family[self.owner]==0)
        self.dv=self.tensor(d['sample_ds'])*(family[self.owner]==1)
        self.phase=self.tensor(d['sample_phase'])+objective.state[self.owner,0].detach()
        self.z=self.tensor(d['sample_z'])
        # Convert before normalization: the baseline multiplies float64 residuals
        # by float32 class weights, then reduces in float64.
        coefficients=objective.quadratic_coefficients()
        for name,value in coefficients.items():setattr(self,'c_'+name,self.tensor(value))

    def scatter(self, values, owner, count=None):
        if count is not None and count==self.objective.raw.numel():
            # Width reductions have millions of contributions but few bins.
            # Independent contiguous groups average ~32 contributions/bin;
            # the partial table is only N/32 scalars, with no N-sized group IDs,
            # sorting, padding, or source copies. Internal inputs are contiguous.
            chunk=32*count;groups=values.numel()//chunk
            if groups:
                end=groups*chunk
                partial=values.new_zeros((groups,count))
                partial.scatter_add_(1,owner[:end].view(groups,chunk),values[:end].view(groups,chunk))
                out=partial.sum(0)
                return out.index_add_(0,owner[end:],values[end:])
        out=values.new_zeros(self.n if count is None else count)
        return out.index_add_(0,owner,values)

    def pair_transpose(self, values):
        out=values.new_zeros(self.n)
        out.index_add_(0,self.i,-values)
        return out.index_add_(0,self.j,values)

    def vertical(self, initial, iterations, tolerance):
        weight=self.c_vertical+self.c_meeting
        unary=self.scatter(self.c_z,self.owner)
        rhs=self.scatter(self.c_z*(self.z-self.dv),self.owner)
        target=torch.where(unary>0,rhs/unary.clamp_min(torch.finfo(initial.dtype).tiny),initial)
        result,used=quadratic_step(initial,self.i,self.j,self.dva-self.dvb,weight,target,unary,
                                  max_cg=iterations,cg_tolerance=tolerance)
        # Report the actual normal-equation residual, not merely a CG cap hit.
        pair_weight=torch.where(self.i==self.j,0.,weight)
        diagonal=unary+self.scatter(pair_weight,self.i)+self.scatter(pair_weight,self.j)
        inv=torch.where(diagonal>0,diagonal.reciprocal(),0.)
        right=rhs+self.pair_transpose(pair_weight*(self.dva-self.dvb))
        def norm(v):
            r=right-unary*v-self.pair_transpose(pair_weight*(v[self.j]-v[self.i]))
            return (r.square()*inv).sum()
        first=norm(initial);last=norm(result)
        relative=torch.sqrt(last/first.clamp_min(torch.finfo(initial.dtype).tiny))
        self.vertical_diagnostics=dict(converged=bool(last<=tolerance**2*first.clamp_min(torch.finfo(initial.dtype).tiny)),
                                       relative_preconditioned_residual=float(relative))
        error=result[self.j]+self.dvb-result[self.i]-self.dva
        self.vertical_energy=float((weight*error.square()).sum()+
                                   (self.c_z*(result[self.owner]+self.dv-self.z).square()).sum())
        return result,used

    def plan(self, x, widths, bounds, owner, offset):
        position=x[owner]+offset
        bucket=(torch.searchsorted(bounds,position.contiguous(),right=True)-1).clamp(0,len(widths)-1)
        inv=widths[bucket].reciprocal()
        fraction=(position-bounds[bucket])*inv
        return owner,bucket,inv,fraction

    @staticmethod
    def q(plan):
        return plan[1]+plan[3]

    @staticmethod
    def tangent(plan, dx, dw):
        owner,bucket,inv,fraction=plan
        prefix=torch.cat((dw.new_zeros(1),dw.cumsum(0)))
        return (dx[owner]-prefix[bucket]-fraction*dw[bucket])*inv

    def transpose(self, plan, values, k):
        owner,bucket,inv,fraction=plan
        scaled=values*inv
        anchors=self.scatter(scaled,owner)
        by_bin=self.scatter(-scaled,bucket,k)
        # Prefix lookup depends on every width STRICTLY before this interval.
        widths=by_bin.flip(0).cumsum(0).flip(0)-by_bin
        widths+=self.scatter(-scaled*fraction,bucket,k)
        return anchors,widths

    def regularizer(self, raw):
        widths=soft.decode_widths(raw,self.config)
        residuals=[]
        if self.config.get('soft_minimum_width',False):
            residuals.append((self.config.get('width_minimum_weight',10.)**.5*
                torch.relu(1.-raw[:1]/self.config.get('minimum_winding_width',1000.))))
        if len(widths)>1:
            residuals.append(torch.diff(widths)/self.objective.initial_width.mean()*
                (self.config.get('width_smooth',.01)/(len(widths)-1))**.5)
        return torch.cat(residuals) if residuals else raw[:0]

    def linearize(self, x, raw, *, width_block=False):
        widths=soft.decode_widths(raw,self.config);bounds=width_boundaries(widths)
        k=len(widths)
        a=self.plan(x,widths,bounds,self.i,self.dua)
        b=self.plan(x,widths,bounds,self.j,self.dub)
        p=self.plan(x,widths,bounds,self.owner,self.du)
        width_jac=torch.func.jacfwd(lambda r:soft.decode_widths(r,self.config))(raw)
        reg=self.regularizer(raw)
        reg_jac=torch.func.jacfwd(self.regularizer)(raw)

        def transpose(edge, prior):
            ax,aw=self.transpose(a,-edge,k);bx,bw=self.transpose(b,edge,k)
            px,pw=self.transpose(p,prior,k)
            return ax+bx+px,width_jac.T@(aw+bw+pw)

        edge_error=self.q(b)-self.q(a)-self.target
        phase_error=soft.circular(self.q(p)-self.phase)
        gx,gr=transpose(self.c_winding*edge_error,self.c_phase*phase_error)
        gx+=self.pair_transpose(self.c_meeting*(x[self.j]+self.dub-x[self.i]-self.dua))
        gr+=reg_jac.T@reg
        gradient=torch.cat((gx,gr))
        diag=self.scatter(self.c_winding*a[2].square(),self.i)+self.scatter(self.c_winding*b[2].square(),self.j)
        diag+=self.scatter(self.c_meeting,self.i)+self.scatter(self.c_meeting,self.j)
        diag+=self.scatter(self.c_phase*p[2].square(),self.owner)
        def operator(step):
            dx,dr=step[:self.n],step[self.n:];dw=width_jac@dr
            edge=self.tangent(b,dx,dw)-self.tangent(a,dx,dw)
            prior=self.tangent(p,dx,dw)
            ox,orr=transpose(self.c_winding*edge,self.c_phase*prior)
            ox+=self.pair_transpose(self.c_meeting*(dx[self.j]-dx[self.i]))
            orr+=reg_jac.T@(reg_jac@dr)
            return torch.cat((ox,orr))
        # Aggregate three scalar moments per width interval. Never create an
        # edges-by-widths Jacobian or apply the full graph once per width.
        # Endpoint cross terms are omitted ONLY from this preconditioner.
        prefix=torch.cat((width_jac.new_zeros(1,k),width_jac.cumsum(0)),dim=0)[:-1]
        width_diagonal=(reg_jac.square()).sum(0)
        block=reg_jac.T@reg_jac if width_block else None
        for plan,weight in ((a,self.c_winding),(b,self.c_winding),(p,self.c_phase)):
            _,bucket,inv,fraction=plan
            w=weight*inv.square()
            s0=self.scatter(w,bucket,k);s1=self.scatter(w*fraction,bucket,k)
            s2=self.scatter(w*fraction.square(),bucket,k)
            width_diagonal+=((s0[:,None]*prefix.square())+
                             (2*s1[:,None]*prefix*width_jac)+
                             (s2[:,None]*width_jac.square())).sum(0)
            if width_block:
                block+=prefix.T@(s0[:,None]*prefix)+prefix.T@(s1[:,None]*width_jac)+\
                    width_jac.T@(s1[:,None]*prefix)+width_jac.T@(s2[:,None]*width_jac)
        if width_block:
            # Four interval-pair moments retain the endpoint cross terms. This
            # is K² storage, not an M×K Jacobian or K graph evaluations.
            bucket=a[1]*k+b[1];weight=self.c_winding*a[2]*b[2]
            moments=[self.scatter(weight*f,bucket,k*k).reshape(k,k)
                     for f in (1.,a[3],b[3],a[3]*b[3])]
            cross=prefix.T@moments[0]@prefix+width_jac.T@moments[1]@prefix+\
                prefix.T@moments[2]@width_jac+width_jac.T@moments[3]@width_jac
            block-=cross+cross.T
            block=(block+block.T)*.5
            width_diagonal=block.diagonal().clamp_min(0)
        diag=torch.cat((diag,width_diagonal.clamp_min(0)))
        # A genuinely unconstrained direction has zero step, not a new prior.
        return (operator,gradient,diag,block) if width_block else (operator,gradient,diag)

    def horizontal_loss(self,x,raw):
        widths=soft.decode_widths(raw,self.config);bounds=width_boundaries(widths)
        a=self.plan(x,widths,bounds,self.i,self.dua);b=self.plan(x,widths,bounds,self.j,self.dub)
        p=self.plan(x,widths,bounds,self.owner,self.du)
        return (self.c_winding*(self.q(b)-self.q(a)-self.target).square()).sum()+\
            (self.c_meeting*(x[self.j]+self.dub-x[self.i]-self.dua).square()).sum()+\
            (self.c_phase*soft.circular(self.q(p)-self.phase).square()).sum()+self.regularizer(raw).square().sum()

    def feasible(self, raw):
        mode=self.config.get('width_parameterization','softplus')
        if mode in ('monotone_additive','monotone_relative'):
            raw=raw.clone();raw[1:]=raw[1:].clamp_min(0)
            if not self.config.get('soft_minimum_width',False):
                raw[0]=raw[0].clamp_min(self.config.get('minimum_winding_width',1000.))
        elif mode=='direct':
            raise ValueError('UV GN requires a positive width parameterization; direct widths are unbounded')
        return raw


@torch.no_grad()
def solve(objective, *, iterations=100, callback=None):
    if not objective.curve.straight:
        raise ValueError('Fixed-winding UV GN requires the straight fixed-length model')
    if iterations<1:raise ValueError('UV GN iterations must be positive')
    config=objective.config;started=time.monotonic();system=StraightSystem(objective)
    curve,raw=objective.curve,objective.raw
    cg_iterations=int(config.get('uv_cg_iterations',100))
    tolerance=float(config.get('uv_cg_tolerance',1e-6))
    v,vcg=system.vertical(curve.v.detach(),int(config.get('uv_v_cg_iterations',1000)),
                         float(config.get('uv_v_cg_tolerance',tolerance)))
    curve.v.copy_(v)
    # The blocks are independent. Do not retain O(events) vertical-only device
    # columns throughout horizontal GN; allocator storage can be reused there.
    del system.c_vertical,system.dva,system.dvb,system.dv,system.z,system.c_z
    widths=soft.decode_widths(raw,config)
    x=soft.uv.q_to_u(objective.state[:,0]+curve.u,widths,width_boundaries(widths))
    current=system.horizontal_loss(x,raw);damping=float(config.get('uv_lm_damping',1e-3))
    if not 0<damping<float('inf'):raise ValueError('UV GN damping must be finite and positive')
    stalled=0;total_cg=vcg;step=0;history=[]
    for step in range(1,iterations+1):
        operator,gradient,diagonal,block=system.linearize(x,raw,width_block=True)
        accepted=False;old=float(current);used=0
        for trial in range(8):
            damp=damping*diagonal
            # Widths are few but cumulatively coupled. Invert that complete
            # block while retaining matrix-free graph coupling in CG itself.
            precondition_block=block+torch.diag(damp[system.n:])
            scale=precondition_block.diagonal().abs().max()
            guard=torch.where(scale>0,scale*torch.finfo(raw.dtype).eps*16,torch.ones_like(scale))
            precondition_block=precondition_block+torch.eye(len(raw),device=raw.device,dtype=raw.dtype)*guard
            factor,info=torch.linalg.cholesky_ex(precondition_block)
            if bool(info):
                damping=min(damping*10,1e12)
                continue
            inv=torch.where(diagonal[:system.n]+damp[:system.n]>0,
                            (diagonal[:system.n]+damp[:system.n]).reciprocal(),0.)
            def precondition(value):
                return torch.cat((value[:system.n]*inv,
                    torch.cholesky_solve(value[system.n:,None],factor)[:,0]))
            delta,used=cg(lambda v:operator(v)+damp*v,-gradient,torch.zeros_like(gradient),
                          diagonal+damp,iterations=cg_iterations,tolerance=tolerance,
                          preconditioner=precondition)
            total_cg+=used
            proposed=system.feasible(raw+delta[system.n:])
            candidate=x+delta[:system.n]
            value=system.horizontal_loss(candidate,proposed)
            if bool(torch.isfinite(value)) and bool(value<=current):
                x=candidate;raw.copy_(proposed);current=value;accepted=True
                damping=max(damping*.3,1e-12);break
            damping=min(damping*10,1e12)
        if not used:
            raise RuntimeError('UV GN width preconditioner remains indefinite after damping retries')
        # Drop the interval plans before constructing the next linearization;
        # otherwise tuple assignment temporarily retains two whole graph plans.
        del operator
        widths=soft.decode_widths(raw,config)
        curve.u.copy_(soft.uv.u_to_q(x,widths,width_boundaries(widths))-objective.state[:,0])
        history.append(dict(iteration=step,loss=float(current)+system.vertical_energy,
                            horizontal_loss=float(current),cg_iterations=used,
                            accepted=accepted,lm_damping=damping,seconds=time.monotonic()-started))
        every=int(config.get('optimizer_progress_every',0))
        if every>0 and step%every==0:
            print(json.dumps(dict(stage='uv_gauss_newton',**history[-1])),flush=True)
        if callback is not None:callback(step)
        stalled=stalled+1 if old-float(current)<=config.get('uv_tolerance',1e-9)*max(1.,abs(old)) else 0
        if stalled>=5:break
    return dict(optimizer='gauss_newton_cg',iterations=step,
                converged=stalled>=5 and system.vertical_diagnostics['converged'],
                stop_reason='objective_plateau' if stalled>=5 else 'iteration_limit',
                horizontal_plateau=stalled>=5,vertical_convergence=system.vertical_diagnostics,
                total_cg_iterations=total_cg,vertical_cg_iterations=vcg,
                seconds=time.monotonic()-started,history=history)
