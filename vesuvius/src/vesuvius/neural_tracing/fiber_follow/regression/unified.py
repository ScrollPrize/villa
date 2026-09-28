"""One observation encoder and path evaluator across space and time.

Memory stores encoder features, never an independently encoded appearance.
The seed is immutable; recent descriptors remain explicit, while adaptive slots
compress temporal context. Identity attention uses the projection supervised by
InfoNCE. Its null key permits a mismatch rather than forcing a reference match.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import AxialEncoder, PathDecoderLayer, UNIFIED_ARCHITECTURE, sample_features, crop_support


def to_device(value, device):
    # Pageable host-to-device copies synchronize the stream; pinned ones queue.
    if torch.device(device).type == 'cuda':
        return value.pin_memory().to(device, non_blocking=True)
    return value.to(device)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.save_for_backward(scale)
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad*ctx.saved_tensors[0].to(grad.dtype), None


def scale_gradient(value, scale):
    """Identity in the forward pass; multiplies the incoming gradient by ``scale``."""
    return _ScaleGradient.apply(value, scale)


def stratified_history(own, budget):
    """Sample chronological valid pairs; weights are inverse inclusion probabilities."""
    n = len(own)
    groups = min(n, budget)
    selected = {}
    for group in range(groups):
        start, stop = group*n//groups, (group+1)*n//groups
        # Use the checkpointed global CPU RNG. Singleton groups need no draw.
        offset = int(torch.randint(stop-start, ())) if stop-start > 1 else 0
        selected[own[start+offset]] = stop-start
    return selected


class UnifiedFollower(nn.Module):
    architecture = UNIFIED_ARCHITECTURE

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, h = cfg.channels, cfg.hidden
        self.encoder = AxialEncoder(cfg)
        self.embedding = nn.Linear(c, cfg.embedding)
        self.observation_projection = nn.Linear(c, h)
        self.metadata = nn.Sequential(nn.Linear(13, h), nn.SiLU(), nn.Linear(h, h))
        self.role = nn.Parameter(torch.randn(4, h)*.02)  # seed, recent, slots, visible history
        self.initial_slots = nn.Parameter(torch.randn(cfg.memory_slots, h)*.02)
        self.write_norm = nn.LayerNorm(h)
        self.write_attention = nn.MultiheadAttention(h, cfg.heads, dropout=0., batch_first=True)
        self.write_gate = nn.Linear(2*h, h)
        self.write_proposal = nn.Linear(2*h, h)
        nn.init.constant_(self.write_gate.bias, -2.)
        self.query = nn.Sequential(nn.Linear(27*(c+1)+3, h), nn.SiLU(), nn.Linear(h, h))
        self.identity_value = nn.Linear(c, h, bias=False)
        self.identity_null = nn.Parameter(torch.zeros(h))
        # Shared by geometry, confidence, supplied candidates, and per-write probes.
        layer = PathDecoderLayer(h, cfg.heads, 2*h, dropout=0., activation='gelu',
                                 batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers, norm=nn.LayerNorm(h))
        self.coordinates = nn.Linear(h, 2)
        self.correction = nn.Linear(h, 2)
        self.confidence_head = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, 1))
        self.probe_head = nn.Linear(h, 4)
        for head in (self.coordinates, self.correction):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)
        self.register_buffer('stencil', torch.tensor([[a*cfg.patch_radius,b*cfg.patch_radius,z]
            for z in (-1.,0.,1.) for a in (-1.,0.,1.) for b in (-1.,0.,1.)]), persistent=False)
        self.register_buffer('planes', torch.arange(1, cfg.n_future+1).float()*cfg.future_step, persistent=False)

    def initial_memory(self, batch, device):
        s, c = self.cfg.memory_slots, self.cfg.channels
        zeros = lambda *shape: torch.zeros(batch, *shape, device=device)
        return dict(slots=self.initial_slots.to(device)[None].expand(batch,-1,-1).clone(),
            anchor=zeros(27,c), anchor_valid=zeros().bool(), anchor_position=zeros(3),
            anchor_frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(),
            recent=zeros(s,c), recent_position=zeros(s,3), recent_frame=torch.eye(3,device=device)[None,None].expand(batch,s,-1,-1).clone(),
            recent_valid=zeros(s).bool(), age=zeros(s), position=zeros(3),
            frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(), seen=zeros().bool())

    def encode(self, image, *, checkpoint_blocks=True):
        # Appearance is independent of the claimed identity for every observation.
        return self.encoder(image, None, None, checkpoint_blocks=checkpoint_blocks)

    def observation(self, image):
        # The caller checkpoints this entire operation. Nested block checkpoints
        # would recompute the axial stack a second time during backward.
        dense, _, tokens = self.encode(image, checkpoint_blocks=False)
        return self.observation_features(dense, tokens)

    def observation_features(self, dense, tokens):
        return sample_features(dense, self.stencil[None].expand(len(dense),-1,-1), self.cfg.fine)[0]

    def seed_descriptor(self, state):
        return state['anchor'][:,13:14]

    def assemble_context(self, tokens, valid, image_tokens, visible, mask):
        return (torch.cat((image_tokens,tokens,visible),1),
                torch.cat((torch.ones(image_tokens.shape[:2],device=mask.device,dtype=torch.bool),valid,mask),1))

    def visible_features(self, local, references):
        meta = torch.cat((references/16,references.new_zeros(*references.shape[:2],10)),-1)
        return self.observation_projection(local)+self.metadata(meta)+self.role[3]

    def encode_observation(self, image, device, gradients):
        image = image.to(device, non_blocking=True)
        if gradients and self.training and torch.is_grad_enabled():
            # Only a small descriptor survives each historical full-crop encoding.
            return checkpoint(self.observation, image, use_reentrant=False)
        return self.observation(image)

    def pose(self, positions, frames, current_pos, current_frame, age):
        delta = torch.einsum('bni,bij->bnj', positions-current_pos[:,None], current_frame)/16
        delta = delta.sign()*torch.log1p(delta.abs())
        rotation = torch.einsum('bji,bnjk->bnik', current_frame, frames).flatten(-2)
        return self.metadata(torch.cat((delta, rotation, torch.log1p(age.clamp_min(0))[...,None]/8),-1))

    def retained(self, state, position, frame):
        b = len(position)
        seed_pose = self.pose(state['anchor_position'][:,None], state['anchor_frame'][:,None],
                              position, frame, torch.zeros(b,1,device=position.device))
        # Seed stencil offsets rotate with its observation frame.
        offsets = torch.einsum('ki,bji->bkj', self.stencil, state['anchor_frame'])
        offsets = torch.einsum('bki,bij->bkj', offsets, frame)/16
        offset_meta = torch.cat((offsets, offsets.new_zeros(b,27,10)),-1)
        anchor = self.observation_projection(state['anchor'])+seed_pose+self.metadata(offset_meta)+self.role[0]
        recent = self.observation_projection(state['recent'])+self.pose(
            state['recent_position'],state['recent_frame'],position,frame,state['age'])+self.role[1]
        slots = state['slots']+self.role[2]+self.pose(state['position'][:,None],
            state['frame'][:,None],position,frame,torch.zeros(b,1,device=position.device))
        tokens = torch.cat((anchor,recent,slots),1)
        valid = torch.cat((state['anchor_valid'][:,None].expand(-1,27),state['recent_valid'],
                           state['seen'][:,None].expand(-1,self.cfg.memory_slots)),1)
        # Explicit descriptors, not compressed slots, participate in InfoNCE space.
        refs = torch.cat((state['anchor'][:,13:14],state['recent']),1)
        ref_valid = torch.cat((state['anchor_valid'][:,None],state['recent_valid']),1)
        ref_roles = torch.cat((self.role[0:1],self.role[1:2].expand(self.cfg.memory_slots,-1)),0)
        return tokens, valid, refs, ref_valid, ref_roles

    def identity_read(self, values, refs, valid, roles):
        query = F.normalize(self.embedding(values).float(),dim=-1)
        keys = F.normalize(self.embedding(refs).float(),dim=-1)
        logits = torch.bmm(query,keys.transpose(1,2))/self.cfg.identity_temperature
        logits = logits.masked_fill(~valid[:,None],-torch.inf)
        logits = torch.cat((logits,logits.new_zeros(*logits.shape[:2],1)),-1)
        v = self.identity_value(refs)+roles[None]
        v = torch.cat((v,self.identity_null[None,None].expand(len(v),1,-1)),1)
        return torch.bmm(logits.softmax(-1).to(v.dtype),v)

    def evaluate(self, points, patches, support, tokens, valid, refs, ref_valid, roles):
        spatial = torch.cat((patches,support[...,None]),-1).flatten(-2)
        query = self.query(torch.cat((spatial,points/16),-1))
        query = query+self.identity_read(patches[:,:,13],refs,ref_valid,roles)
        return self.decoder(query,tokens,memory_key_padding_mask=~valid)

    def write(self, obs, active, position, frame, state, *, probe=True):
        # Sequence slices have different strides from the current observation.
        # Normalize outside Dynamo so they share compiled graphs. Initial state
        # tensors are already materialized with recurrent-output layouts.
        obs, active, position, frame = (v.contiguous() for v in (obs, active, position, frame))
        writer = self._write_probe if probe else self._write_grad if torch.is_grad_enabled() else self._write_burn
        return writer(obs, active, position, frame, state)

    # Separate code objects keep probe scheduling and grad mode out of each
    # compile cache. A non-probe graph can also discard unused probe features.
    # Seed availability and the first differentiable write still legitimately
    # vary requires_grad, even with fixed shapes and strides.
    def _write_grad(self, obs, active, position, frame, state):
        return self._write(obs, active, position, frame, state, probe=False)

    def _write_burn(self, obs, active, position, frame, state):
        return self._write(obs, active, position, frame, state, probe=False)

    def _write_probe(self, obs, active, position, frame, state):
        return self._write(obs, active, position, frame, state, probe=True)

    def _write(self, obs, active, position, frame, state, *, probe):
        position = torch.where(active[:,None],position,state['position'])
        frame = torch.where(active[:,None,None],frame,state['frame'])
        retained, valid, refs, ref_valid, roles = self.retained(state,position,frame)
        observed = self.observation_projection(obs)
        tokens = torch.cat((observed,retained),1)
        valid = torch.cat((torch.ones(len(obs),27,device=obs.device,dtype=torch.bool),valid),1)
        evidence = self.write_attention(self.write_norm(state['slots']),tokens,tokens,
                                        key_padding_mask=~valid,need_weights=False)[0]
        pair = torch.cat((state['slots'],evidence),-1)
        slots = state['slots']+self.write_gate(pair).sigmoid()*(self.write_proposal(pair).tanh()-state['slots'])
        state = dict(state)
        state['slots'] = torch.where(active[:,None,None],slots,state['slots'])
        for key, new in (('recent',obs[:,13]),('recent_position',position),('recent_frame',frame)):
            shifted = torch.cat((new[:,None],state[key][:,:-1]),1)
            mask = active.reshape(len(active),*([1]*(shifted.ndim-1)))
            state[key] = torch.where(mask,shifted,state[key])
        state['recent_valid'] = torch.where(active[:,None],torch.cat((active[:,None],state['recent_valid'][:,:-1]),1),state['recent_valid'])
        state['age'] = torch.where(active[:,None],torch.cat((state['age'].new_zeros(len(obs),1),state['age'][:,:-1]+1),1),state['age'])
        state['position'] = torch.where(active[:,None],position,state['position'])
        state['frame'] = torch.where(active[:,None,None],frame,state['frame'])
        state['seen'] = state['seen'] | active
        # Read pre-write references, before this observation can match itself.
        result = obs.new_zeros(len(obs),4)
        if probe:
            head = obs.new_zeros(len(obs),1,3)
            decoded = self.evaluate(head,obs[:,None],torch.ones(len(obs),1,27,device=obs.device,dtype=torch.bool),
                                    tokens,valid,refs,ref_valid,roles)
            result = self.probe_head(decoded[:,0]).float()
        return state,result

    def schedule(self, x, state=None, probe_mask=None):
        """CPU scheduling metadata, copied once per forward before any encoding."""
        seed = x['memory_seed_valid'].detach().cpu().bool()
        if state is not None:
            seed = seed & ~state['anchor_valid'].detach().cpu()
        return dict(active=x['memory_mask'].detach().cpu().bool(), seed=seed,
                    probe=None if probe_mask is None else probe_mask.detach().cpu().bool())

    # Crops per batched encoding without/with gradients. Checkpoint recompute
    # holds a whole-crop activation stack (about 2.5 GiB per crop in production).
    observation_batch = (8, 1)

    def encode_observations(self, crops, device, gradients):
        """Encode independent full crops (CPU views) in bounded batches.

        An observation depends only on its own crop, so batching changes values
        only by kernel rounding. Pinned loader views copy asynchronously.
        """
        parts = []
        size = self.observation_batch[bool(gradients)]
        for i in range(0, len(crops), size):
            images = torch.stack([c.to(device, non_blocking=True) for c in crops[i:i+size]])
            parts.append(self.encode_observation(images, device, gradients))
        return torch.cat(parts)

    def observe(self, x, current, state=None, probe_mask=None, schedule=None):
        b, t = x['memory_mask'].shape
        device = current.device
        schedule = self.schedule(x, state, probe_mask) if schedule is None else schedule
        state = self.initial_memory(b,device) if state is None else state
        active_cpu = schedule['active']
        burn = max(0,t-self.cfg.memory_grad_steps-1)
        # Inactive (state, step) pairs never reach memory, so they are not encoded.
        seeds = schedule['seed'].nonzero()[:,0].tolist()
        history = [(i,j) for j in range(t-1) for i in range(b) if active_cpu[i,j]]
        old = [(i,j) for i,j in history if j < burn]
        new = [(i,j) for i,j in history if j >= burn]
        k = self.cfg.memory_encoder_grad_steps
        detached, scale = [], None
        if k and self.training and torch.is_grad_enabled():
            # Sampled from the seeded global generator, so runs stay reproducible.
            keep = {}
            for i in range(b):
                own = [pair for pair in new if pair[0] == i]
                keep.update(stratified_history(own, k))
            detached = [pair for pair in new if pair not in keep]
            new = [pair for pair in new if pair in keep]
            scale = torch.tensor([keep[pair] for pair in new], dtype=torch.float32)
        index = lambda pairs: tuple(to_device(torch.tensor(v),device) for v in zip(*pairs))
        observed = current.new_zeros(b,max(t-1,0),*current.shape[1:])
        if old:
            with torch.no_grad():
                encoded = self.encode_observations([x['history_crops'][i,j] for i,j in old],device,False)
            observed = observed.index_put(index(old),encoded)
        if detached:
            # Gradient-window writes still backpropagate; this encoder input does not.
            with torch.no_grad():
                encoded = self.encode_observations([x['history_crops'][i,j] for i,j in detached],device,False)
            observed = observed.index_put(index(detached),encoded)
        if seeds or new:
            encoded = self.encode_observations([x['seed_crop'][i] for i in seeds]+
                                               [x['history_crops'][i,j] for i,j in new],device,True)
            if new:
                encoded_new = encoded[len(seeds):]
                if scale is not None:
                    encoded_new = scale_gradient(encoded_new,to_device(scale,device)[:,None,None])
                observed = observed.index_put(index(new),encoded_new)
            if seeds:
                valid_seed = torch.zeros(b,dtype=torch.bool)
                valid_seed[seeds] = True
                valid_seed = to_device(valid_seed,device)
                seed = current.new_zeros(b,*current.shape[1:]).index_put(index([(i,) for i in seeds]),encoded[:len(seeds)])
                state = dict(state)
                state['anchor'] = torch.where(valid_seed[:,None,None],seed,state['anchor'])
                state['anchor_position'] = torch.where(valid_seed[:,None],x['memory_seed_position'],state['anchor_position'])
                state['anchor_frame'] = torch.where(valid_seed[:,None,None],x['memory_seed_frame'],state['anchor_frame'])
                state['anchor_valid'] = state['anchor_valid'] | valid_seed
        active_steps = active_cpu.any(0).tolist()
        probe_steps = [self.training]*t if schedule['probe'] is None else schedule['probe'].any(0).tolist()
        # Pack time-major once, instead of launching tiny contiguous copies at
        # every write. Each step now has the same layout as the current head.
        observed = observed.transpose(0,1).contiguous()
        active_rows, position_rows, frame_rows = (
            x[key].transpose(0,1).contiguous()
            for key in ('memory_mask','memory_positions','memory_frames'))
        probes = []
        for j in range(t):
            active = active_rows[j].bool()
            if not active_steps[j]:
                probes.append(current.new_zeros(b,4))
                continue
            with torch.set_grad_enabled(torch.is_grad_enabled() and j >= burn):
                obs = current if j == t-1 else observed[j]
                state,probe = self.write(obs,active,position_rows[j],frame_rows[j],state,
                                        probe=self.training and j >= burn and probe_steps[j])
            probes.append(probe)
        return state,torch.stack(probes,1)

    def forward(self, x, hist, hmask, queries=None, candidates=None, memory=None, probe_mask=None,
                candidate_mask=None):
        # Synchronize for scheduling before queueing GPU work, not behind it.
        schedule = self.schedule(x, memory, probe_mask)
        # Training-only work scheduling, never an input to the evaluator. Keep
        # whole curves whenever any point is labeled (including metric points).
        score_candidates = None
        if candidates is not None and candidate_mask is not None:
            if candidate_mask.shape != candidates.shape[:-1]:
                raise ValueError('Candidate scheduling mask must match candidate points')
            score_candidates = candidate_mask.detach().cpu().bool().any(dim=(0,2)).tolist()
        dense, _, image_tokens = self.encode(x['fine'])
        current = self.observation_features(dense, image_tokens)
        # Write the observed head once. Hypothetical candidate scoring never writes.
        state,probes = self.observe(x,current,memory,probe_mask,schedule)
        tokens,valid,refs,ref_valid,roles = self.retained(state,state['position'],state['frame'])
        references = torch.cat((hist,x['seed']),1)
        mask = torch.cat((hmask.bool(),x['seed_mask'].bool()),1) & crop_support(references,self.cfg.fine)
        references = torch.where(mask[...,None],references,0.)
        local,_ = sample_features(dense,references,self.cfg.fine)
        local = torch.where(mask[...,None],local,0.)
        # The seed identity is always its immutable encoded observation, whether
        # it remains visible or not; visible references retain spatial context.
        reference_embedding = F.normalize(self.embedding(local).float(),dim=-1)
        reference_embedding = torch.cat((reference_embedding[:,:-1],
            F.normalize(self.embedding(self.seed_descriptor(state)).float(),dim=-1)),1)
        reference_mask = torch.cat((mask[:,:-1],state['anchor_valid'][:,None]),1)
        visible = self.visible_features(local,references)
        tokens,valid = self.assemble_context(tokens,valid,image_tokens,visible,mask)
        refs = torch.cat((refs,local[:,:-1]),1)
        ref_valid = torch.cat((ref_valid,mask[:,:-1]),1)
        roles = torch.cat((roles,self.role[3:4].expand(self.cfg.n_history,-1)),0)

        def evaluate(points):
            b,k,_ = points.shape
            values,support = sample_features(dense,(points[:,:,None]+self.stencil).reshape(b,k*27,3),self.cfg.fine)
            return self.evaluate(points,values.reshape(b,k,27,-1),support.reshape(b,k,27),tokens,valid,refs,ref_valid,roles)

        points = hist.new_zeros(len(hist),self.cfg.n_future,3)
        points[...,2] = self.planes
        decoded = evaluate(points)
        lateral = self.cfg.lateral_limit*self.coordinates(decoded).float().tanh()
        points = torch.cat((lateral,points[...,2:]),-1)
        initial = points
        refinements = [points]
        if self.cfg.correction:
            for _ in range(self.cfg.correction_steps):
                decoded = evaluate(points)
                delta = self.correction(decoded).float().tanh()*(self.cfg.correction_limit/math.sqrt(2))
                lateral = (points[...,:2]+delta).clamp(-self.cfg.lateral_limit,self.cfg.lateral_limit)
                points = torch.cat((lateral,points[...,2:]),-1)
                refinements.append(points)

        def confidence(curve):
            evidence = evaluate(curve.detach())
            count = torch.arange(1,curve.shape[1]+1,device=curve.device)[None,:,None]
            summary = torch.cat((evidence.cumsum(1)/count,evidence.cummax(1).values),-1)
            return self.confidence_head(summary).squeeze(-1).float()

        logits = confidence(points)
        out = dict(points=points,initial_points=initial,refinement_points=torch.stack(refinements,1),
            confidence_logits=logits,confidence=logits.sigmoid().cummin(-1).values,
            reference_embedding=reference_embedding,reference_mask=reference_mask,
            memory_probe=probes,**{'memory_'+k:v for k,v in state.items()})
        if candidates is not None:
            out['candidate_confidence_logits'] = torch.stack([
                confidence(c) if score_candidates is None or score_candidates[j] else logits.new_zeros(logits.shape)
                for j,c in enumerate(candidates.unbind(1))],1)
        if queries is not None:
            features,support = sample_features(dense,queries,self.cfg.fine)
            out.update(query_embedding=F.normalize(self.embedding(features).float(),dim=-1),query_support=support)
        return out
