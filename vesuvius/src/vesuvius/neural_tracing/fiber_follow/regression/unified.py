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
        return dict(slots=self.initial_slots.to(device)[None].expand(batch,-1,-1),
            anchor=zeros(27,c), anchor_valid=zeros().bool(), anchor_position=zeros(3),
            anchor_frame=torch.eye(3,device=device)[None].expand(batch,-1,-1),
            recent=zeros(s,c), recent_position=zeros(s,3), recent_frame=torch.eye(3,device=device)[None,None].expand(batch,s,-1,-1),
            recent_valid=zeros(s).bool(), age=zeros(s), position=zeros(3),
            frame=torch.eye(3,device=device)[None].expand(batch,-1,-1), seen=zeros().bool())

    def encode(self, image, *, checkpoint_blocks=True):
        # Appearance is independent of the claimed identity for every observation.
        return self.encoder(image, None, None, checkpoint_blocks=checkpoint_blocks)

    def observation(self, image):
        # The caller checkpoints this entire operation. Nested block checkpoints
        # would recompute the axial stack a second time during backward.
        dense, _, _ = self.encode(image, checkpoint_blocks=False)
        return sample_features(dense, self.stencil[None].expand(len(image),-1,-1), self.cfg.fine)[0]

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
        # Auxiliary head uses exactly the same candidate evaluator, before this
        # observation can provide a trivial identity match to its own recent slot.
        result = obs.new_zeros(len(obs),4)
        if probe:
            head = obs.new_zeros(len(obs),1,3)
            decoded = self.evaluate(head,obs[:,None],torch.ones(len(obs),1,27,device=obs.device,dtype=torch.bool),
                                    tokens,valid,refs,ref_valid,roles)
            result = self.probe_head(decoded[:,0]).float()
        return state,result

    def observe(self, x, current, state=None):
        b, t = x['memory_mask'].shape
        device = current.device
        state = self.initial_memory(b,device) if state is None else state
        valid_seed = x['memory_seed_valid'].bool() & ~state['anchor_valid']
        if valid_seed.any():
            clean = torch.where(valid_seed.cpu()[:,None,None,None,None],x['seed_crop'].cpu(),0.)
            seed = self.encode_observation(clean,device,True)
            state = dict(state)
            state['anchor'] = torch.where(valid_seed[:,None,None],seed,state['anchor'])
            state['anchor_position'] = torch.where(valid_seed[:,None],x['memory_seed_position'],state['anchor_position'])
            state['anchor_frame'] = torch.where(valid_seed[:,None,None],x['memory_seed_frame'],state['anchor_frame'])
            state['anchor_valid'] = state['anchor_valid'] | valid_seed
        burn = max(0,t-self.cfg.memory_grad_steps-1)
        probes = []
        for j in range(t):
            active = x['memory_mask'][:,j].bool()
            if not active.any():
                probes.append(current.new_zeros(b,4))
                continue
            with torch.set_grad_enabled(torch.is_grad_enabled() and j >= burn):
                if j == t-1:
                    obs = current
                else:
                    image = x['history_crops'][:,j]
                    clean = torch.where(active.cpu()[:,None,None,None,None],image.cpu(),0.)
                    obs = self.encode_observation(clean,device,j >= burn)
                state,probe = self.write(obs,active,x['memory_positions'][:,j],x['memory_frames'][:,j],state,
                                        probe=self.training and j >= burn)
            probes.append(probe)
        return state,torch.stack(probes,1)

    def forward(self, x, hist, hmask, queries=None, candidates=None, memory=None):
        dense, _, image_tokens = self.encode(x['fine'])
        current = sample_features(dense,self.stencil[None].expand(len(hist),-1,-1),self.cfg.fine)[0]
        # Write the observed head once. Hypothetical candidate scoring never writes.
        state,probes = self.observe(x,current,memory)
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
            F.normalize(self.embedding(state['anchor'][:,13:14]).float(),dim=-1)),1)
        reference_mask = torch.cat((mask[:,:-1],state['anchor_valid'][:,None]),1)
        visible_meta = torch.cat((references/16,references.new_zeros(*references.shape[:2],10)),-1)
        visible = self.observation_projection(local)+self.metadata(visible_meta)+self.role[3]
        tokens = torch.cat((image_tokens,tokens,visible),1)
        valid = torch.cat((torch.ones(image_tokens.shape[:2],device=hist.device,dtype=torch.bool),valid,mask),1)
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
            out['candidate_confidence_logits'] = torch.stack([confidence(c) for c in candidates.unbind(1)],1)
        if queries is not None:
            features,support = sample_features(dense,queries,self.cfg.fine)
            out.update(query_embedding=F.normalize(self.embedding(features).float(),dim=-1),query_support=support)
        return out
