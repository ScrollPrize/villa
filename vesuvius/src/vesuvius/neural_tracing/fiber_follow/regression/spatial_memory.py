"""Seed-conditioned spatial history, with bounded retrieval of older observations.

The archive stores complete axial grids, not pooled replacements. Learned keys
route each candidate to older grids; a soft summary read trains routing even for
unselected entries. Seed/recent grids are always readable. Recurrent slots retain
context after FIFO eviction, but cannot reconstruct evicted spatial evidence.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import SPATIAL_MEMORY_ARCHITECTURE
from .unified import UnifiedFollower


class SpatialMemoryFollower(UnifiedFollower):
    architecture = SPATIAL_MEMORY_ARCHITECTURE
    memory_keys = ('slots','anchor','anchor_valid','anchor_position','anchor_frame',
                   'bank','bank_summary','bank_valid','bank_position','bank_frame',
                   'age','position','frame','seen')

    def __init__(self, cfg):
        super().__init__(cfg)
        h = cfg.hidden
        self.spatial_count = math.prod(cfg.token_shape)
        # Slot zero is the current head. It is excluded when decoding that head.
        self.capacity = 1+cfg.spatial_recent+cfg.spatial_archive
        self.spatial_position = nn.Linear(3, h, bias=False)
        self.summary_query = nn.Parameter(torch.randn(1, h)*.02)
        self.summary_attention = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
        self.retrieval_query = nn.Linear(h, h, bias=False)
        self.retrieval_key = nn.Linear(h, h, bias=False)
        self.archive_attention = nn.MultiheadAttention(h, cfg.heads, batch_first=True)
        self.retrieval_norm = nn.LayerNorm(h)

    def observation_features(self, dense, tokens):
        # Fine head samples supplement the full spatial grid and retain the
        # existing supervised identity space. They are not the history bottleneck.
        fine = super().observation_features(dense, tokens)
        return torch.cat((tokens, F.pad(fine, (0,self.cfg.hidden-self.cfg.channels))),1)

    def head_patches(self, observation):
        return observation[...,self.spatial_count:,:self.cfg.channels]

    def seed_descriptor(self, state):
        return self.head_patches(state['anchor'])[:,13:14]

    def initial_memory(self, batch, device):
        h, n, cap = self.cfg.hidden, self.spatial_count+27, self.capacity
        # Use parameter precision for storage, regardless of whether the state
        # is created inside (training) or outside (tracing) an autocast context.
        # Otherwise the same observation is rounded differently in the two paths.
        dtype = self.initial_slots.dtype
        zeros = lambda *s: torch.zeros(batch,*s,device=device)
        features = lambda *s: torch.zeros(batch,*s,device=device,dtype=dtype)
        return dict(slots=self.initial_slots.to(device)[None].expand(batch,-1,-1).clone(),
            anchor=features(n,h),anchor_valid=zeros().bool(),anchor_position=zeros(3),
            anchor_frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(),
            bank=features(cap,n,h),bank_summary=features(cap,h),bank_valid=zeros(cap).bool(),
            bank_position=zeros(cap,3),bank_frame=torch.eye(3,device=device)[None,None].expand(batch,cap,-1,-1).clone(),
            age=zeros(cap),position=zeros(3),frame=torch.eye(3,device=device)[None].expand(batch,-1,-1).clone(),
            seen=zeros().bool())

    def spatial_tokens(self, observation):
        fine = self.observation_projection(self.head_patches(observation))
        return torch.cat((observation[...,:self.spatial_count,:],fine),-2)

    def located(self, observation, positions, frames, position, frame, age, role):
        """Place full observation grids in the current query's coordinate frame."""
        # [B,O,N,H], with explicit observation origin, orientation and age.
        features = self.spatial_tokens(observation)
        xyz = torch.cat((self.encoder.token_xyz,self.stencil),0)
        world_offset = torch.einsum('ni,boji->bonj',xyz,frames)
        local_offset = torch.einsum('boni,bij->bonj',world_offset,frame)/16
        pose = self.pose(positions,frames,position,frame,age)
        return features+self.spatial_position(local_offset)+pose[:,:,None]+role

    def retained(self, state, position, frame):
        # Only explicit head descriptors participate in pointwise InfoNCE space.
        refs = torch.cat((self.seed_descriptor(state),self.head_patches(state['bank'])[:,:,13]),1)
        valid = torch.cat((state['anchor_valid'][:,None],state['bank_valid']),1)
        roles = torch.cat((self.role[0:1],self.role[1:2].expand(self.capacity,-1)),0)
        return dict(state=state,position=position,frame=frame,start=1),None,refs,valid,roles

    def assemble_context(self, tokens, valid, image_tokens, visible, mask):
        context = dict(tokens)
        context['current'] = torch.cat((image_tokens,visible),1)
        context['current_valid'] = torch.cat((torch.ones(image_tokens.shape[:2],device=mask.device,dtype=torch.bool),mask),1)
        return context,None

    def visible_features(self, local, references):
        meta = references.new_zeros(*references.shape[:2],13)
        meta[...,:3] = references/16
        meta[:,:-1,-1] = torch.arange(1,self.cfg.n_history+1,device=local.device).float().log1p()/8
        return self.observation_projection(local)+self.metadata(meta)+self.role[3]

    def _write(self, obs, active, position, frame, state, *, probe):
        position = torch.where(active[:,None],position,state['position'])
        frame = torch.where(active[:,None,None],frame,state['frame'])
        # Mask before any operation: inactive padding can contain NaN.
        obs = torch.where(active[:,None,None],obs,0.)
        observed = self.spatial_tokens(obs)
        seed = self.observation_projection(self.seed_descriptor(state))
        seed = seed*state['anchor_valid'][:,None,None]
        seed = seed+self.pose(state['anchor_position'][:,None],state['anchor_frame'][:,None],
            position,frame,position.new_zeros(len(obs),1))*state['anchor_valid'][:,None,None]
        previous_pose = self.pose(state['position'][:,None],state['frame'][:,None],position,frame,position.new_zeros(len(obs),1))
        previous_pose = previous_pose*state['seen'][:,None,None]
        evidence = self.write_attention(self.write_norm(state['slots'])+seed+previous_pose,observed,observed,
                                        need_weights=False)[0]
        pair = torch.cat((state['slots'],evidence),-1)
        slots = state['slots']+self.write_gate(pair).sigmoid()*(self.write_proposal(pair).tanh()-state['slots'])
        head = self.observation_projection(self.head_patches(obs)[:,13:14])
        summary = self.summary_attention(self.summary_query[None]+head,observed,observed,need_weights=False)[0][:,0]

        result = obs.new_zeros(len(obs),4)
        if probe:
            # Evaluate against pre-write evidence: no self-match to a just-added
            # descriptor. This uses exactly the path evaluator used at inference.
            context,_,refs,valid,roles = self.retained(state,position,frame)
            context.update(start=0,current=observed,current_valid=torch.ones(observed.shape[:2],device=obs.device,dtype=torch.bool))
            decoded = self.evaluate(obs.new_zeros(len(obs),1,3),self.head_patches(obs)[:,None],
                torch.ones(len(obs),1,27,device=obs.device,dtype=torch.bool),context,None,refs,valid,roles)
            result = self.probe_head(decoded[:,0]).float()
        updated = dict(state)
        updated['slots'] = torch.where(active[:,None,None],slots,state['slots'])
        for key,value in (('bank',obs),('bank_summary',summary),('bank_position',position),('bank_frame',frame)):
            shifted = torch.cat((value[:,None].to(state[key].dtype),state[key][:,:-1]),1)
            updated[key] = torch.where(active.reshape(len(active),*([1]*(shifted.ndim-1))),shifted,state[key])
        updated['bank_valid'] = torch.where(active[:,None],torch.cat((active[:,None],state['bank_valid'][:,:-1]),1),state['bank_valid'])
        updated['age'] = torch.where(active[:,None],torch.cat((state['age'].new_zeros(len(obs),1),state['age'][:,:-1]+1),1),state['age'])
        updated.update(position=position,frame=frame,seen=state['seen'] | active)
        return updated,result

    def _decode_spatial(self, query, context):
        state,position,frame = (context[k] for k in ('state','position','frame'))
        b = len(query)
        start = context['start']
        recent = slice(start,start+self.cfg.spatial_recent)
        archive = slice(start+self.cfg.spatial_recent,start+self.cfg.spatial_recent+self.cfg.spatial_archive)
        seed = self.located(state['anchor'][:,None],state['anchor_position'][:,None],state['anchor_frame'][:,None],
            position,frame,position.new_zeros(b,1),self.role[0])[:,0]
        nearby = self.located(state['bank'][:,recent],state['bank_position'][:,recent],state['bank_frame'][:,recent],
            position,frame,state['age'][:,recent],self.role[1]).flatten(1,2)
        slots = state['slots']+self.role[2]+self.pose(state['position'][:,None],state['frame'][:,None],
            position,frame,position.new_zeros(b,1))
        tokens = torch.cat((context['current'],seed,nearby,slots),1)
        n = self.spatial_count+27
        valid = torch.cat((context['current_valid'],state['anchor_valid'][:,None].expand(-1,n),
            state['bank_valid'][:,recent,None].expand(-1,-1,n).flatten(1),
            state['seen'][:,None].expand(-1,self.cfg.memory_slots)),1)

        # Cheap, seed-conditioned routing. Each path point has its own weights;
        # the candidate reads the union approximated by its K highest-scoring
        # observation blocks. Stable ordering resolves ties toward newer entries.
        seed_query = self.observation_projection(self.seed_descriptor(state))*state['anchor_valid'][:,None,None]
        query = query+seed_query
        summaries = state['bank_summary'][:,archive]+self.pose(state['bank_position'][:,archive],
            state['bank_frame'][:,archive],position,frame,state['age'][:,archive])+self.role[2]
        scores = torch.bmm(self.retrieval_query(query).float(),self.retrieval_key(summaries).float().transpose(1,2))/math.sqrt(self.cfg.hidden)
        archive_valid = state['bank_valid'][:,archive]
        scores = scores.masked_fill(~archive_valid[:,None],-torch.inf)
        # Null observation ensures an empty archive is finite and may be ignored.
        weights = torch.cat((scores,scores.new_zeros(b,query.shape[1],1)),-1).softmax(-1)[...,:-1]
        query = query+torch.bmm(weights.to(summaries.dtype),summaries)
        selected = scores.amax(1).argsort(dim=-1,descending=True,stable=True)[:,:self.cfg.spatial_retrieve]
        row = torch.arange(b,device=query.device)
        reads = torch.zeros_like(query)
        for k in range(self.cfg.spatial_retrieve):
            index = selected[:,k]+start+self.cfg.spatial_recent
            feature = self.located(state['bank'][row,index][:,None],state['bank_position'][row,index][:,None],
                state['bank_frame'][row,index][:,None],position,frame,state['age'][row,index][:,None],self.role[2])[:,0]
            read = self.archive_attention(self.retrieval_norm(query),feature,feature,need_weights=False)[0]
            weight = weights.gather(2,selected[:,k,None,None].expand(-1,query.shape[1],1))
            reads = reads+read*weight.to(read.dtype)
        return self.decoder(query+reads,tokens,memory_key_padding_mask=~valid)

    def evaluate(self, points, patches, support, tokens, valid, refs, ref_valid, roles):
        spatial = torch.cat((patches,support[...,None]),-1).flatten(-2)
        query = self.query(torch.cat((spatial,points/16),-1))
        query = query+self.identity_read(patches[:,:,13],refs,ref_valid,roles)
        if self.training and self.cfg.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._decode_spatial,query,tokens,use_reentrant=False)
        return self._decode_spatial(query,tokens)
