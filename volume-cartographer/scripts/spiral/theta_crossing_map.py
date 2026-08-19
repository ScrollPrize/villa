"""Cached theta=0 crossings for patch and point-collection topology.

The map deliberately stores topology, not transformed geometry.  Node geometry
is supplied by small source adapters and is materialised only while refreshing
the cache.  This keeps the ordinary loss path to int8 edge gathers and a
segmented cumulative sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


_TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class _NodeSource:
    start: int
    count: int
    get_zyxs: Callable[[torch.Tensor], torch.Tensor]


class ThetaCrossingMap:
    """One deduplicated, periodically refreshed crossing cache.

    Edges are stored canonically (low node id, high node id).  ``crossings`` is
    the signed winding adjustment in that canonical direction: +1 for a theta
    delta greater than pi and -1 for a delta less than -pi.
    """

    def __init__(self, device='cuda', update_interval=100, chunk_size=65_536):
        self.device = torch.device(device)
        self.update_interval = self._validate_interval(update_interval)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError('ThetaCrossingMap chunk_size must be positive')
        self._sources: list[_NodeSource] = []
        self._edge_chunks: list[torch.Tensor] = []
        self.num_nodes = 0
        self.edge_nodes = torch.empty((0, 2), dtype=torch.int64, device=self.device)
        self._edge_keys = torch.empty(0, dtype=torch.int64, device=self.device)
        self.node_theta = torch.empty(0, dtype=torch.float32, device=self.device)
        self.crossings = torch.empty(0, dtype=torch.int8, device=self.device)
        self.last_refresh_iteration: int | None = None
        self._fresh_without_iteration = False
        self._topology_dirty = False

    @staticmethod
    def _validate_interval(value):
        value = int(value)
        if value <= 0:
            raise ValueError('theta_crossing_map_update_interval must be positive')
        return value

    def register_nodes(self, count, get_zyxs):
        """Register a node source and return its global starting node id."""
        count = int(count)
        if count < 0:
            raise ValueError('ThetaCrossingMap node count cannot be negative')
        start = self.num_nodes
        self._sources.append(_NodeSource(start, count, get_zyxs))
        self.num_nodes += count
        self._topology_dirty = True
        self.last_refresh_iteration = None
        self._fresh_without_iteration = False
        return start

    def register_edges(self, node_pairs):
        """Register undirected node pairs; duplicates and reverse duplicates are removed."""
        pairs = torch.as_tensor(node_pairs, dtype=torch.int64, device=self.device)
        if pairs.numel() == 0:
            return
        pairs = pairs.reshape(-1, 2)
        if pairs.min().item() < 0 or pairs.max().item() >= self.num_nodes:
            raise ValueError('ThetaCrossingMap edge contains an unregistered node')
        pairs = pairs.sort(dim=1).values
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.numel():
            self._edge_chunks.append(pairs)
            self._topology_dirty = True
            self.last_refresh_iteration = None
            self._fresh_without_iteration = False

    def invalidate(self):
        """Invalidate transformed values without discarding source topology."""
        self.last_refresh_iteration = None
        self._fresh_without_iteration = False

    def _finalize_topology(self):
        if not self._topology_dirty:
            return
        if self._edge_chunks:
            pairs = torch.cat([self.edge_nodes, *self._edge_chunks], dim=0)
            keys = pairs[:, 0] * self.num_nodes + pairs[:, 1]
            keys, order = torch.sort(keys)
            keep = torch.ones_like(keys, dtype=torch.bool)
            keep[1:] = keys[1:] != keys[:-1]
            self._edge_keys = keys[keep]
            self.edge_nodes = pairs[order][keep]
        else:
            self.edge_nodes = torch.empty((0, 2), dtype=torch.int64, device=self.device)
            self._edge_keys = torch.empty(0, dtype=torch.int64, device=self.device)
        self._edge_chunks.clear()
        self.crossings = torch.empty(
            self.edge_nodes.shape[0], dtype=torch.int8, device=self.device)
        self._topology_dirty = False

    def resolve_edges(self, node_pairs):
        """Resolve directed node pairs to ``(edge_ids, directions)``.

        ``directions`` is +1 when the requested direction is canonical and -1
        in reverse.  A sampler emitting an edge absent from source topology is
        a hard error, since silently using sparse theta differences can lose
        multiple wraps.
        """
        self._finalize_topology()
        pairs = torch.as_tensor(node_pairs, dtype=torch.int64, device=self.device)
        original_shape = pairs.shape[:-1]
        pairs = pairs.reshape(-1, 2)
        canonical = pairs.sort(dim=1).values
        keys = canonical[:, 0] * self.num_nodes + canonical[:, 1]
        positions = torch.searchsorted(self._edge_keys, keys)
        safe = positions.clamp_max(max(0, self._edge_keys.numel() - 1))
        found = (positions < self._edge_keys.numel())
        if self._edge_keys.numel():
            found &= self._edge_keys[safe] == keys
        if not bool(found.all()):
            bad = pairs[torch.nonzero(~found, as_tuple=False)[0, 0]].tolist()
            raise RuntimeError(
                'sampled theta walk contains unregistered edge '
                f'{tuple(bad)}; rebuild the crossing topology')
        directions = torch.where(
            pairs[:, 0] == canonical[:, 0], 1, -1).to(torch.int8)
        return positions.reshape(original_shape), directions.reshape(original_shape)

    def resolve_walks(self, node_ids):
        node_ids = torch.as_tensor(node_ids, dtype=torch.int64, device=self.device)
        return self.resolve_edges(torch.stack([node_ids[..., :-1], node_ids[..., 1:]], dim=-1))

    def _set_interval(self, interval):
        interval = self._validate_interval(interval)
        if interval != self.update_interval:
            self.update_interval = interval
            self.last_refresh_iteration = None
            self._fresh_without_iteration = False

    def refresh_if_due(self, iteration, transform, update_interval=None):
        if update_interval is not None:
            self._set_interval(update_interval)
        iteration = int(iteration)
        if self._fresh_without_iteration:
            self.last_refresh_iteration = iteration
            self._fresh_without_iteration = False
            return False
        if (self.last_refresh_iteration is None
                or iteration - self.last_refresh_iteration >= self.update_interval):
            self._refresh(transform)
            self.last_refresh_iteration = iteration
            return True
        return False

    def force_refresh(self, transform):
        self._refresh(transform)
        # A forced diagnostic refresh is current but has no reliable optimiser
        # iteration associated with it.  First subsequent training use sets the
        # schedule anchor without needing another sweep.
        self.last_refresh_iteration = None
        self._fresh_without_iteration = True

    def _refresh(self, transform):
        self._finalize_topology()
        theta = torch.empty(self.num_nodes, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for source in self._sources:
                for lo in range(0, source.count, self.chunk_size):
                    hi = min(source.count, lo + self.chunk_size)
                    local = torch.arange(lo, hi, dtype=torch.int64, device=self.device)
                    zyxs = source.get_zyxs(local).to(
                        device=self.device, dtype=torch.float32)
                    if zyxs.shape != (hi - lo, 3):
                        raise RuntimeError(
                            'ThetaCrossingMap node provider returned shape '
                            f'{tuple(zyxs.shape)}, expected {(hi - lo, 3)}')
                    spiral = transform(zyxs)
                    theta[source.start + lo:source.start + hi] = (
                        torch.atan2(spiral[:, 1], spiral[:, 2]) % _TWO_PI
                    ).to(torch.float32)
            if self.edge_nodes.numel():
                edge_theta = theta[self.edge_nodes]
                delta = edge_theta[:, 1] - edge_theta[:, 0]
                self.crossings = (
                    (delta > np.pi).to(torch.int8)
                    - (delta < -np.pi).to(torch.int8))
            else:
                self.crossings = torch.empty(0, dtype=torch.int8, device=self.device)
        self.node_theta = theta

    def adjustments(self, packed_walks, sampled_theta, dr_per_winding):
        """Gather cumulative crossing adjustments for packed sampled walks.

        Inputs are row-major. Packed edge IDs/directions describe each dense
        node step, and pick positions index nodes in that walk. A nonnegative
        correction-node ID connects a cached patch quad centre to its current
        fractional pick. Ordinary rows are reanchored at their first pick. A
        row with a nonnegative reference-node ID is instead transported from
        that exact node through the start of the dense walk; relative/absolute
        winding losses use this to retain the annotated PCL point's frame even
        when the random samples omit the walk origin.
        """
        if self.last_refresh_iteration is None and self.node_theta.numel() != self.num_nodes:
            raise RuntimeError('ThetaCrossingMap must be refreshed before use')
        edge_ids = packed_walks.edge_ids
        directions = packed_walks.directions
        steps = self.crossings[edge_ids].to(torch.int32) * directions.to(torch.int32)
        steps = steps * packed_walks.edge_valid.to(torch.int32)
        cumulative = torch.cat([
            torch.zeros((*steps.shape[:-1], 1), dtype=torch.int32, device=self.device),
            steps.cumsum(dim=-1),
        ], dim=-1)
        picked = torch.gather(cumulative, -1, packed_walks.pick_positions)
        correction_node_ids = packed_walks.correction_node_ids
        sampled_theta = sampled_theta.detach().to(device=self.device)
        correction_mask = correction_node_ids >= 0
        centre_theta = self.node_theta[correction_node_ids.clamp_min(0)]
        local_delta = sampled_theta - centre_theta
        local = ((local_delta > np.pi).to(torch.int32)
                 - (local_delta < -np.pi).to(torch.int32))
        picked = picked + torch.where(
            correction_mask, local, torch.zeros_like(local))

        reference_node_ids = packed_walks.reference_node_ids
        reference_mask = reference_node_ids >= 0
        reference_theta = self.node_theta[reference_node_ids.clamp_min(0)]
        start_theta = self.node_theta[packed_walks.walk_start_node_ids]
        reference_delta = start_theta - reference_theta
        reference_step = (
            (reference_delta > np.pi).to(torch.int32)
            - (reference_delta < -np.pi).to(torch.int32))
        picked = picked + torch.where(
            reference_mask, reference_step, torch.zeros_like(reference_step)
        )[..., None]
        picked = picked - torch.where(
            reference_mask[..., None], torch.zeros_like(picked[..., :1]),
            picked[..., :1])
        return picked.to(dr_per_winding.dtype) * dr_per_winding.detach()
