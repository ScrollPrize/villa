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
import scipy.sparse
import scipy.sparse.linalg
import torch


_TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class _NodeSource:
    start: int
    count: int
    # Called with a source-local half-open ordinal range [lo, hi); returns the
    # (hi - lo, 3) node geometry. Range-based so providers can slice resident
    # tensors and stream compact storage without an index-tensor round trip.
    get_zyxs: Callable[[int, int], torch.Tensor]


@dataclass(slots=True)
class _UnwrapTree:
    """One preorder-indexed tree used to lift wrapped node theta."""

    node_ids: torch.Tensor
    parent_pairs: torch.Tensor
    subtree_ends: torch.Tensor


@dataclass(slots=True)
class _PotentialSource:
    """Compact tree and neighbour topology owned by an external atlas.

    The callbacks return source-local node ordinals.  Keeping the compact
    representation in the native sampling atlas avoids materialising an
    int64 node map, tree and eight-neighbour edge graph for every patch quad.
    """

    start: int
    count: int
    get_tree_chunk: Callable[[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    get_neighbor_chunk: Callable[[int, int], tuple[int, np.ndarray]]


class ThetaCrossingMap:
    """One deduplicated, periodically refreshed crossing cache.

    Edges are stored canonically (low node id, high node id).  ``crossings`` is
    the signed winding adjustment in that canonical direction: +1 for a theta
    delta greater than pi and -1 for a delta less than -pi.
    """

    def __init__(self, device='cuda', update_interval=100, chunk_size=1_048_576):
        self.device = torch.device(device)
        # Source topology can contain several edges per valid patch quad.  It
        # is immutable, is touched only during refresh/edge resolution, and
        # can dwarf the tensors used by an optimisation step.  Keep it on the
        # host; only the compact int8 crossing values are resident on CUDA.
        self.topology_device = torch.device('cpu')
        self.update_interval = self._validate_interval(update_interval)
        # One chunk bounds the host/device transient (~16 MB of int64 pairs)
        # while amortising per-chunk provider-callback and kernel-launch
        # overhead, and is large enough for the native neighbour provider to
        # fan out across threads.
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError('ThetaCrossingMap chunk_size must be positive')
        self._sources: list[_NodeSource] = []
        self._edge_chunks: list[torch.Tensor] = []
        self._unwrap_trees: list[_UnwrapTree] = []
        self._potential_sources: list[_PotentialSource] = []
        self.num_nodes = 0
        self.edge_nodes = torch.empty(
            (0, 2), dtype=torch.int64, device=self.topology_device)
        self._edge_keys = torch.empty(
            0, dtype=torch.int64, device=self.topology_device)
        self.node_theta = torch.empty(0, dtype=torch.float32, device=self.device)
        self.crossings = torch.empty(0, dtype=torch.int8, device=self.device)
        # Patch quad centres opt into an unwrap tree.  The resulting integer
        # potential is root-relative and is refreshed from the same cached
        # edge crossings as sampled walks.  INT32_MIN marks non-tree nodes
        # (for example PCL topology, which continues to use explicit walks).
        self._unset_potential = torch.iinfo(torch.int32).min
        self.node_winding_potential = torch.empty(
            0, dtype=torch.int32, device=self.device)
        self._any_unset_potential = True
        self._pending_potential_checks = []
        self._potential_node_ids = torch.empty(
            0, dtype=torch.int64, device=self.topology_device)
        self._potential_edge_ids = torch.empty(
            0, dtype=torch.int64, device=self.topology_device)
        self._potential_directions = torch.empty(
            0, dtype=torch.int8, device=self.topology_device)
        self._potential_exit_positions = torch.empty(
            0, dtype=torch.int64, device=self.topology_device)
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
        pairs = torch.as_tensor(
            node_pairs, dtype=torch.int64, device=self.topology_device)
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

    def register_potential_source(
        self, start, count, get_tree_chunk, get_neighbor_chunk,
    ):
        """Register compact patch topology without adding explicit edges.

        ``get_tree_chunk(lo, hi)`` returns preorder node ordinals, parent node
        ordinals and subtree exit positions for ``[lo, hi)``.  The neighbour
        callback consumes a four-neighbour-slot cursor and returns the next
        cursor plus the valid source-local node pairs encountered.  Both APIs
        are deliberately chunked so refresh and consistency checks have a
        fixed-size host transient.
        """
        start = int(start)
        count = int(count)
        if count < 0 or start < 0 or start + count > self.num_nodes:
            raise ValueError('compact potential source is outside registered nodes')
        self._potential_sources.append(_PotentialSource(
            start, count, get_tree_chunk, get_neighbor_chunk))
        self._topology_dirty = True
        self.last_refresh_iteration = None
        self._fresh_without_iteration = False

    def register_unwrap_tree(self, node_ids, parent_positions):
        """Register one connected node tree in depth-first preorder.

        ``node_ids`` contains every tree node exactly once. ``parent_positions``
        indexes that array, with -1 for the root and every other parent strictly
        preceding its child.  The preorder property makes every subtree a
        contiguous range.  At refresh time edge crossing steps are added on
        subtree entry and removed just after subtree exit; one cumulative sum
        then yields a root-relative integer winding potential at every node.
        """
        nodes = torch.as_tensor(
            node_ids, dtype=torch.int64, device=self.topology_device).reshape(-1)
        parents = torch.as_tensor(
            parent_positions, dtype=torch.int64,
            device=self.topology_device).reshape(-1)
        if nodes.numel() == 0 or parents.shape != nodes.shape:
            raise ValueError('unwrap tree nodes and parents must be equal nonempty vectors')
        if nodes.min().item() < 0 or nodes.max().item() >= self.num_nodes:
            raise ValueError('unwrap tree contains an unregistered node')
        if torch.unique(nodes).numel() != nodes.numel():
            raise ValueError('unwrap tree contains duplicate nodes')
        if parents[0].item() != -1:
            raise ValueError('unwrap tree root parent must be -1')
        if parents.numel() > 1:
            positions = torch.arange(
                1, parents.numel(), dtype=torch.int64,
                device=self.topology_device)
            if not bool(((parents[1:] >= 0) & (parents[1:] < positions)).all()):
                raise ValueError('unwrap tree parents must precede their children')

        # In DFS preorder every subtree is contiguous. Compute its end once at
        # topology build time; refreshes then need only a streamed scatter and
        # one cumsum.
        count = nodes.numel()
        if count == 1:
            subtree_ends = torch.zeros(
                1, dtype=torch.int64, device=self.topology_device)
        elif np.array_equal(parents.numpy()[1:], np.arange(count - 1)):
            # A Hamiltonian grid walk is a single chain: every subtree ends at
            # the final node. This is the common fast path for rectangular
            # auto-grown patches.
            subtree_ends = torch.full(
                (count,), count - 1, dtype=torch.int64,
                device=self.topology_device)
        elif count < 10_000:
            # Avoid constructing a sparse triangular system for small ragged
            # patches; a short host loop has much lower fixed overhead.
            parents_np = parents.numpy()
            subtree_sizes = np.ones(count, dtype=np.int64)
            for child in range(count - 1, 0, -1):
                parent = parents_np[child]
                subtree_sizes[parent] += subtree_sizes[child]
            subtree_ends = torch.from_numpy(
                np.arange(count, dtype=np.int64) + subtree_sizes - 1)
        else:
            # subtree_size[parent] = 1 + sum(subtree_size[child]).  Because
            # parents precede children this is an upper-triangular solve.  The
            # scipy implementation performs the reverse accumulation in C;
            # unlike a Python node loop it stays cheap for million-quad bands.
            positions_np = np.arange(count, dtype=np.int64)
            parents_np = parents.numpy()
            rows = np.concatenate([positions_np, parents_np[1:]])
            cols = np.concatenate([positions_np, positions_np[1:]])
            data = np.concatenate([
                np.ones(count, dtype=np.float64),
                -np.ones(count - 1, dtype=np.float64),
            ])
            system = scipy.sparse.csr_matrix(
                (data, (rows, cols)), shape=(count, count))
            subtree_sizes = scipy.sparse.linalg.spsolve_triangular(
                system, np.ones(count, dtype=np.float64), lower=False)
            subtree_sizes = np.rint(subtree_sizes).astype(np.int64)
            subtree_ends = torch.from_numpy(
                positions_np + subtree_sizes - 1)
        if count > 1:
            child_positions = torch.arange(
                1, count, dtype=torch.int64, device=self.topology_device)
            if not bool((subtree_ends[parents[1:]] >= child_positions).all()):
                raise ValueError('unwrap tree nodes must be in depth-first preorder')
        parent_pairs = (
            torch.stack([nodes[parents[1:]], nodes[1:]], dim=1)
            if nodes.numel() > 1
            else torch.empty((0, 2), dtype=torch.int64,
                             device=self.topology_device)
        )
        self._unwrap_trees.append(_UnwrapTree(
            nodes, parent_pairs, subtree_ends))
        self.register_edges(parent_pairs)
        # A single-node tree has no edge through which register_edges can mark
        # the derived potential topology dirty.
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
            self.edge_nodes = torch.empty(
                (0, 2), dtype=torch.int64, device=self.topology_device)
            self._edge_keys = torch.empty(
                0, dtype=torch.int64, device=self.topology_device)
        self._edge_chunks.clear()
        self.crossings = torch.empty(
            self.edge_nodes.shape[0], dtype=torch.int8, device=self.device)
        self._topology_dirty = False

        potential_nodes = []
        potential_edge_ids = []
        potential_directions = []
        potential_exits = []
        offset = 0
        for tree in self._unwrap_trees:
            count = tree.node_ids.numel()
            edge_ids = torch.zeros(
                count, dtype=torch.int64, device=self.topology_device)
            directions = torch.zeros(
                count, dtype=torch.int8, device=self.topology_device)
            if count > 1:
                resolved, resolved_directions = self.resolve_edges(
                    tree.parent_pairs)
                edge_ids[1:] = resolved
                directions[1:] = resolved_directions
            # Each non-root step is removed immediately after its child's
            # complete subtree. Roots carry a zero step, so their exit is inert.
            exits = tree.subtree_ends + offset + 1
            exits[0] = offset
            potential_nodes.append(tree.node_ids)
            potential_edge_ids.append(edge_ids)
            potential_directions.append(directions)
            potential_exits.append(exits)
            offset += count
        if potential_nodes:
            all_nodes = torch.cat(potential_nodes)
            if torch.unique(all_nodes).numel() != all_nodes.numel():
                raise ValueError('unwrap trees overlap at one or more nodes')
            self._potential_node_ids = all_nodes
            self._potential_edge_ids = torch.cat(potential_edge_ids)
            self._potential_directions = torch.cat(potential_directions)
            self._potential_exit_positions = torch.cat(potential_exits)
        else:
            self._potential_node_ids = torch.empty(
                0, dtype=torch.int64, device=self.topology_device)
            self._potential_edge_ids = torch.empty(
                0, dtype=torch.int64, device=self.topology_device)
            self._potential_directions = torch.empty(
                0, dtype=torch.int8, device=self.topology_device)
            self._potential_exit_positions = torch.empty(
                0, dtype=torch.int64, device=self.topology_device)
        self.node_winding_potential = torch.full(
            (self.num_nodes,), self._unset_potential,
            dtype=torch.int32, device=self.device)
        self._any_unset_potential = True

    def resolve_edges(self, node_pairs):
        """Resolve directed node pairs to ``(edge_ids, directions)``.

        ``directions`` is +1 when the requested direction is canonical and -1
        in reverse.  A sampler emitting an edge absent from source topology is
        a hard error, since silently using sparse theta differences can lose
        multiple wraps.
        """
        self._finalize_topology()
        return_device = (
            node_pairs.device
            if isinstance(node_pairs, torch.Tensor)
            else self.topology_device)
        pairs = torch.as_tensor(
            node_pairs, dtype=torch.int64, device=self.topology_device)
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
        return (
            positions.reshape(original_shape).to(return_device),
            directions.reshape(original_shape).to(return_device),
        )

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
        # Resolve outstanding unset-potential verdicts against the outgoing
        # table before it is replaced.
        self._drain_potential_checks(blocking=True)
        self._finalize_topology()
        theta = torch.empty(self.num_nodes, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for source in self._sources:
                for lo in range(0, source.count, self.chunk_size):
                    hi = min(source.count, lo + self.chunk_size)
                    zyxs = source.get_zyxs(lo, hi).to(
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
                # A whole-patch edge table can be multiple GiB as int64 pairs.
                # Stream it through CUDA in bounded chunks instead of
                # materialising the table (and its gather result) there.
                # _finalize_topology already sized this compact cache; fill it
                # in place so refresh does not briefly hold two copies.
                crossings = self.crossings
                for lo in range(0, self.edge_nodes.shape[0], self.chunk_size):
                    hi = min(self.edge_nodes.shape[0], lo + self.chunk_size)
                    edge_nodes = self.edge_nodes[lo:hi].to(self.device)
                    edge_theta = theta[edge_nodes]
                    delta = edge_theta[:, 1] - edge_theta[:, 0]
                    crossings[lo:hi] = (
                        (delta > np.pi).to(torch.int8)
                        - (delta < -np.pi).to(torch.int8))
                self.crossings = crossings
            else:
                self.crossings = torch.empty(0, dtype=torch.int8, device=self.device)

            num_potential_nodes = self._potential_node_ids.numel()
            node_potential = torch.full(
                (self.num_nodes,), self._unset_potential,
                dtype=torch.int32, device=self.device)
            if num_potential_nodes:
                events = torch.zeros(
                    num_potential_nodes + 1,
                    dtype=torch.int32, device=self.device)
                for lo in range(0, num_potential_nodes, self.chunk_size):
                    hi = min(num_potential_nodes, lo + self.chunk_size)
                    edge_ids = self._potential_edge_ids[lo:hi].to(self.device)
                    directions = self._potential_directions[lo:hi].to(self.device)
                    steps = (
                        self.crossings[edge_ids].to(torch.int32)
                        * directions.to(torch.int32)
                        if self.crossings.numel()
                        else torch.zeros_like(directions, dtype=torch.int32))
                    entries = torch.arange(
                        lo, hi, dtype=torch.int64, device=self.device)
                    exits = self._potential_exit_positions[lo:hi].to(self.device)
                    events.index_add_(0, entries, steps)
                    events.index_add_(0, exits, -steps)
                torch.cumsum(events, dim=0, dtype=torch.int32, out=events)
                ordered_potential = events[:-1]
                for lo in range(0, num_potential_nodes, self.chunk_size):
                    hi = min(num_potential_nodes, lo + self.chunk_size)
                    node_ids = self._potential_node_ids[lo:hi].to(self.device)
                    node_potential[node_ids] = ordered_potential[lo:hi]

            # Patch atlases keep their trees in compact native storage.  The
            # callback expands only one bounded chunk of IDs at a time; branch
            # steps are computed straight from theta, so patch edges never
            # enter the explicit PCL/fiber edge table above.
            for source in self._potential_sources:
                if source.count == 0:
                    continue
                events = torch.zeros(
                    source.count + 1, dtype=torch.int32, device=self.device)
                for lo in range(0, source.count, self.chunk_size):
                    hi = min(source.count, lo + self.chunk_size)
                    nodes_np, parents_np, exits_np = source.get_tree_chunk(lo, hi)
                    nodes = torch.as_tensor(
                        nodes_np, dtype=torch.int64, device=self.device)
                    parents = torch.as_tensor(
                        parents_np, dtype=torch.int64, device=self.device)
                    exits = torch.as_tensor(
                        exits_np, dtype=torch.int64, device=self.device)
                    expected = hi - lo
                    if (nodes.numel() != expected
                            or parents.numel() != expected
                            or exits.numel() != expected):
                        raise RuntimeError(
                            'compact theta tree provider returned the wrong chunk size')
                    delta = (
                        theta[source.start + nodes]
                        - theta[source.start + parents])
                    steps = (
                        (delta > np.pi).to(torch.int32)
                        - (delta < -np.pi).to(torch.int32))
                    entries = torch.arange(
                        lo, hi, dtype=torch.int64, device=self.device)
                    events.index_add_(0, entries, steps)
                    events.index_add_(0, exits, -steps)
                torch.cumsum(events, dim=0, dtype=torch.int32, out=events)
                ordered_potential = events[:-1]
                for lo in range(0, source.count, self.chunk_size):
                    hi = min(source.count, lo + self.chunk_size)
                    nodes_np, _, _ = source.get_tree_chunk(lo, hi)
                    nodes = torch.as_tensor(
                        nodes_np, dtype=torch.int64, device=self.device)
                    node_potential[source.start + nodes] = ordered_potential[lo:hi]
            self.node_winding_potential = node_potential
            # One sync per refresh so the common all-registered case skips
            # the per-call unset check below (that check's .any() would
            # otherwise stall every sampler call on the full GPU queue).
            self._any_unset_potential = bool(
                (node_potential == self._unset_potential).any())
        self.node_theta = theta

    def _raise_unset_potential(self, ids):
        values = self.node_winding_potential[ids]
        unset_positions = (values == self._unset_potential).nonzero(
            as_tuple=True)[0]
        if unset_positions.numel():
            bad = f'theta node {int(ids[unset_positions[0]])}'
        else:
            # The table was replaced after the offending gather; the ids are
            # still the batch that failed against the previous table.
            bad = 'a theta node (from a since-replaced potential table)'
        raise RuntimeError(
            f'{bad} has no registered unwrap potential')

    def assert_no_pending_potential_errors(self):
        """Resolve every queued unset-potential verdict now (blocking).

        winding_potentials() defers its unset-node hard error to avoid a
        per-call device synchronisation, so a caller must invoke this at any
        boundary that must not proceed on unvalidated potentials — the
        training loop calls it before each optimizer step, the preview
        export after its diagnostics pass. One-shot tools that call
        winding_potentials()/adjustments_from_potentials() directly should
        call it before returning their results.
        """
        self._drain_potential_checks(blocking=True)

    def _drain_potential_checks(self, blocking=False):
        """Verify completed asynchronous unset-potential checks.

        Non-blocking by default: only verdicts whose copy event has fired
        are consumed, so this never waits on queued GPU work. blocking=True
        forces all pending verdicts to resolve.
        """
        pending = self._pending_potential_checks
        while pending:
            event, verdict, ids = pending[0]
            if not blocking and not event.query():
                break
            event.synchronize()
            if bool(verdict):
                self._raise_unset_potential(ids)
            pending.pop(0)

    def winding_potentials(self, node_ids, sampled_theta=None):
        """Return root-relative integer winding potentials for arbitrary nodes.

        When ``sampled_theta`` is supplied, the cached quad-centre potential is
        transported through the final local centre-to-fractional-sample step.
        """
        ids = torch.as_tensor(node_ids, dtype=torch.int64, device=self.device)
        if self.node_winding_potential.numel() != self.num_nodes:
            raise RuntimeError('ThetaCrossingMap must be refreshed before use')
        values = self.node_winding_potential[ids]
        # Unset entries (a sampler querying a node with no registered
        # potential) are a hard error, but reading the check's verdict here
        # would stall the CPU on all queued GPU work. When the table has no
        # unset entries at all the check is skipped outright; otherwise the
        # verdict is copied to pinned memory asynchronously and raised from
        # a later call, a refresh, or an assert_no_pending_potential_errors()
        # boundary (the training loop places one before each optimizer step,
        # so no parameter update ever applies unvalidated potentials).
        self._drain_potential_checks()
        if self._any_unset_potential:
            if self.device.type == 'cuda':
                verdict = torch.empty((), dtype=torch.bool, pin_memory=True)
                verdict.copy_(
                    (values == self._unset_potential).any(),
                    non_blocking=True)
                event = torch.cuda.Event()
                event.record()
                self._pending_potential_checks.append((event, verdict, ids))
            elif bool((values == self._unset_potential).any()):
                self._raise_unset_potential(ids)
        if sampled_theta is not None:
            theta = torch.as_tensor(
                sampled_theta, device=self.device).detach()
            if theta.shape != ids.shape:
                raise ValueError('sampled theta and potential node IDs must have equal shape')
            local_delta = theta - self.node_theta[ids]
            values = values + (
                (local_delta > np.pi).to(torch.int32)
                - (local_delta < -np.pi).to(torch.int32))
        return values

    def adjustments_from_potentials(
        self, node_ids, sampled_theta, dr_per_winding, *,
        reference_node_ids=None, reference_patch_node_ids=None,
    ):
        """Adjust unordered patch samples using cached per-node potentials.

        Ordinary rows retain the patch tree's root-relative frame. Annotation-led
        rows instead use the exact PCL reference node and its attached patch
        quad, preserving the absolute/relative winding frame without an
        explicit sampled walk from the annotation.
        """
        values = self.winding_potentials(node_ids, sampled_theta)
        has_reference = reference_node_ids is not None
        if has_reference != (reference_patch_node_ids is not None):
            raise ValueError(
                'reference node and reference patch node must be supplied together')
        if has_reference:
            references = torch.as_tensor(
                reference_node_ids, dtype=torch.int64, device=self.device)
            patch_references = torch.as_tensor(
                reference_patch_node_ids, dtype=torch.int64, device=self.device)
            expected_shape = values.shape[:-1]
            if references.shape != expected_shape or patch_references.shape != expected_shape:
                raise ValueError('reference IDs must match the sample row shape')
            patch_potential = self.winding_potentials(patch_references)
            reference_delta = (
                self.node_theta[patch_references] - self.node_theta[references])
            reference_step = (
                (reference_delta > np.pi).to(torch.int32)
                - (reference_delta < -np.pi).to(torch.int32))
            values = (
                values - patch_potential[..., None]
                + reference_step[..., None])
        return values.to(dr_per_winding.dtype) * dr_per_winding.detach()

    def potential_inconsistencies(self):
        """Check potentials and return the nodes on inconsistent patch edges.

        Tree edges are correct by construction; non-tree neighbor edges make
        this a cycle/path-independence diagnostic for large patches.  The node
        IDs are returned on the host so callers can attribute a failure to its
        owning patch without allocating a dense node-to-patch table.
        """
        if self.node_winding_potential.numel() != self.num_nodes:
            raise RuntimeError('ThetaCrossingMap must be refreshed before use')
        checked_total = torch.zeros(
            (), dtype=torch.int64, device=self.device)
        inconsistent_total = torch.zeros(
            (), dtype=torch.int64, device=self.device)
        max_abs_residual_total = torch.zeros(
            (), dtype=torch.int32, device=self.device)

        def edge_chunks():
            for lo in range(0, self.edge_nodes.shape[0], self.chunk_size):
                hi = min(self.edge_nodes.shape[0], lo + self.chunk_size)
                yield (self.edge_nodes[lo:hi].to(self.device),
                       self.crossings[lo:hi].to(torch.int32))
            # Neighbour slots are a pure streaming pass with no resident
            # output, and each slot yields at most one pair, so they can be
            # requested in larger chunks than the geometry/edge loops without
            # raising the peak transient class.
            slot_chunk = self.chunk_size * 8
            for source in self._potential_sources:
                cursor = 0
                end = source.count * 4
                while cursor < end:
                    next_cursor, pairs_np = source.get_neighbor_chunk(
                        cursor, slot_chunk)
                    next_cursor = int(next_cursor)
                    if next_cursor <= cursor or next_cursor > end:
                        raise RuntimeError(
                            'compact theta neighbour provider made invalid progress')
                    cursor = next_cursor
                    pairs = torch.as_tensor(
                        pairs_np, dtype=torch.int64, device=self.device).reshape(-1, 2)
                    if not pairs.numel():
                        continue
                    nodes = pairs + source.start
                    delta = self.node_theta[nodes[:, 1]] - self.node_theta[nodes[:, 0]]
                    crossings = (
                        (delta > np.pi).to(torch.int32)
                        - (delta < -np.pi).to(torch.int32))
                    yield nodes, crossings

        for nodes, crossings in edge_chunks():
            potentials = self.node_winding_potential[nodes]
            valid = (potentials != self._unset_potential).all(dim=1)
            residual = (
                potentials[:, 1] - potentials[:, 0]
                - crossings)
            bad = valid & (residual != 0)
            checked_total += valid.sum()
            inconsistent_total += bad.sum()
            if residual.numel():
                chunk_max = torch.where(
                    valid, residual.abs(), 0).max()
                max_abs_residual_total = torch.maximum(
                    max_abs_residual_total, chunk_max)
        checked = int(checked_total)
        inconsistent = int(inconsistent_total)
        max_abs_residual = int(max_abs_residual_total)
        report = {
            'checked_edges': checked,
            'inconsistent_edges': inconsistent,
            'max_abs_residual': max_abs_residual,
        }
        inconsistent_node_chunks = []
        if inconsistent:
            # Attribution is intentionally a second pass.  The common clean
            # case above performs only three final device synchronisations,
            # rather than one for every streamed edge chunk.
            for nodes, crossings in edge_chunks():
                potentials = self.node_winding_potential[nodes]
                valid = (potentials != self._unset_potential).all(dim=1)
                residual = (
                    potentials[:, 1] - potentials[:, 0]
                    - crossings)
                bad = valid & (residual != 0)
                if bool(bad.any()):
                    inconsistent_node_chunks.append(nodes[bad].cpu())
        inconsistent_nodes = torch.empty(0, dtype=torch.int64)
        if inconsistent_node_chunks:
            inconsistent_nodes = torch.unique(
                torch.cat(inconsistent_node_chunks))
        return report, inconsistent_nodes

    def potential_consistency(self):
        """Return aggregate cycle/path-independence diagnostics."""
        return self.potential_inconsistencies()[0]

    def adjustments(
            self, packed_walks, sampled_theta, dr_per_winding, *,
            return_walk_start_adjustment=False):
        """Gather cumulative crossing adjustments for packed sampled walks.

        Inputs are row-major. Packed edge IDs/directions describe each dense
        node step, and pick positions index nodes in that walk. A nonnegative
        correction-node ID connects a cached patch quad centre to its current
        fractional pick. Ordinary rows are reanchored at their first pick. A
        row with a nonnegative reference-node ID is instead transported from
        that exact node through the start of the dense walk; relative/absolute
        winding losses use this to retain the annotated PCL point's frame even
        when the random samples omit the walk origin. When requested, the second
        return value is the dense walk origin's adjustment in that output frame.
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
        walk_start_adjustment = torch.where(
            reference_mask, reference_step, -picked[..., 0])
        picked = picked + torch.where(
            reference_mask, reference_step, torch.zeros_like(reference_step)
        )[..., None]
        picked = picked - torch.where(
            reference_mask[..., None], torch.zeros_like(picked[..., :1]),
            picked[..., :1])
        result = picked.to(dr_per_winding.dtype) * dr_per_winding.detach()
        if return_walk_start_adjustment:
            return (
                result,
                walk_start_adjustment.to(dr_per_winding.dtype)
                * dr_per_winding.detach(),
            )
        return result
