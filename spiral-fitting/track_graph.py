"""Packed track-crossing graph with stable track and point identities."""

from __future__ import annotations

import itertools
import time

import numpy as np
import rustworkx as rx


class TrackGraph:
    """A rustworkx topology backed by the exact-crossing CSR arrays.

    Graph node indices are rows in ``source_ids``. Node and edge payloads are
    deliberately ``None``; source identities and crossing-local point indices
    remain in packed NumPy arrays instead of millions of Python objects.
    """

    def __init__(
            self, crossing_cache, *, track_chunk_size=250_000,
            node_chunk_size=1_000_000):
        self.source_ids = np.asarray(
            crossing_cache["source_ids"], dtype=np.uint64)
        self.offsets = np.asarray(
            crossing_cache["offsets"], dtype=np.int64)
        self.partners = np.asarray(
            crossing_cache["partners"], dtype=np.int32)
        self.self_local = np.asarray(
            crossing_cache["self_local"], dtype=np.int32)
        self.partner_local = np.asarray(
            crossing_cache["partner_local"], dtype=np.int32)
        self.positions = np.asarray(
            crossing_cache["positions"], dtype=np.float64)
        self.clearances = np.asarray(
            crossing_cache["clearances"], dtype=np.float64)
        self._validate()

        started = time.perf_counter()
        self.graph = rx.PyGraph(multigraph=True)
        for begin in range(0, len(self.source_ids), node_chunk_size):
            count = min(node_chunk_size, len(self.source_ids) - begin)
            self.graph.add_nodes_from((None for _ in range(count)))

        undirected_edges = 0
        for row_begin in range(0, len(self.source_ids), track_chunk_size):
            row_end = min(
                len(self.source_ids), row_begin + track_chunk_size)
            record_begin = int(self.offsets[row_begin])
            record_end = int(self.offsets[row_end])
            counts = np.diff(self.offsets[row_begin:row_end + 1])
            sources = np.repeat(
                np.arange(row_begin, row_end, dtype=np.int32), counts)
            partners = self.partners[record_begin:record_end]
            keep = sources < partners
            if np.any(keep):
                kept_sources = sources[keep]
                kept_partners = partners[keep]
                self.graph.add_edges_from_no_data(
                    zip(kept_sources, kept_partners))
                undirected_edges += len(kept_sources)

        if 2 * undirected_edges != len(self.partners):
            raise ValueError(
                "crossing cache is not a symmetric directed graph")
        self.build_seconds = time.perf_counter() - started

    def _validate(self):
        track_count = len(self.source_ids)
        if self.offsets.shape != (track_count + 1,):
            raise ValueError("crossing offsets are not parallel to source IDs")
        if (track_count and
                np.any(self.source_ids[1:] <= self.source_ids[:-1])):
            raise ValueError("track source IDs must be strictly increasing")
        if (self.offsets[0] != 0
                or np.any(self.offsets[1:] < self.offsets[:-1])):
            raise ValueError("crossing offsets must be monotonic from zero")
        record_count = int(self.offsets[-1])
        for name in (
                "partners", "self_local", "partner_local",
                "positions", "clearances"):
            if getattr(self, name).shape != (record_count,):
                raise ValueError(
                    f"crossing {name} is not parallel to crossing records")
        if (record_count and
                (np.any(self.partners < 0)
                 or np.any(self.partners >= track_count))):
            raise ValueError("crossing partner is outside the graph")
        if (np.any(self.self_local < 0)
                or np.any(self.partner_local < 0)):
            raise ValueError("crossing local indices must be non-negative")

    def __len__(self):
        return self.graph.num_nodes()

    def __getitem__(self, name):
        if name not in {
                "source_ids", "offsets", "partners", "self_local",
                "partner_local", "positions", "clearances"}:
            raise KeyError(name)
        return getattr(self, name)

    @property
    def edge_count(self):
        return self.graph.num_edges()

    def node_for_source_id(self, source_id):
        """Return the graph node for one stable track source ID."""
        source_id = np.uint64(source_id)
        node = int(np.searchsorted(self.source_ids, source_id))
        if node >= len(self.source_ids) or self.source_ids[node] != source_id:
            raise KeyError(int(source_id))
        return node

    def crossing_record(self, track, partner):
        """Return the directed CSR record for track -> partner."""
        track = int(track)
        partner = int(partner)
        if not self.graph.has_node(track):
            raise IndexError(f"track graph has no node {track}")
        begin = int(self.offsets[track])
        end = int(self.offsets[track + 1])
        local_partners = self.partners[begin:end]
        slot = int(np.searchsorted(local_partners, partner))
        if slot >= len(local_partners) or local_partners[slot] != partner:
            raise KeyError((track, partner))
        return begin + slot

    def _bounded_path_to_root(
            self, current, root, remaining_new_tracks, forbidden):
        """Return one simple path suffix to root, or None."""
        for neighbor in self.graph.neighbors(current):
            neighbor = int(neighbor)
            if neighbor == root:
                return (current, root)
            if remaining_new_tracks <= 0 or neighbor in forbidden:
                continue
            forbidden.add(neighbor)
            suffix = self._bounded_path_to_root(
                neighbor, root, remaining_new_tracks - 1, forbidden)
            forbidden.remove(neighbor)
            if suffix is not None:
                return (current, *suffix)
        return None

    def transition_return_cycle_witness(
            self, original, current, candidate, *, visited=None,
            minimum_candidate_travel=20.0, max_cycle_tracks=4):
        """Return a witness proving a candidate could close to original.

        ``visited`` is the current simple walk path from ``original`` through
        ``current``. The witness contains that prefix, ``candidate``, and a
        simple non-backtracking suffix back to ``original``. Only the first
        exit on ``candidate`` is subject to ``minimum_candidate_travel``.
        """
        original = int(original)
        current = int(current)
        candidate = int(candidate)
        max_cycle_tracks = int(max_cycle_tracks)
        minimum_candidate_travel = float(minimum_candidate_travel)
        if (not np.isfinite(minimum_candidate_travel)
                or minimum_candidate_travel < 0):
            raise ValueError(
                "minimum candidate travel must be finite and non-negative")
        if max_cycle_tracks < 3:
            return None
        if visited is None:
            if current != original:
                raise ValueError(
                    "visited is required when current is not original")
            visited = (original,)
        else:
            visited = tuple(map(int, visited))
        if (not visited or visited[0] != original
                or visited[-1] != current):
            raise ValueError(
                "visited must run from original through current")
        if len(set(visited)) != len(visited):
            raise ValueError("visited must not repeat tracks")
        if candidate in visited:
            return None
        if len(visited) + 1 > max_cycle_tracks:
            return None

        try:
            entry_record = self.crossing_record(candidate, current)
        except KeyError:
            return None
        entry_position = float(self.positions[entry_record])
        forbidden = set(visited)
        forbidden.add(candidate)
        remaining_after_candidate = (
            max_cycle_tracks - len(visited) - 1)
        begin = int(self.offsets[candidate])
        end = int(self.offsets[candidate + 1])
        for exit_record in range(begin, end):
            exit_partner = int(self.partners[exit_record])
            if exit_partner == current:
                continue
            if abs(float(self.positions[exit_record]) - entry_position) \
                    < minimum_candidate_travel:
                continue
            if exit_partner == original:
                return (*visited, candidate, original)
            if (remaining_after_candidate <= 0
                    or exit_partner in forbidden):
                continue
            suffix_forbidden = set(forbidden)
            suffix_forbidden.add(exit_partner)
            suffix = self._bounded_path_to_root(
                exit_partner,
                original,
                remaining_after_candidate - 1,
                suffix_forbidden,
            )
            if suffix is not None:
                return (*visited, candidate, *suffix)
        return None

    def transition_has_return_cycle(
            self, original, current, candidate, *, visited=None,
            minimum_candidate_travel=20.0, max_cycle_tracks=4):
        """Whether candidate passes the root-return quality gate."""
        return self.transition_return_cycle_witness(
            original,
            current,
            candidate,
            visited=visited,
            minimum_candidate_travel=minimum_candidate_travel,
            max_cycle_tracks=max_cycle_tracks,
        ) is not None

    def gated_random_walk(
            self, original, steps, *, rng=None,
            minimum_candidate_travel=20.0, max_cycle_tracks=4):
        """Randomly extend a simple walk using the root-return quality gate."""
        original = int(original)
        steps = int(steps)
        if steps < 0:
            raise ValueError("walk steps must be non-negative")
        if not self.graph.has_node(original):
            raise IndexError(f"track graph has no node {original}")
        rng = np.random.default_rng() if rng is None else rng
        visited = [original]
        for _ in range(steps):
            current = visited[-1]
            eligible = [
                int(candidate)
                for candidate in self.graph.neighbors(current)
                if self.transition_has_return_cycle(
                    original,
                    current,
                    int(candidate),
                    visited=visited,
                    minimum_candidate_travel=minimum_candidate_travel,
                    max_cycle_tracks=max_cycle_tracks,
                )
            ]
            if not eligible:
                break
            visited.append(int(rng.choice(eligible)))
        return tuple(visited)

    def _short_cycle_neighbors(self, node, max_tracks):
        node = int(node)
        max_tracks = int(max_tracks)
        if max_tracks < 3:
            return node, max_tracks, [], {}
        if max_tracks > 4:
            raise ValueError(
                "short-cycle queries support at most four tracks")
        if not self.graph.has_node(node):
            raise IndexError(f"track graph has no node {node}")
        root_neighbors = sorted(set(self.graph.neighbors(node)))
        neighbor_sets = {
            neighbor: set(self.graph.neighbors(neighbor))
            for neighbor in root_neighbors
        }
        return node, max_tracks, root_neighbors, neighbor_sets

    def iter_short_cycles_through(
            self, node, *, max_tracks=4, return_source_ids=False):
        """Yield unique simple cycles through one known starting track.

        The start node is present once in each returned tuple and the closing
        edge back to it is implicit. Cycles are simple (no repeated vertices),
        and choosing the smaller of the start node's two cycle-neighbors first
        removes the reverse-orientation duplicate.
        """
        node, max_tracks, root_neighbors, neighbor_sets = \
            self._short_cycle_neighbors(node, max_tracks)
        root_neighbor_set = set(root_neighbors)

        # node -> left -> right -> node
        for left in root_neighbors:
            for right in sorted(
                    root_neighbor_set.intersection(neighbor_sets[left])):
                if left < right:
                    cycle = (node, left, right)
                    yield (
                        tuple(int(self.source_ids[index]) for index in cycle)
                        if return_source_ids else cycle)

        if max_tracks == 4:
            # node -> left -> middle -> right -> node. Iterating unordered
            # pairs of root neighbors gives each reversed cycle only once.
            for left, right in itertools.combinations(root_neighbors, 2):
                middles = (
                    neighbor_sets[left].intersection(neighbor_sets[right])
                    - {node}
                )
                for middle in sorted(middles):
                    cycle = (node, left, middle, right)
                    yield (
                        tuple(int(self.source_ids[index]) for index in cycle)
                        if return_source_ids else cycle)

    def short_cycles_through(
            self, node, *, max_tracks=4, return_source_ids=False,
            limit=None):
        """Return cycles through one start track, optionally capped by limit."""
        cycles = self.iter_short_cycles_through(
            node, max_tracks=max_tracks,
            return_source_ids=return_source_ids)
        if limit is None:
            return list(cycles)
        limit = int(limit)
        if limit < 0:
            raise ValueError("cycle limit must be non-negative")
        return list(itertools.islice(cycles, limit))

    def count_short_cycles_through(self, node, *, max_tracks=4):
        """Count short cycles through one start without materializing them."""
        node, max_tracks, root_neighbors, neighbor_sets = \
            self._short_cycle_neighbors(node, max_tracks)
        root_neighbor_set = set(root_neighbors)
        triangle_count = sum(
            sum(left < right for right in (
                root_neighbor_set.intersection(neighbor_sets[left])))
            for left in root_neighbors
        )
        four_track_count = 0
        if max_tracks == 4:
            for left, right in itertools.combinations(root_neighbors, 2):
                four_track_count += len(
                    neighbor_sets[left].intersection(neighbor_sets[right])
                    - {node})
        return {
            3: triangle_count,
            4: four_track_count,
            "total": triangle_count + four_track_count,
        }

    def _selected_records(self, selected_source_ids):
        """Return selected-row records with partners remapped locally."""
        selected_source_ids = np.asarray(
            selected_source_ids, dtype=np.uint64)
        rows = np.searchsorted(self.source_ids, selected_source_ids)
        valid = rows < len(self.source_ids)
        if not np.all(valid):
            raise ValueError(
                "track graph does not contain every selected track")
        if not np.array_equal(
                self.source_ids[rows], selected_source_ids):
            raise ValueError(
                "track graph does not contain every selected track")

        graph_to_selected = np.full(
            len(self.source_ids), -1, dtype=np.int32)
        graph_to_selected[rows] = np.arange(
            len(rows), dtype=np.int32)
        counts = self.offsets[rows + 1] - self.offsets[rows]
        record_rows = np.repeat(
            np.arange(len(rows), dtype=np.int32), counts)
        record_starts = np.repeat(self.offsets[rows], counts)
        local_starts = np.repeat(
            np.cumsum(np.r_[0, counts[:-1]], dtype=np.int64), counts)
        record_indices = record_starts + (
            np.arange(int(counts.sum()), dtype=np.int64) - local_starts)
        partners = graph_to_selected[self.partners[record_indices]]
        keep = partners >= 0
        return (
            selected_source_ids,
            record_rows[keep],
            partners[keep],
            record_indices[keep],
        )

    @staticmethod
    def _encode_csr(
            source_ids, rows, partners, self_local, partner_local,
            positions, clearances):
        counts = np.bincount(rows, minlength=len(source_ids))
        offsets = np.empty(len(source_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return {
            "source_ids": np.asarray(source_ids, dtype=np.uint64).copy(),
            "offsets": offsets,
            "partners": np.asarray(partners, dtype=np.int32),
            "self_local": np.asarray(self_local, dtype=np.int32),
            "partner_local": np.asarray(
                partner_local, dtype=np.int32),
            "positions": np.asarray(positions, dtype=np.float64),
            "clearances": np.asarray(clearances, dtype=np.float64),
        }

    def restricted_csr(self, selected_source_ids):
        """Restrict crossings to selected tracks without changing points."""
        source_ids, rows, partners, records = self._selected_records(
            selected_source_ids)
        return self._encode_csr(
            source_ids, rows, partners,
            self.self_local[records],
            self.partner_local[records],
            self.positions[records],
            self.clearances[records],
        )

    def clipped_csr(
            self, selected_source_ids, input_offsets, surviving_rows,
            old_point_to_new, output_offsets):
        """Restrict crossings and remap their endpoints after point clipping.

        ``old_point_to_new`` maps the selected tracks' original packed point
        rows to their rows in the compacted output, with ``-1`` for excluded
        points. ``surviving_rows`` maps compacted tracks back to selected rows.
        """
        input_offsets = np.asarray(input_offsets, dtype=np.int64)
        surviving_rows = np.asarray(surviving_rows, dtype=np.int64)
        old_point_to_new = np.asarray(old_point_to_new)
        if old_point_to_new.dtype.kind != "i":
            raise ValueError("point remap must have an integer dtype")
        output_offsets = np.asarray(output_offsets, dtype=np.int64)
        selected_source_ids = np.asarray(
            selected_source_ids, dtype=np.uint64)
        if input_offsets.shape != (len(selected_source_ids) + 1,):
            raise ValueError("input offsets are not parallel to selected tracks")
        if output_offsets.shape != (len(surviving_rows) + 1,):
            raise ValueError("output offsets are not parallel to surviving tracks")
        if old_point_to_new.shape != (int(input_offsets[-1]),):
            raise ValueError("point remap does not cover the selected tracks")

        _, rows, partners, records = self._selected_records(
            selected_source_ids)
        selected_to_output = np.full(
            len(selected_source_ids), -1, dtype=np.int32)
        selected_to_output[surviving_rows] = np.arange(
            len(surviving_rows), dtype=np.int32)
        output_rows = selected_to_output[rows]
        output_partners = selected_to_output[partners]

        old_self_points = (
            input_offsets[rows] + self.self_local[records])
        old_partner_points = (
            input_offsets[partners] + self.partner_local[records])
        valid_bounds = (
            (old_self_points < input_offsets[rows + 1])
            & (old_partner_points < input_offsets[partners + 1])
        )
        mapped_self = np.full(len(records), -1, dtype=np.int64)
        mapped_partner = np.full(len(records), -1, dtype=np.int64)
        mapped_self[valid_bounds] = old_point_to_new[
            old_self_points[valid_bounds]]
        mapped_partner[valid_bounds] = old_point_to_new[
            old_partner_points[valid_bounds]]
        keep = (
            valid_bounds
            & (output_rows >= 0)
            & (output_partners >= 0)
            & (mapped_self >= 0)
            & (mapped_partner >= 0)
        )

        output_rows = output_rows[keep]
        output_partners = output_partners[keep]
        records = records[keep]
        new_self_local = (
            mapped_self[keep] - output_offsets[output_rows])
        new_partner_local = (
            mapped_partner[keep] - output_offsets[output_partners])
        return self._encode_csr(
            selected_source_ids[surviving_rows],
            output_rows,
            output_partners,
            new_self_local,
            new_partner_local,
            self.positions[records],
            self.clearances[records],
        )
