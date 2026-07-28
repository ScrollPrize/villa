"""Packed track-crossing graph with stable track and point identities."""

from __future__ import annotations

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
