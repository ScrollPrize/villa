"""Layout of exported Spiral surface grids.

Surfaces are sampled in spiral order: rows by increasing z, columns by
increasing spiral-space theta. Spiral-space theta always runs outwards (the
ACW mirror is applied before spiral space), so spiral order is innermost wrap
first whatever the sense.

Exports use the orient-segment convention instead: outermost wrap in column 0
(always, by reversing columns) and the scroll top in row 0 (by reversing rows
when z runs bottom to top; left in z order when the z direction is unknown).
Grids are sampled and spliced in spiral order and put in the export layout
only when written; an exported grid's metadata records the z direction under
METADATA_KEY, and a grid without the key is in spiral order.

Handedness never changes the grid: outer-to-inner and top-to-bottom are
physical directions, so the layout is right either way. It only flips the
grid's cross-product normal, which renders correct with
flip-normals = not left_handed_coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


METADATA_KEY = "grid_orientation"


def spiral_outward_sense_for(z_direction_is_top_to_bottom: bool,
                             left_handed_coordinates: bool) -> str:
    """Catalog convention (every scroll shows the same spiral seen from its
    top), as orient-segment applies it: in a right-handed, top-to-bottom volume
    the spiral winds outwards against atan2(y, x), the mirror of the fit's
    canonical spiral (CW, radius growing with atan2(y, x)), so ACW. Each
    property flips that."""
    return "ACW" if z_direction_is_top_to_bottom != left_handed_coordinates else "CW"


def export_metadata(z_direction_is_top_to_bottom: Optional[bool]) -> dict[str, Any]:
    """The METADATA_KEY value of a grid written in the export layout."""
    return {"z_direction_is_top_to_bottom": z_direction_is_top_to_bottom}


@dataclass(frozen=True)
class GridLayout:
    """Which axes of a stored grid are reversed relative to spiral order."""

    reverse_columns: bool = False
    reverse_rows: bool = False

    @classmethod
    def export(cls, z_direction_is_top_to_bottom: Optional[bool]) -> "GridLayout":
        return cls(reverse_columns=True,
                   reverse_rows=z_direction_is_top_to_bottom is False)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "GridLayout":
        if metadata.get(METADATA_KEY) is None:
            return cls()
        return cls.export(metadata[METADATA_KEY].get("z_direction_is_top_to_bottom"))

    def apply(self, grid):
        """Flip a numpy or torch grid (rows, columns, ...) between spiral order
        and this layout; each flip is its own inverse."""
        dims = [axis for axis, flip in ((0, self.reverse_rows), (1, self.reverse_columns)) if flip]
        if not dims:
            return grid
        if isinstance(grid, np.ndarray):
            return np.ascontiguousarray(np.flip(grid, axis=tuple(dims)))
        return grid.flip(dims)

    def rows(self, rows, height):
        return height - 1 - rows if self.reverse_rows else rows

    def columns(self, columns, width):
        return width - 1 - columns if self.reverse_columns else columns

    def column_range(self, begin: int, end: int, width: int) -> list[int]:
        """Map a half-open column range between spiral order and this layout."""
        if self.reverse_columns:
            return [int(width - end), int(width - begin)]
        return [int(begin), int(end)]
