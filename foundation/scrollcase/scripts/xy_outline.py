"""Outline of a scroll's projection along z.

Outlines the union of the x,y coordinates of every point inside the scroll.
For a closed surface that equals the projection of the surface itself, so the
mesh is subdivided finer than the grid, its vertices are binned into a 2D
occupancy grid, holes are filled, and the boundary is traced with marching
squares and resampled to an even arc-length step.

Coordinates are those of the input STL, i.e. for scrollcase output the case
frame (z along the case axis, origin on the axis at the base of the scroll).
The traced outline lies within about half a grid cell of the true projected
boundary; pass --margin to grow it outwards.

Usage:
    python scripts/xy_outline.py <final-dir-or-stl>... --output <dir>

Directories are searched recursively for ``*_scroll.stl``. For each scroll,
``<id>_xy_outline.csv`` and ``<id>_xy_outline.png`` are written to the output
directory, plus ``xy_outlines_overview.png`` covering all of them.
"""

import argparse
import csv
from pathlib import Path

import contourpy
import matplotlib as mpl
import numpy as np
import trimesh
from scipy import ndimage

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_scroll_stls(paths: list[Path]) -> list[Path]:
    """Expand directories into the ``*_scroll.stl`` files they contain."""
    stls = []
    for path in paths:
        if path.is_dir():
            stls.extend(sorted(path.rglob("*_scroll.stl")))
        else:
            stls.append(path)
    return stls


def occupancy_grid(
    mesh: trimesh.Trimesh, cell: float, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise the z-projection of a closed mesh.

    Returns the boolean grid (indexed [iy, ix]) and the x,y of cell (0, 0)'s
    centre.
    """
    # Edges shorter than half a cell guarantee the projected surface leaves no
    # gaps between occupied cells.
    vertices, _ = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=cell / 2
    )
    xy = vertices[:, :2]
    pad = margin + 2 * cell
    origin = np.floor((xy.min(axis=0) - pad) / cell) * cell
    shape = np.ceil((xy.max(axis=0) + pad - origin) / cell).astype(int) + 1
    ix, iy = np.floor((xy - origin) / cell + 0.5).astype(int).T
    grid = np.zeros((shape[1], shape[0]), dtype=bool)
    grid[iy, ix] = True
    grid = ndimage.binary_fill_holes(grid)
    if margin > 0:
        r = margin / cell
        n = int(np.ceil(r))
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        grid = ndimage.binary_dilation(grid, structure=xx**2 + yy**2 <= r**2)
    return grid, origin


def trace_outlines(grid: np.ndarray, origin: np.ndarray, cell: float) -> list[np.ndarray]:
    """Closed boundary polylines of the occupied region, largest first, CCW."""
    labels, n = ndimage.label(grid)
    outlines = []
    for i in range(1, n + 1):
        gen = contourpy.contour_generator(
            z=(labels == i).astype(float), line_type=contourpy.LineType.Separate
        )
        lines = gen.lines(0.5)
        # Holes are filled, so the component has a single (outer) boundary.
        line = max(lines, key=len)
        # contourpy returns (column, row) = (ix, iy) grid coordinates.
        pts = line * cell + origin
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if polygon_area(pts) < 0:
            pts = pts[::-1]
        outlines.append(pts)
    outlines.sort(key=polygon_area, reverse=True)
    return outlines


def polygon_area(pts: np.ndarray) -> float:
    """Signed shoelace area of a closed polygon (positive if CCW)."""
    x, y = pts.T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced points along a closed polyline, spacing at most ``step``."""
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(3, int(np.ceil(s[-1] / step)))
    t = np.arange(n) * s[-1] / n
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def perimeter(pts: np.ndarray) -> float:
    """Length of a closed polyline."""
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def write_csv(path: Path, scroll_id: str, outlines: list[np.ndarray]) -> None:
    """Write the outline points, one row per point."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scroll_id", "contour", "point", "x_mm", "y_mm"])
        for c, pts in enumerate(outlines):
            for p, (x, y) in enumerate(pts):
                writer.writerow([scroll_id, c, p, f"{x:.3f}", f"{y:.3f}"])


def plot_outline(ax, mesh, grid, origin, cell, outlines, scroll_id) -> None:
    """Draw the occupancy grid, mesh footprint coloured by z, and outline."""
    h, w = grid.shape
    extent = [
        origin[0] - cell / 2,
        origin[0] + (w - 0.5) * cell,
        origin[1] - cell / 2,
        origin[1] + (h - 0.5) * cell,
    ]
    ax.imshow(
        np.where(grid, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=mpl.colors.ListedColormap(["#f5e9cf"]),
        interpolation="nearest",
    )
    v = mesh.vertices
    order = np.argsort(v[:, 2])
    # Drop the near-white end of Blues so low points stay visible on the fill.
    blues = mpl.colors.ListedColormap(mpl.colormaps["Blues"](np.linspace(0.35, 1, 256)))
    sc = ax.scatter(
        v[order, 0], v[order, 1], c=v[order, 2], s=0.3, cmap=blues, linewidths=0
    )
    for pts in outlines:
        loop = np.vstack([pts, pts[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="#c0392b", lw=1)
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=2, color="#c0392b")
    ax.axhline(0, color="#888", lw=0.5)
    ax.axvline(0, color="#888", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    n_pts = sum(len(p) for p in outlines)
    area = sum(polygon_area(p) for p in outlines)
    ax.set_title(
        f"{scroll_id}: {len(outlines)} contour(s), {n_pts} pts, "
        f"area {area:.0f} mm², z {v[:, 2].min():.1f}–{v[:, 2].max():.1f} mm",
        fontsize=9,
    )
    return sc


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="scroll STLs or dirs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=float, default=0.5, help="grid cell (mm)")
    parser.add_argument("--step", type=float, default=1.0, help="max point spacing (mm)")
    parser.add_argument("--margin", type=float, default=0.0, help="outward offset (mm)")
    args = parser.parse_args()

    stls = find_scroll_stls(args.inputs)
    if not stls:
        parser.error("no *_scroll.stl found")
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for stl in stls:
        scroll_id = stl.stem.removesuffix("_scroll")
        mesh = trimesh.load_mesh(stl)
        if not mesh.is_watertight:
            print(f"warning: {stl} is not watertight; holes in the projection are filled anyway")
        grid, origin = occupancy_grid(mesh, args.cell, args.margin)
        outlines = [resample_closed(p, args.step) for p in trace_outlines(grid, origin, args.cell)]
        write_csv(args.output / f"{scroll_id}_xy_outline.csv", scroll_id, outlines)

        fig, ax = plt.subplots(figsize=(7, 7))
        sc = plot_outline(ax, mesh, grid, origin, args.cell, outlines, scroll_id)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="vertex z (mm)")
        fig.savefig(args.output / f"{scroll_id}_xy_outline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        results.append((scroll_id, mesh, grid, origin, outlines))
        lengths = ", ".join(f"{perimeter(p):.1f}" for p in outlines)
        print(f"{scroll_id}: {len(outlines)} contour(s), perimeter {lengths} mm")

    n = len(results)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax, (scroll_id, mesh, grid, origin, outlines) in zip(axes.flat, results):
        plot_outline(ax, mesh, grid, origin, args.cell, outlines, scroll_id)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "xy_outlines_overview.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
