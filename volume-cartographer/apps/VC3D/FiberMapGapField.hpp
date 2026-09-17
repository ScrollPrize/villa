#pragma once

#include <QRectF>

#include <cstddef>
#include <vector>

#include "FiberNetworkLayout.hpp"

// The Fiber Map's gap heat map: for every cell of the global map's extent, a
// surrogate for the shortest distance from that spot on the sheet to any
// drawn fiber, counting fibers on neighbouring windings.
//
// The global map draws fibers at x = (s*theta + 2*pi*k) * rRef, y = z, so one
// winding is exactly one period P = 2*pi*rRef of x, and "the same angle on the
// winding k away" is the map position x + k*P. GlobalResult also carries the
// Archimedean sheet model r(W) = radius0 + pitch*W with sheetDistanceVx(), a
// monotonic warp u(x) in which a horizontal step is arclength (the integral of
// r dtheta) rather than the map's rRef-scaled angle. The field is
//
//     D(cell) = min over k of  sqrt( S_k^2 + (|k| * acrossWeight * pitch)^2 )
//
// where S_k is the distance, in (u, z), from the cell's position shifted k
// periods to the nearest rasterised fiber, and |k|*pitch is the modelled radial
// separation of k sheets. acrossWeight = 1 reads as the model's own radial
// spacing; larger values push the field toward per-winding coverage (only a
// fiber on the cell's own winding counts). acrossWeight == 0, or a degenerate
// sheet model (pitch 0), disables the fold: D = S_0, and `folded` is false.
//
// What this is and is not. It is one number per cell, in voxels, exact up to
// the rasterisation (fibers are drawn one cell wide, distances are read from
// an exact per-cell Euclidean distance transform and interpolated between
// columns) and clamped at `saturationVx`. It is NOT a 3D chord distance: the in-sheet part is
// arclength of r dtheta (not of the spiral's true arc), the radial part is the
// global linear model's spacing (not the local one), and S_k is measured at the
// arc scale of the winding the query lands on rather than the cell's own,
// which overstates by about |k|*pitch/r. Cells where the model's radius is not
// positive have no meaningful sheet position and read NaN.
//
// Coverage is the layout's padded extent (x0Vx..x1Vx, yMinVx..yMaxVx), not the
// scroll's full height. Seeds are the runs of every placed fiber whose anchor
// is resolved (GlobalAnchor::Unresolved fibers sit at an arbitrary winding and
// never seed); runs shorter than two points are not drawn and do not seed
// either; interpolated runs seed only when `seedInterpolated`.
//
// Everything here is voxels, like the layout; the intents behind the defaults
// are noted per field at 2.4 um/voxel. Deterministic: every cell is a pure
// function of its own inputs, so the thread count never changes the output.
namespace vc3d::fiber_map::gaps
{

struct GapFieldParams {
    // Output and raster cell. Intent: 0.05 cm.
    double cellVx = 208.0;
    // Distances clamp here; also bounds the fold search. Intent: 1 cm.
    double saturationVx = 4167.0;
    // Multiplier on the model pitch for the across-sheet term (0 disables).
    double acrossWeight = 1.0;
    bool seedInterpolated = true;
    // Hard cap on |k|. The saturation already bounds k at ceil(sat/across)-1;
    // this guards a tiny fitted pitch. foldTruncated reports when it may
    // have bitten.
    int maxFoldWindings = 128;
    // Budget for cols*rows of the output and of the seed raster; the cell
    // doubles until both fit (cellCoarsened reports it).
    std::size_t maxCells = 24000000;
};

struct GapField {
    // Grid origin is the extent's corner: cell (row i, col j) is centred at
    // (x0Vx + (j + 0.5) * cellVx, y0Vx + (i + 0.5) * cellVx), y being +z like
    // every placed coordinate (the scene negates y once when drawing).
    double x0Vx = 0.0;
    double y0Vx = 0.0;
    double cellVx = 0.0;
    int cols = 0;
    int rows = 0;
    // Row-major, rows * cols; clamped to saturationVx; NaN outside the model's
    // valid domain.
    std::vector<float> distanceVx;
    double saturationVx = 0.0;
    // False when the fold was disabled (acrossWeight or pitch is 0).
    bool folded = false;
    // Conservative: some cell's search stopped at maxFoldWindings while an
    // omitted winding both lands in the seed raster and has an across term
    // below that cell's result, so a closer seed there is possible (not
    // established - the in-sheet part of the omitted candidates is never
    // evaluated). False means the cap provably changed nothing.
    bool foldTruncated = false;
    // cellVx is larger than requested because of maxCells.
    bool cellCoarsened = false;
    int seedFiberCount = 0;
    int skippedUnresolvedCount = 0;

    [[nodiscard]] bool empty() const { return distanceVx.empty(); }
    [[nodiscard]] float at(int row, int col) const
    {
        return distanceVx[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                          static_cast<std::size_t>(col)];
    }
};

// Throws std::invalid_argument for non-finite or non-positive cell/saturation,
// negative or non-finite acrossWeight, negative maxFoldWindings, or a cell
// budget no cell size can meet (the seed raster is never narrower than a few
// cells). An empty layout (no fibers, or a degenerate extent or reference
// radius) yields an empty field.
[[nodiscard]] GapField buildGapField(const GlobalResult& layout, const GapFieldParams& params);

// Whether two captures of the heat-map settings would build the same field:
// both off, or both on with equal parameters. This is the workspace's one
// test for "the published field is the one the toolbar asks for", used both
// to show a retained field again without a rebuild and to keep a rebuild
// queued behind a running build when the settings moved while it ran.
[[nodiscard]] inline bool sameGapSettings(bool wantA, const GapFieldParams& a, bool wantB,
                                          const GapFieldParams& b)
{
    if (wantA != wantB) {
        return false;
    }
    if (!wantA) {
        return true;
    }
    return a.cellVx == b.cellVx && a.saturationVx == b.saturationVx &&
           a.acrossWeight == b.acrossWeight && a.seedInterpolated == b.seedInterpolated &&
           a.maxFoldWindings == b.maxFoldWindings && a.maxCells == b.maxCells;
}

// One pixmap tile of the field: columns [colBegin, colEnd) and the scene rect
// they fill, in scene coordinates (y = -z, so the rect's top is the field's
// last row). Image row 0 of a tile is field row rows-1.
struct GapFieldTile {
    int colBegin = 0;
    int colEnd = 0;
    QRectF sceneRect;
};

// Splits the field into tiles of at most maxTileCols columns (>= 1), left to
// right, edge to edge. An empty field yields no tiles.
[[nodiscard]] std::vector<GapFieldTile> gapFieldTiles(const GapField& field, int maxTileCols);

}  // namespace vc3d::fiber_map::gaps
