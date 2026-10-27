#pragma once

#include "vc/core/types/SheetCollection.hpp"

#include <cstdint>
#include <vector>

// Experimental sheet stitcher: fuse several .asv sheet meshes (each a nu*nv grid
// of native-L0 XYZ) into ONE coherent surface grid that a VC3D segment
// (QuadSurface / tifxyz) can be built from. The method is a feathered splat-blend
// into a shared tangent-plane parametrisation:
//   1. PCA a common frame (e1, e2 in-plane; n = surface normal) from all vertices.
//   2. Project every vertex to (u, v) in that plane; rasterise a shared grid at the
//      sheets' native spacing.
//   3. Splat each vertex bilinearly, weighted by a feather ramp that falls to 0 at
//      the sheet's footprint boundary, so overlaps become a seam-free weighted
//      average while interiors keep each sheet's own geometry.
// This is deliberately a first, swappable version (the UI calls only stitchSheets);
// it assumes the selected sheets are locally co-oriented (a height field over one
// plane). Strongly curved unions become multivalued over the plane -- reported via
// StitchResult::foldCells rather than silently mangled.
namespace vc::sheets
{

struct StitchParams {
    double cell = 0.0;      // (u, v) grid spacing in L0 voxels; <= 0 => auto (median sheet spacing)
    double feather = 4.0;   // feather radius in grid cells for overlap blending
    double margin = 1.0;    // extra cells padded around the union
};

// A stitched surface grid: rows*cols nodes, row-major (row = v, col = u). Invalid
// nodes (no contribution) are left as the sentinel {-1, -1, -1}, matching what the
// QuadSurface loader treats as an empty cell. `cell` is the (u, v) spacing in L0
// voxels, so the QuadSurface scale is 1/cell on both axes.
struct StitchResult {
    int cols = 0;           // grid width  (u direction)
    int rows = 0;           // grid height (v direction)
    double cell = 0.0;      // chosen (u, v) spacing in L0 voxels
    std::vector<Point> grid;      // size rows*cols; node(r, c) = grid[r*cols + c]
    std::vector<uint8_t> valid;   // size rows*cols; 1 where a point was blended in

    // The tangent frame used for the parametrisation (for provenance / debugging).
    Point origin{}, e1{}, e2{}, n{};

    // Diagnostics the UI can surface.
    int inputSheets = 0;
    int64_t inputVerts = 0;
    int filledCells = 0;
    int foldCells = 0;          // nodes whose contributions span > ~1 cell along n (fold / multivalued)
    double maxSpreadVox = 0.0;  // worst within-node spread along n (L0 voxels)

    bool ok() const { return cols > 0 && rows > 0 && filledCells > 0; }
    const Point& at(int r, int c) const { return grid[size_t(r) * cols + c]; }
    bool validAt(int r, int c) const { return valid[size_t(r) * cols + c] != 0; }
};

// Stitch the given sheet meshes into one grid. Empty input (or all-empty footprints)
// yields a result with ok() == false. Never throws for ordinary input.
StitchResult stitchSheets(const std::vector<Mesh>& sheets, const StitchParams& params = {});

}  // namespace vc::sheets
