#pragma once

#include <cstdint>
#include <optional>
#include <vector>

// The Fiber Map's kollesis seams: where one papyrus sheet was glued onto the
// next. H fibers end at a seam, and an annotator tags that end with
// kollesis_termination; this reads those tags back into seams, and from the
// sheet lengths between them predicts where seams nobody has tagged yet
// should be.
//
// In the unrolled map a seam is vertical: one sheet distance, the full scroll
// height. A tag at a fiber's smaller-x end says the sheet STARTS there (a
// left termination); at its larger-x end, that the sheet ENDS there (a right
// termination). Where two sheets overlap at a kollesis the left terminations
// of the newer sheet and the right terminations of the older one lie close in
// x, so terminations are grouped by proximity along the map. Within a group
// each side's bound is the vertical through that side's mean x; the seam is
// the vertical centred between the two bounds, or the one bound when only one
// side was tagged.
//
// Prediction: the gaps between neighbouring seams are read as whole numbers
// of sheets. The unit sheet length is estimated from the gaps (a low
// percentile as the seed, then the per-sheet mean once every gap has been
// assigned its sheet count), and a gap of k >= 2 sheets gets k - 1 predicted
// seams spaced evenly inside it. Beyond the last seam, the requested number of
// further seams are extrapolated one unit apart. Each prediction carries a
// spread: the sample deviation of the per-sheet lengths, growing with the
// square root of the sheets stepped for extrapolations.
//
// Nothing here knows about Qt scenes; the caller supplies both the map x
// (arclength at the reference radius, in which the grouping gap is a number
// of windings) and the scene x (sheet distance, in which seams are placed and
// lengths read).
namespace vc3d::fiber_map::kollesis
{

struct Termination {
    uint64_t fiberId = 0;
    int controlIndex = -1;
    // Map x (theta * rRef): the grouping coordinate, since the gap threshold
    // is a number of windings.
    double xMapVx = 0.0;
    // Scene x (sheet distance from winding 0): where the seam is placed.
    double xSceneVx = 0.0;
    // Scroll z, voxels, growing upward (kept for the tooltip and tests; the
    // bounds are vertical and do not use it).
    double zVx = 0.0;
    // Tagged at the fiber's smaller-x end (the sheet starts here) rather
    // than its larger-x end (the sheet ends here).
    bool left = false;
};

// One side's edge: the vertical through the side's mean scene x.
struct Bound {
    double xVx = 0.0;
    int count = 0;
};

struct Seam {
    std::optional<Bound> left;
    std::optional<Bound> right;
    // Every termination of the group, ascending in xMapVx.
    std::vector<Termination> members;
    // The seam's scene x: centred between the bounds, else the one bound.
    [[nodiscard]] double xVx() const;
    [[nodiscard]] bool hasBand() const { return left && right; }
};

// What the gaps between seams said about the sheet length.
struct SheetStatistics {
    // Gaps between neighbouring seams (sheetLengthVx), read as this many
    // sheets each; sheetCounts.size() == gap count.
    std::vector<int> sheetCounts;
    // The unit sheet length: the mean of every gap divided by its sheet
    // count, weighted by sheet count (total gap length over total sheets).
    double unitLengthVx = 0.0;
    // Sample standard deviation of the per-gap sheet lengths (gap / count);
    // 0 with fewer than two gaps.
    double spreadVx = 0.0;
    [[nodiscard]] bool valid() const { return unitLengthVx > 0.0; }
};

struct PredictedSeam {
    double xVx = 0.0;
    // +- this much, scene voxels (0 when the data gave no spread).
    double spreadVx = 0.0;
    // Beyond the last tagged seam (else inside a gap between two).
    bool extrapolated = false;
    // Interior: the gap (index into Model::sheetLengthVx) it sits in and its
    // sheet number within that gap, 1-based. Extrapolated: sheets past the
    // last seam, 1-based, gapIndex -1.
    int gapIndex = -1;
    int step = 0;
};

struct Params {
    // Terminations farther apart than this along the map start a new seam.
    double gapWindings = 0.75;
    // Map x per winding (2 * pi * rRef). A non-positive width groups nothing:
    // every distinct x is its own seam.
    double windingWidthVx = 0.0;
    // Interior predictions on gaps read as several sheets.
    bool predictInterior = true;
    // How many seams to extrapolate past the last tagged one (0: none).
    int extrapolateCount = 0;
};

struct Model {
    // Ascending in scene x.
    std::vector<Seam> seams;
    // sheetLengthVx[i] is the gap between seams i and i+1; seams.size() - 1
    // entries, empty with fewer than two seams.
    std::vector<double> sheetLengthVx;
    SheetStatistics statistics;
    // Ascending in scene x; empty without statistics.
    std::vector<PredictedSeam> predicted;
};

// The vertical through the points' mean scene x. Empty input is count 0 at
// x 0.
[[nodiscard]] Bound fitBound(const std::vector<Termination>& points);

// Sheet counts and unit length from the gaps; invalid (unit 0) without a
// positive gap.
[[nodiscard]] SheetStatistics estimateSheetStatistics(const std::vector<double>& gapsVx);

[[nodiscard]] Model buildModel(std::vector<Termination> terminations, const Params& params);

}  // namespace vc3d::fiber_map::kollesis
