#include "FiberMapGapField.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vc3d::fiber_map::gaps
{

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
constexpr float kOutside = std::numeric_limits<float>::quiet_NaN();

// One rasterised polyline segment in (u, z) voxels.
struct Segment {
    double uA = 0.0;
    double yA = 0.0;
    double uB = 0.0;
    double yB = 0.0;
};

int cellIndex(double coordinate, double origin, double cell, int count)
{
    const double raw = std::floor((coordinate - origin) / cell);
    if (!(raw >= 0.0)) {
        return 0;
    }
    const double capped = std::min(raw, static_cast<double>(count - 1));
    return static_cast<int>(capped);
}

// Continuous raster column coordinate of u (integer values are column
// centres), or NaN when u lies more than half a cell outside the raster,
// where by construction of the raster's halo every seed is beyond the
// saturation distance.
float rasterColumnOf(double u, double uRaster0, double cell, int rasterCols)
{
    const double g = (u - uRaster0) / cell - 0.5;
    if (!(g >= -0.5) || !(g <= static_cast<double>(rasterCols) - 0.5)) {
        return kOutside;
    }
    return static_cast<float>(std::clamp(g, 0.0, static_cast<double>(rasterCols - 1)));
}

// Linear interpolation of one raster row of the distance transform between
// its two nearest columns (rows coincide with the output grid, so nothing
// interpolates in z).
float sampleRow(const float* row, int rasterCols, float g)
{
    const int c0 = std::min(static_cast<int>(g), rasterCols - 1);
    const int c1 = std::min(c0 + 1, rasterCols - 1);
    const float t = g - static_cast<float>(c0);
    return row[c0] + t * (row[c1] - row[c0]);
}

// "No seed here" for the squared-distance transform: large and finite, so
// the parabola intersections below stay finite arithmetic.
constexpr double kFar = 1e20;

// Felzenszwalb-Huttenlocher one-dimensional squared distance transform of
// the sampled function f (n samples at unit spacing) into d: d[q] = min over
// p of (q - p)^2 + f[p]. Exact in double, O(n). v and z are scratch of n and
// n + 1 entries. Chosen over cv::distanceTransform because the field must be
// bitwise identical whatever the thread count, and OpenCV's precise
// transform is not.
void squaredDistanceTransform1d(const double* f, int n, double* d, int* v, double* z)
{
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < n; ++q) {
        const double fq = f[q] + static_cast<double>(q) * static_cast<double>(q);
        for (;;) {
            const int p = v[k];
            const double fp = f[p] + static_cast<double>(p) * static_cast<double>(p);
            const double s = (fq - fp) / (2.0 * static_cast<double>(q - p));
            if (s <= z[k] && k > 0) {
                --k;
                continue;
            }
            if (s <= z[k]) {
                // k == 0: the new parabola dominates everything so far.
                v[0] = q;
                z[0] = -std::numeric_limits<double>::infinity();
                z[1] = std::numeric_limits<double>::infinity();
            } else {
                ++k;
                v[k] = q;
                z[k] = s;
                z[k + 1] = std::numeric_limits<double>::infinity();
            }
            break;
        }
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<double>(q)) {
            ++k;
        }
        const double dq = static_cast<double>(q - v[k]);
        d[q] = dq * dq + f[v[k]];
    }
}

// Euclidean distance, in cells, from every cell of the seed mask (0 = seed)
// to the nearest seed cell centre: the separable exact transform, rows then
// columns, each line independent so the thread count cannot change a bit.
std::vector<float> exactDistanceTransform(const cv::Mat& seeds)
{
    const int rows = seeds.rows;
    const int cols = seeds.cols;
    std::vector<double> squared(static_cast<std::size_t>(rows) * cols);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        std::vector<double> f(static_cast<std::size_t>(cols));
        std::vector<int> v(static_cast<std::size_t>(cols));
        std::vector<double> z(static_cast<std::size_t>(cols) + 1);
        for (int i = range.start; i < range.end; ++i) {
            const unsigned char* mask = seeds.ptr<unsigned char>(i);
            for (int j = 0; j < cols; ++j) {
                f[static_cast<std::size_t>(j)] = mask[j] == 0 ? 0.0 : kFar;
            }
            squaredDistanceTransform1d(f.data(), cols,
                                       squared.data() + static_cast<std::size_t>(i) * cols,
                                       v.data(), z.data());
        }
    });
    std::vector<float> distance(squared.size());
    cv::parallel_for_(cv::Range(0, cols), [&](const cv::Range& range) {
        std::vector<double> f(static_cast<std::size_t>(rows));
        std::vector<double> d(static_cast<std::size_t>(rows));
        std::vector<int> v(static_cast<std::size_t>(rows));
        std::vector<double> z(static_cast<std::size_t>(rows) + 1);
        for (int j = range.start; j < range.end; ++j) {
            for (int i = 0; i < rows; ++i) {
                f[static_cast<std::size_t>(i)] = squared[static_cast<std::size_t>(i) * cols + j];
            }
            squaredDistanceTransform1d(f.data(), rows, d.data(), v.data(), z.data());
            for (int i = 0; i < rows; ++i) {
                distance[static_cast<std::size_t>(i) * cols + j] =
                    static_cast<float>(std::sqrt(d[static_cast<std::size_t>(i)]));
            }
        }
    });
    return distance;
}

}  // namespace

GapField buildGapField(const GlobalResult& layout, const GapFieldParams& params)
{
    if (!std::isfinite(params.cellVx) || !(params.cellVx > 0.0)) {
        throw std::invalid_argument("gap field: cell size must be a positive finite length");
    }
    if (!std::isfinite(params.saturationVx) || !(params.saturationVx > 0.0)) {
        throw std::invalid_argument("gap field: saturation must be a positive finite length");
    }
    if (!std::isfinite(params.acrossWeight) || params.acrossWeight < 0.0) {
        throw std::invalid_argument("gap field: across-sheet weight must be finite and >= 0");
    }
    if (params.maxFoldWindings < 0) {
        throw std::invalid_argument("gap field: maxFoldWindings must be >= 0");
    }
    if (params.maxCells == 0) {
        throw std::invalid_argument("gap field: maxCells must be > 0");
    }

    GapField field;
    field.saturationVx = params.saturationVx;
    const SheetModel model = sheetModelOf(layout);
    if (layout.fibers.empty() || !(model.rRefVx > 0.0) || !(layout.x1Vx > layout.x0Vx) ||
        !(layout.yMaxVx > layout.yMinVx)) {
        return field;
    }

    const double period = kTwoPi * model.rRefVx;
    const double across = params.acrossWeight * model.pitchVx;
    const bool fold = std::isfinite(across) && across > 0.0;
    // The model's radius is positive only past this x (no bound when the
    // pitch is zero, where the radius is the constant radius0 > 0 or the
    // fallback rRef).
    const double xMinValid = model.pitchVx > 0.0
        ? -(model.radius0Vx / model.pitchVx) * period
        : -std::numeric_limits<double>::infinity();
    const auto inDomain = [xMinValid](double x) { return x > xMinValid; };

    // Seeds: every drawn run of a resolved fiber, in (u, z); a point outside
    // the model domain breaks the run (it has no sheet position).
    std::vector<Segment> segments;
    double uLo = std::numeric_limits<double>::infinity();
    double uHi = -std::numeric_limits<double>::infinity();
    for (const GlobalPlacedFiber& placed : layout.fibers) {
        if (placed.meta.anchor == GlobalAnchor::Unresolved) {
            ++field.skippedUnresolvedCount;
            continue;
        }
        bool seeded = false;
        for (const Run& run : placed.fiber.runs) {
            if (run.points.size() < 2 || (!run.traced && !params.seedInterpolated)) {
                continue;
            }
            for (std::size_t i = 1; i < run.points.size(); ++i) {
                const QPointF& a = run.points[i - 1];
                const QPointF& b = run.points[i];
                if (!inDomain(a.x()) || !inDomain(b.x())) {
                    continue;
                }
                Segment segment;
                segment.uA = sheetDistanceVx(model, a.x());
                segment.yA = a.y();
                segment.uB = sheetDistanceVx(model, b.x());
                segment.yB = b.y();
                if (!std::isfinite(segment.uA) || !std::isfinite(segment.uB) ||
                    !std::isfinite(segment.yA) || !std::isfinite(segment.yB)) {
                    continue;
                }
                uLo = std::min({uLo, segment.uA, segment.uB});
                uHi = std::max({uHi, segment.uA, segment.uB});
                segments.push_back(segment);
                seeded = true;
            }
        }
        if (seeded) {
            ++field.seedFiberCount;
        }
    }
    const bool haveSeeds = !segments.empty();

    // Grid dimensions under the cell budget. The seed raster spans the seeds'
    // u-extent grown by a halo of saturation + 2 cells (clamped to the model
    // domain), so a query landing outside it is more than the saturation from
    // every seed in u alone and can be skipped exactly.
    double cell = params.cellVx;
    int cols = 0;
    int rows = 0;
    int rasterCols = 0;
    double uRaster0 = 0.0;
    const double uDomainLo = std::isfinite(xMinValid)
        ? sheetDistanceVx(model, xMinValid)
        : -std::numeric_limits<double>::infinity();
    for (int doublings = 0;; ++doublings) {
        // The seed raster is at least the halo's 4 cells wide plus the seeds,
        // so a budget of a handful of cells is never met; and a cell that has
        // doubled past every finite length has stopped meaning anything.
        if (doublings > 64 || !std::isfinite(cell)) {
            throw std::invalid_argument("gap field: maxCells too small for any cell size");
        }
        // Counts stay in double until the budget has passed: a tiny cell
        // against a wide extent would overflow an int long before it fails
        // the budget.
        const auto count = [](double length, double c) {
            return std::max(1.0, std::ceil(length / c));
        };
        const double colsD = count(layout.x1Vx - layout.x0Vx, cell);
        const double rowsD = count(layout.yMaxVx - layout.yMinVx, cell);
        double rasterColsD = 1.0;
        if (haveSeeds) {
            const double halo = params.saturationVx + 2.0 * cell;
            uRaster0 = std::max(uLo - halo, uDomainLo);
            rasterColsD = count((uHi + halo) - uRaster0, cell);
        }
        const double budget = static_cast<double>(params.maxCells);
        if (colsD * rowsD <= budget && rasterColsD * rowsD <= budget) {
            cols = static_cast<int>(colsD);
            rows = static_cast<int>(rowsD);
            rasterCols = static_cast<int>(rasterColsD);
            break;
        }
        // Once the output is a single cell, doubling only nudges the raster
        // toward its 4-cell halo floor: the budget cannot be met at any
        // cell size that still measures anything.
        if (colsD <= 1.0 && rowsD <= 1.0) {
            throw std::invalid_argument("gap field: maxCells too small for any cell size");
        }
        cell *= 2.0;
        field.cellCoarsened = true;
    }
    field.x0Vx = layout.x0Vx;
    field.y0Vx = layout.yMinVx;
    field.cellVx = cell;
    field.cols = cols;
    field.rows = rows;
    field.folded = fold && haveSeeds;
    const double saturation = params.saturationVx;
    field.distanceVx.assign(
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows),
        static_cast<float>(saturation));

    // Out-of-domain cells are NaN whether or not there are seeds.
    std::vector<unsigned char> columnInDomain(static_cast<std::size_t>(cols), 0);
    for (int j = 0; j < cols; ++j) {
        const double x = layout.x0Vx + (static_cast<double>(j) + 0.5) * cell;
        columnInDomain[static_cast<std::size_t>(j)] = inDomain(x) ? 1 : 0;
    }
    if (!haveSeeds) {
        for (int i = 0; i < rows; ++i) {
            float* out = field.distanceVx.data() + static_cast<std::size_t>(i) * cols;
            for (int j = 0; j < cols; ++j) {
                if (!columnInDomain[static_cast<std::size_t>(j)]) {
                    out[j] = kOutside;
                }
            }
        }
        return field;
    }

    // Rasterise the seeds one cell wide and take the distance to them.
    cv::Mat seeds(rows, rasterCols, CV_8UC1, cv::Scalar(255));
    for (const Segment& segment : segments) {
        const cv::Point a(cellIndex(segment.uA, uRaster0, cell, rasterCols),
                          cellIndex(segment.yA, layout.yMinVx, cell, rows));
        const cv::Point b(cellIndex(segment.uB, uRaster0, cell, rasterCols),
                          cellIndex(segment.yB, layout.yMinVx, cell, rows));
        cv::line(seeds, a, b, cv::Scalar(0), 1, cv::LINE_8);
    }
    const std::vector<float> distance = exactDistanceTransform(seeds);

    // The fold search: k with |k| * across >= saturation can never beat the
    // clamp, and the cap guards a tiny fitted pitch.
    int foldWindings = 0;
    bool capBinds = false;
    if (field.folded) {
        const double bySaturation = std::ceil(saturation / across) - 1.0;
        const double capped = std::min(bySaturation, static_cast<double>(params.maxFoldWindings));
        foldWindings = static_cast<int>(std::max(0.0, capped));
        capBinds = bySaturation > static_cast<double>(params.maxFoldWindings);
    }

    // Column shift maps: where output column j lands in the raster when read
    // k windings over, the same for every row.
    const int shiftCount = 2 * foldWindings + 1;
    std::vector<float> columnMap(static_cast<std::size_t>(shiftCount) * cols, kOutside);
    for (int k = -foldWindings; k <= foldWindings; ++k) {
        float* map = columnMap.data() + static_cast<std::size_t>(k + foldWindings) * cols;
        for (int j = 0; j < cols; ++j) {
            const double x =
                layout.x0Vx + (static_cast<double>(j) + 0.5) * cell + static_cast<double>(k) * period;
            if (!inDomain(x)) {
                continue;
            }
            map[j] = rasterColumnOf(sheetDistanceVx(model, x), uRaster0, cell, rasterCols);
        }
    }

    // When the cap binds: per column, the across term of the nearest omitted
    // winding (|k| > foldWindings) whose shifted query would still land in
    // the raster. Only such a winding can hold a seed the search never saw;
    // one landing outside is beyond the saturation by the halo argument.
    // +inf when no omitted winding lands in the raster.
    std::vector<float> omittedAcross(static_cast<std::size_t>(cols),
                                     std::numeric_limits<float>::infinity());
    if (capBinds) {
        const double uRaster1 = uRaster0 + static_cast<double>(rasterCols) * cell;
        double xRasterLo = sheetXForDistanceVx(model, uRaster0);
        double xRasterHi = sheetXForDistanceVx(model, uRaster1);
        if (!std::isfinite(xRasterLo)) {
            xRasterLo = xMinValid;
        }
        if (std::isfinite(xRasterHi)) {
            for (int j = 0; j < cols; ++j) {
                const double x = layout.x0Vx + (static_cast<double>(j) + 0.5) * cell;
                const double kLo = std::ceil((xRasterLo - x) / period);
                const double kHi = std::floor((xRasterHi - x) / period);
                const double beyond = static_cast<double>(foldWindings) + 1.0;
                double nearest = std::numeric_limits<double>::infinity();
                const double outwardLo = std::max(kLo, beyond);
                if (outwardLo <= kHi) {
                    nearest = std::min(nearest, outwardLo);
                }
                const double inwardHi = std::min(kHi, -beyond);
                if (kLo <= inwardHi) {
                    nearest = std::min(nearest, -inwardHi);
                }
                if (std::isfinite(nearest)) {
                    omittedAcross[static_cast<std::size_t>(j)] =
                        static_cast<float>(nearest * across);
                }
            }
        }
    }

    std::vector<unsigned char> rowTruncated(static_cast<std::size_t>(rows), 0);
    const float cellF = static_cast<float>(cell);
    const float saturationF = static_cast<float>(saturation);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const float* row = distance.data() + static_cast<std::size_t>(i) * rasterCols;
            float* out = field.distanceVx.data() + static_cast<std::size_t>(i) * cols;
            bool truncated = false;
            for (int j = 0; j < cols; ++j) {
                if (!columnInDomain[static_cast<std::size_t>(j)]) {
                    out[j] = kOutside;
                    continue;
                }
                float best = saturationF;
                const float g0 = columnMap[static_cast<std::size_t>(foldWindings) * cols + j];
                if (!std::isnan(g0)) {
                    best = std::min(best, sampleRow(row, rasterCols, g0) * cellF);
                }
                int m = 1;
                for (; m <= foldWindings; ++m) {
                    const float acrossM = static_cast<float>(m) * static_cast<float>(across);
                    if (acrossM >= best) {
                        break;
                    }
                    for (const int k : {m, -m}) {
                        const float g =
                            columnMap[static_cast<std::size_t>(k + foldWindings) * cols + j];
                        if (std::isnan(g)) {
                            continue;
                        }
                        const float s = sampleRow(row, rasterCols, g) * cellF;
                        best = std::min(best, std::hypot(s, acrossM));
                    }
                }
                // The cap bit if the search ran off its end while an omitted
                // winding that lands in the raster could still beat best.
                if (capBinds && m > foldWindings && omittedAcross[static_cast<std::size_t>(j)] < best) {
                    truncated = true;
                }
                out[j] = best;
            }
            if (truncated) {
                rowTruncated[static_cast<std::size_t>(i)] = 1;
            }
        }
    });
    field.foldTruncated =
        std::any_of(rowTruncated.begin(), rowTruncated.end(), [](unsigned char v) { return v != 0; });
    return field;
}

std::vector<GapFieldTile> gapFieldTiles(const GapField& field, int maxTileCols)
{
    std::vector<GapFieldTile> tiles;
    if (field.empty() || field.cols <= 0 || field.rows <= 0) {
        return tiles;
    }
    const int width = std::max(1, maxTileCols);
    const double top = -(field.y0Vx + static_cast<double>(field.rows) * field.cellVx);
    const double bottom = -field.y0Vx;
    for (int colBegin = 0; colBegin < field.cols; colBegin += width) {
        GapFieldTile tile;
        tile.colBegin = colBegin;
        tile.colEnd = std::min(field.cols, colBegin + width);
        tile.sceneRect = QRectF(
            QPointF(field.x0Vx + static_cast<double>(tile.colBegin) * field.cellVx, top),
            QPointF(field.x0Vx + static_cast<double>(tile.colEnd) * field.cellVx, bottom));
        tiles.push_back(tile);
    }
    return tiles;
}

}  // namespace vc3d::fiber_map::gaps
