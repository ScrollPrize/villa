#include "Fiber3DRibbon.hpp"
#include "Fiber3DGeometry.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/fiber_tracer/FiberDisplay.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::fibers3d::ribbon
{
namespace
{
constexpr size_t maxRows = 262144;
constexpr size_t maxTexels = 8 * 1024 * 1024;
void check(const std::atomic_bool* cancelled)
{
    if (cancelled && cancelled->load(std::memory_order_relaxed))
        throw std::runtime_error("CT ribbon cancelled");
}
cv::Vec3d vec(const Point& p)
{
    return {p[0], p[1], p[2]};
}
Point point(cv::Vec3d p)
{
    return {p[0], p[1], p[2]};
}
class CallbackSampler : public vc::lasagna::NormalSampler
{
public:
    explicit CallbackSampler(const NormalAt& at) : at_(at) {}
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& p) const override
    {
        const auto n = at_(p);
        return n ? vc::lasagna::NormalSample{*n, true, {}} : vc::lasagna::NormalSample{};
    }

private:
    const NormalAt& at_;
};
}  // namespace

Frames framesFromSurface(const cv::Mat_<cv::Vec3f>& grid, const cv::Mat_<cv::Vec3f>& sideGrid, double spacing, bool sampledNormals, const std::atomic_bool* cancelled)
{
    Frames result;
    if (grid.rows != vc::lasagna::kLineViewCrossSampleCount || grid.cols < 2 || sideGrid.size() != grid.size() || !std::isfinite(spacing) ||
        spacing < 1 || spacing > maxRows)
        throw std::invalid_argument("Invalid Line Annotation surface grid");
    const int mid = grid.rows / 2;
    const int subdivisions = int(std::ceil(spacing));
    Strip rows;
    for (int col = 0; col + 1 < grid.cols; ++col) {
        check(cancelled);
        for (int k = 0; k < subdivisions + (col + 2 == grid.cols ? 1 : 0); ++k) {
            const double t = double(k) / subdivisions;
            const auto at = [&](const cv::Mat_<cv::Vec3f>& g, int r) {
                return cv::Vec3d(g(r, col)) * (1 - t) + cv::Vec3d(g(r, col + 1)) * t;
            };
            Row row;
            row.center = at(grid, mid);
            for (int cross = 0; cross < grid.rows; ++cross)
                row.surface[cross] = at(grid, cross);
            // Keep the builder's interpolated vectors, including their
            // length. Renormalizing them would change its CT coordinates.
            row.side = (at(grid, mid + 1) - at(grid, mid)) / vc::lasagna::kLineViewCrossRowSpacingBaseVoxels;
            row.normal = (at(sideGrid, mid + 1) - at(sideGrid, mid)) / vc::lasagna::kLineViewCrossRowSpacingBaseVoxels;
            row.tangent = row.side.cross(row.normal);
            row.tangent /= cv::norm(row.tangent);
            row.sampledNormal = sampledNormals;
            rows.push_back(row);
            if (row.sampledNormal)
                ++result.sampledNormals;
            if (rows.size() + result.rows > maxRows)
                throw std::runtime_error("CT ribbon is too long; narrow the displayed region");
        }
    }
    result.rows += rows.size();
    result.strips.push_back(std::move(rows));
    return result;
}

Frames makeFrames(
    const Fiber& fiber,
    const std::optional<Bounds>& clip,
    const NormalAt& normalAt,
    const std::atomic_bool* cancelled,
    const ModelForPoints& modelForPoints,
    const std::function<cv::Vec3f(const cv::Vec3f&)>& centerReference)
{
    Frames result;
    std::vector<cv::Vec3d> points;
    size_t total = 0;
    auto flush = [&] {
        if (points.size() < 2) {
            points.clear();
            return;
        }
        check(cancelled);
        CallbackSampler sampler(normalAt);
        auto model = modelForPoints ? modelForPoints(points)
                     : normalAt     ? vc::lasagna::lineModelFromSampledPoints(points, &sampler)
                                    : vc::lasagna::lineModelForInspection(points);
        vc::lasagna::LineViewConfig config;
        config.buildLineZSlices = false;
        config.orientedPointNormals = vc::lasagna::orientedLineNormals(model, centerReference);
        // An AFV without annotation controls opens with its two endpoints.
        // With controls, keep the exact same nearest-source-point supports.

        for (const auto& marker : fiber.markers) {
            if (marker.kind != MarkerKind::ControlPoint)
                continue;
            size_t closest = 0;
            double distance = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < points.size(); ++i) {
                const double d = cv::norm(points[i] - vec(marker.point));
                if (d < distance) {
                    distance = d;
                    closest = i;
                }
            }
            config.controlPointLinePositions.push_back(double(closest));
        }
        if (std::any_of(fiber.displayNormals.begin(), fiber.displayNormals.end(), [](const auto& n) { return n.has_value(); })) {
            std::vector<cv::Vec3f> displayPoints(points.begin(), points.end());
            config.displayPointNormals =
                vc::fiber_tracer::fiberDisplayField(
                    displayPoints, config.orientedPointNormals, config.controlPointLinePositions, fiber.displayNormals, fiber.controlDirections)
                    .normals;
        }
        if (config.controlPointLinePositions.empty())
            config.controlPointLinePositions = {0., double(points.size() - 1)};
        const auto views = vc::lasagna::buildLineViewSurfaces(model, config);
        auto part =
            framesFromSurface(views.lineSurface->rawPoints(), views.lineSideSlice->rawPoints(), views.stripPositionMap.stripGridSpacingBaseVoxels, bool(modelForPoints || normalAt), cancelled);
        result.rows += part.rows;
        result.sampledNormals += part.sampledNormals;
        if (result.rows > maxRows)
            throw std::runtime_error("CT ribbon is too long; narrow the displayed region");
        result.strips.insert(result.strips.end(), std::make_move_iterator(part.strips.begin()), std::make_move_iterator(part.strips.end()));
        points.clear();
    };
    auto append = [&](cv::Vec3d p) {
        if (++total > maxRows)
            throw std::runtime_error("CT ribbon is too long; narrow the displayed region");
        points.push_back(p);
    };
    // AFV storage blocks are not breaks in a fiber. Build one frame sequence
    // across blocks sharing an original point, even if the spatial query
    // returned those blocks out of order. Missing indices or geometry still
    // break the strip; proximity alone never joins separate runs.
    std::vector<const Run*> ordered;
    ordered.reserve(fiber.runs.size());
    for (const auto& run : fiber.runs)
        ordered.push_back(&run);
    std::stable_sort(ordered.begin(), ordered.end(), [](const Run* a, const Run* b) { return a->firstPoint < b->firstPoint; });
    const Run* previous = nullptr;
    for (const auto* source : ordered) {
        const auto& run = *source;
        if (run.points.size() < 2) {
            flush();
            previous = nullptr;
            continue;
        }
        const bool contiguous = previous && previous->firstPoint >= 0 && run.firstPoint >= previous->firstPoint &&
                                uint64_t(run.firstPoint) - uint64_t(previous->firstPoint) == previous->points.size() - 1 &&
                                previous->points.back() == run.points.front();
        if (!contiguous)
            flush();
        for (size_t i = 1; i < run.points.size(); ++i) {
            check(cancelled);
            const auto style = i - 1 < run.styles.size() ? run.styles[i - 1] : SpanStyle::Solid;
            const auto& a = run.points[i - 1];
            const auto& b = run.points[i];
            if (!finite(a) || !finite(b) || style == SpanStyle::Gap || style == SpanStyle::InferredGap) {
                flush();
                continue;
            }
            const auto p = vec(a), q = vec(b);
            const double length = cv::norm(q - p);
            if (length < 1e-8)
                continue;
            if (!points.empty() && cv::norm(points.back() - p) > 1e-6)
                flush();
            if (points.empty())
                append(p);
            if (length > double(maxRows))
                throw std::runtime_error("CT ribbon segment is too long");
            append(q);
        }
        previous = source;
    }
    flush();
    if (clip) {
        // Crop the already-built canonical surface. Rebuilding its normal
        // field on a cropped curve would change the Line Annotation cut.
        Frames cropped;
        for (const auto& strip : result.strips) {
            Strip current;
            const auto flushClipped = [&] {
                if (current.size() >= 2) {
                    cropped.rows += current.size();
                    for (const auto& row : current)
                        if (row.sampledNormal)
                            ++cropped.sampledNormals;
                    cropped.strips.push_back(std::move(current));
                }
                current = {};
            };
            for (size_t i = 1; i < strip.size(); ++i) {
                double lo = 0, hi = 1;
                const auto& a = strip[i - 1];
                const auto& b = strip[i];
                if (!clipSegment(point(a.center), point(b.center), *clip, lo, hi)) {
                    flushClipped();
                    continue;
                }
                const auto at = [&](double t) {
                    Row row;
                    row.center = a.center * (1 - t) + b.center * t;
                    row.normal = a.normal * (1 - t) + b.normal * t;
                    row.side = a.side * (1 - t) + b.side * t;
                    row.tangent = a.tangent * (1 - t) + b.tangent * t;
                    row.tangent /= cv::norm(row.tangent);
                    row.sampledNormal = a.sampledNormal || b.sampledNormal;
                    for (size_t j = 0; j < row.surface.size(); ++j)
                        row.surface[j] = a.surface[j] * (1 - t) + b.surface[j] * t;
                    return row;
                };
                if (current.empty())
                    current.push_back(at(lo));
                current.push_back(at(hi));
                if (hi < 1)
                    flushClipped();
            }
            flushClipped();
        }
        return cropped;
    }
    return result;
}

cv::Vec3d position(const Row& row, double offsetVoxels, double rotationDegrees)
{
    // Interpolate the actual Line Annotation support vertices, rather than
    // recomputing their plane from normalized vectors (even their rounding
    // must not cause a different CT lookup).
    const double y = double(row.surface.size() / 2) + offsetVoxels / vc::lasagna::kLineViewCrossRowSpacingBaseVoxels;
    const int i = std::clamp(int(std::floor(y)), 0, int(row.surface.size()) - 2);
    const double t = y - i;
    const auto p = row.surface[i] * (1 - t) + row.surface[i + 1] * t;
    if (rotationDegrees == 0)
        return p;
    const double a = rotationDegrees * std::acos(-1.) / 180.;
    const auto v = p - row.center;
    return row.center + v * std::cos(a) + row.tangent.cross(v) * std::sin(a) + row.tangent * (row.tangent.dot(v) * (1 - std::cos(a)));
}

std::shared_ptr<Image> sampleProgressive(const Frames& frames, Volume& volume, int widthVoxels, int rotationDegrees, int level, const std::atomic_bool* cancelled)
{
    check(cancelled);
    if (widthVoxels < 1 || widthVoxels > 512)
        throw std::invalid_argument("CT ribbon width must be 1–512 voxels");
    if (!volume.hasScaleLevel(level))
        throw std::runtime_error("Requested CT ribbon level is unavailable");
    const auto dtype = volume.dtype();
    if (dtype != vc::render::ChunkDtype::UInt8 && dtype != vc::render::ChunkDtype::UInt16)
        throw std::runtime_error("CT ribbon supports uint8 and uint16 CT volumes");
    const int width = widthVoxels + 1;
    // Include the repeated tile-border rows in the cap.
    size_t texels = 0;
    for (const auto& strip : frames.strips)
        texels += (strip.size() + strip.size() / 1023) * size_t(width);
    if (texels > maxTexels)
        throw std::runtime_error("CT ribbon exceeds 8 million samples; reduce width or region");
    auto image = std::make_shared<Image>();
    image->rows = frames.rows;
    image->sampledNormals = frames.sampledNormals;
    image->widthVoxels = widthVoxels;
    image->rotationDegrees = rotationDegrees;
    image->level = level;
    for (const auto& strip : frames.strips)
        for (size_t start = 0; start + 1 < strip.size(); start += 1023) {
            check(cancelled);
            Tile tile;
            tile.width = width;
            tile.height = int(std::min(size_t(1024), strip.size() - start));
            tile.pixels.resize(size_t(tile.width) * tile.height * 2);
            for (int r = 0; r < tile.height; ++r) {
                const auto& row = strip[start + r];
                tile.left.push_back(point(position(row, -widthVoxels * .5, rotationDegrees)));
                tile.right.push_back(point(position(row, widthVoxels * .5, rotationDegrees)));
            }
            const auto progress = ct::sampleTileProgressive(tile, *volume.chunkedCache(), level, cancelled);
            image->validTexels += progress.valid;
            image->coveredTexels += progress.covered;
            image->requestedLevelTexels += progress.requestedLevel;
            image->errors += progress.errors;
            image->complete = image->complete && progress.requestedLevel == progress.valid;
            image->tiles.push_back(std::move(tile));
        }
    check(cancelled);
    if (image->tiles.empty())
        throw std::runtime_error("No continuous fiber segment in the displayed region");
    return image;
}
}  // namespace vc3d::fibers3d::ribbon
