#include "Fiber3DRibbon.hpp"
#include "Fiber3DGeometry.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/LineFrameGeometry.hpp"
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
cv::Vec3d initialNormal(cv::Vec3d tangent)
{
    int axis = 0;
    for (int i = 1; i < 3; ++i)
        if (std::abs(tangent[i]) < std::abs(tangent[axis]))
            axis = i;
    cv::Vec3d n{};
    n[axis] = 1;
    return *vc::fiber_tracer::projectDisplayNormal(n, tangent);
}
}  // namespace

Frames makeFrames(const Fiber& fiber, const std::optional<Bounds>& clip, const NormalAt& normalAt, const std::atomic_bool* cancelled)
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
        Strip rows(points.size());
        size_t firstValid = points.size();
        for (size_t i = 0; i < rows.size(); ++i) {
            check(cancelled);
            auto& r = rows[i];
            r.center = points[i];
            r.tangent = vc::geometry::lineTangent(points.size(), i, [&](size_t k) { return points[k]; });
            if (normalAt)
                if (const auto sampled = normalAt(r.center))
                    if (const auto n = vc::fiber_tracer::projectDisplayNormal(*sampled, r.tangent)) {
                        r.normal = *n;
                        r.sampledNormal = true;
                        firstValid = std::min(firstValid, i);
                        ++result.sampledNormals;
                    }
        }
        // Anchor missing initial rows to the first real Lasagna normal, when
        // there is one. Never flip a texture just because a normal changes sign.
        const size_t anchor = firstValid == rows.size() ? 0 : firstValid;
        if (firstValid == rows.size())
            rows[0].normal = initialNormal(rows[0].tangent);
        for (size_t i = anchor; i > 0; --i)
            rows[i - 1].normal = vc::geometry::transportFrameNormal(rows[i].normal, rows[i].tangent, rows[i - 1].tangent);
        for (size_t i = anchor + 1; i < rows.size(); ++i) {
            auto n = vc::geometry::transportFrameNormal(rows[i - 1].normal, rows[i - 1].tangent, rows[i].tangent);
            if (rows[i].sampledNormal) {
                if (rows[i].normal.dot(n) < 0)
                    rows[i].normal *= -1;
            } else
                rows[i].normal = n;
        }
        result.rows += rows.size();
        result.strips.push_back(std::move(rows));
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
            double begin = 0, end = 1;
            const auto& a = run.points[i - 1];
            const auto& b = run.points[i];
            if (!finite(a) || !finite(b) || style == SpanStyle::Gap || style == SpanStyle::InferredGap ||
                (clip && !clipSegment(a, b, *clip, begin, end))) {
                flush();
                continue;
            }
            const auto p = vec(interpolate(a, b, begin)), q = vec(interpolate(a, b, end));
            const double length = cv::norm(q - p);
            if (length < 1e-8)
                continue;
            if (!points.empty() && cv::norm(points.back() - p) > 1e-6)
                flush();
            if (points.empty())
                append(p);
            if (length > double(maxRows))
                throw std::runtime_error("CT ribbon segment is too long");
            const size_t steps = size_t(std::ceil(length));
            for (size_t k = 1; k <= steps; ++k)
                append(p + (q - p) * (double(k) / steps));
        }
        previous = source;
    }
    flush();
    return result;
}

cv::Vec3d across(const Row& row, double rotationDegrees)
{
    const auto n = vc::fiber_tracer::rotateDisplayNormal(row.normal, row.tangent, rotationDegrees * std::acos(-1.) / 180.);
    return n.cross(row.tangent);
}

std::shared_ptr<Image> sample(const Frames& frames, Volume& volume, int widthVoxels, int rotationDegrees, const std::atomic_bool* cancelled)
{
    check(cancelled);
    if (widthVoxels < 1 || widthVoxels > 512)
        throw std::invalid_argument("CT ribbon width must be 1–512 voxels");
    if (!volume.hasScaleLevel(0))
        throw std::runtime_error("CT ribbon needs level 0 of the active CT");
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
    for (const auto& strip : frames.strips)
        for (size_t start = 0; start + 1 < strip.size(); start += 1023) {
            check(cancelled);
            Tile tile;
            tile.width = width;
            tile.height = int(std::min(size_t(1024), strip.size() - start));
            tile.pixels.resize(size_t(tile.width) * tile.height * 2);
            for (int r = 0; r < tile.height; ++r) {
                const auto& row = strip[start + r];
                const auto axis = across(row, rotationDegrees);
                tile.left.push_back(point(row.center - axis * (widthVoxels * .5)));
                tile.right.push_back(point(row.center + axis * (widthVoxels * .5)));
            }
            ct::sampleTile(tile, volume, 0, cancelled);
            image->tiles.push_back(std::move(tile));
        }
    check(cancelled);
    if (image->tiles.empty())
        throw std::runtime_error("No continuous fiber segment in the displayed region");
    return image;
}
}  // namespace vc3d::fibers3d::ribbon
