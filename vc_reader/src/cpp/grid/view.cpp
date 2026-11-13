// Views over volumes: which pyramid level a view samples, and frames drawn from the
// caches alone whose missing pixels are read in the background.
#include "vc_reader/grid.hpp"

#include "../array.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vcr::grid {

std::array<double, 3> voxelExtent(const Level& level)
{
    std::array<double, 3> extent{};
    if (!level.linear) {
        for (std::size_t a = 0; a < 3; ++a)
            extent[a] = 1.0 / std::abs(level.scale[a]);
        return extent;
    }
    // index = L p + offset: one index step along axis a moves p by column a of L^-1.
    const auto& L = *level.linear;
    const double det = L[0] * (L[4] * L[8] - L[5] * L[7]) - L[1] * (L[3] * L[8] - L[5] * L[6]) +
                       L[2] * (L[3] * L[7] - L[4] * L[6]);
    if (det == 0.0 || !std::isfinite(det))
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    const std::array<double, 9> inv{
        (L[4] * L[8] - L[5] * L[7]) / det, (L[2] * L[7] - L[1] * L[8]) / det, (L[1] * L[5] - L[2] * L[4]) / det,
        (L[5] * L[6] - L[3] * L[8]) / det, (L[0] * L[8] - L[2] * L[6]) / det, (L[2] * L[3] - L[0] * L[5]) / det,
        (L[3] * L[7] - L[4] * L[6]) / det, (L[1] * L[6] - L[0] * L[7]) / det, (L[0] * L[4] - L[1] * L[3]) / det};
    for (std::size_t a = 0; a < 3; ++a)
        extent[a] = std::sqrt(inv[a] * inv[a] + inv[3 + a] * inv[3 + a] + inv[6 + a] * inv[6 + a]);
    return extent;
}

namespace {

double largestExtent(const Level& level)
{
    const auto extent = voxelExtent(level);
    return *std::max_element(extent.begin(), extent.end());
}

}  // namespace

int levelForView(const std::vector<Level>& levels, float pixelsPerVoxel, float qualityBias)
{
    if (!std::isfinite(pixelsPerVoxel) || !std::isfinite(qualityBias) || pixelsPerVoxel <= 0.0f ||
        qualityBias <= 0.0f)
        return 0;
    const double maximumExtent = 1.0 / (double(pixelsPerVoxel) * double(qualityBias));
    int selected = -1;
    for (std::size_t level = 0; level < levels.size(); ++level) {
        if (!levels[level].array)
            continue;
        // At high zoom, keep the finest level even when none meets the threshold.
        if (selected < 0 || largestExtent(levels[level]) <= maximumExtent)
            selected = int(level);
    }
    return std::max(selected, 0);
}

int fallbackLevels(const std::vector<Level>& levels, int start, int width, int height, float pixelsPerVoxel,
                   int maximum)
{
    const int count = int(levels.size());
    if (start < 0 || start >= count || maximum <= 0)
        return 0;
    const int last = std::min(count - 1, start + maximum);
    if (width <= 0 || height <= 0 || !std::isfinite(pixelsPerVoxel) || pixelsPerVoxel <= 0.0f)
        return last - start;
    const double viewExtent = double(std::max(width, height)) / double(pixelsPerVoxel);
    for (int level = start; level <= last; ++level) {
        const auto& L = levels[std::size_t(level)];
        if (!L.array)
            continue;
        // A representative chunk edge in working voxels: the mean over the axes.
        const auto extent = voxelExtent(L);
        const auto& chunks = L.array->spec().chunks;
        const std::size_t first = L.channel ? 1 : 0;
        if (chunks.size() != first + 3)
            throw std::invalid_argument("view levels must be ZYX arrays or selected CZYX channels");
        double chunkExtent = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis)
            chunkExtent += double(chunks[first + axis]) * extent[axis];
        if (chunkExtent / 3.0 >= viewExtent)
            return level - start;
    }
    return last - start;
}

std::array<float, 18> magnificationPoints(const Raster& raster)
{
    // Off-centre points too: the view's centre may be over a hole or a boundary.
    std::array<float, 18> xy{};
    std::size_t i = 0;
    for (const double y : {0.25, 0.5, 0.75})
        for (const double x : {0.25, 0.5, 0.75}) {
            xy[i++] = float(raster.x0 + x * raster.width * raster.sx);
            xy[i++] = float(raster.y0 + y * raster.height * raster.sy);
        }
    return xy;
}

float pixelsPerVoxel(const float* columnTangents, const float* rowTangents, std::size_t n, double sx, double sy)
{
    double magnification = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double dx[3], dy[3];
        for (std::size_t a = 0; a < 3; ++a) {
            dx[a] = double(columnTangents[i * 3 + a]) * sx;
            dy[a] = double(rowTangents[i * 3 + a]) * sy;
        }
        const double a = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        const double b = dx[0] * dy[0] + dx[1] * dy[1] + dx[2] * dy[2];
        const double c = dy[0] * dy[0] + dy[1] * dy[1] + dy[2] * dy[2];
        // Smaller singular value: the most magnified direction, including grids whose
        // row and column tangents are not orthogonal.
        const double extent2 = 0.5 * (a + c - std::hypot(a - c, 2 * b));
        if (std::isfinite(extent2) && extent2 > 0)
            magnification = std::max(magnification, 1.0 / std::sqrt(extent2));
    }
    return float(magnification);
}

float pixelsPerVoxel(const Surface& surface, const Raster& raster)
{
    const auto xy = magnificationPoints(raster);
    float du[27], dv[27];
    surface.pointsAt(xy.data(), 9, nullptr, nullptr, du, dv);
    return pixelsPerVoxel(du, dv, 9, raster.sx, raster.sy);
}

Result sample(const Frame& frame, const Request& request)
{
    if (frame.plane)
        return sample(*frame.plane, request);
    if (!frame.surface)
        throw std::invalid_argument("a frame needs a plane or a surface");
    return sample(*frame.surface, frame.raster, request);
}

Result sampleCached(const Frame& frame, Request request, Pending* pending)
{
    if (request.levels.empty())
        return {};
    request.cachedOnly = true;
    auto result = sample(frame, request);
    if (!pending)
        return result;

    // Pixels with a layer from a coarser level, or none, read the view's level; pixels
    // with a layer no level held also read the coarsest level, for a quick preview.
    const std::size_t pixels = result.level.size();
    const bool coarser = request.levels.size() > 1;
    std::vector<std::uint8_t> target(pixels, 0), coarse(pixels, 0);
    bool anyTarget = false, anyCoarse = false;
    for (std::size_t q = 0; q < pixels; ++q) {
        const auto level = result.level[q];
        const bool missing = level == Result::kMissing;
        if (!missing && (level == Result::kNone || level == 0))
            continue;
        target[q] = 1;
        anyTarget = true;
        if (missing && coarser) {
            coarse[q] = 1;
            anyCoarse = true;
        }
    }
    // Sampled on the CPU, the frame names the chunks it lacked: reads fetch exactly those.
    const bool chunkReads = result.missingChunks.size() == request.levels.size();
    const auto read = [&](std::size_t index, std::vector<std::uint8_t> mask) {
        Pending::Read r{frame, {}, std::move(mask), {}};
        r.request.levels = {request.levels[index]};
        r.request.order = request.order;
        r.request.offsets = request.offsets;
        if (chunkReads) {
            r.chunks = result.missingChunks[index];
        }
        return r;
    };
    if (anyCoarse)
        pending->coarse.push_back(read(request.levels.size() - 1, std::move(coarse)));
    if (anyTarget)
        pending->target.push_back(read(0, std::move(target)));
    return result;
}

namespace {

// Brings the chunks into the array's caches: Array::read_many with null outputs
// resolves each chunk once, in parallel, and copies nothing.
void readChunks(Array& array, const std::vector<std::int64_t>& keys, const std::atomic<bool>* cancel)
{
    const auto& grid = array.chunk_grid();
    const auto& shape = array.spec().chunks;
    std::vector<Index> starts, sizes(keys.size(), Index(shape.begin(), shape.end()));
    for (auto key : keys) {
        Index start(grid.size());
        for (std::size_t axis = grid.size(); axis-- > 0;) {
            start[axis] = key % grid[axis] * shape[axis];
            key /= grid[axis];
        }
        starts.push_back(std::move(start));
    }
    array.read_many(starts, sizes, std::vector<std::uint8_t*>(keys.size(), nullptr), nullptr, false, cancel);
}

}  // namespace

std::string load(const Pending& pending, const std::function<bool()>& afterPart, const std::atomic<bool>* cancel)
{
    std::string failure;
    for (const auto* part : {&pending.coarse, &pending.target}) {
        try {
            for (const auto& read : *part) {
                if (!read.chunks.empty()) {
                    readChunks(*read.request.levels.front().array, read.chunks, cancel);
                    continue;
                }
                auto request = read.request;
                request.cachedOnly = false;
                request.pixels = read.pixels.empty() ? nullptr : read.pixels.data();
                request.cancel = cancel;
                const auto result = sample(read.frame, request);
                if (failure.empty())
                    failure = result.cudaFailure;
            }
        } catch (const Cancelled&) {
        }
        if (!part->empty() && afterPart && !afterPart())
            break;
        if (cancel && cancel->load(std::memory_order_acquire))
            break;
    }
    return failure;
}

}  // namespace vcr::grid
