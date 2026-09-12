#include "vc/lasagna/ModelPrefetch.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_set>

namespace vc::lasagna {

ModelPrefetchWindow::ModelPrefetchWindow(std::vector<ModelPrefetchSource> sources,
                                       ModelPrefetchWindowOptions options)
    : sources_(std::move(sources)), options_(options)
{
    std::erase_if(sources_, [](const auto& source) { return !source.remote; });
    if (!modelPrefetchEnabled() || !std::isfinite(options_.lookahead) ||
        options_.lookahead <= 0 || !std::isfinite(options_.refreshDistance) ||
        options_.refreshDistance <= 0)
        sources_.clear();
}

ModelPrefetchReport ModelPrefetchWindow::advance(
    const cv::Vec3d& origin, const cv::Vec3d& direction, double remainingDistance,
    const std::atomic<bool>* cancelled) noexcept
{
    ModelPrefetchReport delta;
    if (sources_.empty() || (cancelled && cancelled->load(std::memory_order_relaxed)))
        return delta;
    const auto started = std::chrono::steady_clock::now();
    try {
        if (!plan_ || cv::norm(origin - plannedOrigin_) >= options_.refreshDistance) {
            const double ahead = std::clamp(remainingDistance, 0.0, options_.lookahead);
            plan_ = std::make_unique<ModelPrefetchPlan>(sources_,
                std::vector<cv::Vec3d>{origin, origin + direction * ahead}, options_.corridor);
            plannedOrigin_ = origin;
        }
        const auto before = plan_->report();
        (void)plan_->pump(cancelled);
        delta.submitted = plan_->report().submitted - before.submitted;
        delta.rejected = plan_->report().rejected - before.rejected;
        delta.alreadyResolved = plan_->report().alreadyResolved - before.alreadyResolved;
    } catch (...) {
        sources_.clear();
        plan_.reset();
    }
    delta.planningMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return delta;
}

namespace {

using vc::render::ChunkKey;

bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

// Clip in source-voxel coordinates before computing step counts or converting
// to integer chunk coordinates. Long off-volume tails must not consume the
// entire planner budget before reaching a useful part of the line.
bool clipSegment(cv::Vec3d& a, cv::Vec3d& b,
                 const std::array<int, 3>& shape, double radius)
{
    if (!finitePoint(a) || !finitePoint(b))
        return false;
    const cv::Vec3d delta = b - a;
    if (!finitePoint(delta))
        return false;
    double first = 0.0;
    double last = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double low = -radius;
        const double high = static_cast<double>(shape[2 - axis] - 1) + radius;
        if (delta[axis] == 0.0) {
            if (a[axis] < low || a[axis] > high)
                return false;
            continue;
        }
        double entry = (low - a[axis]) / delta[axis];
        double exit = (high - a[axis]) / delta[axis];
        if (entry > exit)
            std::swap(entry, exit);
        first = std::max(first, entry);
        last = std::min(last, exit);
        if (first > last)
            return false;
    }
    b = a + last * delta;
    a += first * delta;
    return finitePoint(a) && finitePoint(b);
}

size_t chunkBytes(const std::array<int, 3>& shape, vc::render::ChunkDtype dtype)
{
    size_t bytes = dtype == vc::render::ChunkDtype::UInt16 ? 2 : 1;
    for (const int extent : shape) {
        if (extent <= 0 || bytes > std::numeric_limits<size_t>::max() /
                                      static_cast<size_t>(extent))
            return 0;
        bytes *= static_cast<size_t>(extent);
    }
    return bytes;
}

std::vector<ChunkKey> corridorKeys(
    const ModelPrefetchSource& source, const std::vector<cv::Vec3d>& polyline,
    double radius, size_t maxKeys, size_t workBudget, bool& truncated)
{
    std::vector<ChunkKey> keys;
    if (polyline.empty() || maxKeys == 0 || workBudget == 0)
        return keys;
    const auto shape = source.cache->shape(0);
    const auto chunks = source.cache->chunkShape(0);
    for (int axis = 0; axis < 3; ++axis) {
        if (shape[axis] <= 0 || chunks[axis] <= 0)
            return keys;
    }
    // Half a chunk per step bounds extra work on sparse line segments.
    // The swept subsegment box below covers the entire connecting
    // segment, with a one-voxel halo for trilinear interpolation.
    const double step = 0.5 * std::min({chunks[0], chunks[1], chunks[2]});
    const double halo = radius / source.spacing + 1.0;
    if (!std::isfinite(halo))
        return keys;
    std::unordered_set<ChunkKey, vc::render::ChunkKeyHash> seen;
    std::array<int, 3> previousLo{}, previousHi{};
    bool havePreviousBox = false;
    const auto addBox = [&](const cv::Vec3d& a, const cv::Vec3d& b) {
        std::array<int, 3> lo{}, hi{};
        for (int axis = 0; axis < 3; ++axis) {
            const int xyz = 2 - axis;
            const double minimum = std::min(a[xyz], b[xyz]) - halo;
            const double maximum = std::max(a[xyz], b[xyz]) + halo;
            if (maximum < 0.0 || minimum > shape[axis] - 1.0)
                return true;
            lo[axis] = static_cast<int>(std::floor(
                std::clamp(minimum, 0.0, shape[axis] - 1.0) / chunks[axis]));
            hi[axis] = static_cast<int>(std::floor(
                std::clamp(maximum, 0.0, shape[axis] - 1.0) / chunks[axis]));
        }
        // Dense saved curves often have thousands of vertices in a few dozen
        // chunks. Identical adjacent boxes are already fully covered, including
        // bends within that box; no geometric approximation is involved.
        if (havePreviousBox && lo == previousLo && hi == previousHi)
            return true;
        previousLo = lo;
        previousHi = hi;
        havePreviousBox = true;
        for (int z = lo[0]; z <= hi[0]; ++z) {
            for (int y = lo[1]; y <= hi[1]; ++y) {
                for (int x = lo[2]; x <= hi[2]; ++x) {
                    if (workBudget == 0 || keys.size() >= maxKeys)
                        return false;
                    --workBudget;
                    const ChunkKey key{0, z, y, x};
                    if (seen.insert(key).second)
                        keys.push_back(key);
                }
            }
        }
        return true;
    };
    const size_t segments = std::max<size_t>(1, polyline.size() - 1);
    for (size_t index = 0; index < segments; ++index) {
        if (workBudget == 0 || keys.size() >= maxKeys) {
            truncated = true;
            break;
        }
        --workBudget; // Invalid, duplicate and fully outside segments count too.
        cv::Vec3d a = polyline[index] / source.spacing;
        cv::Vec3d b = polyline[std::min(index + 1, polyline.size() - 1)] /
                     source.spacing;
        if (!clipSegment(a, b, shape, halo))
            continue;
        const cv::Vec3d delta = b - a;
        const double length = std::hypot(delta[0], delta[1], delta[2]);
        const double steps = std::max(1.0, std::ceil(length / step));
        // Never turn an untrusted/unusually long distance into an integer
        // loop bound. The independent work budget is the only loop bound.
        double t = 0.0;
        while (t < 1.0) {
            if (workBudget == 0 || keys.size() >= maxKeys) {
                truncated = true;
                return keys;
            }
            --workBudget;
            const double nextT = std::min(1.0, t + 1.0 / steps);
            if (nextT <= t || !addBox(a + delta * t, a + delta * nextT)) {
                truncated = true;
                return keys;
            }
            t = nextT;
        }
    }
    return keys;
}

} // namespace

bool modelPrefetchEnabled()
{
    static const bool enabled = [] {
        const char* setting = std::getenv("VC3D_LINE_MODEL_PREFETCH");
        return setting == nullptr || setting[0] != '0';
    }();
    return enabled;
}

ModelPrefetchPlan::ModelPrefetchPlan(
    std::vector<ModelPrefetchSource> sources,
    const std::vector<cv::Vec3d>& polyline, ModelPrefetchOptions options)
{
    if (!modelPrefetchEnabled())
        return;
    const auto start = std::chrono::steady_clock::now();
    std::erase_if(sources, [](const ModelPrefetchSource& source) {
        return !source.remote || !source.cache || !(source.spacing > 0.0) ||
               !std::isfinite(source.spacing) || source.cache->numLevels() == 0;
    });
    if (sources.empty() || !std::isfinite(options.radius) || options.radius < 0.0 ||
        options.maxRequests == 0 || options.maxPlanningSteps == 0)
        return;
    std::vector<std::vector<ChunkKey>> bySource(sources.size());
    std::vector<size_t> bytes(sources.size());
    const size_t perSourceKeys = options.maxRequests / sources.size();
    const size_t perSourceWork = options.maxPlanningSteps / sources.size();
    size_t maxCount = 0;
    for (size_t index = 0; index < sources.size(); ++index) {
        bytes[index] = chunkBytes(sources[index].cache->chunkShape(0),
                                  sources[index].cache->dtype());
        // The admission API has a 32 MiB ceiling. A larger single chunk can
        // never be admitted; do not leave a plan permanently parked on it.
        if (bytes[index] == 0 || bytes[index] > 32ULL * 1024ULL * 1024ULL)
            continue;
        bySource[index] = corridorKeys(sources[index], polyline, options.radius,
                                      perSourceKeys, perSourceWork, report_.truncated);
        maxCount = std::max(maxCount, bySource[index].size());
    }
    // Interleave model channels so a full admission window cannot be spent on
    // just one channel while all the others needed for scoring remain cold.
    for (size_t keyIndex = 0; keyIndex < maxCount; ++keyIndex) {
        for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
            if (keyIndex >= bySource[sourceIndex].size())
                continue;
            if (bytes[sourceIndex] > options.maxPlannedBytes - report_.plannedBytes) {
                report_.truncated = true;
                continue;
            }
            requests_.push_back({sources[sourceIndex].cache,
                                 bySource[sourceIndex][keyIndex]});
            report_.plannedBytes += bytes[sourceIndex];
        }
    }
    report_.planned = requests_.size();
    report_.planningMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

bool ModelPrefetchPlan::pump(const std::atomic<bool>* cancelled)
{
    using Status = vc::render::ChunkCache::SpeculativePrefetchStatus;
    for (size_t considered = 0; next_ < requests_.size() && considered < 128;
         ++considered) {
        if (cancelled && cancelled->load(std::memory_order_relaxed)) {
            next_ = requests_.size();
            return true;
        }
        const auto& request = requests_[next_];
        switch (request.cache->prefetchSpeculativeChunk(request.key)) {
        case Status::Submitted: ++report_.submitted; break;
        case Status::AlreadyQueued: ++report_.alreadyQueued; break;
        case Status::AlreadyResolved: ++report_.alreadyResolved; break;
        case Status::Skipped: ++report_.skipped; break;
        case Status::Rejected:
            ++report_.rejected;
            return false;
        }
        ++next_;
    }
    return next_ == requests_.size();
}

} // namespace vc::lasagna
