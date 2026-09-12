#include "vc/lasagna/ModelPrefetch.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace vc::lasagna {

namespace {
bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}
}

ModelPrefetchProjection modelPrefetchProjection()
{
    static const auto mode = [] {
        const char* setting = std::getenv("VC3D_LINE_MODEL_PROJECTION");
        const std::string_view value = setting ? setting : "straight";
        if (value == "guided") return ModelPrefetchProjection::Guided;
        if (value == "curved") return ModelPrefetchProjection::Curved;
        return ModelPrefetchProjection::Straight;
    }();
    return mode;
}

const char* modelPrefetchProjectionName()
{
    switch (modelPrefetchProjection()) {
    case ModelPrefetchProjection::Guided: return "guided";
    case ModelPrefetchProjection::Curved: return "curved";
    default: return "straight";
    }
}

ModelPrefetchPredictor::ModelPrefetchPredictor(
    ModelPrefetchWindowOptions options, ModelPrefetchReference reference)
    : options_(options)
{
    if (options_.projection == ModelPrefetchProjection::Straight ||
        !(reference.scale > 0) || !std::isfinite(reference.scale))
        return;
    const size_t count = std::min<size_t>(2048, reference.points.size());
    reference_.reserve(count);
    arcs_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = reference.reverse ? reference.points.size() - 1 - i : i;
        const auto point = reference.points[index] * reference.scale;
        // Keep only the valid oriented prefix: no shortcut across invalid data
        // and no distant endpoint appended after the input-inspection limit.
        if (!finitePoint(point)) break;
        const double distance = reference_.empty() ? 0 : cv::norm(point - reference_.back());
        if (!std::isfinite(distance)) break;
        if (!reference_.empty() && distance <= 1e-12) continue;
        const double arc = arcs_.empty() ? 0 : arcs_.back() + distance;
        if (!std::isfinite(arc)) break;
        reference_.push_back(point);
        arcs_.push_back(arc);
    }
    if (!reference_.empty()) matchedOrigin_ = reference_.front();
}

void ModelPrefetchPredictor::observe(const cv::Vec3d& origin, const cv::Vec3d& direction)
{
    if (historySize_ != 0) {
        const auto displacement = origin - history_[historySize_ - 1].origin;
        const double distance = cv::norm(displacement);
        if (distance < options_.refreshDistance / 8) return;
        if (distance > options_.refreshDistance * 2 ||
            displacement.dot(direction) < distance * 0.5 ||
            direction.dot(history_[historySize_ - 1].direction) < 0.5)
            historySize_ = 0; // reversal or discontinuous leading-beam jump
    }
    if (historySize_ == history_.size()) {
        std::move(history_.begin() + 1, history_.end(), history_.begin());
        --historySize_;
    }
    history_[historySize_++] = {origin, direction};
}

bool ModelPrefetchPredictor::followReference(
    const cv::Vec3d& origin, const cv::Vec3d& direction, double ahead,
    std::vector<cv::Vec3d>& points)
{
    if (reference_.size() < 2 || ahead <= 0) return false;
    // Cursor starts at the known control end, never at a global nearest point.
    // Monotone progress and a bounded arc interval avoid jumping across distant
    // self-approaches. Equal-distance ties retain the earlier segment.
    const double allowance = std::clamp(2 * cv::norm(origin - matchedOrigin_),
        options_.refreshDistance, options_.refreshDistance * 4);
    const double limit = progress_ + allowance;
    double bestDistance = std::numeric_limits<double>::infinity();
    size_t best = cursor_;
    double bestProgress = progress_;
    cv::Vec3d anchor;
    for (size_t i = cursor_, steps = 0; i + 1 < reference_.size() && steps < 128; ++i, ++steps) {
        if (arcs_[i] > limit) break;
        const double length = arcs_[i + 1] - arcs_[i];
        const auto tangent = (reference_[i + 1] - reference_[i]) / length;
        if (tangent.dot(direction) < 0.5) continue;
        const double low = std::max(0.0, progress_ - arcs_[i]);
        const double high = std::min(length, limit - arcs_[i]);
        if (high < low) continue;
        const double along = std::clamp((origin - reference_[i]).dot(tangent), low, high);
        const auto projected = reference_[i] + tangent * along;
        const double distance = cv::norm(projected - origin);
        if (distance < bestDistance) {
            bestDistance = distance;
            best = i;
            bestProgress = arcs_[i] + along;
            anchor = projected;
        }
    }
    if (!std::isfinite(bestDistance) || bestDistance > options_.corridor.radius) return false;
    cursor_ = best;
    progress_ = bestProgress;
    matchedOrigin_ = origin;
    const auto offset = origin - anchor;
    points = {origin};
    auto point = anchor;
    for (size_t i = best, steps = 0; i + 1 < reference_.size() && steps < 128 && ahead > 0; ++i, ++steps) {
        const auto delta = reference_[i + 1] - point;
        const double length = cv::norm(delta);
        if (length <= 1e-12) continue;
        const double advance = std::min(length, ahead);
        point += delta * (advance / length);
        points.push_back(point + offset); // continuous at the actual trace origin
        ahead -= advance;
    }
    return points.size() > 1;
}

bool ModelPrefetchPredictor::extrapolateCurve(
    const cv::Vec3d& origin, const cv::Vec3d& direction, double ahead,
    std::vector<cv::Vec3d>& points) const
{
    if (historySize_ < 4 || ahead <= 0) return false;
    cv::Vec3d rotation{0, 0, 0};
    double totalTurn = 0, distance = 0;
    for (size_t i = 1; i < historySize_; ++i) {
        const auto turn = history_[i - 1].direction.cross(history_[i].direction);
        rotation += turn;
        totalTurn += cv::norm(turn);
        distance += cv::norm(history_[i].origin - history_[i - 1].origin);
    }
    const double coherentTurn = cv::norm(rotation);
    if (coherentTurn < 0.01 || coherentTurn < 0.75 * totalTurn || distance <= 0)
        return false; // noisy alternating turns do not justify a curved forecast
    const auto axis = rotation / coherentTurn;
    // Cap the undamped turn at 30 degrees; damping caps integrated turn at 15.
    const double rate = std::min(coherentTurn / distance, 0.5235987755982988 / ahead);
    points = {origin};
    auto point = origin;
    auto tangent = direction;
    const double step = ahead / 8;
    for (int i = 0; i < 8; ++i) {
        const double angle = rate * step * (1 - (i + 0.5) / 8);
        const auto next = tangent * std::cos(angle) + axis.cross(tangent) * std::sin(angle) +
            axis * (axis.dot(tangent) * (1 - std::cos(angle)));
        const auto middle = tangent + next;
        point += middle * (step / cv::norm(middle));
        points.push_back(point);
        tangent = next;
    }
    return true;
}

std::optional<ModelPrefetchPrediction> ModelPrefetchPredictor::update(
    const cv::Vec3d& origin, const cv::Vec3d& direction, double remainingDistance)
{
    if (!finitePoint(origin) || !finitePoint(direction) || !std::isfinite(remainingDistance) ||
        !(options_.lookahead > 0) || !std::isfinite(options_.lookahead) ||
        !(options_.corridor.radius >= 0) || !std::isfinite(options_.corridor.radius) ||
        !(options_.refreshDistance > 0) || !std::isfinite(options_.refreshDistance))
        return std::nullopt;
    const double displacement = cv::norm(origin - plannedOrigin_);
    const bool guided = options_.projection != ModelPrefetchProjection::Straight;
    const double norm = cv::norm(direction);
    if (guided && (!(norm > 0) || !std::isfinite(norm))) return std::nullopt;
    const auto unit = guided ? direction / norm : direction;
    if (options_.projection == ModelPrefetchProjection::Curved) observe(origin, unit);
    // A matched reference plan already includes its bends. Keep distance-only
    // refresh while following it; an unmatched forecast can refresh on turns.
    // A stale reference is reconsidered at the next bounded distance refresh.
    const bool turn = guided && planned_ && !plannedReference_ &&
        displacement >= options_.refreshDistance / 4 &&
        unit.dot(plannedDirection_) < 0.9659258262890683; // accumulated turn >15 degrees
    if (planned_ && displacement < options_.refreshDistance && !turn) return std::nullopt;
    const double ahead = std::clamp(remainingDistance, 0.0, options_.lookahead);
    ModelPrefetchPrediction prediction;
    prediction.turnRefresh = turn;
    prediction.reference = guided && followReference(origin, unit, ahead, prediction.points);
    prediction.referenceFallback = guided && ahead > 0 && reference_.size() >= 2 && !prediction.reference;
    prediction.curved = !prediction.reference && options_.projection == ModelPrefetchProjection::Curved &&
        extrapolateCurve(origin, unit, ahead, prediction.points);
    if (!prediction.reference && !prediction.curved)
        prediction.points = {origin, origin + direction * ahead};
    planned_ = true;
    plannedReference_ = prediction.reference;
    plannedOrigin_ = origin;
    plannedDirection_ = unit;
    return prediction;
}

ModelPrefetchWindow::ModelPrefetchWindow(std::vector<ModelPrefetchSource> sources,
                                       ModelPrefetchWindowOptions options,
                                       ModelPrefetchReference reference)
    : sources_(std::move(sources)), options_(options),
      predictor_(options, modelPrefetchEnabled() && std::any_of(sources_.begin(), sources_.end(),
          [](const auto& source) { return source.remote; }) ? reference : ModelPrefetchReference{})
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
        if (auto prediction = predictor_.update(origin, direction, remainingDistance)) {
            plan_ = std::make_unique<ModelPrefetchPlan>(sources_, prediction->points, options_.corridor);
            delta.replans = 1;
            delta.turnRefreshes = prediction->turnRefresh;
            delta.referencePlans = prediction->reference;
            delta.referenceFallbacks = prediction->referenceFallback;
            delta.curvaturePlans = prediction->curved;
        }
        if (!plan_) return delta;
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
