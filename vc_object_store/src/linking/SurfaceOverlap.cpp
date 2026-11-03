#include "linking/StoredGeometry.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <future>
#include <thread>
#include <functional>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "spatial/SurfaceGrid.hpp"
#include "spatial/SurfaceIndex.hpp"
#include "vc_object_store/Uuid.hpp"

namespace vc::object_links {
namespace {

struct PhaseTimer {
    using Clock = std::chrono::steady_clock;
    double* seconds;
    Clock::time_point start;
    explicit PhaseTimer(double* value) : seconds(value), start(value ? Clock::now() : Clock::time_point{}) {}
    ~PhaseTimer() { if (seconds) *seconds += std::chrono::duration<double>(Clock::now() - start).count(); }
};
struct Profile {
    bool enabled = std::getenv("VC_SURFACE_OVERLAP_PROFILE") != nullptr;
    double loading = 0, index = 0, harvest = 0, fit = 0, agreement = 0;
    uint64_t bboxPairs = 0, queries = 0, fitPairs = 0, samples = 0;
    ~Profile() {
        if (enabled) {
            const nlohmann::json out{{"loading_s", loading}, {"index_s", index}, {"harvest_s", harvest},
                {"fit_s", fit}, {"agreement_s", agreement}, {"bbox_pairs", bboxPairs},
                {"queries", queries}, {"fit_pairs", fitPairs}, {"samples", samples}};
            std::fprintf(stderr, "SURFACE_OVERLAP_PROFILE %s\n", out.dump().c_str());
        }
    }
};

struct Vec2 {
    double x = 0.0, y = 0.0;
    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
};
double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
double normSq(Vec2 a) { return dot(a, a); }

struct Pose {
    double r00 = 1, r01 = 0, r10 = 0, r11 = 1, tx = 0, ty = 0;
    Vec2 apply(Vec2 v) const { return {r00 * v.x + r01 * v.y + tx, r10 * v.x + r11 * v.y + ty}; }
    Pose inverse() const
    {
        Pose inv;
        inv.r00 = r00; inv.r01 = r10; inv.r10 = r01; inv.r11 = r11;
        inv.tx = -(inv.r00 * tx + inv.r01 * ty);
        inv.ty = -(inv.r10 * tx + inv.r11 * ty);
        return inv;
    }
    bool reflected() const { return r00 * r11 - r01 * r10 < 0.0; }
};

struct SamplePoint {
    int row = 0, col = 0;  // stored grid vertex
    Vec2 uv;  // metric (voxel) parameterization: grid index / scale
    cv::Vec3f xyz;
    cv::Vec3f tangentU;  // d(xyz)/d(u), d(xyz)/d(v)
    cv::Vec3f tangentV;
};

struct LoadedSurface {
    std::string uuid;
    std::shared_ptr<SurfaceGrid> grid;
    const cv::Mat_<cv::Vec3f>* points = nullptr; // stable for this read-only detector run
    nlohmann::json coordinateProvenance;
    cv::Vec2f scale = {1.0f, 1.0f};
    std::vector<SamplePoint> samples;                  // strided by sampleSpacing
    cv::Vec3f bboxLow, bboxHigh;
};

std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Tangent frame at a grid vertex, derivatives w.r.t. metric UV.
std::optional<std::pair<cv::Vec3f, cv::Vec3f>> tangentsAt(
    const cv::Mat_<cv::Vec3f>& points, int row, int col, const cv::Vec2f& scale)
{
    auto valid = [&](int r, int c) { return vc::geometry::validSurfacePoint(points(r, c)); };
    auto derivU = [&]() -> std::optional<cv::Vec3f> {
        if (col > 0 && col + 1 < points.cols && valid(row, col - 1) && valid(row, col + 1))
            return (points(row, col + 1) - points(row, col - 1)) * (scale[0] * 0.5f);
        if (col + 1 < points.cols && valid(row, col + 1))
            return (points(row, col + 1) - points(row, col)) * scale[0];
        if (col > 0 && valid(row, col - 1))
            return (points(row, col) - points(row, col - 1)) * scale[0];
        return std::nullopt;
    };
    auto derivV = [&]() -> std::optional<cv::Vec3f> {
        if (row > 0 && row + 1 < points.rows && valid(row - 1, col) && valid(row + 1, col))
            return (points(row + 1, col) - points(row - 1, col)) * (scale[1] * 0.5f);
        if (row + 1 < points.rows && valid(row + 1, col))
            return (points(row + 1, col) - points(row, col)) * scale[1];
        if (row > 0 && valid(row - 1, col))
            return (points(row, col) - points(row - 1, col)) * scale[1];
        return std::nullopt;
    };
    auto du = derivU();
    auto dv = derivV();
    if (!du || !dv) return std::nullopt;
    return std::pair{*du, *dv};
}

// Bilinear xyz sample at metric UV; nullopt on invalid quad.
std::optional<cv::Vec3f> sampleXyz(const LoadedSurface& surface, Vec2 uv)
{
    const auto* points = surface.points;
    double col = uv.x * surface.scale[0];
    double row = uv.y * surface.scale[1];
    const int c0 = int(std::floor(col));
    const int r0 = int(std::floor(row));
    if (c0 < 0 || r0 < 0 || c0 + 1 >= points->cols || r0 + 1 >= points->rows) return std::nullopt;
    const auto& p00 = (*points)(r0, c0);
    const auto& p10 = (*points)(r0, c0 + 1);
    const auto& p01 = (*points)(r0 + 1, c0);
    const auto& p11 = (*points)(r0 + 1, c0 + 1);
    if (p00[0] == -1.0f || p10[0] == -1.0f || p01[0] == -1.0f || p11[0] == -1.0f) return std::nullopt;
    const double fx = col - c0, fy = row - r0;
    auto interp = [&](float a, float b, float c, float d) {
        return float((1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d));
    };
    return cv::Vec3f{interp(p00[0], p10[0], p01[0], p11[0]),
                     interp(p00[1], p10[1], p01[1], p11[1]),
                     interp(p00[2], p10[2], p01[2], p11[2])};
}

std::optional<LoadedSurface> loadSurface(detail::StoredGeometry& inputs, const objects::ObjectRecord& object, const SurfaceOverlapParams& params,
                                     DetectorReport& report, double sourceScale,
                                     const std::optional<std::string>& geometryRevision = std::nullopt)
{
    LoadedSurface surface;
    try {
        surface.grid = detail::scaledSurfaceGrid(detail::surfaceGrid(inputs.load(object, geometryRevision)), sourceScale,
                                                 params.volumeUuid.has_value());
        surface.coordinateProvenance = detail::coordinateProvenance(object, params);
        surface.uuid = object.uuid;
        surface.scale = surface.grid->scale();
        if (!(surface.scale[0] > 0) || !(surface.scale[1] > 0)) {
            detail::reportFailure(report, "load", object.uuid, "invalid scale", object.uuid + ": invalid scale, skipped");
            return std::nullopt;
        }

    } catch (const std::exception& e) {
        detail::reportFailure(report, "load", object.uuid, e.what(), object.uuid + ": load failed: " + e.what());
        return std::nullopt;
    }
    const auto* points = surface.grid->rawPointsPtr();
    surface.points = points;
    const int stepR = std::max(1, int(std::llround(params.sampleSpacing * surface.scale[1])));
    const int stepC = std::max(1, int(std::llround(params.sampleSpacing * surface.scale[0])));
    surface.bboxLow = {1e30f, 1e30f, 1e30f};
    surface.bboxHigh = {-1e30f, -1e30f, -1e30f};
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f p = (*points)(row, col);
            if (p[0] == -1.0f || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            surface.bboxLow = {std::min(surface.bboxLow[0], p[0]), std::min(surface.bboxLow[1], p[1]), std::min(surface.bboxLow[2], p[2])};
            surface.bboxHigh = {std::max(surface.bboxHigh[0], p[0]), std::max(surface.bboxHigh[1], p[1]), std::max(surface.bboxHigh[2], p[2])};
            if (row % stepR != 0 || col % stepC != 0) continue;
            auto tangents = tangentsAt(*points, row, col, surface.scale);
            if (!tangents) continue;
            SamplePoint sample;
            sample.row = row;
            sample.col = col;
            sample.uv = {col / double(surface.scale[0]), row / double(surface.scale[1])};
            sample.xyz = p;
            sample.tangentU = tangents->first;
            sample.tangentV = tangents->second;
            surface.samples.push_back(std::move(sample));
        }
    }
    if (surface.samples.empty()) {
        detail::reportDiagnostic(report, "load", object.uuid, "no usable samples", object.uuid + ": no usable samples, skipped");
        return std::nullopt;
    }
    return surface;
}

struct Correspondence {
    Vec2 target;  // target metric UV
    Vec2 seed;    // seed metric UV
    cv::Vec3f targetTangentU, targetTangentV;
    cv::Vec3f seedTangentU, seedTangentV;
    float distance = 0.0f;
    std::uint64_t cell = 0;
};

// Keep the lowest splitmix64(cell) ranks, discarding cells that cannot survive
// once the correspondence cap is reached.
// This bounds memory when collecting many targets during one seed pass.
class CorrespondenceCollector {
public:
    explicit CorrespondenceCollector(size_t limit) : limit_(limit) {}
    void add(Correspondence c)
    {
        auto found = byCell_.find(c.cell);
        if (found != byCell_.end()) {
            if (c.distance < found->second.distance) found->second = c;
            return;
        }
        if (byCell_.size() == limit_) {
            if (ranks_.empty())
                for (const auto& [cell, value] : byCell_) ranks_.emplace(splitmix64(cell), cell);
            const auto rank = splitmix64(c.cell);
            auto worst = std::prev(ranks_.end());
            if (rank > worst->first) return; // splitmix64 is a bijection: no ties
            byCell_.erase(worst->second);
            ranks_.erase(worst);
            ranks_.emplace(rank, c.cell);
        }
        byCell_.emplace(c.cell, c);
    }
    void merge(const CorrespondenceCollector& other)
    {
        for (const auto& [cell, c] : other.byCell_) add(c);
    }
    std::vector<Correspondence> take() const
    {
        std::vector<Correspondence> out;
        out.reserve(byCell_.size());
        for (const auto& [cell, c] : byCell_) out.push_back(c);
        return out;
    }
private:
    size_t limit_;
    std::map<uint64_t, Correspondence> byCell_;
    std::map<uint64_t, uint64_t> ranks_;
};

void addCorrespondence(const SamplePoint& sample, const LoadedSurface& target,
    const SurfaceIndex::LookupResult& hit, const SurfaceOverlapParams& params,
    CorrespondenceCollector& collector)
{
    const auto* targetPoints = target.points;
    const cv::Vec2f grid = hit.grid;
    const Vec2 targetUv{grid[0] / double(target.scale[0]), grid[1] / double(target.scale[1])};
    const auto targetXyz = sampleXyz(target, targetUv);
    if (!targetXyz) return;
    const cv::Vec3f delta = *targetXyz - sample.xyz;
    const float distance = std::sqrt(delta.dot(delta));
    if (distance > params.tolerance) return;
    const int row = std::clamp(int(std::lround(grid[1])), 0, targetPoints->rows - 1);
    const int col = std::clamp(int(std::lround(grid[0])), 0, targetPoints->cols - 1);
    const auto tangents = tangentsAt(*targetPoints, row, col, target.scale);
    if (!tangents) return;
    Correspondence c;
    c.seed = sample.uv;
    c.target = targetUv;
    c.seedTangentU = sample.tangentU;
    c.seedTangentV = sample.tangentV;
    c.targetTangentU = tangents->first;
    c.targetTangentV = tangents->second;
    c.distance = distance;
    const auto cx = std::uint32_t(std::int32_t(std::floor(c.seed.x / params.thinningSpacing)));
    const auto cy = std::uint32_t(std::int32_t(std::floor(c.seed.y / params.thinningSpacing)));
    c.cell = (std::uint64_t(cy) << 32) | cx;
    collector.add(c);
}

std::map<size_t, CorrespondenceCollector> harvestTargets(const LoadedSurface& seed,
    const std::vector<LoadedSurface>& surfaces, const std::vector<size_t>& targets,
    const SurfaceIndex& index, const SurfaceOverlapParams& params,
    const std::atomic_bool* cancel, Profile& profile, const objects::util::ProgressCallback& progress)
{
    std::map<size_t, CorrespondenceCollector> matches;
    if (targets.empty()) return matches;
    std::unordered_map<SurfaceGrid*, size_t> targetIndex;
    std::unordered_set<SurfaceGrid*> included;
    for (const auto ti : targets) {
        targetIndex.emplace(surfaces[ti].grid.get(), ti);
        included.insert(surfaces[ti].grid.get());
    }
    using Matches = std::map<size_t, CorrespondenceCollector>;
    std::atomic_size_t completed{0};
    const auto label = seed.uuid + " -> " + std::to_string(targets.size()) + " surfaces";
    auto harvestRange = [&](size_t begin, size_t end) {
        Matches local;
        SurfaceIndex::SurfaceFilter filter;
        filter.includeRaw = &included;
        std::array<cv::Vec3f, 512> points;
        for (size_t first = begin; first < end; first += points.size()) {
            if (cancel && cancel->load()) return local;
            const size_t count = std::min(points.size(), end - first);
            for (size_t i = 0; i < count; ++i) points[i] = seed.samples[first+i].xyz;
            index.locateAllBatch(std::span(points.data(), count), float(params.tolerance), filter,
                [&](size_t i, const auto& hits) {
                    const auto& sample = seed.samples[first+i];
                    // Same traversal and tie handling as individual locate()
                    // calls, amortizing the shared lock across this batch.
                    for (const auto& hit : hits) {
                        const auto ti = targetIndex.at(hit.surface.get());
                        auto [it, inserted] = local.try_emplace(ti, params.maxCorrespondences);
                        addCorrespondence(sample, surfaces[ti], hit, params, it->second);
                    }
                });
            completed.fetch_add(count, std::memory_order_relaxed);
        }
        return local;
    };
    const size_t requested = params.workers ? size_t(params.workers)
        : std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    const size_t workers = std::min(requested, std::max(size_t(1), seed.samples.size() / 16384));
    objects::util::reportProgress(progress, "searching", 0, seed.samples.size(), label);
    std::vector<std::future<Matches>> jobs;
    // Even one worker uses an asynchronous range so the caller can keep
    // reporting progress and checking cancellation during a long search.
    for (size_t worker = 0; worker < workers; ++worker) {
        const size_t begin = seed.samples.size() * worker / workers;
        const size_t end = seed.samples.size() * (worker + 1) / workers;
        jobs.push_back(std::async(std::launch::async, harvestRange, begin, end));
    }
    for (auto& job : jobs) {
        while (job.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
            objects::util::reportProgress(progress, "searching",
                completed.load(std::memory_order_relaxed), seed.samples.size(), label);
        // Ranges merge in source order, preserving first-sample wins on exact
        // distance ties. A globally retained cell must be in each range's top K
        // if present there, so the per-range caps preserve the final selection.
        for (auto& [ti, collector] : job.get()) {
            auto [it, inserted] = matches.try_emplace(ti, params.maxCorrespondences);
            it->second.merge(collector);
        }
    }
    profile.queries += completed.load();
    objects::util::reportProgress(progress, "searching",
        completed.load(), seed.samples.size(), label);
    return matches;
}

double tangentMismatchSq(const Pose& pose, const Correspondence& v)
{
    auto normDiffSq = [](const cv::Vec3f& a, const cv::Vec3f& b) {
        const double an = std::sqrt(double(a.dot(a)));
        const double bn = std::sqrt(double(b.dot(b)));
        if (!(an > 1e-8) || !(bn > 1e-8)) return 0.0;
        const cv::Vec3d d = cv::Vec3d(a) / an - cv::Vec3d(b) / bn;
        return d.dot(d);
    };
    // target(t) == seed(R t + tr), hence J_target == J_seed R.
    const cv::Vec3f predU = float(pose.r00) * v.seedTangentU + float(pose.r10) * v.seedTangentV;
    const cv::Vec3f predV = float(pose.r01) * v.seedTangentU + float(pose.r11) * v.seedTangentV;
    return normDiffSq(v.targetTangentU, predU) + normDiffSq(v.targetTangentV, predV);
}

Pose twoPointPose(const Correspondence& a, const Correspondence& b, bool reflection, bool& valid)
{
    const Vec2 sourceDelta = b.target - a.target;
    const Vec2 targetDelta = b.seed - a.seed;
    if (normSq(sourceDelta) < 1e-12 || normSq(targetDelta) < 1e-12) {
        valid = false;
        return {};
    }
    const double angle = reflection
        ? std::atan2(targetDelta.y, targetDelta.x) + std::atan2(sourceDelta.y, sourceDelta.x)
        : std::atan2(targetDelta.y, targetDelta.x) - std::atan2(sourceDelta.y, sourceDelta.x);
    const double c = std::cos(angle), s = std::sin(angle);
    Pose pose;
    if (reflection) {
        pose.r00 = c; pose.r01 = s; pose.r10 = s; pose.r11 = -c;
    } else {
        pose.r00 = c; pose.r01 = -s; pose.r10 = s; pose.r11 = c;
    }
    const Vec2 rotated = pose.apply((a.target + b.target) * 0.5);
    const Vec2 mid = (a.seed + b.seed) * 0.5;
    pose.tx += mid.x - rotated.x;
    pose.ty += mid.y - rotated.y;
    valid = true;
    return pose;
}

Pose weightedPose(const std::vector<Correspondence>& values, const std::vector<std::uint8_t>& included,
                  const std::vector<double>& weights, bool reflection)
{
    double total = 0;
    Vec2 sourceMean, targetMean;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!included[i]) continue;
        total += weights[i];
        sourceMean = sourceMean + values[i].target * weights[i];
        targetMean = targetMean + values[i].seed * weights[i];
    }
    sourceMean = sourceMean / total;
    targetMean = targetMean / total;
    double cosineTerm = 0, sineTerm = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!included[i]) continue;
        const Vec2 src = values[i].target - sourceMean;
        const Vec2 tgt = values[i].seed - targetMean;
        if (reflection) {
            cosineTerm += weights[i] * (tgt.x * src.x - tgt.y * src.y);
            sineTerm += weights[i] * (tgt.x * src.y + tgt.y * src.x);
        } else {
            cosineTerm += weights[i] * (tgt.x * src.x + tgt.y * src.y);
            sineTerm += weights[i] * (tgt.y * src.x - tgt.x * src.y);
        }
    }
    const double angle = std::atan2(sineTerm, cosineTerm);
    const double c = std::cos(angle), s = std::sin(angle);
    Pose pose;
    if (reflection) {
        pose.r00 = c; pose.r01 = s; pose.r10 = s; pose.r11 = -c;
    } else {
        pose.r00 = c; pose.r01 = -s; pose.r10 = s; pose.r11 = c;
    }
    const Vec2 rotated = pose.apply(sourceMean);
    pose.tx += targetMean.x - rotated.x;
    pose.ty += targetMean.y - rotated.y;
    return pose;
}

std::pair<double, double> spreads(const std::vector<Correspondence>& values,
                                  const std::vector<std::uint8_t>& included)
{
    Vec2 mean;
    size_t count = 0;
    for (size_t i = 0; i < values.size(); ++i)
        if (included[i]) {
            mean = mean + values[i].seed;
            ++count;
        }
    if (count == 0) return {0, 0};
    mean = mean / double(count);
    double xx = 0, xy = 0, yy = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!included[i]) continue;
        const Vec2 d = values[i].seed - mean;
        xx += d.x * d.x; xy += d.x * d.y; yy += d.y * d.y;
    }
    const double angle = 0.5 * std::atan2(2 * xy, xx - yy);
    const Vec2 major{std::cos(angle), std::sin(angle)};
    const Vec2 minor{-major.y, major.x};
    double majLo = 1e300, majHi = -1e300, minLo = 1e300, minHi = -1e300;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!included[i]) continue;
        const double a = dot(values[i].seed, major);
        const double b = dot(values[i].seed, minor);
        majLo = std::min(majLo, a); majHi = std::max(majHi, a);
        minLo = std::min(minLo, b); minHi = std::max(minHi, b);
    }
    double first = majHi - majLo, second = minHi - minLo;
    if (second > first) std::swap(first, second);
    return {first, second};
}

struct FitResult {
    bool accepted = false;
    Pose pose;
    std::size_t inliers = 0;
    double rms = std::numeric_limits<double>::infinity();
    double majorSpread = 0, minorSpread = 0;
    std::string rejection;
};

FitResult fitPair(const std::vector<Correspondence>& values, const SurfaceOverlapParams& params, std::uint64_t seed)
{
    FitResult result;
    if (values.size() < params.minInliers) {
        result.rejection = "insufficient_correspondences";
        return result;
    }
    const double thresholdSq = params.uvInlierTolerance * params.uvInlierTolerance;
    constexpr double tangentThresholdSq = 1.0;
    auto isInlier = [&](const Pose& pose, const Correspondence& v) {
        return normSq(pose.apply(v.target) - v.seed) <= thresholdSq && tangentMismatchSq(pose, v) <= tangentThresholdSq;
    };
    std::size_t bestCount = 0;
    double bestError = std::numeric_limits<double>::infinity();
    Pose bestPose;
    std::size_t limit = params.ransacMaxHypotheses;
    std::size_t hypotheses = 0;
    std::uint64_t state = seed;
    while (hypotheses < limit) {
        state = splitmix64(state);
        const std::size_t first = state % values.size();
        state = splitmix64(state);
        std::size_t second = state % (values.size() - 1);
        if (second >= first) ++second;
        for (int reflected = 0; reflected <= (params.allowReflection ? 1 : 0) && hypotheses < limit;
             ++reflected, ++hypotheses) {
            bool valid = false;
            const Pose pose = twoPointPose(values[first], values[second], reflected != 0, valid);
            if (!valid) continue;
            std::size_t count = 0;
            double error = 0;
            for (const auto& v : values)
                if (isInlier(pose, v)) {
                    ++count;
                    error += normSq(pose.apply(v.target) - v.seed);
                }
            if (count > bestCount || (count == bestCount && error < bestError)) {
                bestCount = count;
                bestError = error;
                bestPose = pose;
                const double ratio = double(count) / double(values.size());
                const double success = ratio * ratio;
                if (success >= 1.0) {
                    limit = std::min(limit, hypotheses + 1);
                } else if (success > 0.0) {
                    const double needed = std::ceil(std::log(1.0 - params.ransacConfidence) / std::log(1.0 - success));
                    limit = std::min(limit, std::max(hypotheses + 1, std::size_t(needed)));
                }
            }
        }
    }
    if (bestCount < params.minInliers) {
        result.rejection = "insufficient_inliers";
        result.inliers = bestCount;
        return result;
    }
    const bool reflection = bestPose.reflected();
    const double huberDelta = std::min(1.0, params.uvInlierTolerance);
    std::vector<std::uint8_t> included(values.size(), 0);
    std::vector<double> weights(values.size(), 0.0);
    Pose refitted = bestPose;
    auto refitRounds = [&](int rounds) {
        for (int it = 0; it < rounds; ++it) {
            for (size_t i = 0; i < values.size(); ++i) {
                included[i] = isInlier(refitted, values[i]) ? 1 : 0;
                if (!included[i]) continue;
                const double residual = std::sqrt(normSq(refitted.apply(values[i].target) - values[i].seed));
                weights[i] = (residual <= huberDelta || residual == 0.0) ? 1.0 : huberDelta / residual;
            }
            refitted = weightedPose(values, included, weights, reflection);
        }
    };
    refitRounds(6);
    std::size_t finalCount = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        included[i] = isInlier(refitted, values[i]) ? 1 : 0;
        finalCount += included[i] ? 1 : 0;
    }
    if (finalCount < params.minInliers) {
        result.rejection = "insufficient_inliers_after_refit";
        result.inliers = finalCount;
        return result;
    }
    refitRounds(3);
    double squaredError = 0;
    for (size_t i = 0; i < values.size(); ++i)
        if (included[i]) squaredError += normSq(refitted.apply(values[i].target) - values[i].seed);
    result.rms = std::sqrt(squaredError / double(finalCount));
    result.inliers = finalCount;
    result.pose = refitted;
    const auto [major, minor] = spreads(values, included);
    result.majorSpread = major;
    result.minorSpread = minor;
    if (major + 1e-9 < params.minMajorSpread) {
        result.rejection = "insufficient_major_axis_spread";
        return result;
    }
    if (minor + 1e-9 < params.minMinorSpread) {
        result.rejection = "insufficient_minor_axis_spread";
        return result;
    }
    if (result.rms > params.maxRefitRms) {
        result.rejection = "refit_rms";
        return result;
    }
    result.accepted = true;
    return result;
}

// Dense 3D agreement counts: {overlap samples, agreements, total samples}.
struct Consistency {
    std::size_t samples = 0, agreements = 0, total = 0;
};

// Traverse seed UV -> target UV once, sharing validity and distance gates
// between detector statistics and materialized correspondences.
template <class Accept>
Consistency visitOverlapAgreements(const LoadedSurface& seed, const LoadedSurface& target, const Pose& seedToTarget,
                                   double agreementRadius, Accept accept)
{
    Consistency result;
    const double agreementSq = agreementRadius * agreementRadius;
    for (const auto& sample : seed.samples) {
        ++result.total;
        const Vec2 uv = seedToTarget.apply(sample.uv);
        const auto other = sampleXyz(target, uv);
        if (!other) continue;
        ++result.samples;
        const cv::Vec3f d = *other - sample.xyz;
        const double distanceSq = double(d.dot(d));
        if (distanceSq <= agreementSq) {
            ++result.agreements;
            accept(sample, uv, *other, distanceSq);
        }
    }
    return result;
}

Consistency overlapConsistency(const LoadedSurface& seed, const LoadedSurface& target, const Pose& targetToSeed,
                               double agreementRadius)
{
    return visitOverlapAgreements(seed, target, targetToSeed.inverse(), agreementRadius,
        [](const SamplePoint&, Vec2, const cv::Vec3f&, double) {});
}

bool bboxOverlap(const LoadedSurface& a, const LoadedSurface& b, double pad)
{
    for (int i = 0; i < 3; ++i)
        if (a.bboxLow[i] - pad > b.bboxHigh[i] || b.bboxLow[i] - pad > a.bboxHigh[i]) return false;
    return true;
}

void validateParams(const SurfaceOverlapParams& params)
{
    auto positive = [](double value, const char* name) {
        if (!(value > 0.0) || !std::isfinite(value))
            throw std::invalid_argument(std::string("graph links: ") + name + " must be positive and finite");
    };
    positive(params.tolerance, "tolerance");
    positive(params.sampleSpacing, "sampleSpacing");
    positive(params.thinningSpacing, "thinningSpacing");
    positive(params.uvInlierTolerance, "uvInlierTolerance");
    positive(params.minMajorSpread, "minMajorSpread");
    positive(params.minMinorSpread, "minMinorSpread");
    positive(params.maxRefitRms, "maxRefitRms");
    if (params.workers < 0) throw std::invalid_argument("graph links: workers must be nonnegative");
    if (params.maxCorrespondences == 0 || params.minInliers < 2 || params.ransacMaxHypotheses == 0)
        throw std::invalid_argument("graph links: correspondence/inlier/hypothesis counts must be positive");
    if (!(params.ransacConfidence > 0.0 && params.ransacConfidence < 1.0))
        throw std::invalid_argument("graph links: ransacConfidence must be between zero and one");
    if (!(params.minAgreementRatio >= 0.0 && params.minAgreementRatio <= 1.0))
        throw std::invalid_argument("graph links: minAgreementRatio must be in [0,1]");
    if (!(params.minOverlapFraction >= 0.0 && params.minOverlapFraction <= 1.0))
        throw std::invalid_argument("graph links: minOverlapFraction must be in [0,1]");
    if (params.edgeKind.empty())
        throw std::invalid_argument("graph links: edgeKind must be non-empty");
}

}  // namespace

DetectorReport linkSurfaceOverlaps(objects::ObjectStore& db,
                                 const SurfaceOverlapParams& params, const std::atomic_bool* cancel)
{
    validateParams(params);
    Profile profile;
    const auto& progress = params.progress;
    objects::util::reportProgress(progress, "selecting stored surfaces");
    DetectorReport report;
    auto cancelled = [&] { return cancel && cancel->load(); };

    detail::StoredGeometry inputs(db);
    const auto selected = inputs.select("surface", params.surfaceUuids);
    const auto fresh = detail::newAmong(params.newUuids, {&selected}, "surface-overlap");
    const auto scales = detail::coordinateScales(selected, params, report);
    if (report.failed()) return report;

    // An incremental run loads the new surfaces and the selected surfaces
    // whose stored bounds come within the pair search's bbox pad of one.
    // Each keeps its index in the selection, which seeds its pairs' fits.
    std::vector<objects::ObjectRecord> objects;
    std::vector<size_t> selectionIndex;
    if (fresh) {
        objects::util::reportProgress(progress, "finding partners of new surfaces");
        std::set<std::string> wanted = *fresh, all;
        for (const auto& object : selected) all.insert(object.uuid);
        for (const auto& object : selected) {
            if (!fresh->contains(object.uuid)) continue;
            const auto box = inputs.workingBounds(object, params);
            if (!box) { wanted = all; break; }  // no stored bounds: every surface is a partner
            for (const auto& uuid : inputs.objectsNear("surface", *box, params.tolerance, params))
                if (all.contains(uuid)) wanted.insert(uuid);
        }
        for (size_t i = 0; i < selected.size(); ++i)
            if (wanted.contains(selected[i].uuid)) { objects.push_back(selected[i]); selectionIndex.push_back(i); }
    } else {
        objects = selected;
    }

    std::vector<LoadedSurface> surfaces;
    std::vector<size_t> seedIndex;  // per loaded surface: its index for pair seeds
    size_t loadedCount = 0;
    for (size_t oi = 0; oi < objects.size(); ++oi) {
        const auto& object = objects[oi];
        objects::util::reportProgress(progress, "loading surfaces", loadedCount++, objects.size(), object.uuid);
        if (cancelled()) {
            detail::reportFailure(report, "cancel", "", "cancelled", "surface-overlap: cancelled during loading");
            return report;
        }
        PhaseTimer loadingTimer(profile.enabled ? &profile.loading : nullptr);
        if (auto loaded = loadSurface(inputs, object, params, report, scales.at(object.uuid))) {
            surfaces.push_back(std::move(*loaded));
            seedIndex.push_back(fresh ? selectionIndex[oi] : surfaces.size() - 1);
        }
    }
    objects::util::reportProgress(progress, "loading surfaces", objects.size(), objects.size());
    std::vector<char> isNew(surfaces.size(), 0);
    if (fresh)
        for (size_t i = 0; i < surfaces.size(); ++i) isNew[i] = fresh->contains(surfaces[i].uuid);
    // Pairs examined: all of them, or those with a new surface.
    auto pairsAfter = [&](size_t si) -> uint64_t {
        if (!fresh || isNew[si]) return surfaces.size() - si - 1;
        return uint64_t(std::count(isNew.begin() + si + 1, isNew.end(), 1));
    };
    report.objectsConsidered = int64_t(surfaces.size());
    for (const auto& surface : surfaces) profile.samples += surface.samples.size();

    // Triangle-accurate spatial index over every loaded surface, as in
    // vc_seg_add_overlap; each seed sample queries all eligible targets at once.
    objects::util::reportProgress(progress, "building surface spatial index");
    SurfaceIndex index;

    {
        std::vector<SurfaceIndex::SurfacePtr> grids;
        grids.reserve(surfaces.size());
        for (const auto& surface : surfaces) grids.push_back(surface.grid);
        PhaseTimer indexTimer(profile.enabled ? &profile.index : nullptr);
        index.rebuild(grids);
    }

    const double agreementRadius = params.tolerance + params.uvInlierTolerance;
    std::vector<LinkCandidate> candidates;
    std::map<std::string, std::size_t> rejections;
    uint64_t totalPairs = 0;
    for (size_t si = 0; si < surfaces.size(); ++si) totalPairs += pairsAfter(si);
    uint64_t processedPairs = 0;
    objects::util::reportProgress(progress, "checking surface pairs", 0, totalPairs);
    for (std::size_t si = 0; si < surfaces.size(); ++si) {
        const LoadedSurface& seed = surfaces[si];
        const auto item = "surface " + std::to_string(si + 1) + "/" + std::to_string(surfaces.size());
        const auto surfaceProgress = objects::util::subtaskProgress(progress, "checking surface pairs",
            processedPairs, totalPairs, item);
        std::vector<size_t> targets;
        for (size_t ti = si + 1; ti < surfaces.size(); ++ti)
            if ((!fresh || isNew[si] || isNew[ti]) && bboxOverlap(seed, surfaces[ti], params.tolerance))
                targets.push_back(ti);
        profile.bboxPairs += targets.size();
        auto matches = [&] {
            PhaseTimer timer(profile.enabled ? &profile.harvest : nullptr);
            return harvestTargets(seed, surfaces, targets, index, params, cancel, profile, surfaceProgress);
        }();
        if (cancelled()) {
            detail::reportFailure(report, "cancel", "", "cancelled", "surface-overlap: cancelled");
            return report;
        }
        objects::util::reportProgress(surfaceProgress, "fitting overlaps");
        for (auto& [ti, collector] : matches) {
            if (cancelled()) {
                detail::reportFailure(report, "cancel", "", "cancelled", "surface-overlap: cancelled");
                return report;
            }
            const LoadedSurface& target = surfaces[ti];
            const auto correspondences = collector.take();
            if (correspondences.size() < params.minInliers) continue;
            const std::uint64_t pairSeed = splitmix64(std::uint64_t(seedIndex[si] + 1) * 0x9E3779B9ULL ^
                                                      std::uint64_t(seedIndex[ti] + 1));
            ++profile.fitPairs;
            FitResult fit = [&] {
                PhaseTimer timer(profile.enabled ? &profile.fit : nullptr);
                return fitPair(correspondences, params, pairSeed);
            }();
            if (!fit.accepted) {
                ++rejections[fit.rejection.empty() ? "rejected" : fit.rejection];
                continue;
            }
            const Consistency consistency = [&] {
                PhaseTimer timer(profile.enabled ? &profile.agreement : nullptr);
                return overlapConsistency(seed, target, fit.pose, agreementRadius);
            }();
            if (consistency.samples < params.minInliers) {
                ++rejections["insufficient_overlap_samples"];
                continue;
            }
            const double agreementRatio = double(consistency.agreements) / double(consistency.samples);
            if (agreementRatio < params.minAgreementRatio) {
                ++rejections["overlap_conflict"];
                continue;
            }
            double overlapFraction = 0.0;
            if (params.minOverlapFraction > 0.0) {
                PhaseTimer timer(profile.enabled ? &profile.agreement : nullptr);
                const Consistency reverse = overlapConsistency(target, seed, fit.pose.inverse(), agreementRadius);
                const double seedFraction = consistency.total ? double(consistency.agreements) / double(consistency.total) : 0.0;
                const double targetFraction = reverse.total ? double(reverse.agreements) / double(reverse.total) : 0.0;
                overlapFraction = std::min(seedFraction, targetFraction);
                if (overlapFraction < params.minOverlapFraction) {
                    ++rejections["insufficient_overlap_fraction"];
                    continue;
                }
            }

            LinkCandidate candidate;
            candidate.aUuid = seed.uuid;
            candidate.bUuid = target.uuid;
            candidate.kind = params.edgeKind;
            candidate.aAnchor.wholeGeometry = true;
            candidate.bAnchor.wholeGeometry = true;
            candidate.score = agreementRatio;
            nlohmann::json fitMeta{{"correspondences", correspondences.size()},
                                   {"inliers", fit.inliers},
                                   {"rms", fit.rms},
                                   {"major_spread", fit.majorSpread},
                                   {"minor_spread", fit.minorSpread},
                                   {"agreement_ratio", agreementRatio},
                                   {"reflected", fit.pose.reflected()},
                                   {"pose", {fit.pose.r00, fit.pose.r01, fit.pose.tx,
                                             fit.pose.r10, fit.pose.r11, fit.pose.ty}}};
            if (params.minOverlapFraction > 0.0) fitMeta["overlap_fraction"] = overlapFraction;
            auto metadata = nlohmann::json{{"fit", std::move(fitMeta)}, {"coordinate_frame", nullptr}};
            if (params.volumeUuid) {
                metadata["coordinate_frame"] = {{"volume_uuid", *params.volumeUuid},
                    {"objects", {{seed.uuid, seed.coordinateProvenance}, {target.uuid, target.coordinateProvenance}}}};
                metadata["fit"]["pose_units"] = "volume_voxels";
                // Stored edges reorder endpoints, so name the pose direction by UUID.
                metadata["fit"]["pose_from"] = target.uuid;
                metadata["fit"]["pose_to"] = seed.uuid;
                metadata["fit"]["grid_scales"] = {{seed.uuid, {seed.scale[0], seed.scale[1]}},
                                                  {target.uuid, {target.scale[0], target.scale[1]}}};
                metadata["fit"]["agreement_radius"] = agreementRadius;
            }
            candidate.meta = metadata.dump();
            candidates.push_back(std::move(candidate));
        }
        processedPairs += pairsAfter(si);
        objects::util::reportProgress(progress, "checking surface pairs", processedPairs, totalPairs, item);
    }
    objects::util::reportProgress(progress, "checking surface pairs", totalPairs, totalPairs);
    for (const auto& [reason, count] : rejections)
        detail::reportDiagnostic(report, "discover", "", reason,
            "surface-overlap: " + std::to_string(count) + " pair(s) rejected: " + reason, int64_t(count));

    StaleEdgeScope scope{{params.edgeKind}, {}};
    if (fresh) {
        // Edges of a new surface, with both endpoints selected and loaded or
        // not examined (a partner that failed to load keeps its edges).
        const auto failed = detail::failedObjects(report);
        for (const auto& object : selected)
            if (!failed.contains(object.uuid)) scope.objectUuids.push_back(object.uuid);
        for (const auto& uuid : *fresh)
            if (!failed.contains(uuid)) scope.touching.push_back(uuid);
        if (scope.touching.empty()) scope.objectUuids.clear();  // no new surface loaded: remove nothing
    } else {
        for (const auto& surface : surfaces) scope.objectUuids.push_back(surface.uuid);
    }
    auto txn = db.write();
    DetectorReport committed = detail::commitStoredCandidates(db, inputs, txn, kSurfaceOverlapDetector, kSurfaceOverlapVersion, candidates, params.replaceStalePending, &scope, progress);
    if (cancelled()) throw std::runtime_error("surface-overlap: cancelled before commit");
    txn.commit();
    report.committed = true;
    report.candidatesFound = committed.candidatesFound;
    report.edgesCreated = committed.edgesCreated;
    report.edgesUpdated = committed.edgesUpdated;
    report.edgesSkippedExisting = committed.edgesSkippedExisting;
    report.staleRemoved = committed.staleRemoved;
    return report;
}

namespace {
// One overlap edge to materialize: its record's endpoints, anchors and
// metadata, its endpoints' objects, and their geometry revisions (current
// ones when unset).
struct OverlapEdgeView {
    std::string uuid, aUuid, bUuid;
    bool wholeGeometry = false;
    bool stale = false;
    std::string meta;
    std::function<objects::ObjectRecord(const std::string& uuid)> object;  // read when first needed
    std::optional<std::string> aGeometryRevision, bGeometryRevision;
};

class OverlapMaterializer {
public:
    OverlapMaterializer(objects::ObjectStore& db, const OverlapCorrespondenceParams& params)
        : db_(db), params_(params), inputs_(db)
    {
        if (params.volumeUuid.empty()) throw std::invalid_argument("overlap correspondences: volumeUuid is required");
        if (!(params.spacing > 0) || !std::isfinite(params.spacing) || params.maxPairs == 0)
            throw std::invalid_argument("overlap correspondences: spacing and maxPairs must be positive");
        loading_.volumeUuid = params.volumeUuid;
        loading_.sampleSpacing = params.spacing;
    }
    detail::StoredGeometry& inputs() { return inputs_; }

    // Throws the edge's error; `out` names the edge and its endpoints.
    void materialize(const std::function<OverlapEdgeView()>& read, OverlapCorrespondences& out)
    {
        using Json = nlohmann::json;
        const OverlapEdgeView edge = read();
        out.edgeUuid = edge.uuid; out.aUuid = edge.aUuid; out.bUuid = edge.bUuid;
        if (!edge.wholeGeometry || edge.aUuid == edge.bUuid)
            throw std::runtime_error("not a whole-surface overlap edge");
        const auto meta = Json::parse(edge.meta);
        const auto fit = meta.value("fit", Json());
        if (!fit.is_object() || !fit.contains("pose")) throw std::runtime_error("edge has no overlap pose");
        const auto frame = meta.value("coordinate_frame", Json());
        if (!frame.is_object() || frame.value("volume_uuid", "") != params_.volumeUuid ||
            fit.value("pose_units", "") != "volume_voxels")
            throw std::runtime_error("edge pose is not in volume " + params_.volumeUuid +
                " voxels; rerun surface-overlap detection with that volume_uuid");
        if (!fit.contains("pose_from") || !fit.contains("pose_to") || !fit.contains("grid_scales") ||
            !fit.contains("agreement_radius"))
            throw std::runtime_error("edge does not record its pose direction, grid scales and agreement radius; "
                                     "rerun surface-overlap detection");
        const auto from = fit.at("pose_from").get<std::string>(), to = fit.at("pose_to").get<std::string>();
        const auto& scales = fit.at("grid_scales");
        const auto agreementRadius = fit.at("agreement_radius").get<double>();
        if (!(agreementRadius > 0) || !std::isfinite(agreementRadius))
            throw std::runtime_error("invalid overlap agreement radius");
        if (std::minmax(from, to) != std::minmax(edge.aUuid, edge.bUuid))
            throw std::runtime_error("pose endpoints do not match the edge");
        if (edge.stale) throw std::runtime_error("edge geometry is stale");
        const auto& a = surface(edge, edge.aUuid, edge.aGeometryRevision);
        const auto& b = surface(edge, edge.bUuid, edge.bGeometryRevision);
        for (const auto* loaded : {&a, &b}) {
            const auto recorded = scales.at(loaded->uuid).get<std::array<double, 2>>();
            for (int axis = 0; axis < 2; ++axis)
                if (std::abs(recorded[axis] - loaded->scale[axis]) > 1e-6 * std::abs(recorded[axis]))
                    throw std::runtime_error(loaded->uuid + ": grid scale differs from the detected pose; refresh the edge");
        }
        const auto values = fit.at("pose").get<std::vector<double>>();
        if (values.size() != 6 || !std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }))
            throw std::runtime_error("invalid overlap pose");
        const Pose pose{values[0], values[1], values[3], values[4], values[2], values[5]};
        const Pose aToB = from == edge.aUuid ? pose : pose.inverse();
        std::vector<OverlapCorrespondence> agreeing;
        visitOverlapAgreements(a, b, aToB, agreementRadius,
            [&](const SamplePoint& sample, Vec2 uv, const cv::Vec3f& other, double distanceSq) {
                OverlapCorrespondence pair;
                pair.aGrid = {double(sample.row), double(sample.col)};
                pair.bGrid = {uv.y * b.scale[1], uv.x * b.scale[0]};
                pair.aXyz = {sample.xyz[0], sample.xyz[1], sample.xyz[2]};
                pair.bXyz = {other[0], other[1], other[2]};
                pair.distance = std::sqrt(distanceSq);
                agreeing.push_back(pair);
            });
        if (agreeing.empty()) throw std::runtime_error("no agreeing correspondences under the stored pose");
        const auto count = std::min(agreeing.size(), params_.maxPairs);
        for (size_t i = 0; i < count; ++i) out.pairs.push_back(agreeing[i * agreeing.size() / count]);
        out.aGeometryRevision = revisionOf(edge.aUuid, edge.aGeometryRevision);
        out.bGeometryRevision = revisionOf(edge.bUuid, edge.bGeometryRevision);
    }

private:
    const LoadedSurface& surface(const OverlapEdgeView& edge, const std::string& uuid,
                                 const std::optional<std::string>& revision)
    {
        const auto key = uuid + "@" + revision.value_or("");
        if (const auto found = surfaces_.find(key); found != surfaces_.end()) return found->second;
        const auto object = edge.object(uuid);
        double scale;
        try {
            scale = objects::geometryScaleToFrame(nlohmann::json::parse(object.meta), params_.volumeUuid);
        } catch (const std::exception& error) {
            throw std::runtime_error(object.uuid + ": " + error.what());
        }
        DetectorReport report;
        auto loaded = loadSurface(inputs_, object, loading_, report, scale, revision);
        if (!loaded)
            throw std::runtime_error(report.issues.empty() ? object.uuid + ": surface could not be loaded"
                                                           : report.issues.front().message);
        return surfaces_.emplace(key, std::move(*loaded)).first->second;
    }
    std::string revisionOf(const std::string& uuid, const std::optional<std::string>& revision)
    {
        return revision ? *revision : db_.currentGeometry(inputs_.read(), uuid)->uuid;
    }

    objects::ObjectStore& db_;
    const OverlapCorrespondenceParams& params_;
    detail::StoredGeometry inputs_;
    SurfaceOverlapParams loading_;
    std::map<std::string, LoadedSurface> surfaces_;
};

template <typename Edge>
std::vector<OverlapCorrespondences> materializeEach(OverlapMaterializer& materializer, const std::vector<Edge>& edges,
    const std::function<std::string(const Edge&)>& name, const std::function<OverlapEdgeView(const Edge&)>& read)
{
    std::vector<OverlapCorrespondences> result;
    for (const auto& edge : edges) {
        OverlapCorrespondences out;
        out.edgeUuid = name(edge);
        try {
            materializer.materialize([&] { return read(edge); }, out);
        } catch (const std::exception& error) {
            out.pairs.clear();
            out.error = error.what();
        }
        result.push_back(std::move(out));
    }
    return result;
}
}  // namespace

std::vector<OverlapCorrespondences> materializeOverlapCorrespondences(objects::ObjectStore& db,
    const std::vector<std::string>& edgeUuids, const OverlapCorrespondenceParams& params)
{
    OverlapMaterializer materializer(db, params);
    auto& inputs = materializer.inputs();
    return materializeEach<std::string>(materializer, edgeUuids, [](const std::string& id) { return id; },
        [&](const std::string& id) {
            const auto edge = db.edgeByUuid(inputs.read(), id);
            if (!edge) throw std::runtime_error("edge not found");
            OverlapEdgeView view;
            view.uuid = edge->uuid; view.aUuid = edge->aUuid; view.bUuid = edge->bUuid;
            view.wholeGeometry = edge->aAnchor.wholeGeometry && edge->bAnchor.wholeGeometry;
            view.stale = db.edgeGeometryState(inputs.read(), edge->uuid).stale;
            view.meta = edge->meta;
            view.object = [&inputs](const std::string& uuid) { return inputs.select("surface", {uuid}).front(); };
            return view;
        });
}

std::vector<OverlapCorrespondences> materializeOverlapCorrespondences(objects::ObjectStore& db,
    const std::vector<OverlapEdgeAt>& edges, const OverlapCorrespondenceParams& params)
{
    OverlapMaterializer materializer(db, params);
    return materializeEach<OverlapEdgeAt>(materializer, edges, [](const OverlapEdgeAt& edge) { return edge.uuid; },
        [](const OverlapEdgeAt& edge) {
            OverlapEdgeView view;
            view.uuid = edge.uuid; view.aUuid = edge.aUuid; view.bUuid = edge.bUuid;
            view.wholeGeometry = edge.wholeGeometry;
            view.meta = edge.meta;
            view.object = [&edge](const std::string& uuid) {
                objects::ObjectRecord record;
                record.uuid = uuid;
                record.kind = "surface";
                record.meta = uuid == edge.aUuid ? edge.aObjectMeta : edge.bObjectMeta;
                return record;
            };
            view.aGeometryRevision = edge.aGeometryRevision;
            view.bGeometryRevision = edge.bGeometryRevision;
            return view;
        });
}

}  // namespace vc::object_links
