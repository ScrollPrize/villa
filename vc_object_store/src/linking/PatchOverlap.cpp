#include "linking/StoredGeometry.hpp"
// Patch-overlap detector for vc_object_links.
//
// Pair acceptance follows spiral-fitting's patch_merger (merging-patches
// branch): correspondences harvested by 3D proximity, a two-point
// RANSAC over 2D rigid/reflected poses between the patches'
// parameterizations gated on 3D tangent-frame agreement, Huber-weighted
// refits, principal-axis spread gates on the inliers (major = overlap
// length, minor = overlap "thickness"), and a dense 3D agreement check
// under the fitted pose. Accepted pairs become pending graph edges via
// commitCandidates(); no merged geometry is produced here.

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
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "spatial/PatchGrid.hpp"
#include "spatial/PatchIndex.hpp"
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
    bool enabled = std::getenv("VC_PATCH_OVERLAP_PROFILE") != nullptr;
    double loading = 0, index = 0, harvest = 0, fit = 0, agreement = 0;
    uint64_t bboxPairs = 0, queries = 0, fitPairs = 0, samples = 0;
    ~Profile() {
        if (enabled) {
            const nlohmann::json out{{"loading_s", loading}, {"index_s", index}, {"harvest_s", harvest},
                {"fit_s", fit}, {"agreement_s", agreement}, {"bbox_pairs", bboxPairs},
                {"queries", queries}, {"fit_pairs", fitPairs}, {"samples", samples}};
            std::fprintf(stderr, "PATCH_OVERLAP_PROFILE %s\n", out.dump().c_str());
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
    Vec2 uv;  // metric (voxel) parameterization: grid index / scale
    cv::Vec3f xyz;
    cv::Vec3f tangentU;  // d(xyz)/d(u), d(xyz)/d(v)
    cv::Vec3f tangentV;
};

struct LoadedPatch {
    std::string uuid;
    std::shared_ptr<PatchGrid> surface;
    const cv::Mat_<cv::Vec3f>* points = nullptr; // stable for this read-only detector run
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
    auto valid = [&](int r, int c) { return points(r, c)[0] != -1.0f; };
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
std::optional<cv::Vec3f> sampleXyz(const LoadedPatch& patch, Vec2 uv)
{
    const auto* points = patch.points;
    double col = uv.x * patch.scale[0];
    double row = uv.y * patch.scale[1];
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

std::optional<LoadedPatch> loadPatch(detail::StoredGeometry& inputs, const objects::ObjectRecord& object, const PatchOverlapParams& params,
                                     DetectorReport& report)
{
    LoadedPatch patch;
    try {
        patch.surface = detail::patchSurface(inputs.load(object));
        patch.uuid = object.uuid;
        patch.scale = patch.surface->scale();
        if (!(patch.scale[0] > 0) || !(patch.scale[1] > 0)) {
            report.problems.push_back(object.uuid + ": invalid scale, skipped");
            return std::nullopt;
        }

    } catch (const std::exception& e) {
        report.problems.push_back(object.uuid + ": load failed: " + e.what());
        return std::nullopt;
    }
    const auto* points = patch.surface->rawPointsPtr();
    patch.points = points;
    const int stepR = std::max(1, int(std::llround(params.sampleSpacing * patch.scale[1])));
    const int stepC = std::max(1, int(std::llround(params.sampleSpacing * patch.scale[0])));
    patch.bboxLow = {1e30f, 1e30f, 1e30f};
    patch.bboxHigh = {-1e30f, -1e30f, -1e30f};
    for (int row = 0; row < points->rows; ++row) {
        for (int col = 0; col < points->cols; ++col) {
            const cv::Vec3f p = (*points)(row, col);
            if (p[0] == -1.0f || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            patch.bboxLow = {std::min(patch.bboxLow[0], p[0]), std::min(patch.bboxLow[1], p[1]), std::min(patch.bboxLow[2], p[2])};
            patch.bboxHigh = {std::max(patch.bboxHigh[0], p[0]), std::max(patch.bboxHigh[1], p[1]), std::max(patch.bboxHigh[2], p[2])};
            if (row % stepR != 0 || col % stepC != 0) continue;
            auto tangents = tangentsAt(*points, row, col, patch.scale);
            if (!tangents) continue;
            SamplePoint sample;
            sample.uv = {col / double(patch.scale[0]), row / double(patch.scale[1])};
            sample.xyz = p;
            sample.tangentU = tangents->first;
            sample.tangentV = tangents->second;
            patch.samples.push_back(std::move(sample));
        }
    }
    if (patch.samples.empty()) {
        report.problems.push_back(object.uuid + ": no usable samples, skipped");
        return std::nullopt;
    }
    return patch;
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

void addCorrespondence(const SamplePoint& sample, const LoadedPatch& target,
    const PatchIndex::LookupResult& hit, const PatchOverlapParams& params,
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

std::map<size_t, CorrespondenceCollector> harvestTargets(const LoadedPatch& seed,
    const std::vector<LoadedPatch>& patches, const std::vector<size_t>& targets,
    const PatchIndex& index, const PatchOverlapParams& params,
    const std::atomic_bool* cancel, Profile& profile, const objects::util::ProgressCallback& progress)
{
    std::map<size_t, CorrespondenceCollector> matches;
    if (targets.empty()) return matches;
    std::unordered_map<PatchGrid*, size_t> targetIndex;
    std::unordered_set<PatchGrid*> included;
    for (const auto ti : targets) {
        targetIndex.emplace(patches[ti].surface.get(), ti);
        included.insert(patches[ti].surface.get());
    }
    using Matches = std::map<size_t, CorrespondenceCollector>;
    std::atomic_size_t completed{0};
    const auto label = seed.uuid + " -> " + std::to_string(targets.size()) + " patches";
    auto harvestRange = [&](size_t begin, size_t end) {
        Matches local;
        PatchIndex::SurfaceFilter filter;
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
                        addCorrespondence(sample, patches[ti], hit, params, it->second);
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

FitResult fitPair(const std::vector<Correspondence>& values, const PatchOverlapParams& params, std::uint64_t seed)
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

// Dense 3D agreement of one patch against another under a pose mapping
// target UV -> seed UV: {overlap samples, agreements, total samples}.
struct Consistency {
    std::size_t samples = 0, agreements = 0, total = 0;
};

Consistency overlapConsistency(const LoadedPatch& seed, const LoadedPatch& target, const Pose& targetToSeed,
                               double agreementRadius)
{
    Consistency result;
    const Pose seedToTarget = targetToSeed.inverse();
    const double agreementSq = agreementRadius * agreementRadius;
    for (const auto& sample : seed.samples) {
        ++result.total;
        const auto other = sampleXyz(target, seedToTarget.apply(sample.uv));
        if (!other) continue;
        ++result.samples;
        const cv::Vec3f d = *other - sample.xyz;
        if (double(d.dot(d)) <= agreementSq) ++result.agreements;
    }
    return result;
}

bool bboxOverlap(const LoadedPatch& a, const LoadedPatch& b, double pad)
{
    for (int i = 0; i < 3; ++i)
        if (a.bboxLow[i] - pad > b.bboxHigh[i] || b.bboxLow[i] - pad > a.bboxHigh[i]) return false;
    return true;
}

void validateParams(const PatchOverlapParams& params)
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

DetectorReport linkPatchOverlaps(objects::ObjectStore& db,
                                 const PatchOverlapParams& params, const std::atomic_bool* cancel)
{
    validateParams(params);
    Profile profile;
    const auto& progress = params.progress;
    objects::util::reportProgress(progress, "selecting stored patches");
    DetectorReport report;
    auto cancelled = [&] { return cancel && cancel->load(); };

    detail::StoredGeometry inputs(db);
    const auto objects = inputs.select("patch", params.patchUuids);

    std::vector<LoadedPatch> patches;
    size_t loadedCount = 0;
    for (const auto& object : objects) {
        objects::util::reportProgress(progress, "loading patches", loadedCount++, objects.size(), object.uuid);
        if (cancelled()) {
            report.problems.push_back("patch-overlap: cancelled during loading");
            return report;
        }
        PhaseTimer loadingTimer(profile.enabled ? &profile.loading : nullptr);
        if (auto loaded = loadPatch(inputs, object, params, report))
            patches.push_back(std::move(*loaded));
    }
    objects::util::reportProgress(progress, "loading patches", objects.size(), objects.size());
    report.objectsConsidered = int64_t(patches.size());
    for (const auto& patch : patches) profile.samples += patch.samples.size();

    // Triangle-accurate spatial index over every loaded patch, as in
    // vc_seg_add_overlap; each seed sample queries all eligible targets at once.
    objects::util::reportProgress(progress, "building patch spatial index");
    PatchIndex index;

    {
        std::vector<PatchIndex::SurfacePtr> surfaces;
        surfaces.reserve(patches.size());
        for (const auto& patch : patches) surfaces.push_back(patch.surface);
        PhaseTimer indexTimer(profile.enabled ? &profile.index : nullptr);
        index.rebuild(surfaces);
    }

    const double agreementRadius = params.tolerance + params.uvInlierTolerance;
    std::vector<LinkCandidate> candidates;
    std::map<std::string, std::size_t> rejections;
    const uint64_t totalPairs = patches.empty() ? 0 : uint64_t(patches.size()) * (patches.size() - 1) / 2;
    uint64_t processedPairs = 0;
    objects::util::reportProgress(progress, "checking patch pairs", 0, totalPairs);
    for (std::size_t si = 0; si < patches.size(); ++si) {
        const LoadedPatch& seed = patches[si];
        const auto item = "patch " + std::to_string(si + 1) + "/" + std::to_string(patches.size());
        const auto patchProgress = objects::util::subtaskProgress(progress, "checking patch pairs",
            processedPairs, totalPairs, item);
        std::vector<size_t> targets;
        for (size_t ti = si + 1; ti < patches.size(); ++ti)
            if (bboxOverlap(seed, patches[ti], params.tolerance)) targets.push_back(ti);
        profile.bboxPairs += targets.size();
        auto matches = [&] {
            PhaseTimer timer(profile.enabled ? &profile.harvest : nullptr);
            return harvestTargets(seed, patches, targets, index, params, cancel, profile, patchProgress);
        }();
        if (cancelled()) {
            report.problems.push_back("patch-overlap: cancelled");
            return report;
        }
        objects::util::reportProgress(patchProgress, "fitting overlaps");
        for (auto& [ti, collector] : matches) {
            if (cancelled()) {
                report.problems.push_back("patch-overlap: cancelled");
                return report;
            }
            const LoadedPatch& target = patches[ti];
            const auto correspondences = collector.take();
            if (correspondences.size() < params.minInliers) continue;
            const std::uint64_t pairSeed = splitmix64(std::uint64_t(si + 1) * 0x9E3779B9ULL ^ std::uint64_t(ti + 1));
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
            candidate.meta = nlohmann::json{{"fit", std::move(fitMeta)}}.dump();
            candidates.push_back(std::move(candidate));
        }
        processedPairs += patches.size() - si - 1;
        objects::util::reportProgress(progress, "checking patch pairs", processedPairs, totalPairs, item);
    }
    objects::util::reportProgress(progress, "checking patch pairs", totalPairs, totalPairs);
    for (const auto& [reason, count] : rejections)
        report.problems.push_back("patch-overlap: " + std::to_string(count) + " pair(s) rejected: " + reason);

    StaleEdgeScope scope{{params.edgeKind}, {}};
    for (const auto& patch : patches) scope.objectUuids.push_back(patch.uuid);
    auto txn = db.write();
    DetectorReport committed = detail::commitStoredCandidates(db, inputs, txn, kPatchOverlapDetector, 1, candidates, params.replaceStalePending, &scope, progress);
    if (cancelled()) throw std::runtime_error("patch-overlap: cancelled before commit");
    txn.commit();
    report.candidatesFound = committed.candidatesFound;
    report.edgesCreated = committed.edgesCreated;
    report.edgesUpdated = committed.edgesUpdated;
    report.edgesSkippedExisting = committed.edgesSkippedExisting;
    report.staleRemoved = committed.staleRemoved;
    return report;
}

}  // namespace vc::object_links

