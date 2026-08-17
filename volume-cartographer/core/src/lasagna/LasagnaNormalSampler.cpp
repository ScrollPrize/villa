#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/render/DecodedChunkCacheBudget.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] size_t normalBatchWorkerCount(
    size_t itemCount,
    int parallelThreads,
    size_t fallbackWorkers)
{
    if (itemCount < 2) {
        return 1;
    }
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const size_t requested = parallelThreads > 0
        ? static_cast<size_t>(parallelThreads)
        : (hardwareThreads == 0 ? fallbackWorkers : static_cast<size_t>(hardwareThreads));
    return std::clamp<size_t>(requested, 1, itemCount);
}

struct PreparedNormalPoint {
    LasagnaCubeRequest gradMag;
    LasagnaCubeRequest nx;
    LasagnaCubeRequest ny;
};

} // namespace

double requiredPositiveManifestDouble(
    const LasagnaDatasetManifest& manifest,
    const char* key)
{
    const auto it = manifest.raw.find(key);
    if (it == manifest.raw.end() || !it->is_number()) {
        throw std::runtime_error(
            std::string("Lasagna manifest is missing numeric field '") + key + "'");
    }
    const double value = it->get<double>();
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(
            std::string("Lasagna manifest field '") + key + "' must be positive and finite");
    }
    return value;
}

class LasagnaNormalSampler::Impl {
public:
    Impl(const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options)
        : gradMagDecodeScale_(
              requiredPositiveManifestDouble(dataset.manifest(), "grad_mag_encode_scale") /
              requiredPositiveManifestDouble(dataset.manifest(), "grad_mag_factor"))
        , options_(options)
        , cache_(sharedLasagnaChannelChunkCache(options.maxCachedBytes))
    {
        const LasagnaDatasetManifest& manifest = dataset.manifest();
        auto nxFuture = std::async(std::launch::async,
            [&manifest]() { return bindLasagnaChannel(manifest, "nx"); });
        auto nyFuture = std::async(std::launch::async,
            [&manifest]() { return bindLasagnaChannel(manifest, "ny"); });
        auto gradMagFuture = std::async(std::launch::async,
            [&manifest]() { return bindLasagnaChannel(manifest, "grad_mag"); });
        nx_ = nxFuture.get();
        ny_ = nyFuture.get();
        gradMag_ = gradMagFuture.get();

        if (sameLasagnaSamplingGrid(gradMag_, nx_) &&
            sameLasagnaSamplingGrid(gradMag_, ny_)) {
            gradMagCorners_ = std::make_unique<LasagnaChannelCornerSampler>(
                gradMag_);
            nxCorners_ = std::make_unique<LasagnaChannelCornerSampler>(
                nx_);
            nyCorners_ = std::make_unique<LasagnaChannelCornerSampler>(
                ny_);
        }

        if (nx_.shapeZYX != ny_.shapeZYX) {
            throw std::runtime_error("Lasagna nx and ny channels must have matching spatial shapes");
        }
        if (manifest.groupForChannel("pred_dt") != nullptr) {
            predDt_ = bindLasagnaChannel(manifest, "pred_dt");
        }
    }

    [[nodiscard]] std::optional<double> sampleWindingDensity(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleLasagnaChannel(gradMag_, *cache_, volumePoint);
        if (!gradMag.has_value() || *gradMag < 0.0) {
            return std::nullopt;
        }
        return *gradMag / gradMagDecodeScale_;
    }

    [[nodiscard]] std::optional<double> samplePredDt(const cv::Vec3d& volumePoint) const
    {
        if (!predDt_.has_value()) {
            return std::nullopt;
        }
        return sampleLasagnaChannel(*predDt_, *cache_, volumePoint);
    }

    [[nodiscard]] bool hasPredDtChannel() const
    {
        return predDt_.has_value();
    }

    [[nodiscard]] std::optional<double> predDtSpacing() const
    {
        if (!predDt_.has_value()) {
            return std::nullopt;
        }
        return predDt_->spacing;
    }

    [[nodiscard]] double windingDistance(const cv::Vec3d& a,
                                         const cv::Vec3d& b,
                                         double stepVx) const
    {
        const cv::Vec3d delta = b - a;
        const double distanceVx = length(delta);
        if (!(distanceVx > kEpsilon) || !std::isfinite(distanceVx)) {
            return 0.0;
        }
        const double step = std::isfinite(stepVx) && stepVx > 0.0 ? stepVx : 8.0;
        const int intervals = std::max(1, static_cast<int>(std::ceil(distanceVx / step)));
        double integral = 0.0;
        for (int i = 0; i < intervals; ++i) {
            const double t0 = static_cast<double>(i) / static_cast<double>(intervals);
            const double t1 = static_cast<double>(i + 1) / static_cast<double>(intervals);
            const cv::Vec3d p0 = a * (1.0 - t0) + b * t0;
            const cv::Vec3d p1 = a * (1.0 - t1) + b * t1;
            const auto d0 = sampleWindingDensity(p0);
            const auto d1 = sampleWindingDensity(p1);
            if (!d0.has_value() || !d1.has_value()) {
                return std::numeric_limits<double>::infinity();
            }
            integral += 0.5 * (*d0 + *d1) * (distanceVx / static_cast<double>(intervals));
        }
        return integral;
    }

    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleLasagnaChannel(gradMag_, *cache_, volumePoint);
        if (!gradMag.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"};
        }
        if (*gradMag <= 0.0) {
            return {{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"};
        }

        const auto normal = sampleLasagnaCompactAxisTensor(nx_, ny_, *cache_, volumePoint);
        if (!normal.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"};
        }
        if (length(*normal) <= kEpsilon) {
            return {{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"};
        }
        return {*normal, true, {}};
    }

    [[nodiscard]] NormalSampleWithDerivative sampleNormalWithDerivative(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleLasagnaChannel(gradMag_, *cache_, volumePoint);
        if (!gradMag.has_value()) {
            return {{{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        if (*gradMag <= 0.0) {
            return {{{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"},
                    cv::Matx33d::zeros(),
                    false};
        }

        const auto normal = sampleLasagnaCompactAxisTensor(nx_, ny_, *cache_, volumePoint);
        if (!normal.has_value()) {
            return {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        if (length(*normal) <= kEpsilon) {
            return {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                    cv::Matx33d::zeros(),
                    false};
        }
        return {{*normal, true, {}}, cv::Matx33d::zeros(), false};
    }

    [[nodiscard]] NormalPrefetchReport prefetchNormalSamples(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative) const
    {
        if (volumePoints.empty()) {
            return {};
        }

        std::vector<LasagnaChannelChunkKey> gradMagKeys;
        std::vector<LasagnaChannelChunkKey> nxKeys;
        std::vector<LasagnaChannelChunkKey> nyKeys;
        gradMagKeys.reserve(volumePoints.size() * 8);
        nxKeys.reserve(volumePoints.size() * 8);
        nyKeys.reserve(volumePoints.size() * 8);

        for (const auto& point : volumePoints) {
            appendLasagnaInterpolationChunkKeys(gradMag_, point, gradMagKeys);
            appendLasagnaInterpolationChunkKeys(nx_, point, nxKeys);
            appendLasagnaInterpolationChunkKeys(ny_, point, nyKeys);
        }

        std::vector<LasagnaChannelChunkCache::PrefetchRequest> requests;
        requests.reserve(gradMagKeys.size() + nxKeys.size() + nyKeys.size());
        const size_t keyCount = std::max({
            gradMagKeys.size(), nxKeys.size(), nyKeys.size()});
        for (size_t index = 0; index < keyCount; ++index) {
            if (index < gradMagKeys.size()) {
                requests.emplace_back(&gradMag_, gradMagKeys[index]);
            }
            if (index < nxKeys.size()) {
                requests.emplace_back(&nx_, nxKeys[index]);
            }
            if (index < nyKeys.size()) {
                requests.emplace_back(&ny_, nyKeys[index]);
            }
        }
        (void)withDerivative;
        return cache_->prefetchInterleaved(requests);
    }

    [[nodiscard]] std::array<const LasagnaChannelCornerSampler*, 3>
    groupedCornerSamplers() const noexcept
    {
        return {gradMagCorners_.get(), nxCorners_.get(), nyCorners_.get()};
    }

    void materializeGroupedCorners(
        const std::vector<std::vector<LasagnaCornerSample>>& corners,
        size_t firstVolume,
        int parallelThreads,
        std::vector<LasagnaNormalSampler::FloatNormalSample>& samples) const
    {
        if (firstVolume + 3 > corners.size())
            throw std::invalid_argument("normal corner batch is missing channel volumes");
        const auto& gradMagCorners = corners[firstVolume];
        const auto& nxCorners = corners[firstVolume + 1];
        const auto& nyCorners = corners[firstVolume + 2];
        if (nxCorners.size() != gradMagCorners.size() ||
            nyCorners.size() != gradMagCorners.size()) {
            throw std::invalid_argument("normal corner channel sample counts differ");
        }
        samples.resize(gradMagCorners.size());
        const int workers = std::clamp(
            parallelThreads, 1, static_cast<int>(std::max<size_t>(1, samples.size())));
        auto materialize = [&](size_t index) {
            if (!gradMagCorners[index].valid ||
                !(interpolateLasagnaCorners(gradMagCorners[index]) > 0.0f) ||
                !nxCorners[index].valid ||
                !nyCorners[index].valid) {
                samples[index] = {};
                return;
            }
            const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(
                nxCorners[index], nyCorners[index]);
            samples[index] = {normal, normal.dot(normal) > 1.0e-12f};
        };
#ifdef _OPENMP
        if (workers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(samples.size());
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                materialize(static_cast<size_t>(rawIndex));
            return;
        }
#endif
        for (size_t index = 0; index < samples.size(); ++index)
            materialize(index);
    }

    void materializeGroupedCorners(
        const LasagnaCornerBatch& corners,
        size_t firstVolume,
        int parallelThreads,
        std::vector<LasagnaNormalSampler::FloatNormalSample>& samples) const
    {
        if (firstVolume + 3 > corners.values.size())
            throw std::invalid_argument("normal corner batch is missing channel volumes");
        const size_t count = corners.fractionsXYZ.size();
        if (corners.valid.size() != count ||
            corners.values[firstVolume].size() != count ||
            corners.values[firstVolume + 1].size() != count ||
            corners.values[firstVolume + 2].size() != count) {
            throw std::invalid_argument("normal corner channel sample counts differ");
        }
        samples.resize(count);
        const int workers = std::clamp(
            parallelThreads, 1, static_cast<int>(std::max<size_t>(1, count)));
        auto materialize = [&](size_t index) {
            if (corners.valid[index] == 0) {
                samples[index] = {};
                return;
            }
            const cv::Vec3f fraction = corners.fractionsXYZ[index];
            const LasagnaCornerSample gradMag{
                corners.values[firstVolume][index], fraction, true};
            if (!(interpolateLasagnaCorners(gradMag) > 0.0f)) {
                samples[index] = {};
                return;
            }
            const LasagnaCornerSample nx{
                corners.values[firstVolume + 1][index], fraction, true};
            const LasagnaCornerSample ny{
                corners.values[firstVolume + 2][index], fraction, true};
            const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(nx, ny);
            samples[index] = {normal, normal.dot(normal) > 1.0e-12f};
        };
#ifdef _OPENMP
        if (workers > 1) {
            const auto rawCount = static_cast<std::ptrdiff_t>(count);
            #pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t rawIndex = 0; rawIndex < rawCount; ++rawIndex)
                materialize(static_cast<size_t>(rawIndex));
            return;
        }
#endif
        for (size_t index = 0; index < count; ++index)
            materialize(index);
    }

    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3f>& volumePoints,
        int parallelThreads,
        std::vector<LasagnaNormalSampler::FloatNormalSample>& samples) const
    {
        using Clock = std::chrono::steady_clock;
        NormalBatchReport report;
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty())
            return report;

        if (!gradMagCorners_ || !nxCorners_ || !nyCorners_) {
            std::vector<cv::Vec3d> doublePoints;
            doublePoints.reserve(volumePoints.size());
            for (const auto& point : volumePoints)
                doublePoints.emplace_back(point[0], point[1], point[2]);
            std::vector<NormalSampleWithDerivative> fallback;
            report = sampleNormalBatch(
                doublePoints, false, parallelThreads, fallback);
            for (size_t index = 0; index < fallback.size(); ++index) {
                const auto& sample = fallback[index].sample;
                samples[index] = {
                    {static_cast<float>(sample.normal[0]),
                     static_cast<float>(sample.normal[1]),
                     static_cast<float>(sample.normal[2])},
                    sample.valid};
            }
            return report;
        }

        const auto directStart = Clock::now();
        std::vector<std::vector<LasagnaCornerSample>> groupedCorners;
        report.prefetch = sampleLasagnaChannelCornerBatch(
            {gradMagCorners_.get(), nxCorners_.get(), nyCorners_.get()},
            volumePoints,
            groupedCorners,
            parallelThreads);
        materializeGroupedCorners(groupedCorners, 0, parallelThreads, samples);
        report.materializeMs = std::chrono::duration<double, std::milli>(
            Clock::now() - directStart).count();
        return report;
    }

    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative,
        int parallelThreads,
        std::vector<NormalSampleWithDerivative>& samples) const
    {
        using Clock = std::chrono::steady_clock;
        NormalBatchReport report;
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty()) {
            return report;
        }

        const size_t readWorkers = lasagnaReadWorkersPerChannel();
        const size_t batchWorkers =
            normalBatchWorkerCount(volumePoints.size(), parallelThreads, readWorkers);
        const bool sharedSpatialGrid =
            sameLasagnaSamplingGrid(gradMag_, nx_) &&
            sameLasagnaSamplingGrid(gradMag_, ny_);
        if (gradMagCorners_ && nxCorners_ && nyCorners_) {
        const auto directStart = Clock::now();
        std::vector<cv::Vec3f> floatPoints;
        floatPoints.reserve(volumePoints.size());
        for (const auto& point : volumePoints) {
            floatPoints.push_back({
                static_cast<float>(point[0]),
                static_cast<float>(point[1]),
                static_cast<float>(point[2])});
        }
        std::vector<LasagnaCornerSample> gradMagCorners;
        std::vector<LasagnaCornerSample> nxCorners;
        std::vector<LasagnaCornerSample> nyCorners;
        std::vector<std::vector<LasagnaCornerSample>> groupedCorners;
        report.prefetch = sampleLasagnaChannelCornerBatch(
            {gradMagCorners_.get(), nxCorners_.get(), nyCorners_.get()},
            floatPoints,
            groupedCorners,
            parallelThreads);
        gradMagCorners = std::move(groupedCorners[0]);
        nxCorners = std::move(groupedCorners[1]);
        nyCorners = std::move(groupedCorners[2]);

        auto materializeDirect = [&](size_t index) {
            if (!gradMagCorners[index].valid) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            const float gradMag = interpolateLasagnaCorners(gradMagCorners[index]);
            if (!(gradMag > 0.0f)) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            if (!nxCorners[index].valid || !nyCorners[index].valid) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(
                nxCorners[index], nyCorners[index]);
            if (normal.dot(normal) <= 1.0e-12f) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            samples[index] = {{{normal[0], normal[1], normal[2]}, true, {}},
                              cv::Matx33d::zeros(), false};
        };

        if (batchWorkers <= 1) {
            for (size_t index = 0; index < volumePoints.size(); ++index)
                materializeDirect(index);
        } else {
#ifdef _OPENMP
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel for schedule(static) num_threads(static_cast<int>(batchWorkers))
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex)
                materializeDirect(static_cast<size_t>(rawIndex));
#else
            std::vector<std::future<void>> futures;
            futures.reserve(batchWorkers);
            std::atomic<size_t> next{0};
            for (size_t worker = 0; worker < batchWorkers; ++worker) {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const size_t index = next.fetch_add(1);
                        if (index >= volumePoints.size())
                            return;
                        materializeDirect(index);
                    }
                }));
            }
            for (auto& future : futures)
                future.get();
#endif
        }
        const auto directEnd = Clock::now();
        report.materializeMs = std::chrono::duration<double, std::milli>(
            directEnd - directStart).count();
        (void)withDerivative;
        return report;
        }

        const auto prefetchStart = Clock::now();
        std::vector<PreparedNormalPoint> prepared(volumePoints.size());
        std::vector<LasagnaChannelChunkKey> gradMagKeys;
        std::vector<LasagnaChannelChunkKey> nxKeys;
        std::vector<LasagnaChannelChunkKey> nyKeys;
        const auto preparePoint = [&](size_t index,
                                      std::vector<LasagnaChannelChunkKey>& localGradMagKeys,
                                      std::vector<LasagnaChannelChunkKey>& localNxKeys,
                                      std::vector<LasagnaChannelChunkKey>& localNyKeys) {
            auto& point = prepared[index];
            point.gradMag = prepareLasagnaCubeRequest(gradMag_, volumePoints[index]);
            if (sharedSpatialGrid) {
                point.nx = cloneLasagnaCubeRequestForBinding(point.gradMag, nx_);
                point.ny = cloneLasagnaCubeRequestForBinding(point.gradMag, ny_);
            } else {
                point.nx = prepareLasagnaCubeRequest(nx_, volumePoints[index]);
                point.ny = prepareLasagnaCubeRequest(ny_, volumePoints[index]);
            }
            appendUniqueLasagnaCubeRequestChunkKeys(point.gradMag, localGradMagKeys);
            appendUniqueLasagnaCubeRequestChunkKeys(point.nx, localNxKeys);
            appendUniqueLasagnaCubeRequestChunkKeys(point.ny, localNyKeys);
        };

#ifdef _OPENMP
        if (batchWorkers > 1) {
            std::vector<std::vector<LasagnaChannelChunkKey>> gradMagKeysByWorker(batchWorkers);
            std::vector<std::vector<LasagnaChannelChunkKey>> nxKeysByWorker(batchWorkers);
            std::vector<std::vector<LasagnaChannelChunkKey>> nyKeysByWorker(batchWorkers);
            const size_t reservePerWorker =
                volumePoints.size() / batchWorkers + 16;
            for (size_t worker = 0; worker < batchWorkers; ++worker) {
                gradMagKeysByWorker[worker].reserve(reservePerWorker);
                nxKeysByWorker[worker].reserve(reservePerWorker);
                nyKeysByWorker[worker].reserve(reservePerWorker);
            }
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel num_threads(static_cast<int>(batchWorkers))
            {
                const size_t worker = static_cast<size_t>(omp_get_thread_num());
                auto& localGradMagKeys = gradMagKeysByWorker[worker];
                auto& localNxKeys = nxKeysByWorker[worker];
                auto& localNyKeys = nyKeysByWorker[worker];
                #pragma omp for schedule(static)
                for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                    preparePoint(
                        static_cast<size_t>(rawIndex),
                        localGradMagKeys,
                        localNxKeys,
                        localNyKeys);
                }
                deduplicateLasagnaChunkKeysInPlace(localGradMagKeys);
                deduplicateLasagnaChunkKeysInPlace(localNxKeys);
                deduplicateLasagnaChunkKeysInPlace(localNyKeys);
            }
            auto mergeKeys = [](std::vector<LasagnaChannelChunkKey>& out,
                                std::vector<std::vector<LasagnaChannelChunkKey>>& byWorker) {
                size_t total = 0;
                for (const auto& local : byWorker) {
                    total += local.size();
                }
                out.reserve(total);
                for (auto& local : byWorker) {
                    out.insert(
                        out.end(),
                        std::make_move_iterator(local.begin()),
                        std::make_move_iterator(local.end()));
                }
            };
            mergeKeys(gradMagKeys, gradMagKeysByWorker);
            mergeKeys(nxKeys, nxKeysByWorker);
            mergeKeys(nyKeys, nyKeysByWorker);
        } else
#endif
        {
            gradMagKeys.reserve(volumePoints.size());
            nxKeys.reserve(volumePoints.size());
            nyKeys.reserve(volumePoints.size());
            for (size_t index = 0; index < volumePoints.size(); ++index) {
                preparePoint(index, gradMagKeys, nxKeys, nyKeys);
            }
            deduplicateLasagnaChunkKeysInPlace(gradMagKeys);
            deduplicateLasagnaChunkKeysInPlace(nxKeys);
            deduplicateLasagnaChunkKeysInPlace(nyKeys);
        }

        LasagnaChannelChunkCache::ResolvedChunkMap gradMagChunks;
        LasagnaChannelChunkCache::ResolvedChunkMap nxChunks;
        LasagnaChannelChunkCache::ResolvedChunkMap nyChunks;
        auto gradMagPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(
                gradMag_, *gradMag_.array, gradMagKeys, readWorkers, gradMagChunks);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(nx_, *nx_.array, nxKeys, readWorkers, nxChunks);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(ny_, *ny_.array, nyKeys, readWorkers, nyChunks);
        });
        const NormalPrefetchReport gradMagReport = gradMagPrefetch.get();
        const NormalPrefetchReport nxReport = nxPrefetch.get();
        const NormalPrefetchReport nyReport = nyPrefetch.get();
        report.prefetch.requestedChunks = gradMagReport.requestedChunks +
                                          nxReport.requestedChunks +
                                          nyReport.requestedChunks;
        report.prefetch.chunksRead = gradMagReport.chunksRead +
                                     nxReport.chunksRead +
                                     nyReport.chunksRead;

        const auto assignPointChunks = [&](size_t index) {
            auto& point = prepared[index];
            assignResolvedLasagnaCubeRequestChunks(point.gradMag, gradMagChunks);
            assignResolvedLasagnaCubeRequestChunks(point.nx, nxChunks);
            assignResolvedLasagnaCubeRequestChunks(point.ny, nyChunks);
        };
#ifdef _OPENMP
        if (batchWorkers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(prepared.size());
            #pragma omp parallel for schedule(static) num_threads(static_cast<int>(batchWorkers))
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                assignPointChunks(static_cast<size_t>(rawIndex));
            }
        } else
#endif
        {
            for (size_t index = 0; index < prepared.size(); ++index) {
                assignPointChunks(index);
            }
        }
        const auto prefetchEnd = Clock::now();

        const size_t materializeWorkers = batchWorkers;
        const auto materializeOne = [&](size_t index) {
            const PreparedNormalPoint& point = prepared[index];
            const auto gradMag = sampleLasagnaChannel(gradMag_, point.gradMag);
            if (!gradMag.has_value()) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            if (*gradMag <= 0.0) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }

            const auto normal = sampleLasagnaCompactAxisTensor(nx_, ny_, point.nx, point.ny);
            if (!normal.has_value()) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "missing Lasagna nx/ny sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            if (length(*normal) <= kEpsilon) {
                samples[index] = {{{0.0, 0.0, 0.0}, false, "degenerate Lasagna normal sample"},
                                  cv::Matx33d::zeros(),
                                  false};
                return;
            }
            samples[index] = {{*normal, true, {}}, cv::Matx33d::zeros(), false};
        };

        if (materializeWorkers <= 1) {
            for (size_t index = 0; index < volumePoints.size(); ++index) {
                materializeOne(index);
            }
        } else {
#ifdef _OPENMP
            std::atomic<bool> failed{false};
            std::exception_ptr firstError;
            const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
            #pragma omp parallel for schedule(static) num_threads(static_cast<int>(materializeWorkers))
            for (std::ptrdiff_t rawIndex = 0; rawIndex < count; ++rawIndex) {
                if (failed.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    materializeOne(static_cast<size_t>(rawIndex));
                } catch (...) {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true)) {
                        #pragma omp critical(lasagna_normal_batch_error)
                        {
                            if (!firstError) {
                                firstError = std::current_exception();
                            }
                        }
                    }
                }
            }
            if (firstError) {
                std::rethrow_exception(firstError);
            }
#else
            std::vector<std::future<void>> futures;
            futures.reserve(materializeWorkers);
            std::atomic<size_t> next{0};
            for (size_t worker = 0; worker < materializeWorkers; ++worker) {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true) {
                        const size_t index = next.fetch_add(1);
                        if (index >= volumePoints.size()) {
                            return;
                        }
                        materializeOne(index);
                    }
                }));
            }
            for (auto& future : futures) {
                future.get();
            }
#endif
        }
        const auto materializeEnd = Clock::now();
        report.prefetchMs = std::chrono::duration<double, std::milli>(
            prefetchEnd - prefetchStart).count();
        report.materializeMs = std::chrono::duration<double, std::milli>(
            materializeEnd - prefetchEnd).count();
        (void)withDerivative;
        return report;
    }

private:
    LasagnaChannelBinding nx_;
    LasagnaChannelBinding ny_;
    LasagnaChannelBinding gradMag_;
    std::optional<LasagnaChannelBinding> predDt_;
    double gradMagDecodeScale_ = 1000.0;
    LasagnaNormalSamplerOptions options_;
    std::shared_ptr<LasagnaChannelChunkCache> cache_;
    std::unique_ptr<LasagnaChannelCornerSampler> gradMagCorners_;
    std::unique_ptr<LasagnaChannelCornerSampler> nxCorners_;
    std::unique_ptr<LasagnaChannelCornerSampler> nyCorners_;
};

LasagnaNormalSampler::LasagnaNormalSampler(
    const LasagnaDataset& dataset,
    LasagnaNormalSamplerOptions options)
    : impl_(std::make_unique<Impl>(dataset, options))
{
}

LasagnaNormalSampler::~LasagnaNormalSampler() = default;

LasagnaNormalSampler::LasagnaNormalSampler(LasagnaNormalSampler&&) noexcept = default;

LasagnaNormalSampler& LasagnaNormalSampler::operator=(LasagnaNormalSampler&&) noexcept = default;

NormalSample LasagnaNormalSampler::sampleNormal(const cv::Vec3d& volumePoint) const
{
    return impl_->sampleNormal(volumePoint);
}

std::optional<double> LasagnaNormalSampler::sampleWindingDensity(const cv::Vec3d& volumePoint) const
{
    return impl_->sampleWindingDensity(volumePoint);
}

std::optional<double> LasagnaNormalSampler::samplePredDt(const cv::Vec3d& volumePoint) const
{
    return impl_->samplePredDt(volumePoint);
}

bool LasagnaNormalSampler::hasPredDtChannel() const
{
    return impl_->hasPredDtChannel();
}

std::optional<double> LasagnaNormalSampler::predDtSpacing() const
{
    return impl_->predDtSpacing();
}

double LasagnaNormalSampler::windingDistance(const cv::Vec3d& a,
                                             const cv::Vec3d& b,
                                             double stepVx) const
{
    return impl_->windingDistance(a, b, stepVx);
}

NormalSampleWithDerivative LasagnaNormalSampler::sampleNormalWithDerivative(
    const cv::Vec3d& volumePoint) const
{
    return impl_->sampleNormalWithDerivative(volumePoint);
}

NormalPrefetchReport LasagnaNormalSampler::prefetchNormalSamples(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative) const
{
    return impl_->prefetchNormalSamples(volumePoints, withDerivative);
}

NormalBatchReport LasagnaNormalSampler::sampleNormalBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative,
    std::vector<NormalSampleWithDerivative>& samples) const
{
    return impl_->sampleNormalBatch(volumePoints, withDerivative, 0, samples);
}

NormalBatchReport LasagnaNormalSampler::sampleNormalBatch(
    const std::vector<cv::Vec3d>& volumePoints,
    bool withDerivative,
    int parallelThreads,
    std::vector<NormalSampleWithDerivative>& samples) const
{
    return impl_->sampleNormalBatch(volumePoints, withDerivative, parallelThreads, samples);
}

NormalBatchReport LasagnaNormalSampler::sampleNormalBatch(
    const std::vector<cv::Vec3f>& volumePoints,
    int parallelThreads,
    std::vector<FloatNormalSample>& samples) const
{
    return impl_->sampleNormalBatch(volumePoints, parallelThreads, samples);
}

std::array<const LasagnaChannelCornerSampler*, 3>
LasagnaNormalSampler::groupedCornerSamplers() const noexcept
{
    return impl_->groupedCornerSamplers();
}

void LasagnaNormalSampler::materializeGroupedCorners(
    const std::vector<std::vector<LasagnaCornerSample>>& corners,
    size_t firstVolume,
    int parallelThreads,
    std::vector<FloatNormalSample>& samples) const
{
    impl_->materializeGroupedCorners(
        corners, firstVolume, parallelThreads, samples);
}

void LasagnaNormalSampler::materializeGroupedCorners(
    const LasagnaCornerBatch& corners,
    size_t firstVolume,
    int parallelThreads,
    std::vector<FloatNormalSample>& samples) const
{
    impl_->materializeGroupedCorners(
        corners, firstVolume, parallelThreads, samples);
}

} // namespace vc::lasagna
