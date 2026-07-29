#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
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

    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative,
        std::vector<NormalSampleWithDerivative>& samples) const
    {
        using Clock = std::chrono::steady_clock;
        NormalBatchReport report;
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty()) {
            return report;
        }

        const auto prefetchStart = Clock::now();
        std::vector<PreparedNormalPoint> prepared(volumePoints.size());
        std::vector<LasagnaChannelChunkKey> gradMagKeys;
        std::vector<LasagnaChannelChunkKey> nxKeys;
        std::vector<LasagnaChannelChunkKey> nyKeys;
        gradMagKeys.reserve(volumePoints.size() * 8);
        nxKeys.reserve(volumePoints.size() * 8);
        nyKeys.reserve(volumePoints.size() * 8);
        for (size_t index = 0; index < volumePoints.size(); ++index) {
            auto& point = prepared[index];
            point.gradMag = prepareLasagnaCubeRequest(gradMag_, volumePoints[index]);
            point.nx = prepareLasagnaCubeRequest(nx_, volumePoints[index]);
            point.ny = prepareLasagnaCubeRequest(ny_, volumePoints[index]);
            if (point.gradMag.valid) {
                gradMagKeys.insert(
                    gradMagKeys.end(), point.gradMag.keys.begin(), point.gradMag.keys.end());
            }
            if (point.nx.valid) {
                nxKeys.insert(nxKeys.end(), point.nx.keys.begin(), point.nx.keys.end());
            }
            if (point.ny.valid) {
                nyKeys.insert(nyKeys.end(), point.ny.keys.begin(), point.ny.keys.end());
            }
        }

        const size_t workers = lasagnaReadWorkersPerChannel();

        LasagnaChannelChunkCache::ResolvedChunkMap gradMagChunks;
        LasagnaChannelChunkCache::ResolvedChunkMap nxChunks;
        LasagnaChannelChunkCache::ResolvedChunkMap nyChunks;
        auto gradMagPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(
                gradMag_, *gradMag_.array, gradMagKeys, workers, gradMagChunks);
        });
        auto nxPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(nx_, *nx_.array, nxKeys, workers, nxChunks);
        });
        auto nyPrefetch = std::async(std::launch::async, [&]() {
            return cache_->prefetchResolved(ny_, *ny_.array, nyKeys, workers, nyChunks);
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

        const auto assignChunks = [](std::vector<PreparedNormalPoint>& points,
                                     const LasagnaChannelChunkCache::ResolvedChunkMap& chunks,
                                     LasagnaCubeRequest PreparedNormalPoint::*member) {
            for (auto& point : points) {
                LasagnaCubeRequest& request = point.*member;
                if (!request.valid) {
                    continue;
                }
                for (size_t cubeIndex = 0; cubeIndex < request.keys.size(); ++cubeIndex) {
                    auto it = chunks.find(request.keys[cubeIndex]);
                    if (it != chunks.end()) {
                        request.chunks[cubeIndex] = it->second;
                    }
                }
            }
        };
        assignChunks(prepared, gradMagChunks, &PreparedNormalPoint::gradMag);
        assignChunks(prepared, nxChunks, &PreparedNormalPoint::nx);
        assignChunks(prepared, nyChunks, &PreparedNormalPoint::ny);
        const auto prefetchEnd = Clock::now();

        const size_t materializeWorkers = std::min(volumePoints.size(), workers);
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
    return impl_->sampleNormalBatch(volumePoints, withDerivative, samples);
}

} // namespace vc::lasagna
