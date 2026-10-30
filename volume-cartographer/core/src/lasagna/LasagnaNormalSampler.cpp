#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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

[[nodiscard]] size_t normalBatchWorkerCount(size_t itemCount, int parallelThreads)
{
    if (itemCount < 2) {
        return 1;
    }
    // Default (parallelThreads <= 0) honors the OpenMP ICV the way
    // FiberTrace::traceWorkerCount does, instead of hardware_concurrency().
    // The num_threads clause overrides omp_set_num_threads(1) per region, so
    // a full-width team per calling thread would oversubscribe every core
    // during a line solve (up to four calling threads) and starve the GUI and
    // render pools. The materialize loops are schedule(static) with per-slot
    // writes, so results are identical at any width. Callers that measured a
    // benefit can still request more threads explicitly.
    size_t requested = 1;
    if (parallelThreads > 0) {
        requested = static_cast<size_t>(parallelThreads);
    } else {
#ifdef _OPENMP
        const int ompThreads = omp_get_max_threads();
        requested = ompThreads > 0 ? static_cast<size_t>(ompThreads) : size_t{1};
#endif
    }
    return std::clamp<size_t>(requested, 1, itemCount);
}

template <class Fn>
void parallelFor(size_t count, size_t workers, Fn&& fn)
{
#ifdef _OPENMP
    if (workers > 1) {
        const auto signedCount = static_cast<std::ptrdiff_t>(count);
        #pragma omp parallel for schedule(static) num_threads(static_cast<int>(workers))
        for (std::ptrdiff_t index = 0; index < signedCount; ++index)
            fn(static_cast<size_t>(index));
        return;
    }
#endif
    (void)workers;
    for (size_t index = 0; index < count; ++index)
        fn(index);
}

[[nodiscard]] NormalSampleWithDerivative invalidNormal(const char* reason)
{
    return {{{0.0, 0.0, 0.0}, false, reason}, cv::Matx33d::zeros(), false};
}

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

} // namespace

void validateLasagnaNormalDatasetStructure(const LasagnaDataset& dataset)
{
    const auto& manifest = dataset.manifest();
    if (!manifest.baseShapeZYX.has_value())
        throw std::invalid_argument("normal manifest must declare base_shape_zyx");
    requiredPositiveManifestDouble(manifest, "grad_mag_encode_scale");
    requiredPositiveManifestDouble(manifest, "grad_mag_factor");

    LasagnaGroupArrays arrays(0);
    const auto nx = bindLasagnaChannel(manifest, "nx", arrays);
    const auto ny = bindLasagnaChannel(manifest, "ny", arrays);
    const auto gradMag = bindLasagnaChannel(manifest, "grad_mag", arrays);
    const auto validate = [&](const LasagnaChannelBinding& binding,
                              std::string_view channel) {
        if (binding.group == nullptr || (manifest.version != 3 && binding.group->channels.size() != 1)) {
            throw std::invalid_argument(
                "normal channel '" + std::string(channel) +
                "' must use its own 3D Lasagna group");
        }
        const double baseSpacing =
            static_cast<double>(binding.group->scaleFactor()) *
            manifest.sourceToBase;
        if (!lasagnaChannelShapeCompatible(
                *manifest.baseShapeZYX,
                baseSpacing,
                binding.shapeZYX,
                binding.chunksZYX)) {
            throw std::invalid_argument(
                "normal channel '" + std::string(channel) +
                "' shape is incompatible with base_shape_zyx and scale");
        }
    };
    validate(nx, "nx");
    validate(ny, "ny");
    validate(gradMag, "grad_mag");
    const double nxBaseSpacing = nx.spacing * manifest.workingToBaseScale;
    const double nyBaseSpacing = ny.spacing * manifest.workingToBaseScale;
    if (nx.shapeZYX != ny.shapeZYX ||
        std::abs(nxBaseSpacing - nyBaseSpacing) >
            1.0e-12 * std::max({1.0, std::abs(nxBaseSpacing),
                                std::abs(nyBaseSpacing)})) {
        throw std::invalid_argument(
            "normal nx and ny channels must have matching shape and base scale");
    }
}

class LasagnaNormalSampler::Impl {
public:
    bool masksInvalidCorners() const noexcept { return maskInvalidCorners_; }
    Impl(const LasagnaDataset& dataset, const LasagnaNormalSamplerOptions& options)
        : arrays_(options.memoryCache ? options.memoryCache
                                      : std::make_shared<vcr::MemoryCache>(options.maxCachedBytes)),
          gradMagDecodeScale_(
              requiredPositiveManifestDouble(dataset.manifest(), "grad_mag_encode_scale") /
              requiredPositiveManifestDouble(dataset.manifest(), "grad_mag_factor"))
    {
        const LasagnaDatasetManifest& manifest = dataset.manifest();
        maskInvalidCorners_ = manifest.raw.value("mask_invalid_normal_corners", false);
        nx_ = bindLasagnaChannel(manifest, "nx", arrays_);
        ny_ = bindLasagnaChannel(manifest, "ny", arrays_);
        gradMag_ = bindLasagnaChannel(manifest, "grad_mag", arrays_);
        sharedGrid_ = sameLasagnaSamplingGrid(gradMag_, nx_) && sameLasagnaSamplingGrid(gradMag_, ny_);
        if (maskInvalidCorners_ && !sharedGrid_) {
            throw std::invalid_argument("masked normal channels must share the validity sampling grid");
        }
        if (nx_.shapeZYX != ny_.shapeZYX) {
            throw std::runtime_error("Lasagna nx and ny channels must have matching spatial shapes");
        }
    }

    [[nodiscard]] std::optional<double> sampleWindingDensity(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleLasagnaChannel(gradMag_, volumePoint);
        if (!gradMag.has_value() || *gradMag < 0.0) {
            return std::nullopt;
        }
        return *gradMag / gradMagDecodeScale_;
    }

    [[nodiscard]] double windingDistance(const cv::Vec3d& a,
                                         const cv::Vec3d& b,
                                         double stepVx,
                                         bool alignWithNormal) const
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
            auto d0 = sampleWindingDensity(p0);
            auto d1 = sampleWindingDensity(p1);
            if (!d0.has_value() || !d1.has_value()) {
                return std::numeric_limits<double>::infinity();
            }
            if (alignWithNormal) {
                const cv::Vec3d direction = delta * (1.0 / distanceVx);
                const auto n0 = sampleLasagnaCompactAxisTensor(
                    nx_, ny_, p0);
                const auto n1 = sampleLasagnaCompactAxisTensor(
                    nx_, ny_, p1);
                if (!n0.has_value() || !n1.has_value() ||
                    length(*n0) <= kEpsilon || length(*n1) <= kEpsilon) {
                    return std::numeric_limits<double>::infinity();
                }
                *d0 *= std::abs(direction.dot(*n0) / length(*n0));
                *d1 *= std::abs(direction.dot(*n1) / length(*n1));
            }
            integral += 0.5 * (*d0 + *d1) * (distanceVx / static_cast<double>(intervals));
        }
        return integral;
    }

    [[nodiscard]] std::vector<double> normalAlignedWindingDistancesBatch(
        const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
        double stepVx,
        int parallelThreads) const
    {
        if (!(stepVx > 0.0) || !std::isfinite(stepVx))
            throw std::invalid_argument("winding integration step must be positive and finite");
        std::vector<double> result(
            connectors.size(), std::numeric_limits<double>::infinity());
        if (connectors.empty())
            return result;
        const bool sharedGrid = sharedGrid_;

        struct Layout {
            size_t firstPoint = 0;
            size_t intervals = 0;
            double intervalLength = 0.0;
            cv::Vec3d direction{0.0, 0.0, 0.0};
        };
        std::vector<Layout> layouts(connectors.size());
        size_t totalPoints = 0;
        for (size_t index = 0; index < connectors.size(); ++index) {
            const cv::Vec3d delta = connectors[index].second - connectors[index].first;
            const double distance = length(delta);
            if (!std::isfinite(distance))
                continue;
            if (distance <= kEpsilon) {
                result[index] = 0.0;
                continue;
            }
            const size_t intervals = std::max<size_t>(
                1, static_cast<size_t>(std::ceil(distance / stepVx)));
            if (totalPoints > std::numeric_limits<size_t>::max() - intervals - 1)
                throw std::overflow_error("batched winding sample count exceeds size_t");
            layouts[index] = {
                totalPoints,
                intervals,
                distance / static_cast<double>(intervals),
                delta * (1.0 / distance),
            };
            totalPoints += intervals + 1;
        }

        std::vector<cv::Vec3f> points;
        std::vector<size_t> pointConnectors;
        points.reserve(totalPoints);
        pointConnectors.reserve(totalPoints);
        for (size_t connector = 0; connector < connectors.size(); ++connector) {
            const auto& layout = layouts[connector];
            if (layout.intervals == 0)
                continue;
            for (size_t sample = 0; sample <= layout.intervals; ++sample) {
                const double t = static_cast<double>(sample) /
                    static_cast<double>(layout.intervals);
                const cv::Vec3d point = connectors[connector].first * (1.0 - t) +
                    connectors[connector].second * t;
                points.push_back({
                    static_cast<float>(point[0]),
                    static_cast<float>(point[1]),
                    static_cast<float>(point[2]),
                });
                pointConnectors.push_back(connector);
            }
        }

        const int workers = static_cast<int>(normalBatchWorkerCount(
            points.size(), parallelThreads));
        std::vector<double> alignedDensity(points.size(), 0.0);
        std::vector<uint8_t> valid(points.size(), uint8_t{0});
        constexpr size_t kBatchPoints = 262144;
        for (size_t begin = 0; begin < points.size(); begin += kBatchPoints) {
            const size_t end = std::min(points.size(), begin + kBatchPoints);
            std::vector<cv::Vec3f> batchPoints(
                points.begin() + static_cast<std::ptrdiff_t>(begin),
                points.begin() + static_cast<std::ptrdiff_t>(end));
            LasagnaCornerBatch groupedCorners;
            LasagnaCornerBatch densityCorners;
            LasagnaCornerBatch normalCorners;
            if (sharedGrid) {
                sampleLasagnaChannelCornerBatch(
                    {&gradMag_, &nx_, &ny_}, batchPoints, groupedCorners);
            } else {
                sampleLasagnaChannelCornerBatch({&gradMag_}, batchPoints, densityCorners);
                sampleLasagnaChannelCornerBatch({&nx_, &ny_}, batchPoints, normalCorners);
            }
            const auto materialize = [&](size_t local) {
                const auto& densityBatch = sharedGrid
                    ? groupedCorners
                    : densityCorners;
                const auto& normalBatch = sharedGrid
                    ? groupedCorners
                    : normalCorners;
                if (densityBatch.valid[local] == 0 ||
                    normalBatch.valid[local] == 0)
                    return;
                const auto densityWeights = lasagnaCornerWeights(
                    densityBatch.fractionsXYZ[local]);
                const auto normalWeights = lasagnaCornerWeights(
                    normalBatch.fractionsXYZ[local]);
                const double encodedDensity = interpolateLasagnaCorners(
                    densityBatch.values[0][local], densityWeights);
                if (!(encodedDensity >= 0.0))
                    return;
                const size_t normalOffset = sharedGrid ? 1 : 0;
                const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(
                    normalBatch.values[normalOffset][local],
                    normalBatch.values[normalOffset + 1][local],
                    normalWeights);
                const double magnitude = std::sqrt(
                    static_cast<double>(normal.dot(normal)));
                if (!(magnitude > kEpsilon))
                    return;
                const cv::Vec3d normalDouble{normal[0], normal[1], normal[2]};
                const cv::Vec3d& direction =
                    layouts[pointConnectors[begin + local]].direction;
                alignedDensity[begin + local] =
                    encodedDensity / gradMagDecodeScale_ *
                    std::abs(direction.dot(normalDouble) / magnitude);
                valid[begin + local] = uint8_t{1};
            };
#ifdef _OPENMP
            if (workers > 1) {
                const auto count = static_cast<std::ptrdiff_t>(end - begin);
#pragma omp parallel for schedule(static) num_threads(workers)
                for (std::ptrdiff_t local = 0; local < count; ++local)
                    materialize(static_cast<size_t>(local));
            } else
#endif
            {
                for (size_t local = 0; local < end - begin; ++local)
                    materialize(local);
            }
        }

        const auto integrate = [&](size_t connector) {
            const auto& layout = layouts[connector];
            if (layout.intervals == 0)
                return;
            double integral = 0.0;
            for (size_t sample = 0; sample < layout.intervals; ++sample) {
                const size_t first = layout.firstPoint + sample;
                if (valid[first] == 0 || valid[first + 1] == 0)
                    return;
                integral += 0.5 *
                    (alignedDensity[first] + alignedDensity[first + 1]) *
                    layout.intervalLength;
            }
            result[connector] = integral;
        };
#ifdef _OPENMP
        if (workers > 1) {
            const auto count = static_cast<std::ptrdiff_t>(connectors.size());
#pragma omp parallel for schedule(static) num_threads(workers)
            for (std::ptrdiff_t connector = 0; connector < count; ++connector)
                integrate(static_cast<size_t>(connector));
        } else
#endif
        {
            for (size_t connector = 0; connector < connectors.size(); ++connector)
                integrate(connector);
        }
        return result;
    }

    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const
    {
        const auto gradMag = sampleLasagnaChannel(gradMag_, volumePoint);
        if (!gradMag.has_value()) {
            return {{0.0, 0.0, 0.0}, false, "missing Lasagna grad_mag sample"};
        }
        if (*gradMag <= 0.0) {
            return {{0.0, 0.0, 0.0}, false, "Lasagna grad_mag sample is zero"};
        }

        const auto normal = sampleLasagnaCompactAxisTensor(nx_, ny_, volumePoint, maskInvalidCorners_ ? &gradMag_ : nullptr);
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
        const auto gradMag = sampleLasagnaChannel(gradMag_, volumePoint);
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

        const auto normal = sampleLasagnaCompactAxisTensor(nx_, ny_, volumePoint, maskInvalidCorners_ ? &gradMag_ : nullptr);
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

    [[nodiscard]] std::array<const LasagnaChannelBinding*, 3>
    groupedCornerBindings() const noexcept
    {
        if (!sharedGrid_)
            return {nullptr, nullptr, nullptr};
        return {&gradMag_, &nx_, &ny_};
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
            const auto weights = lasagnaCornerWeights(corners.fractionsXYZ[index]);
            const auto& gradMag = corners.values[firstVolume][index];
            if (!(interpolateLasagnaCorners(gradMag, weights) > 0.0f)) {
                samples[index] = {};
                return;
            }
            const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(
                corners.values[firstVolume + 1][index], corners.values[firstVolume + 2][index],
                maskInvalidCorners_ ? validLasagnaCornerWeights(weights, gradMag) : weights);
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

        if (!sharedGrid_) {
            std::vector<cv::Vec3d> doublePoints(volumePoints.begin(), volumePoints.end());
            std::vector<NormalSampleWithDerivative> fallback;
            report = sampleNormalBatch(doublePoints, false, parallelThreads, fallback);
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

        const auto start = Clock::now();
        LasagnaCornerBatch corners;
        sampleLasagnaChannelCornerBatch({&gradMag_, &nx_, &ny_}, volumePoints, corners);
        materializeGroupedCorners(corners, 0, parallelThreads, samples);
        report.materializeMs = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        return report;
    }

    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative,
        int parallelThreads,
        std::vector<NormalSampleWithDerivative>& samples) const
    {
        (void)withDerivative;
        using Clock = std::chrono::steady_clock;
        NormalBatchReport report;
        samples.clear();
        samples.resize(volumePoints.size());
        if (volumePoints.empty()) {
            return report;
        }
        const size_t workers = normalBatchWorkerCount(volumePoints.size(), parallelThreads);
        const auto start = Clock::now();
        if (sharedGrid_) {
            const std::vector<cv::Vec3f> floatPoints(volumePoints.begin(), volumePoints.end());
            LasagnaCornerBatch corners;
            sampleLasagnaChannelCornerBatch({&gradMag_, &nx_, &ny_}, floatPoints, corners);
            parallelFor(volumePoints.size(), workers, [&](size_t index) {
                if (!corners.valid[index]) {
                    samples[index] = invalidNormal("missing Lasagna grad_mag sample");
                    return;
                }
                const auto weights = lasagnaCornerWeights(corners.fractionsXYZ[index]);
                if (!(interpolateLasagnaCorners(corners.values[0][index], weights) > 0.0f)) {
                    samples[index] = invalidNormal("Lasagna grad_mag sample is zero");
                    return;
                }
                const cv::Vec3f normal = interpolateLasagnaCompactAxisCorners(
                    corners.values[1][index], corners.values[2][index],
                    maskInvalidCorners_ ? validLasagnaCornerWeights(weights, corners.values[0][index]) : weights);
                if (normal.dot(normal) <= 1.0e-12f) {
                    samples[index] = invalidNormal("degenerate Lasagna normal sample");
                    return;
                }
                samples[index] = {{{normal[0], normal[1], normal[2]}, true, {}},
                                  cv::Matx33d::zeros(), false};
            });
        } else {
            LasagnaCubes gradMag, nx, ny;
            readLasagnaCubes(gradMag_, volumePoints, gradMag);
            readLasagnaCubes(nx_, volumePoints, nx);
            readLasagnaCubes(ny_, volumePoints, ny);
            parallelFor(volumePoints.size(), workers, [&](size_t index) {
                if (!gradMag.valid[index]) {
                    samples[index] = invalidNormal("missing Lasagna grad_mag sample");
                    return;
                }
                if (interpolateLasagnaCube(gradMag.values[index], gradMag.fractionsXYZ[index]) <= 0.0) {
                    samples[index] = invalidNormal("Lasagna grad_mag sample is zero");
                    return;
                }
                const auto normal = nx.valid[index] && ny.valid[index]
                    ? lasagnaCompactAxisCube(nx.values[index], ny.values[index], nx.fractionsXYZ[index])
                    : std::nullopt;
                if (!normal.has_value()) {
                    samples[index] = invalidNormal("missing Lasagna nx/ny sample");
                    return;
                }
                if (length(*normal) <= kEpsilon) {
                    samples[index] = invalidNormal("degenerate Lasagna normal sample");
                    return;
                }
                samples[index] = {{*normal, true, {}}, cv::Matx33d::zeros(), false};
            });
        }
        report.materializeMs = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        return report;
    }

private:
    LasagnaGroupArrays arrays_;
    LasagnaChannelBinding nx_;
    LasagnaChannelBinding ny_;
    LasagnaChannelBinding gradMag_;
    bool sharedGrid_ = false;
    bool maskInvalidCorners_ = false;
    double gradMagDecodeScale_ = 1000.0;
};

LasagnaNormalSampler::LasagnaNormalSampler(
    const LasagnaDataset& dataset)
    : LasagnaNormalSampler(dataset, LasagnaNormalSamplerOptions{})
{
}

LasagnaNormalSampler::~LasagnaNormalSampler() = default;

LasagnaNormalSampler::LasagnaNormalSampler(
    const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options)
    : impl_(std::make_unique<Impl>(dataset, options))
{
}

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

double LasagnaNormalSampler::windingDistance(const cv::Vec3d& a,
                                             const cv::Vec3d& b,
                                             double stepVx) const
{
    return impl_->windingDistance(a, b, stepVx, false);
}

double LasagnaNormalSampler::normalAlignedWindingDistance(
    const cv::Vec3d& a,
    const cv::Vec3d& b,
    double stepVx) const
{
    return impl_->windingDistance(a, b, stepVx, true);
}

std::vector<double> LasagnaNormalSampler::normalAlignedWindingDistancesBatch(
    const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors,
    double stepVx,
    int parallelThreads) const
{
    return impl_->normalAlignedWindingDistancesBatch(
        connectors, stepVx, parallelThreads);
}

NormalSampleWithDerivative LasagnaNormalSampler::sampleNormalWithDerivative(
    const cv::Vec3d& volumePoint) const
{
    return impl_->sampleNormalWithDerivative(volumePoint);
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

bool LasagnaNormalSampler::masksInvalidCorners() const noexcept
{
    return impl_->masksInvalidCorners();
}

std::array<const LasagnaChannelBinding*, 3>
LasagnaNormalSampler::groupedCornerBindings() const noexcept
{
    return impl_->groupedCornerBindings();
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
