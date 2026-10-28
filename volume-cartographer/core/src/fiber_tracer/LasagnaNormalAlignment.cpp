#include "vc/fiber_tracer/LasagnaNormalAlignment.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <queue>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr std::size_t kProgressStride = 65536;

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const float squared = value.dot(value);
    if (!std::isfinite(squared) || !(squared > 1.0e-12F))
        throw std::invalid_argument("Normal alignment sample is degenerate");
    return value * (1.0F / std::sqrt(squared));
}

std::size_t checkedProduct(const std::array<std::size_t, 3>& shape)
{
    std::size_t result = 1;
    for (const std::size_t extent : shape) {
        if (extent == 0 || result > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::invalid_argument("Normal alignment lattice is empty or exceeds size_t");
        }
        result *= extent;
    }
    return result;
}

std::size_t checkedSum(std::initializer_list<std::size_t> values)
{
    std::size_t result = 0;
    for (const std::size_t value : values) {
        if (value > std::numeric_limits<std::size_t>::max() - result) {
            throw std::invalid_argument(
                "Normal alignment progress work exceeds size_t");
        }
        result += value;
    }
    return result;
}

void notifyProgress(
    const LasagnaNormalAlignmentProgressCallback& progress,
    LasagnaNormalAlignmentProgressPhase phase,
    std::size_t completed,
    std::size_t total,
    double residual = 0.0)
{
    if (progress)
        progress({phase, completed, total, residual});
}

}  // namespace

LasagnaNormalLattice makeLasagnaNormalLattice(const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ, double spacingBaseVoxels)
{
    if (!std::isfinite(spacingBaseVoxels) || !(spacingBaseVoxels > 0.0)) {
        throw std::invalid_argument("Normal alignment lattice spacing must be finite and positive");
    }
    LasagnaNormalLattice lattice;
    lattice.spacingBaseVoxels = spacingBaseVoxels;
    std::array<std::int64_t, 3> end{};
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(minimumBaseXYZ[axis]) || !std::isfinite(maximumBaseXYZ[axis]) || !(maximumBaseXYZ[axis] > minimumBaseXYZ[axis])) {
            throw std::invalid_argument("Normal alignment bbox must have finite increasing XYZ bounds");
        }
        lattice.beginXYZ[axis] = static_cast<std::int64_t>(std::ceil(minimumBaseXYZ[axis] / spacingBaseVoxels));
        end[axis] = static_cast<std::int64_t>(std::ceil(maximumBaseXYZ[axis] / spacingBaseVoxels));
        if (end[axis] <= lattice.beginXYZ[axis]) {
            throw std::invalid_argument("Normal alignment bbox contains no globally anchored samples");
        }
        lattice.shapeXYZ[axis] = static_cast<std::size_t>(end[axis] - lattice.beginXYZ[axis]);
    }
    lattice.positionsBaseXYZ.reserve(checkedProduct(lattice.shapeXYZ));
    for (std::size_t z = 0; z < lattice.shapeXYZ[2]; ++z) {
        for (std::size_t y = 0; y < lattice.shapeXYZ[1]; ++y) {
            for (std::size_t x = 0; x < lattice.shapeXYZ[0]; ++x) {
                lattice.positionsBaseXYZ.push_back({
                    static_cast<float>(static_cast<double>(lattice.beginXYZ[0] + static_cast<std::int64_t>(x)) * spacingBaseVoxels),
                    static_cast<float>(static_cast<double>(lattice.beginXYZ[1] + static_cast<std::int64_t>(y)) * spacingBaseVoxels),
                    static_cast<float>(static_cast<double>(lattice.beginXYZ[2] + static_cast<std::int64_t>(z)) * spacingBaseVoxels),
                });
            }
        }
    }
    return lattice;
}

std::vector<BinaryPairwiseFactor> makeLasagnaNormalLatticeFactors(
    const LasagnaNormalLattice& lattice,
    std::span<const std::size_t> nodeByLatticeSample,
    std::span<const cv::Vec3f> retainedNormals,
    int neighborRadius,
    const LasagnaNormalAlignmentProgressCallback& progress)
{
    const std::size_t sampleCount = checkedProduct(lattice.shapeXYZ);
    if (lattice.positionsBaseXYZ.size() != sampleCount || nodeByLatticeSample.size() != sampleCount) {
        throw std::invalid_argument("Normal alignment lattice arrays have inconsistent sizes");
    }
    if (neighborRadius < 1) {
        throw std::invalid_argument("Normal alignment neighbor radius must be positive");
    }
    constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
    for (const std::size_t node : nodeByLatticeSample) {
        if (node != missing && node >= retainedNormals.size()) {
            throw std::invalid_argument("Normal alignment lattice references an invalid retained sample");
        }
    }
    const auto linear = [&](std::size_t x, std::size_t y, std::size_t z) { return (z * lattice.shapeXYZ[1] + y) * lattice.shapeXYZ[0] + x; };
    std::vector<BinaryPairwiseFactor> factors;
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Factors,
        0,
        sampleCount);
    std::size_t nextProgress = std::min(kProgressStride, sampleCount);
    for (std::size_t z = 0; z < lattice.shapeXYZ[2]; ++z) {
        for (std::size_t y = 0; y < lattice.shapeXYZ[1]; ++y) {
            for (std::size_t x = 0; x < lattice.shapeXYZ[0]; ++x) {
                const std::size_t a = nodeByLatticeSample[linear(x, y, z)];
                if (a == missing)
                    continue;
                for (int dz = -neighborRadius; dz <= neighborRadius; ++dz) {
                    for (int dy = -neighborRadius; dy <= neighborRadius; ++dy) {
                        for (int dx = -neighborRadius; dx <= neighborRadius; ++dx) {
                            if (dz < 0 || (dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx <= 0)) {
                                continue;
                            }
                            const auto neighborX = static_cast<std::int64_t>(x) + dx;
                            const auto neighborY = static_cast<std::int64_t>(y) + dy;
                            const auto neighborZ = static_cast<std::int64_t>(z) + dz;
                            if (neighborX < 0 || neighborY < 0 || neighborZ < 0 || neighborX >= static_cast<std::int64_t>(lattice.shapeXYZ[0]) ||
                                neighborY >= static_cast<std::int64_t>(lattice.shapeXYZ[1]) ||
                                neighborZ >= static_cast<std::int64_t>(lattice.shapeXYZ[2])) {
                                continue;
                            }
                            const std::size_t b =
                                nodeByLatticeSample[linear(static_cast<std::size_t>(neighborX), static_cast<std::size_t>(neighborY), static_cast<std::size_t>(neighborZ))];
                            if (b == missing)
                                continue;
                            if (auto factor = makeLasagnaNormalAlignmentFactor(a, b, retainedNormals[a], retainedNormals[b])) {
                                factors.push_back(*factor);
                            }
                        }
                    }
                }
            }
        }
        const std::size_t completed = std::min(
            sampleCount,
            (z + 1) * lattice.shapeXYZ[0] * lattice.shapeXYZ[1]);
        if (completed >= nextProgress || completed == sampleCount) {
            notifyProgress(
                progress,
                LasagnaNormalAlignmentProgressPhase::Factors,
                completed,
                sampleCount);
            nextProgress = completed >
                    std::numeric_limits<std::size_t>::max() - kProgressStride
                ? sampleCount
                : std::min(sampleCount, completed + kProgressStride);
        }
    }
    return factors;
}

std::optional<BinaryPairwiseFactor> makeLasagnaNormalAlignmentFactor(std::size_t a, std::size_t b, const cv::Vec3f& normalA, const cv::Vec3f& normalB)
{
    if (a == b)
        throw std::invalid_argument("Normal alignment factor requires two samples");
    const cv::Vec3f first = normalized(normalA);
    const cv::Vec3f second = normalized(normalB);
    const double dot = std::clamp(static_cast<double>(first.dot(second)), -1.0, 1.0);
    if (dot == 0.0)
        return std::nullopt;
    return BinaryPairwiseFactor{
        a,
        b,
        0.5 * (1.0 - dot),
        0.5 * (1.0 + dot),
    };
}

LasagnaNormalAlignmentReport alignLasagnaNormalSamples(
    std::span<const cv::Vec3f> normals,
    std::span<const BinaryPairwiseFactor> neighborhoodFactors,
    const LasagnaNormalAlignmentConfig& config,
    const LasagnaNormalAlignmentProgressCallback& progress)
{
    if (normals.empty())
        throw std::invalid_argument("Normal alignment requires samples");

    const std::size_t componentWork = checkedSum({
        normals.size(), neighborhoodFactors.size(), normals.size()});
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Components,
        0,
        componentWork);
    std::size_t completedWork = 0;
    std::size_t nextProgress = std::min(kProgressStride, componentWork);
    const auto advanceProgress = [&](std::size_t amount) {
        completedWork += amount;
        if (completedWork >= nextProgress || completedWork == componentWork) {
            notifyProgress(
                progress,
                LasagnaNormalAlignmentProgressPhase::Components,
                completedWork,
                componentWork);
            nextProgress = completedWork >
                    std::numeric_limits<std::size_t>::max() - kProgressStride
                ? componentWork
                : std::min(componentWork, completedWork + kProgressStride);
        }
    };

    std::vector<cv::Vec3f> normalizedNormals;
    normalizedNormals.reserve(normals.size());
    for (const auto& normal : normals) {
        normalizedNormals.push_back(normalized(normal));
        advanceProgress(1);
    }

    std::vector<std::vector<std::size_t>> adjacency(normals.size());
    for (std::size_t index = 0; index < neighborhoodFactors.size(); ++index) {
        const auto& factor = neighborhoodFactors[index];
        if (factor.a >= normals.size() || factor.b >= normals.size() || factor.a == factor.b) {
            throw std::invalid_argument("Normal alignment factor references an invalid sample pair");
        }
        adjacency[factor.a].push_back(index);
        adjacency[factor.b].push_back(index);
        advanceProgress(1);
    }

    LasagnaNormalAlignmentReport report;
    report.fixedStates.assign(normals.size(), BinaryBeliefState::Free);
    report.componentByNode.assign(normals.size(), 0);
    std::vector<unsigned char> visited(normals.size(), 0);
    for (std::size_t start = 0; start < normals.size(); ++start) {
        if (visited[start] != 0)
            continue;
        ++report.connectedComponents;
        const std::size_t component = report.connectedComponents - 1;
        if (adjacency[start].empty())
            ++report.isolatedSamples;
        report.fixedStates[start] = BinaryBeliefState::Zero;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1;
        report.componentByNode[start] = component;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            advanceProgress(1);
            for (const std::size_t factorIndex : adjacency[node]) {
                const auto& factor = neighborhoodFactors[factorIndex];
                const std::size_t neighbor = factor.a == node ? factor.b : factor.a;
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    report.componentByNode[neighbor] = component;
                    pending.push(neighbor);
                }
            }
        }
    }

    BinaryBeliefPropagationProgressCallback binaryProgress;
    if (progress) {
        binaryProgress = [&](const BinaryBeliefPropagationProgress& current) {
            if (current.complete)
                return;
            notifyProgress(
                progress,
                LasagnaNormalAlignmentProgressPhase::Messages,
                current.messageIteration,
                current.maximumMessageIterations,
                current.messageResidual);
        };
    }
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Messages,
        0,
        config.beliefPropagation.maximumMessageIterations);
    report.beliefPropagation = solveBinaryPairwiseSumProduct(
        normals.size(),
        neighborhoodFactors,
        report.fixedStates,
        config.beliefPropagation,
        binaryProgress);
    report.flipProbability = report.beliefPropagation.probabilityOne;
    report.alignedNormals = std::move(normalizedNormals);
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Finalize,
        0,
        report.alignedNormals.size());
    nextProgress = std::min(kProgressStride, report.alignedNormals.size());
    for (std::size_t node = 0; node < report.alignedNormals.size(); ++node) {
        if (report.flipProbability[node] > 0.5) {
            report.alignedNormals[node] *= -1.0F;
            ++report.flippedSamples;
        }
        const std::size_t completed = node + 1;
        if (completed >= nextProgress ||
            completed == report.alignedNormals.size()) {
            notifyProgress(
                progress,
                LasagnaNormalAlignmentProgressPhase::Finalize,
                completed,
                report.alignedNormals.size());
            nextProgress = completed >
                    std::numeric_limits<std::size_t>::max() - kProgressStride
                ? report.alignedNormals.size()
                : std::min(
                      report.alignedNormals.size(),
                      completed + kProgressStride);
        }
    }
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Complete,
        1,
        1,
        report.beliefPropagation.messageResidual);
    return report;
}

std::optional<AlignedLasagnaNormalSample> LasagnaNormalAlignmentField::nearest(
    const cv::Vec3d& pointBaseXYZ) const
{
    if (!(lattice.spacingBaseVoxels > 0.0) ||
        nodeByLatticeSample.empty() ||
        alignment.alignedNormals.size() != positionsBaseXYZ.size() ||
        alignment.componentByNode.size() != positionsBaseXYZ.size()) {
        return std::nullopt;
    }
    std::array<std::size_t, 3> local{};
    cv::Vec3d latticePoint;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(pointBaseXYZ[axis]))
            return std::nullopt;
        const auto global = static_cast<std::int64_t>(
            std::llround(pointBaseXYZ[axis] / lattice.spacingBaseVoxels));
        const auto offset = global - lattice.beginXYZ[axis];
        if (offset < 0 ||
            offset >= static_cast<std::int64_t>(lattice.shapeXYZ[axis])) {
            return std::nullopt;
        }
        local[axis] = static_cast<std::size_t>(offset);
        latticePoint[axis] = static_cast<double>(global) *
            lattice.spacingBaseVoxels;
    }
    const double maximumDistance =
        std::sqrt(3.0) * 0.5 * lattice.spacingBaseVoxels +
        1.0e-9 * std::max(1.0, lattice.spacingBaseVoxels);
    if (cv::norm(pointBaseXYZ - latticePoint) > maximumDistance)
        return std::nullopt;
    const std::size_t linear =
        (local[2] * lattice.shapeXYZ[1] + local[1]) *
            lattice.shapeXYZ[0] +
        local[0];
    if (linear >= nodeByLatticeSample.size())
        return std::nullopt;
    const std::size_t node = nodeByLatticeSample[linear];
    if (node == std::numeric_limits<std::size_t>::max() ||
        node >= alignment.alignedNormals.size()) {
        return std::nullopt;
    }
    return AlignedLasagnaNormalSample{
        alignment.alignedNormals[node],
        alignment.componentByNode[node],
        node,
    };
}

LasagnaNormalAlignmentField sampleAndAlignLasagnaNormalLattice(
    const vc::lasagna::LasagnaNormalSampler& sampler,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ,
    double spacingBaseVoxels,
    int neighborRadius,
    int parallelThreads,
    const LasagnaNormalAlignmentConfig& config,
    const LasagnaNormalAlignmentProgressCallback& progress)
{
    if (parallelThreads < 1)
        throw std::invalid_argument("Normal alignment worker count must be positive");
    LasagnaNormalAlignmentField field;
    field.lattice = makeLasagnaNormalLattice(
        minimumBaseXYZ, maximumBaseXYZ, spacingBaseVoxels);
    field.candidateSamples = field.lattice.positionsBaseXYZ.size();
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Sampling,
        0,
        field.candidateSamples);
    std::vector<vc::lasagna::LasagnaNormalSampler::FloatNormalSample> sampled;
    const auto sampling = sampler.sampleNormalBatch(
        field.lattice.positionsBaseXYZ, parallelThreads, sampled);
    field.prefetchMilliseconds = sampling.prefetchMs;
    field.materializeMilliseconds = sampling.materializeMs;
    notifyProgress(
        progress,
        LasagnaNormalAlignmentProgressPhase::Sampling,
        field.candidateSamples,
        field.candidateSamples);
    field.nodeByLatticeSample.assign(
        field.candidateSamples, std::numeric_limits<std::size_t>::max());
    field.positionsBaseXYZ.reserve(field.candidateSamples);
    field.rawNormals.reserve(field.candidateSamples);
    for (std::size_t candidate = 0; candidate < sampled.size(); ++candidate) {
        if (!sampled[candidate].valid)
            continue;
        field.nodeByLatticeSample[candidate] = field.positionsBaseXYZ.size();
        field.positionsBaseXYZ.push_back(field.lattice.positionsBaseXYZ[candidate]);
        field.rawNormals.push_back(sampled[candidate].normal);
    }
    if (field.positionsBaseXYZ.empty())
        throw std::invalid_argument("Lasagna normal alignment crop contains no valid samples");
    const auto factors = makeLasagnaNormalLatticeFactors(
        field.lattice,
        field.nodeByLatticeSample,
        field.rawNormals,
        neighborRadius,
        progress);
    auto solveConfig = config;
    solveConfig.beliefPropagation.parallelWorkers =
        static_cast<std::size_t>(parallelThreads);
    field.alignment = alignLasagnaNormalSamples(
        field.rawNormals, factors, solveConfig, progress);
    return field;
}

void writeNormalGlyphObj(const std::filesystem::path& path, std::span<const cv::Vec3f> positionsBaseXYZ, std::span<const cv::Vec3f> normals, const NormalGlyphObjConfig& config)
{
    if (positionsBaseXYZ.size() != normals.size()) {
        throw std::invalid_argument("Normal OBJ positions and normals have different sizes");
    }
    if (!std::isfinite(config.baseRadius) || !(config.baseRadius > 0.0) || !std::isfinite(config.directionLength) || !(config.directionLength > 0.0)) {
        throw std::invalid_argument("Normal OBJ glyph dimensions must be finite and positive");
    }
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Failed to open normal OBJ: " + path.string());
    output << std::setprecision(9);

    std::vector<std::array<cv::Vec3f, 4>> bases;
    bases.reserve(normals.size());
    for (const auto& rawNormal : normals) {
        const cv::Vec3f normal = normalized(rawNormal);
        const cv::Vec3f absolute{std::abs(normal[0]), std::abs(normal[1]), std::abs(normal[2])};
        cv::Vec3f reference{1.0F, 0.0F, 0.0F};
        if (absolute[1] <= absolute[0] && absolute[1] <= absolute[2])
            reference = {0.0F, 1.0F, 0.0F};
        else if (absolute[2] <= absolute[0] && absolute[2] <= absolute[1])
            reference = {0.0F, 0.0F, 1.0F};
        const cv::Vec3f first = normalized(normal.cross(reference));
        const cv::Vec3f second = normalized(normal.cross(first));
        bases.push_back({
            first * static_cast<float>(config.baseRadius),
            first * static_cast<float>(-config.baseRadius),
            second * static_cast<float>(config.baseRadius),
            second * static_cast<float>(-config.baseRadius),
        });
    }

    std::size_t vertex = 1;
    output << "g normal_bases\n";
    for (std::size_t index = 0; index < normals.size(); ++index) {
        const auto& center = positionsBaseXYZ[index];
        for (const auto& offset : bases[index]) {
            const cv::Vec3f point = center + offset;
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        }
        output << "l " << vertex << ' ' << vertex + 1 << '\n';
        output << "l " << vertex + 2 << ' ' << vertex + 3 << '\n';
        vertex += 4;
    }
    output << "g normal_directions\n";
    for (std::size_t index = 0; index < normals.size(); ++index) {
        const cv::Vec3f normal = normalized(normals[index]);
        const cv::Vec3f end = positionsBaseXYZ[index] + normal * static_cast<float>(config.directionLength);
        const auto& center = positionsBaseXYZ[index];
        output << "v " << center[0] << ' ' << center[1] << ' ' << center[2] << '\n';
        output << "v " << end[0] << ' ' << end[1] << ' ' << end[2] << '\n';
        output << "l " << vertex << ' ' << vertex + 1 << '\n';
        vertex += 2;
    }
    if (!output)
        throw std::runtime_error("Failed to write normal OBJ: " + path.string());
}

}  // namespace vc::fiber_tracer
