#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/util/PortableTensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <opencv2/core.hpp>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr float kFloatEpsilon = 1.0e-6f;

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    const double len = length(v);
    if (!(len > kEpsilon) || !std::isfinite(len)) {
        return {0.0, 0.0, 0.0};
    }
    return v / len;
}

struct DecodedCompactNormal {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool valid = false;
};

[[nodiscard]] const std::array<DecodedCompactNormal, 256 * 256>&
decodedCompactNormalTable()
{
    static const auto table = [] {
        std::array<DecodedCompactNormal, 256 * 256> out{};
        for (size_t rawNx = 0; rawNx < 256; ++rawNx) {
            for (size_t rawNy = 0; rawNy < 256; ++rawNy) {
                double nx = decodeCompactNormalComponent(static_cast<double>(rawNx));
                double ny = decodeCompactNormalComponent(static_cast<double>(rawNy));
                const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
                double nz = std::sqrt(nzSq);
                const double normalLength = std::sqrt(nx * nx + ny * ny + nz * nz);
                auto& entry = out[(rawNx << 8) | rawNy];
                if (!(normalLength > kEpsilon) || !std::isfinite(normalLength)) {
                    continue;
                }
                const double invLength = 1.0 / normalLength;
                entry.x = nx * invLength;
                entry.y = ny * invLength;
                entry.z = nz * invLength;
                entry.valid = true;
            }
        }
        return out;
    }();
    return table;
}

[[nodiscard]] cv::Vec3d fallbackPrincipalAxis(
    double a00,
    double a11,
    double a22)
{
    int axis = 0;
    double value = a00;
    if (a11 > value) {
        axis = 1;
        value = a11;
    }
    if (a22 > value) {
        axis = 2;
    }
    cv::Vec3d result{0.0, 0.0, 0.0};
    result[axis] = 1.0;
    return result;
}

[[nodiscard]] cv::Vec3d principalCompactTensorAxisFromComponents(
    double a00,
    double a01,
    double a02,
    double a11,
    double a12,
    double a22,
    const cv::Vec3d& hint)
{
    if (!std::isfinite(a00) || !std::isfinite(a01) || !std::isfinite(a02) ||
        !std::isfinite(a11) || !std::isfinite(a12) || !std::isfinite(a22)) {
        return {0.0, 0.0, 0.0};
    }

    cv::Vec3d axis{0.0, 0.0, 0.0};
    const double offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    if (offDiagonal <= kEpsilon) {
        axis = fallbackPrincipalAxis(a00, a11, a22);
    } else {
        const double q = (a00 + a11 + a22) / 3.0;
        const double b00 = a00 - q;
        const double b11 = a11 - q;
        const double b22 = a22 - q;
        const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * offDiagonal;
        const double p = std::sqrt(std::max(0.0, p2 / 6.0));
        if (p <= kEpsilon) {
            axis = fallbackPrincipalAxis(a00, a11, a22);
        } else {
            const double invP = 1.0 / p;
            const double c00 = b00 * invP;
            const double c01 = a01 * invP;
            const double c02 = a02 * invP;
            const double c11 = b11 * invP;
            const double c12 = a12 * invP;
            const double c22 = b22 * invP;
            const double detC =
                c00 * (c11 * c22 - c12 * c12) -
                c01 * (c01 * c22 - c12 * c02) +
                c02 * (c01 * c12 - c11 * c02);
            const double r = std::clamp(0.5 * detC, -1.0, 1.0);
            const double phi = std::acos(r) / 3.0;
            const double lambda = q + 2.0 * p * std::cos(phi);
            const cv::Vec3d row0{a00 - lambda, a01, a02};
            const cv::Vec3d row1{a01, a11 - lambda, a12};
            const cv::Vec3d row2{a02, a12, a22 - lambda};
            const std::array<cv::Vec3d, 3> candidates = {
                row0.cross(row1),
                row0.cross(row2),
                row1.cross(row2),
            };
            double bestNorm2 = -1.0;
            for (const auto& candidate : candidates) {
                const double norm2 = candidate.dot(candidate);
                if (norm2 > bestNorm2) {
                    bestNorm2 = norm2;
                    axis = candidate;
                }
            }
            if (!(bestNorm2 > kEpsilon * kEpsilon)) {
                axis = fallbackPrincipalAxis(a00, a11, a22);
            }
        }
    }
    if (length(axis) <= kEpsilon) {
        axis = fallbackPrincipalAxis(a00, a11, a22);
    }
    axis = normalizedOrZero(axis);
    if (length(axis) <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    const cv::Vec3d normalizedHint = normalizedOrZero(hint);
    if (length(normalizedHint) > kEpsilon) {
        if (axis.dot(normalizedHint) < 0.0) {
            axis *= -1.0;
        }
    } else if (axis[2] < 0.0) {
        axis *= -1.0;
    }
    return axis;
}

// A byte channel's value: uint8, or float32 holding an integer in [0, 255].
[[nodiscard]] uint8_t byteValue(double value)
{
    if (!std::isfinite(value) || value < 0 || value > 255)
        throw std::runtime_error("invalid packed encoded byte value");
    return static_cast<uint8_t>(std::lround(value));
}

// The distinct cubes (lower corners x, y, z) of a batch and each point's cube:
// nearby candidate points share few cubes, which are read once.
struct DistinctCubes {
    std::vector<std::array<int64_t, 3>> lower;
    std::vector<uint32_t> of;  // per point: its cube in `lower`
    void add(int64_t x, int64_t y, int64_t z)
    {
        if (2 * (lower.size() + 1) > slots.size()) {
            slots.assign(std::max<size_t>(64, 4 * slots.size()), kEmpty);
            for (uint32_t i = 0; i < lower.size(); ++i)
                slotOf(key(lower[i][0], lower[i][1], lower[i][2])) = i;
        }
        uint32_t& slot = slotOf(key(x, y, z));
        if (slot == kEmpty) {
            slot = uint32_t(lower.size());
            lower.push_back({x, y, z});
        }
        of.push_back(slot);
    }

private:
    static constexpr uint32_t kEmpty = UINT32_MAX;
    std::vector<uint32_t> slots;
    // Voxel indices fit 21 bits per axis.
    static uint64_t key(int64_t x, int64_t y, int64_t z) { return uint64_t(z) << 42 | uint64_t(y) << 21 | uint64_t(x); }
    uint32_t& slotOf(uint64_t k)
    {
        const size_t mask = slots.size() - 1;
        for (size_t i = (k * 0x9e3779b97f4a7c15ull) >> 20 & mask;; i = (i + 1) & mask) {
            const uint32_t at = slots[i];
            if (at == kEmpty)
                return slots[i];
            const auto& l = lower[at];
            if (key(l[0], l[1], l[2]) == k)
                return slots[i];
        }
    }
};

// Point p of a corner read: corner p % 8 of cube p / 8 in voxel coordinates,
// (z, y, x) or (c, z, y, x) for packed groups, the upper corners clamped to the
// last voxel (a vcr coordinate provider; no coordinate buffer).
struct CornerCoordinates {
    using value_type = int64_t;
    const std::vector<std::array<int64_t, 3>>* lower;
    std::array<int64_t, 3> last;  // x, y, z
    int64_t channel = -1;         // packed groups: the channel index
    int64_t base(size_t) const { return 0; }
    const int64_t* operator()(size_t p, int64_t* c) const
    {
        const auto& l = (*lower)[p >> 3];
        const size_t k = p & 7;
        int64_t* zyx = c;
        if (channel >= 0) *zyx++ = channel;
        zyx[0] = std::min(l[2] + int64_t(k >> 2), last[2]);
        zyx[1] = std::min(l[1] + int64_t((k >> 1) & 1), last[1]);
        zyx[2] = std::min(l[0] + int64_t(k & 1), last[0]);
        return c;
    }
    void prefetch(size_t) const {}
};

// The eight corners of each cube (corner k at cube * 8 + k), read with one
// Array::sample_nearest_coordinates call.
template <class T>
void readCorners(const LasagnaChannelBinding& binding,
                 const std::vector<std::array<int64_t, 3>>& lower, std::vector<T>& out)
{
    const CornerCoordinates coords{&lower,
        {int64_t(binding.shapeZYX[2]) - 1, int64_t(binding.shapeZYX[1]) - 1, int64_t(binding.shapeZYX[0]) - 1},
        binding.packed ? int64_t(binding.channelIndex) : -1};
    const size_t n = lower.size() * 8;
    out.resize(n);
    if (binding.float32) {
        std::vector<float> values(n);
        binding.array->sample_nearest_coordinates(coords, n, reinterpret_cast<uint8_t*>(values.data()));
        for (size_t i = 0; i < n; ++i)
            out[i] = std::is_same_v<T, uint8_t> ? T(byteValue(values[i])) : T(values[i]);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        binding.array->sample_nearest_coordinates(coords, n, out.data());
    } else {
        std::vector<uint8_t> values(n);
        binding.array->sample_nearest_coordinates(coords, n, values.data());
        std::copy(values.begin(), values.end(), out.begin());
    }
}

template <class Real>
[[nodiscard]] bool insideGrid(const LasagnaChannelBinding& binding, Real x, Real y, Real z)
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x >= 0 && y >= 0 && z >= 0 &&
           x <= static_cast<Real>(binding.shapeZYX[2] - 1) && y <= static_cast<Real>(binding.shapeZYX[1] - 1) &&
           z <= static_cast<Real>(binding.shapeZYX[0] - 1);
}

template <class Real>
[[nodiscard]] Real cornerWeight(size_t k, const cv::Vec<Real, 3>& f)
{
    return ((k & 1) ? f[0] : 1 - f[0]) * ((k & 2) ? f[1] : 1 - f[1]) * ((k & 4) ? f[2] : 1 - f[2]);
}

using vc::core::portable::normalizedFloatOrZero;

[[nodiscard]] std::array<float, 8> cornerWeights(const cv::Vec3f& fractionXYZ)
{
    const float fx = fractionXYZ[0];
    const float fy = fractionXYZ[1];
    const float fz = fractionXYZ[2];
    const float ax = 1.0f - fx;
    const float ay = 1.0f - fy;
    const float az = 1.0f - fz;
    return {
        az * ay * ax, az * ay * fx, az * fy * ax, az * fy * fx,
        fz * ay * ax, fz * ay * fx, fz * fy * ax, fz * fy * fx};
}

struct DecodedCompactNormalFloat {
    // Two padding lanes permit one vector load, no overread into the next entry.
    std::array<float, 8> tensor{};
    bool valid = false;
};

[[nodiscard]] const std::array<DecodedCompactNormalFloat, 256 * 256>&
decodedCompactNormalFloatTable()
{
    static const auto table = [] {
        std::array<DecodedCompactNormalFloat, 256 * 256> out{};
        for (size_t rawNx = 0; rawNx < 256; ++rawNx) {
            for (size_t rawNy = 0; rawNy < 256; ++rawNy) {
                const float x =
                    (static_cast<float>(rawNx) - 128.0f) / 127.0f;
                const float y =
                    (static_cast<float>(rawNy) - 128.0f) / 127.0f;
                const float z = std::sqrt(std::max(
                    0.0f, 1.0f - x * x - y * y));
                auto& decoded = out[(rawNx << 8) | rawNy];
                const cv::Vec3f axis = normalizedFloatOrZero(cv::Vec3f{x, y, z});
                decoded.valid = axis.dot(axis) >
                    kFloatEpsilon * kFloatEpsilon;
                if (decoded.valid) {
                    decoded.tensor = {
                        axis[0] * axis[0],
                        axis[0] * axis[1],
                        axis[0] * axis[2],
                        axis[1] * axis[1],
                        axis[1] * axis[2],
                        axis[2] * axis[2]};
                }
            }
        }
        return out;
    }();
    return table;
}

using vc::core::portable::principalFloatTensorAxis;

} // namespace

LasagnaChannelBinding bindLasagnaChannel(
    const LasagnaDatasetManifest& manifest,
    std::string_view channel, LasagnaGroupArrays& arrays, bool allowFloat32)
{
    const LasagnaChannelGroup* group = manifest.groupForChannel(channel);
    if (group == nullptr) {
        throw std::runtime_error("Lasagna dataset missing required channel '" + std::string(channel) + "'");
    }
    const auto channelIndex = group->channelIndex(channel);
    if (!channelIndex.has_value()) {
        throw std::runtime_error("Internal Lasagna channel lookup failure");
    }
    auto& opened = arrays.arrays[group];
    if (!opened.array) {
        auto array = openLasagnaGroupArray(*group, arrays.cache);
        opened = {std::shared_ptr<vcr::Array>(std::move(array.array)), std::move(array.meta)};
    }
    const auto& meta = opened.meta;
    LasagnaChannelBinding binding;
    binding.group = group;
    binding.channelIndex = *channelIndex;
    binding.array = opened.array;
    binding.spacing = static_cast<double>(group->scaleFactor()) *
                      manifest.sourceToBase / manifest.workingToBaseScale;
    binding.float32 = meta.dtype == "<f4";
    if (binding.float32 ? !(allowFloat32 || meta.shape.size() == 4) : meta.dtype != "|u1")
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' must be uint8");
    binding.packed = meta.shape.size() == 4 && meta.chunks.size() == 4;
    if ((binding.packed && manifest.version != 3 && !manifest.raw.value("packed_channel_view", false)) ||
        (!binding.packed && (meta.shape.size() != 3 || meta.chunks.size() != 3)) ||
        (binding.packed && (size_t(meta.shape[0]) != group->channels.size() || meta.chunks[0] != meta.shape[0]))) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) +
                                 "' must reference a 3D (Z,Y,X) zarr");
    }
    const size_t first = binding.packed ? 1 : 0;
    for (size_t axis = 0; axis < 3; ++axis) {
        binding.shapeZYX[axis] = size_t(meta.shape[first + axis]);
        binding.chunksZYX[axis] = size_t(meta.chunks[first + axis]);
    }
    if (binding.spacing <= 0.0 || !std::isfinite(binding.spacing)) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has invalid spacing");
    }
    if (binding.shapeZYX[0] == 0 || binding.shapeZYX[1] == 0 || binding.shapeZYX[2] == 0 ||
        binding.chunksZYX[0] == 0 || binding.chunksZYX[1] == 0 || binding.chunksZYX[2] == 0) {
        throw std::runtime_error("Lasagna channel '" + std::string(channel) + "' has empty zarr shape/chunks");
    }
    return binding;
}

std::vector<LasagnaChannelBinding> bindLasagnaEmbeddingChannels(
    const LasagnaDatasetManifest& manifest, const LasagnaChannelBinding& lattice,
    LasagnaGroupArrays& arrays)
{
    size_t count = 0;
    for (const auto& group : manifest.groups)
        for (const auto& channel : group.channels)
            if (channel.starts_with("embedding_")) ++count;
    if (count == 0 || count > 256)
        throw std::runtime_error("embedding tracing requires 1..256 contiguous embedding_NNN channels");
    std::vector<LasagnaChannelBinding> bindings;
    for (size_t i = 0; i < count; ++i) {
        const auto digits = std::to_string(i);
        auto binding = bindLasagnaChannel(manifest, "embedding_" + std::string(3-digits.size(), '0') + digits,
                                          arrays, true);
        if (!binding.float32 || binding.spacing != lattice.spacing || binding.shapeZYX != lattice.shapeZYX)
            throw std::runtime_error("embedding channels must be float32 on the prediction lattice");
        bindings.push_back(std::move(binding));
    }
    return bindings;
}

bool sameLasagnaSamplingGrid(
    const LasagnaChannelBinding& a,
    const LasagnaChannelBinding& b)
{
    const double spacingTolerance =
        1.0e-12 * std::max({1.0, std::abs(a.spacing), std::abs(b.spacing)});
    return a.shapeZYX == b.shapeZYX &&
           a.chunksZYX == b.chunksZYX &&
           std::abs(a.spacing - b.spacing) <= spacingTolerance;
}

void sampleLasagnaChannelCornerBatch(
    const std::vector<const LasagnaChannelBinding*>& bindings,
    const std::vector<cv::Vec3f>& volumePoints,
    LasagnaCornerBatch& samples)
{
    samples.values.assign(bindings.size(), std::vector<std::array<uint8_t, 8>>(volumePoints.size()));
    samples.fractionsXYZ.assign(volumePoints.size(), cv::Vec3f{0.0f, 0.0f, 0.0f});
    samples.valid.assign(volumePoints.size(), uint8_t{0});
    if (bindings.empty() || volumePoints.empty())
        return;
    for (const auto* binding : bindings) {
        if (binding == nullptr || !sameLasagnaSamplingGrid(*bindings.front(), *binding))
            throw std::invalid_argument("grouped corner samplers must share one spatial grid");
    }
    const auto& grid = *bindings.front();
    const float spacing = static_cast<float>(grid.spacing);
    DistinctCubes cubes;
    std::vector<size_t> points;
    points.reserve(volumePoints.size());
    for (size_t index = 0; index < volumePoints.size(); ++index) {
        const cv::Vec3f point = volumePoints[index] / spacing;
        if (!insideGrid(grid, point[0], point[1], point[2]))
            continue;
        const cv::Vec3f base{std::floor(point[0]), std::floor(point[1]), std::floor(point[2])};
        samples.fractionsXYZ[index] = point - base;
        samples.valid[index] = 1;
        cubes.add(int64_t(base[0]), int64_t(base[1]), int64_t(base[2]));
        points.push_back(index);
    }
    std::vector<uint8_t> values;
    for (size_t b = 0; b < bindings.size(); ++b) {
        readCorners(*bindings[b], cubes.lower, values);
        auto& out = samples.values[b];
        for (size_t i = 0; i < points.size(); ++i)
            std::copy_n(values.begin() + std::ptrdiff_t(cubes.of[i]) * 8, 8, out[points[i]].begin());
    }
}

void visitLasagnaChannelCorners(
    const std::vector<const LasagnaChannelBinding*>& bindings,
    const std::vector<cv::Vec3f>& volumePoints,
    void* visitorContext,
    LasagnaCornerPointVisitor visitor,
    int parallelThreads,
    LasagnaCornerBatch& scratch)
{
    sampleLasagnaChannelCornerBatch(bindings, volumePoints, scratch);
    const auto visit = [&](size_t point, std::vector<std::array<uint8_t, 8>>& corners) {
        for (size_t b = 0; b < bindings.size(); ++b)
            corners[b] = scratch.values[b][point];
        visitor(visitorContext, point, scratch.fractionsXYZ[point], scratch.valid[point] != 0, corners);
    };
    const auto count = static_cast<std::ptrdiff_t>(volumePoints.size());
    const int workers = std::clamp(parallelThreads, 1, int(std::max<std::ptrdiff_t>(1, count)));
#ifdef _OPENMP
    if (workers > 1) {
        #pragma omp parallel num_threads(workers)
        {
            std::vector<std::array<uint8_t, 8>> corners(bindings.size());
            #pragma omp for schedule(static)
            for (std::ptrdiff_t point = 0; point < count; ++point)
                visit(size_t(point), corners);
        }
        return;
    }
#endif
    std::vector<std::array<uint8_t, 8>> corners(bindings.size());
    for (std::ptrdiff_t point = 0; point < count; ++point)
        visit(size_t(point), corners);
}

void readLasagnaCubes(
    const LasagnaChannelBinding& binding,
    std::span<const cv::Vec3d> volumePoints,
    LasagnaCubes& cubes)
{
    cubes.values.assign(volumePoints.size(), {});
    cubes.fractionsXYZ.assign(volumePoints.size(), cv::Vec3d{0.0, 0.0, 0.0});
    cubes.valid.assign(volumePoints.size(), uint8_t{0});
    DistinctCubes distinct;
    std::vector<size_t> points;
    for (size_t index = 0; index < volumePoints.size(); ++index) {
        const cv::Vec3d point = volumePoints[index] / binding.spacing;
        if (!insideGrid(binding, point[0], point[1], point[2]))
            continue;
        const cv::Vec3d base{std::floor(point[0]), std::floor(point[1]), std::floor(point[2])};
        cubes.fractionsXYZ[index] = point - base;
        cubes.valid[index] = 1;
        distinct.add(int64_t(base[0]), int64_t(base[1]), int64_t(base[2]));
        points.push_back(index);
    }
    std::vector<double> values;
    readCorners(binding, distinct.lower, values);
    for (size_t i = 0; i < points.size(); ++i)
        std::copy_n(values.begin() + std::ptrdiff_t(distinct.of[i]) * 8, 8, cubes.values[points[i]].begin());
}

double interpolateLasagnaCube(const std::array<double, 8>& values, const cv::Vec3d& f)
{
    const double c00 = values[0] * (1.0 - f[0]) + values[1] * f[0];
    const double c01 = values[2] * (1.0 - f[0]) + values[3] * f[0];
    const double c10 = values[4] * (1.0 - f[0]) + values[5] * f[0];
    const double c11 = values[6] * (1.0 - f[0]) + values[7] * f[0];
    const double c0 = c00 * (1.0 - f[1]) + c01 * f[1];
    const double c1 = c10 * (1.0 - f[1]) + c11 * f[1];
    return c0 * (1.0 - f[2]) + c1 * f[2];
}

std::optional<cv::Vec3d> lasagnaCompactAxisCube(
    const std::array<double, 8>& nxValues,
    const std::array<double, 8>& nyValues,
    const cv::Vec3d& fraction,
    const std::array<double, 8>* validity)
{
    const auto& normalTable = decodedCompactNormalTable();
    double t00 = 0.0;
    double t01 = 0.0;
    double t02 = 0.0;
    double t11 = 0.0;
    double t12 = 0.0;
    double t22 = 0.0;
    cv::Vec3d hint{0.0, 0.0, 0.0};
    double totalWeight = 0.0;
    for (size_t k = 0; k < 8; ++k) {
        const double weight = cornerWeight(k, fraction);
        if (weight <= 0.0 || (validity && byteValue((*validity)[k]) == 0))
            continue;
        const auto& decoded =
            normalTable[(static_cast<size_t>(byteValue(nxValues[k])) << 8) |
                        static_cast<size_t>(byteValue(nyValues[k]))];
        if (!decoded.valid)
            continue;
        const double nx = decoded.x;
        const double ny = decoded.y;
        const double nz = decoded.z;
        t00 += weight * nx * nx;
        t01 += weight * nx * ny;
        t02 += weight * nx * nz;
        t11 += weight * ny * ny;
        t12 += weight * ny * nz;
        t22 += weight * nz * nz;
        hint += cv::Vec3d{nx, ny, nz} * weight;
        totalWeight += weight;
    }
    if (totalWeight <= kEpsilon) {
        return std::nullopt;
    }
    const cv::Vec3d normal = principalCompactTensorAxisFromComponents(
        t00, t01, t02, t11, t12, t22, hint);
    if (length(normal) <= kEpsilon) {
        return std::nullopt;
    }
    return normal;
}

std::optional<double> sampleLasagnaChannel(
    const LasagnaChannelBinding& binding,
    const cv::Vec3d& volumePoint)
{
    LasagnaCubes cubes;
    readLasagnaCubes(binding, std::span(&volumePoint, 1), cubes);
    if (!cubes.valid[0])
        return std::nullopt;
    return interpolateLasagnaCube(cubes.values[0], cubes.fractionsXYZ[0]);
}

std::optional<cv::Vec3d> sampleLasagnaCompactAxisTensor(
    const LasagnaChannelBinding& nxBinding,
    const LasagnaChannelBinding& nyBinding,
    const cv::Vec3d& volumePoint, const LasagnaChannelBinding* validity)
{
    if (validity && (!sameLasagnaSamplingGrid(nxBinding, *validity) || !sameLasagnaSamplingGrid(nyBinding, *validity)))
        throw std::invalid_argument("validity must share axis sampling grid");
    LasagnaCubes nx, ny, mask;
    readLasagnaCubes(nxBinding, std::span(&volumePoint, 1), nx);
    readLasagnaCubes(nyBinding, std::span(&volumePoint, 1), ny);
    if (!nx.valid[0] || !ny.valid[0])
        return std::nullopt;
    if (validity) {
        readLasagnaCubes(*validity, std::span(&volumePoint, 1), mask);
        if (!mask.valid[0])
            return std::nullopt;
    }
    return lasagnaCompactAxisCube(nx.values[0], ny.values[0], nx.fractionsXYZ[0],
                                  validity ? &mask.values[0] : nullptr);
}

void sampleLasagnaVectorBatch(const std::vector<LasagnaChannelBinding>& channels,
    const std::vector<cv::Vec3f>& points, std::vector<std::vector<float>>& vectors, bool normalize,
    const LasagnaChannelBinding* validity)
{
    if (channels.empty()) throw std::invalid_argument("vector sampling requires channels");
    if (validity)
        for (const auto& c : channels)
            if (c.spacing != validity->spacing || c.shapeZYX != validity->shapeZYX)
                throw std::invalid_argument("validity must share scalar sampling lattice");
    const std::vector<cv::Vec3d> volumePoints(points.begin(), points.end());
    LasagnaCubes mask;
    if (validity) readLasagnaCubes(*validity, volumePoints, mask);
    std::vector<LasagnaCubes> cubes(channels.size());
    for (size_t c = 0; c < channels.size(); ++c)
        readLasagnaCubes(channels[c], volumePoints, cubes[c]);
    vectors.assign(points.size(), {});
    for (size_t i = 0; i < points.size(); ++i) {
        auto& vector = vectors[i];
        if (validity && !mask.valid[i]) continue;
        const bool masked = validity && std::any_of(mask.values[i].begin(), mask.values[i].end(),
                                                    [](double v) { return byteValue(v) == 0; });
        double norm2 = 0;
        for (size_t c = 0; c < channels.size(); ++c) {
            if (!cubes[c].valid[i]) { vector.clear(); break; }
            const auto& values = cubes[c].values[i];
            const auto& f = cubes[c].fractionsXYZ[i];
            std::optional<double> value = interpolateLasagnaCube(values, f);
            if (masked) {
                double sum = 0, denom = 0;
                for (size_t k = 0; k < 8; ++k) if (byteValue(mask.values[i][k])) {
                    const double w = cornerWeight(k, f);
                    if (w > 0) { sum += w * values[k]; denom += w; }
                }
                value = denom > 0 ? std::optional<double>(sum / denom) : std::nullopt;
            }
            if (!value || !std::isfinite(*value)) { vector.clear(); break; }
            vector.push_back(static_cast<float>(*value));
            norm2 += *value * *value;
        }
        if (vector.size() != channels.size() || !std::isfinite(norm2) || (normalize && norm2 <= 1e-16)) vector.clear();
        else if (normalize) for (auto& v : vector) v /= static_cast<float>(std::sqrt(norm2));
    }
}

std::array<float, 8> lasagnaCornerWeights(const cv::Vec3f& fractionXYZ)
{
    return cornerWeights(fractionXYZ);
}

float interpolateLasagnaCorners(
    const std::array<uint8_t, 8>& values,
    const std::array<float, 8>& weights)
{
    float value = 0.0f;
    for (size_t corner = 0; corner < weights.size(); ++corner)
        value = std::fma(weights[corner], static_cast<float>(values[corner]), value);
    return value;
}

std::array<float, 8> validLasagnaCornerWeights(
    const std::array<float, 8>& weights, const std::array<uint8_t, 8>& validity)
{
    auto result=weights;
    for (size_t i=0;i<8;++i) if (!validity[i]) result[i]=0;
    return result; // Tensor eigenvector is scale invariant; do not perturb all-valid weights.
}

cv::Vec3f interpolateLasagnaCompactAxisCorners(
    const std::array<uint8_t, 8>& nx,
    const std::array<uint8_t, 8>& ny,
    const std::array<float, 8>& weights,
    const cv::Vec3f& hint, bool vectorized)
{
    float a00 = 0.0f;
    float a01 = 0.0f;
    float a02 = 0.0f;
    float a11 = 0.0f;
    float a12 = 0.0f;
    float a22 = 0.0f;
    const auto& normalTable = decodedCompactNormalFloatTable();
#if defined(__AVX2__) && defined(__FMA__)
    if (vectorized) {
        // Each lane retains the scalar corner order and fused rounding exactly.
        __m256 tensor=_mm256_setzero_ps();
        for(size_t corner=0;corner<weights.size();++corner) {
            const auto& decoded=normalTable[(size_t(nx[corner])<<8)|size_t(ny[corner])];
            if(decoded.valid)
                tensor=_mm256_fmadd_ps(_mm256_set1_ps(weights[corner]),_mm256_loadu_ps(decoded.tensor.data()),tensor);
        }
        alignas(32) float values[8];_mm256_store_ps(values,tensor);
        if(!(values[0]+values[3]+values[5]>0.f)) return {0.f,0.f,0.f};
        return principalFloatTensorAxis(values[0],values[1],values[2],values[3],values[4],values[5],hint);
    }
#else
    (void)vectorized; // Portable scalar reference on builds without AVX2/FMA.
#endif
    for (size_t corner = 0; corner < weights.size(); ++corner) {
        const auto& decoded = normalTable[
            (static_cast<size_t>(nx[corner]) << 8) |
            static_cast<size_t>(ny[corner])];
        if (!decoded.valid)
            continue;
        const float weight = weights[corner];
        a00 = std::fma(weight, decoded.tensor[0], a00);
        a01 = std::fma(weight, decoded.tensor[1], a01);
        a02 = std::fma(weight, decoded.tensor[2], a02);
        a11 = std::fma(weight, decoded.tensor[3], a11);
        a12 = std::fma(weight, decoded.tensor[4], a12);
        a22 = std::fma(weight, decoded.tensor[5], a22);
    }
    if (!(a00 + a11 + a22 > 0.0f))
        return {0.0f, 0.0f, 0.0f};
    return principalFloatTensorAxis(a00, a01, a02, a11, a12, a22, hint);
}

double decodeCompactNormalComponent(double raw)
{
    return (raw - 128.0) / 127.0;
}

cv::Vec3d decodeCompactNormalFromRaw(double rawNx, double rawNy)
{
    const double nx = decodeCompactNormalComponent(rawNx);
    const double ny = decodeCompactNormalComponent(rawNy);
    const double nzSq = std::max(0.0, 1.0 - nx * nx - ny * ny);
    return normalizedOrZero({nx, ny, std::sqrt(nzSq)});
}

std::optional<std::array<uint8_t, 2>>
encodeCompactNormalToRaw(const cv::Vec3d& input)
{
    cv::Vec3d normal = normalizedOrZero(input);
    const double norm2 = normal.dot(normal);
    if (!(norm2 > kEpsilon * kEpsilon) ||
        !std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2])) {
        return std::nullopt;
    }
    if (normal[2] < 0.0)
        normal *= -1.0;
    const auto encode = [](double component) {
        const long raw = std::lround(component * 127.0 + 128.0);
        return static_cast<uint8_t>(std::clamp(raw, 0L, 255L));
    };
    return std::array<uint8_t, 2>{encode(normal[0]), encode(normal[1])};
}

cv::Vec3d principalCompactTensorAxis(const cv::Matx33d& tensor, const cv::Vec3d& hint)
{
    auto fallbackTensorAxis = [](const cv::Matx33d& t) {
        int axis = 0;
        double value = t(0, 0);
        if (t(1, 1) > value) {
            axis = 1;
            value = t(1, 1);
        }
        if (t(2, 2) > value) {
            axis = 2;
        }
        cv::Vec3d result{0.0, 0.0, 0.0};
        result[axis] = 1.0;
        return result;
    };

    bool finiteTensor = true;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            finiteTensor = finiteTensor && std::isfinite(tensor(row, col));
        }
    }
    if (!finiteTensor) {
        return {0.0, 0.0, 0.0};
    }

    cv::Vec3d axis{0.0, 0.0, 0.0};

    const double a00 = tensor(0, 0);
    const double a01 = 0.5 * (tensor(0, 1) + tensor(1, 0));
    const double a02 = 0.5 * (tensor(0, 2) + tensor(2, 0));
    const double a11 = tensor(1, 1);
    const double a12 = 0.5 * (tensor(1, 2) + tensor(2, 1));
    const double a22 = tensor(2, 2);
    const double offDiagonal = a01 * a01 + a02 * a02 + a12 * a12;
    if (offDiagonal <= kEpsilon) {
        axis = fallbackTensorAxis(tensor);
    } else {
        const double q = (a00 + a11 + a22) / 3.0;
        const double b00 = a00 - q;
        const double b11 = a11 - q;
        const double b22 = a22 - q;
        const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * offDiagonal;
        const double p = std::sqrt(std::max(0.0, p2 / 6.0));
        if (p <= kEpsilon) {
            axis = fallbackTensorAxis(tensor);
        } else {
            const double invP = 1.0 / p;
            const double c00 = b00 * invP;
            const double c01 = a01 * invP;
            const double c02 = a02 * invP;
            const double c11 = b11 * invP;
            const double c12 = a12 * invP;
            const double c22 = b22 * invP;
            const double detC =
                c00 * (c11 * c22 - c12 * c12) -
                c01 * (c01 * c22 - c12 * c02) +
                c02 * (c01 * c12 - c11 * c02);
            const double r = std::clamp(0.5 * detC, -1.0, 1.0);
            const double phi = std::acos(r) / 3.0;
            const double lambda = q + 2.0 * p * std::cos(phi);
            const cv::Vec3d row0{a00 - lambda, a01, a02};
            const cv::Vec3d row1{a01, a11 - lambda, a12};
            const cv::Vec3d row2{a02, a12, a22 - lambda};
            const std::array<cv::Vec3d, 3> candidates = {
                row0.cross(row1),
                row0.cross(row2),
                row1.cross(row2),
            };
            double bestNorm2 = -1.0;
            for (const auto& candidate : candidates) {
                const double norm2 = candidate.dot(candidate);
                if (norm2 > bestNorm2) {
                    bestNorm2 = norm2;
                    axis = candidate;
                }
            }
            if (!(bestNorm2 > kEpsilon * kEpsilon)) {
                axis = fallbackTensorAxis(tensor);
            }
        }
    }
    if (length(axis) <= kEpsilon) {
        axis = fallbackTensorAxis(tensor);
    }
    axis = normalizedOrZero(axis);
    if (length(axis) <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    const cv::Vec3d normalizedHint = normalizedOrZero(hint);
    if (length(normalizedHint) > kEpsilon) {
        if (axis.dot(normalizedHint) < 0.0) {
            axis *= -1.0;
        }
    } else if (axis[2] < 0.0) {
        axis *= -1.0;
    }
    return axis;
}

} // namespace vc::lasagna
