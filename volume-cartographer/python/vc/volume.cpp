#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace nb = nanobind;
using namespace nb::literals;

void bindNormalSampler(nb::module_& m);

namespace {

template <typename T>
nb::ndarray<T, nb::numpy, nb::c_contig> makeNumpyArray(std::vector<T>&& data,
                                                       std::array<size_t, 3> shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::ndarray<T, nb::numpy, nb::c_contig>(
        heap->data(),
        {shape[0], shape[1], shape[2]},
        owner);
}


using FloatCoords = nb::ndarray<float, nb::numpy, nb::c_contig>;
using BoolMask = nb::ndarray<bool, nb::numpy, nb::c_contig>;

template <typename T>
nb::ndarray<T, nb::numpy, nb::c_contig> makeNumpyArray4(
    std::vector<T>&& data,
    const std::array<size_t, 4>& shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::ndarray<T, nb::numpy, nb::c_contig>(
        heap->data(), {shape[0], shape[1], shape[2], shape[3]}, owner);
}

void requireFloatShape(const FloatCoords& value,
                       const std::vector<size_t>& expected,
                       const char* name)
{
    if (value.ndim() != expected.size())
        throw nb::value_error((std::string(name) + " has the wrong rank").c_str());
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        if (value.shape(axis) != expected[axis]) {
            throw nb::value_error(
                (std::string(name) + " has the wrong shape").c_str());
        }
    }
}

void requireFinite(const FloatCoords& value, const char* name)
{
    size_t count = 1;
    for (size_t axis = 0; axis < value.ndim(); ++axis)
        count *= value.shape(axis);
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(value.data()[index])) {
            throw nb::value_error(
                (std::string(name) + " must contain only finite values").c_str());
        }
    }
}

float sampleControl1D(const FloatCoords& control,
                      float coordinate,
                      float origin,
                      float spacing)
{
    const size_t count = control.shape(0);
    if (count == 1)
        return control.data()[0];
    const float u = std::clamp(
        (coordinate - origin) / spacing, 0.0f, static_cast<float>(count - 1));
    const size_t lo = static_cast<size_t>(std::floor(u));
    const size_t hi = std::min(lo + 1, count - 1);
    const float fraction = u - static_cast<float>(lo);
    return control.data()[lo] * (1.0f - fraction) +
           control.data()[hi] * fraction;
}

float sampleControl2D(const FloatCoords& control,
                      float coordinateA,
                      float coordinateB,
                      float originA,
                      float originB,
                      float spacingA,
                      float spacingB)
{
    const size_t height = control.shape(0);
    const size_t width = control.shape(1);
    const float ua = std::clamp(
        (coordinateA - originA) / spacingA,
        0.0f,
        static_cast<float>(height - 1));
    const float ub = std::clamp(
        (coordinateB - originB) / spacingB,
        0.0f,
        static_cast<float>(width - 1));
    const size_t a0 = static_cast<size_t>(std::floor(ua));
    const size_t b0 = static_cast<size_t>(std::floor(ub));
    const size_t a1 = std::min(a0 + 1, height - 1);
    const size_t b1 = std::min(b0 + 1, width - 1);
    const float wa = ua - static_cast<float>(a0);
    const float wb = ub - static_cast<float>(b0);
    const float* data = control.data();
    const float c00 = data[a0 * width + b0];
    const float c01 = data[a0 * width + b1];
    const float c10 = data[a1 * width + b0];
    const float c11 = data[a1 * width + b1];
    const float c0 = c00 * (1.0f - wb) + c01 * wb;
    const float c1 = c10 * (1.0f - wb) + c11 * wb;
    return c0 * (1.0f - wa) + c1 * wa;
}

nb::tuple generateFiber3DCoords(
    const std::array<int, 3>& outputShapeZyx,
    const FloatCoords& cpLocalZyx,
    const FloatCoords& cpVolumeZyx,
    const FloatCoords& outputToSourceZyx,
    const std::string& smoothMode,
    const nb::list& smoothControls,
    const FloatCoords& smoothOriginZyx,
    const FloatCoords& smoothSpacingZyx,
    float volumeSpacingBase,
    const std::array<int, 3>& baseShapeZyx)
{
    requireFloatShape(cpLocalZyx, {3}, "cp_local_zyx");
    requireFloatShape(cpVolumeZyx, {3}, "cp_volume_zyx");
    requireFloatShape(outputToSourceZyx, {3, 3}, "output_to_source_zyx");
    requireFloatShape(smoothOriginZyx, {3}, "smooth_origin_zyx");
    requireFloatShape(smoothSpacingZyx, {3}, "smooth_spacing_zyx");
    requireFinite(cpLocalZyx, "cp_local_zyx");
    requireFinite(cpVolumeZyx, "cp_volume_zyx");
    requireFinite(outputToSourceZyx, "output_to_source_zyx");
    requireFinite(smoothOriginZyx, "smooth_origin_zyx");
    requireFinite(smoothSpacingZyx, "smooth_spacing_zyx");
    if (!std::isfinite(volumeSpacingBase) || volumeSpacingBase <= 0.0f)
        throw nb::value_error("volume_spacing_base must be finite and positive");

    size_t pointCount = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (outputShapeZyx[axis] <= 0)
            throw nb::value_error("output_shape_zyx values must be positive");
        if (baseShapeZyx[axis] <= 0)
            throw nb::value_error("base_shape_zyx values must be positive");
        if (!(smoothSpacingZyx.data()[axis] > 0.0f))
            throw nb::value_error("smooth_spacing_zyx values must be positive");
        const size_t dim = static_cast<size_t>(outputShapeZyx[axis]);
        if (pointCount > std::numeric_limits<size_t>::max() / dim)
            throw nb::value_error("output_shape_zyx allocation overflows size_t");
        pointCount *= dim;
    }
    if (pointCount > std::numeric_limits<size_t>::max() / 3 / sizeof(float))
        throw nb::value_error("coordinate allocation overflows size_t");

    size_t expectedControls = 0;
    if (smoothMode == "none")
        expectedControls = 0;
    else if (smoothMode == "1d" || smoothMode == "2d")
        expectedControls = 1;
    else if (smoothMode == "3d")
        expectedControls = 3;
    else
        throw nb::value_error("smooth_mode must be one of: none, 1d, 2d, 3d");
    if (smoothControls.size() != expectedControls)
        throw nb::value_error("smooth_controls count does not match smooth_mode");

    std::vector<FloatCoords> controls;
    controls.reserve(expectedControls);
    for (size_t index = 0; index < expectedControls; ++index) {
        auto control = nb::cast<FloatCoords>(smoothControls[index]);
        const size_t expectedRank = smoothMode == "1d" ? 1 : 2;
        if (control.ndim() != expectedRank)
            throw nb::value_error("smooth control has the wrong rank");
        for (size_t axis = 0; axis < control.ndim(); ++axis) {
            if (control.shape(axis) == 0)
                throw nb::value_error("smooth controls must not be empty");
        }
        requireFinite(control, "smooth control");
        controls.emplace_back(std::move(control));
    }

    std::vector<float> coords(pointCount * 3);
    std::vector<uint8_t> valid(pointCount);
    const float* cpLocal = cpLocalZyx.data();
    const float* cpVolume = cpVolumeZyx.data();
    const float* matrix = outputToSourceZyx.data();
    const float* origin = smoothOriginZyx.data();
    const float* spacing = smoothSpacingZyx.data();
    const size_t outY = static_cast<size_t>(outputShapeZyx[1]);
    const size_t outX = static_cast<size_t>(outputShapeZyx[2]);
    {
        nb::gil_scoped_release release;
        size_t point = 0;
        for (int z = 0; z < outputShapeZyx[0]; ++z) {
            const float relativeZ = static_cast<float>(z) - cpLocal[0];
            for (int y = 0; y < outputShapeZyx[1]; ++y) {
                const float relativeY = static_cast<float>(y) - cpLocal[1];
                for (int x = 0; x < outputShapeZyx[2]; ++x, ++point) {
                    const float relativeX = static_cast<float>(x) - cpLocal[2];
                    float sourceZ = matrix[0] * relativeZ + matrix[1] * relativeY +
                                    matrix[2] * relativeX;
                    float sourceY = matrix[3] * relativeZ + matrix[4] * relativeY +
                                    matrix[5] * relativeX;
                    float sourceX = matrix[6] * relativeZ + matrix[7] * relativeY +
                                    matrix[8] * relativeX;
                    if (smoothMode == "1d") {
                        sourceZ -= sampleControl1D(
                            controls[0], sourceX, origin[2], spacing[2]);
                    } else if (smoothMode == "2d") {
                        sourceZ -= sampleControl2D(
                            controls[0], sourceY, sourceX,
                            origin[1], origin[2], spacing[1], spacing[2]);
                    } else if (smoothMode == "3d") {
                        sourceX -= sampleControl2D(
                            controls[2], sourceZ, sourceY,
                            origin[0], origin[1], spacing[0], spacing[1]);
                        sourceY -= sampleControl2D(
                            controls[1], sourceZ, sourceX,
                            origin[0], origin[2], spacing[0], spacing[2]);
                        sourceZ -= sampleControl2D(
                            controls[0], sourceY, sourceX,
                            origin[1], origin[2], spacing[1], spacing[2]);
                    }

                    const float baseZ = (cpVolume[0] + sourceZ) * volumeSpacingBase;
                    const float baseY = (cpVolume[1] + sourceY) * volumeSpacingBase;
                    const float baseX = (cpVolume[2] + sourceX) * volumeSpacingBase;
                    const size_t offset = point * 3;
                    coords[offset] = baseZ;
                    coords[offset + 1] = baseY;
                    coords[offset + 2] = baseX;
                    valid[point] = static_cast<uint8_t>(
                        std::isfinite(baseZ) && std::isfinite(baseY) &&
                        std::isfinite(baseX) && baseZ >= 0.0f && baseY >= 0.0f &&
                        baseX >= 0.0f &&
                        baseZ <= static_cast<float>(baseShapeZyx[0] - 1) &&
                        baseY <= static_cast<float>(baseShapeZyx[1] - 1) &&
                        baseX <= static_cast<float>(baseShapeZyx[2] - 1));
                }
            }
        }
    }

    auto coordsArray = makeNumpyArray4<float>(
        std::move(coords),
        {static_cast<size_t>(outputShapeZyx[0]), outY, outX, size_t{3}});
    auto validArray = makeNumpyArray<uint8_t>(
        std::move(valid),
        {static_cast<size_t>(outputShapeZyx[0]), outY, outX});
    return nb::make_tuple(coordsArray, validArray);
}

struct BlurStencilEntry {
    int z;
    int y;
    int x;
    float weight;
};

nb::ndarray<float, nb::numpy, nb::c_contig> fiber3DValueTransform(
    const FloatCoords& volume,
    const BoolMask& validMask,
    const std::string& normalization,
    float contrast,
    float brightness,
    float gamma)
{
    if (volume.ndim() != 3 || validMask.ndim() != 3)
        throw nb::value_error("volume and valid_mask must have shape (Z,Y,X)");
    std::array<size_t, 3> shape{};
    size_t pointCount = 1;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        shape[dimension] = volume.shape(dimension);
        if (shape[dimension] == 0)
            throw nb::value_error("volume dimensions must be positive");
        if (validMask.shape(dimension) != shape[dimension])
            throw nb::value_error("valid_mask shape must match volume");
        if (pointCount > std::numeric_limits<size_t>::max() / shape[dimension])
            throw nb::value_error("volume allocation overflows size_t");
        pointCount *= shape[dimension];
    }
    if (pointCount > std::numeric_limits<size_t>::max() / sizeof(float))
        throw nb::value_error("volume allocation overflows size_t");
    if (normalization != "none" && normalization != "raw" &&
        normalization != "identity" && normalization != "zscore" &&
        normalization != "minmax") {
        throw nb::value_error("unsupported image normalization");
    }
    if (!std::isfinite(contrast) || contrast <= 0.0f ||
        !std::isfinite(brightness) ||
        !std::isfinite(gamma) || gamma <= 0.0f) {
        throw nb::value_error(
            "contrast and gamma must be positive and all parameters finite");
    }

    std::vector<float> output(pointCount);
    const float* source = volume.data();
    const bool* mask = validMask.data();
    {
        nb::gil_scoped_release release;
        size_t validCount = 0;
        double sum = 0.0;
        float sourceMin = std::numeric_limits<float>::infinity();
        float sourceMax = -std::numeric_limits<float>::infinity();
        for (size_t index = 0; index < pointCount; ++index) {
            if (!mask[index])
                continue;
            const float value = source[index];
            sum += static_cast<double>(value);
            sourceMin = std::min(sourceMin, value);
            sourceMax = std::max(sourceMax, value);
            ++validCount;
        }
        const bool anyValid = validCount != 0;
        if (!anyValid) {
            for (size_t index = 0; index < pointCount; ++index) {
                const float value = source[index];
                sum += static_cast<double>(value);
                sourceMin = std::min(sourceMin, value);
                sourceMax = std::max(sourceMax, value);
            }
            validCount = pointCount;
        }

        const float sourceMean = static_cast<float>(
            sum / static_cast<double>(validCount));
        float normalizeDivisor = 1.0f;
        const bool useZscore = anyValid && normalization == "zscore";
        const bool useMinmax = anyValid && normalization == "minmax";
        if (useZscore) {
            double squared = 0.0;
            size_t varianceCount = 0;
            for (size_t index = 0; index < pointCount; ++index) {
                if (mask[index]) {
                    const double delta = static_cast<double>(source[index]) -
                                         static_cast<double>(sourceMean);
                    squared += delta * delta;
                    ++varianceCount;
                }
            }
            const float standardDeviation = std::max(
                1.0e-6f,
                static_cast<float>(std::sqrt(
                    squared / static_cast<double>(varianceCount))));
            normalizeDivisor = standardDeviation;
        } else if (useMinmax) {
            normalizeDivisor = std::max(1.0e-6f, sourceMax - sourceMin);
        }

        const auto normalizeValue = [&](float value) {
            if (useZscore)
                return (value - sourceMean) / normalizeDivisor;
            if (useMinmax)
                return (value - sourceMin) / normalizeDivisor;
            return value;
        };
        const float normalizedMean = normalizeValue(sourceMean);
        const auto transformValue = [&](float value) {
            const float normalized = normalizeValue(value);
            return (normalized - normalizedMean) * contrast +
                   normalizedMean + brightness;
        };
        const float transformedMin = transformValue(sourceMin);
        const float transformedMax = transformValue(sourceMax);
        const float transformedRange = std::max(
            1.0e-6f, transformedMax - transformedMin);
        for (size_t index = 0; index < pointCount; ++index) {
            float value = transformValue(source[index]);
            if (gamma != 1.0f) {
                const float unit = std::clamp(
                    (value - transformedMin) / transformedRange, 0.0f, 1.0f);
                value = std::pow(unit, gamma) *
                            (transformedMax - transformedMin) +
                        transformedMin;
            }
            output[index] = value;
        }
    }
    return makeNumpyArray<float>(std::move(output), shape);
}

std::array<float, 3> normalizeVector(const std::array<float, 3>& value,
                                     const char* name)
{
    const float norm = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (!std::isfinite(norm) || norm <= 0.0f)
        throw nb::value_error(
            (std::string(name) + " must have positive finite norm").c_str());
    return {value[0] / norm, value[1] / norm, value[2] / norm};
}

std::array<float, 3> crossVector(const std::array<float, 3>& a,
                                 const std::array<float, 3>& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

std::vector<BlurStencilEntry> makeBlurStencil(const std::array<float, 3>& axis,
                                              float sigma)
{
    if (sigma <= 0.0f)
        return {{0, 0, 0, 1.0f}};
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float kernelSum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float scaled = static_cast<float>(offset) / sigma;
        const float weight = std::exp(-0.5f * scaled * scaled);
        kernel[static_cast<size_t>(offset + radius)] = weight;
        kernelSum += weight;
    }
    for (float& weight : kernel)
        weight /= kernelSum;

    std::vector<BlurStencilEntry> stencil;
    stencil.reserve(kernel.size() * 8);
    for (int tap = -radius; tap <= radius; ++tap) {
        const std::array<float, 3> delta{
            static_cast<float>(tap) * axis[0],
            static_cast<float>(tap) * axis[1],
            static_cast<float>(tap) * axis[2],
        };
        const std::array<int, 3> low{
            static_cast<int>(std::floor(delta[0])),
            static_cast<int>(std::floor(delta[1])),
            static_cast<int>(std::floor(delta[2])),
        };
        const std::array<float, 3> fraction{
            delta[0] - static_cast<float>(low[0]),
            delta[1] - static_cast<float>(low[1]),
            delta[2] - static_cast<float>(low[2]),
        };
        const float tapWeight = kernel[static_cast<size_t>(tap + radius)];
        for (int cornerZ = 0; cornerZ < 2; ++cornerZ) {
            const float weightZ = cornerZ != 0 ? fraction[0] : 1.0f - fraction[0];
            for (int cornerY = 0; cornerY < 2; ++cornerY) {
                const float weightY = cornerY != 0 ? fraction[1] : 1.0f - fraction[1];
                for (int cornerX = 0; cornerX < 2; ++cornerX) {
                    const float weightX = cornerX != 0 ? fraction[2] : 1.0f - fraction[2];
                    const BlurStencilEntry candidate{
                        low[0] + cornerZ,
                        low[1] + cornerY,
                        low[2] + cornerX,
                        tapWeight * weightZ * weightY * weightX,
                    };
                    if (candidate.weight == 0.0f)
                        continue;
                    auto existing = std::find_if(
                        stencil.begin(), stencil.end(), [&](const BlurStencilEntry& entry) {
                            return entry.z == candidate.z && entry.y == candidate.y &&
                                   entry.x == candidate.x;
                        });
                    if (existing == stencil.end())
                        stencil.push_back(candidate);
                    else
                        existing->weight += candidate.weight;
                }
            }
        }
    }
    return stencil;
}

std::optional<size_t> axisAlignedDimension(const std::array<float, 3>& axis)
{
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        const size_t a = (dimension + 1) % 3;
        const size_t b = (dimension + 2) % 3;
        if (std::abs(std::abs(axis[dimension]) - 1.0f) <= 1.0e-6f &&
            std::abs(axis[a]) <= 1.0e-6f && std::abs(axis[b]) <= 1.0e-6f) {
            return dimension;
        }
    }
    return std::nullopt;
}

std::vector<float> makeGaussianKernel(float sigma)
{
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float scaled = static_cast<float>(offset) / sigma;
        const float weight = std::exp(-0.5f * scaled * scaled);
        kernel[static_cast<size_t>(offset + radius)] = weight;
        sum += weight;
    }
    for (float& weight : kernel)
        weight /= sum;
    return kernel;
}

void applyAxisAlignedBlur(const float* source,
                          float* destination,
                          const std::array<size_t, 3>& shape,
                          size_t dimension,
                          float sigma)
{
    const auto kernel = makeGaussianKernel(sigma);
    const int radius = static_cast<int>((kernel.size() - 1) / 2);
    const int depth = static_cast<int>(shape[0]);
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    const size_t plane = shape[1] * shape[2];
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = 0.0f;
                for (int offset = -radius; offset <= radius; ++offset) {
                    int sourceZ = z;
                    int sourceY = y;
                    int sourceX = x;
                    if (dimension == 0)
                        sourceZ = std::clamp(z + offset, 0, depth - 1);
                    else if (dimension == 1)
                        sourceY = std::clamp(y + offset, 0, height - 1);
                    else
                        sourceX = std::clamp(x + offset, 0, width - 1);
                    const size_t index = static_cast<size_t>(sourceZ) * plane +
                                         static_cast<size_t>(sourceY) * shape[2] +
                                         static_cast<size_t>(sourceX);
                    value += source[index] *
                             kernel[static_cast<size_t>(offset + radius)];
                }
                destination[static_cast<size_t>(z) * plane +
                            static_cast<size_t>(y) * shape[2] +
                            static_cast<size_t>(x)] = value;
            }
        }
    }
}

void applyBlurStencil(const float* source,
                      float* destination,
                      const std::array<size_t, 3>& shape,
                      const std::vector<BlurStencilEntry>& stencil)
{
    const int depth = static_cast<int>(shape[0]);
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);
    const size_t plane = shape[1] * shape[2];
    int minZ = 0;
    int minY = 0;
    int minX = 0;
    int maxZ = 0;
    int maxY = 0;
    int maxX = 0;
    std::vector<std::ptrdiff_t> linearOffsets;
    linearOffsets.reserve(stencil.size());
    for (const auto& entry : stencil) {
        minZ = std::min(minZ, entry.z);
        minY = std::min(minY, entry.y);
        minX = std::min(minX, entry.x);
        maxZ = std::max(maxZ, entry.z);
        maxY = std::max(maxY, entry.y);
        maxX = std::max(maxX, entry.x);
        linearOffsets.push_back(
            static_cast<std::ptrdiff_t>(entry.z) *
                static_cast<std::ptrdiff_t>(plane) +
            static_cast<std::ptrdiff_t>(entry.y) *
                static_cast<std::ptrdiff_t>(shape[2]) +
            static_cast<std::ptrdiff_t>(entry.x));
    }
    const int interiorZ0 = std::min(depth, std::max(0, -minZ));
    const int interiorY0 = std::min(height, std::max(0, -minY));
    const int interiorX0 = std::min(width, std::max(0, -minX));
    const int interiorZ1 = std::max(interiorZ0, std::min(depth, depth - maxZ));
    const int interiorY1 = std::max(interiorY0, std::min(height, height - maxY));
    const int interiorX1 = std::max(interiorX0, std::min(width, width - maxX));
    const auto writeBoundary = [&](int z, int y, int x) {
        float value = 0.0f;
        for (const auto& entry : stencil) {
            const int sourceZ = std::clamp(z + entry.z, 0, depth - 1);
            const int sourceY = std::clamp(y + entry.y, 0, height - 1);
            const int sourceX = std::clamp(x + entry.x, 0, width - 1);
            const size_t index = static_cast<size_t>(sourceZ) * plane +
                                 static_cast<size_t>(sourceY) * shape[2] +
                                 static_cast<size_t>(sourceX);
            value += source[index] * entry.weight;
        }
        destination[static_cast<size_t>(z) * plane +
                    static_cast<size_t>(y) * shape[2] +
                    static_cast<size_t>(x)] = value;
    };
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            if (z < interiorZ0 || z >= interiorZ1 ||
                y < interiorY0 || y >= interiorY1 ||
                interiorX0 >= interiorX1) {
                for (int x = 0; x < width; ++x)
                    writeBoundary(z, y, x);
                continue;
            }
            for (int x = 0; x < interiorX0; ++x)
                writeBoundary(z, y, x);
            for (int x = interiorX1; x < width; ++x)
                writeBoundary(z, y, x);

            float* destinationRow = destination + static_cast<size_t>(z) * plane +
                                    static_cast<size_t>(y) * shape[2];
            std::fill(destinationRow + interiorX0, destinationRow + interiorX1, 0.0f);
            for (size_t tap = 0; tap < stencil.size(); ++tap) {
                const float* sourceRow = source +
                    static_cast<std::ptrdiff_t>(static_cast<size_t>(z) * plane +
                                                static_cast<size_t>(y) * shape[2]) +
                    linearOffsets[tap];
                const float weight = stencil[tap].weight;
                for (int x = interiorX0; x < interiorX1; ++x)
                    destinationRow[x] += sourceRow[x] * weight;
            }
        }
    }
}

nb::ndarray<float, nb::numpy, nb::c_contig> anisotropicBlur3D(
    const FloatCoords& volume,
    const FloatCoords& axisZyx,
    float sigmaAlong,
    float sigmaAcross)
{
    if (volume.ndim() != 3)
        throw nb::value_error("volume must have shape (Z,Y,X)");
    requireFloatShape(axisZyx, {3}, "axis_zyx");
    requireFinite(axisZyx, "axis_zyx");
    if (!std::isfinite(sigmaAlong) || sigmaAlong < 0.0f ||
        !std::isfinite(sigmaAcross) || sigmaAcross < 0.0f) {
        throw nb::value_error("blur sigmas must be finite and nonnegative");
    }
    std::array<size_t, 3> shape{};
    size_t pointCount = 1;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        shape[dimension] = volume.shape(dimension);
        if (shape[dimension] == 0)
            throw nb::value_error("volume dimensions must be positive");
        if (shape[dimension] > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw nb::value_error("volume dimensions exceed native index range");
        if (pointCount > std::numeric_limits<size_t>::max() / shape[dimension])
            throw nb::value_error("volume allocation overflows size_t");
        pointCount *= shape[dimension];
    }
    if (pointCount > std::numeric_limits<size_t>::max() / sizeof(float))
        throw nb::value_error("volume allocation overflows size_t");

    const float* rawAxis = axisZyx.data();
    const auto along = normalizeVector(
        {rawAxis[0], rawAxis[1], rawAxis[2]}, "axis_zyx");
    const std::array<float, 3> reference =
        std::abs(along[0]) > 0.9f
            ? std::array<float, 3>{0.0f, 1.0f, 0.0f}
            : std::array<float, 3>{1.0f, 0.0f, 0.0f};
    const auto acrossA = normalizeVector(
        crossVector(along, reference), "perpendicular axis");
    const auto acrossB = normalizeVector(
        crossVector(along, acrossA), "perpendicular axis");
    const std::array<std::array<float, 3>, 3> axes{along, acrossA, acrossB};
    const std::array<float, 3> sigmas{sigmaAlong, sigmaAcross, sigmaAcross};

    std::vector<float> first(pointCount);
    std::vector<float> second(pointCount);
    const float* source = volume.data();
    float* destination = first.data();
    size_t completedPasses = 0;
    {
        nb::gil_scoped_release release;
        for (size_t pass = 0; pass < axes.size(); ++pass) {
            if (sigmas[pass] <= 0.0f)
                continue;
            const auto alignedDimension = axisAlignedDimension(axes[pass]);
            if (alignedDimension) {
                applyAxisAlignedBlur(
                    source, destination, shape, *alignedDimension, sigmas[pass]);
            } else {
                const auto stencil = makeBlurStencil(axes[pass], sigmas[pass]);
                applyBlurStencil(source, destination, shape, stencil);
            }
            source = destination;
            destination = destination == first.data() ? second.data() : first.data();
            ++completedPasses;
        }
        if (completedPasses == 0)
            std::copy(volume.data(), volume.data() + pointCount, first.data());
        else if (source != first.data())
            first.swap(second);
    }
    return makeNumpyArray<float>(std::move(first), shape);
}


} // namespace

NB_MODULE(volume, m)
{
    bindNormalSampler(m);
    m.doc() = "Fiber3D training kernels and the Lasagna normal sampler";
    m.def("generate_fiber3d_coords",
          &generateFiber3DCoords,
          "output_shape_zyx"_a,
          "cp_local_zyx"_a,
          "cp_volume_zyx"_a,
          "output_to_source_zyx"_a,
          "smooth_mode"_a,
          "smooth_controls"_a,
          "smooth_origin_zyx"_a,
          "smooth_spacing_zyx"_a,
          "volume_spacing_base"_a,
          "base_shape_zyx"_a,
          "Generate Fiber3D base coordinates and validity in one native pass.");
    m.def("anisotropic_blur_3d",
          &anisotropicBlur3D,
          "volume"_a,
          "axis_zyx"_a,
          "sigma_along"_a,
          "sigma_across"_a,
          "Apply the Fiber3D three-pass oriented Gaussian blur on one CPU thread.");
    m.def("fiber3d_value_transform",
          &fiber3DValueTransform,
          "volume"_a,
          "valid_mask"_a,
          "normalization"_a,
          "contrast"_a,
          "brightness"_a,
          "gamma"_a,
          "Apply Fiber3D normalization, contrast, brightness, and gamma in one CPU pass.");
}
