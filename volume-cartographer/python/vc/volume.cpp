#include <vc_reader/transforms.hpp>
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
    if (normalization != "none" && normalization != "raw" && normalization != "identity" &&
        normalization != "zscore" && normalization != "minmax")
        throw nb::value_error("unsupported image normalization");
    if (!std::isfinite(contrast) || contrast <= 0 || !std::isfinite(brightness) ||
        !std::isfinite(gamma) || gamma <= 0)
        throw nb::value_error("contrast and gamma must be positive and all parameters finite");
    std::vector<float> output(pointCount);
    {
        nb::gil_scoped_release release;
        namespace tr = vcr::transforms;
        tr::Shape dimensions{1, shape[0], shape[1], shape[2]};
        tr::Array image(dimensions, 'f', sizeof(float)), mask(dimensions, 'u', 1);
        std::memcpy(image.data(), volume.data(), pointCount * sizeof(float));
        std::memcpy(mask.data(), validMask.data(), pointCount);
        tr::Target imageTarget, maskTarget;
        imageTarget.refill = false;
        maskTarget.role = tr::Role::Valid;
        tr::Sample sample{{"image", {std::move(image), imageTarget}},
                          {"padding_mask", {std::move(mask), maskTarget}}};
        tr::Random random{0};
        const auto apply = [&](const auto& operation) {
            tr::apply(operation, sample, tr::sample(operation, sample, random));
        };
        if (std::any_of(validMask.data(), validMask.data() + pointCount, [](bool v) { return v; }) &&
            (normalization == "zscore" || normalization == "minmax")) {
            tr::Normalize op;
            op.mode = normalization == "zscore" ? tr::Normalization::ZScore : tr::Normalization::MinMax;
            op.nonzero = false; op.masked = true; op.eps = 1e-6;
            apply(op);
        }
        tr::Contrast contrastOp;
        contrastOp.factor = double(contrast); contrastOp.preserve_range = false; contrastOp.masked = true;
        apply(contrastOp);
        tr::BrightnessAdd brightnessOp;
        brightnessOp.shift = double(brightness);
        apply(brightnessOp);
        if (gamma != 1) {
            tr::Gamma op;
            op.gamma = double(gamma); op.masked = true; op.clip = true; op.eps = 1e-6;
            apply(op);
        }
        std::memcpy(output.data(), sample.at("image").array.data(), pointCount * sizeof(float));
    }

    return makeNumpyArray<float>(std::move(output), shape);
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
