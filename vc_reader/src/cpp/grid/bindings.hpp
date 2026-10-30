#pragma once
// Python access to vcr::grid: a surface given as its vertex grid (and optional vertex
// normals), resampled at a raster's pixels or sampled from one volume level along it.
#include "vc_reader/grid.hpp"

#include <nanobind/stl/array.h>

#include <memory>

namespace vcr::grid::python {

using Floats = nb::ndarray<float, nb::c_contig, nb::device::cpu>;

struct Input {
    std::unique_ptr<Surface> surface;
    Raster raster;
};

// raster: x0, y0, sx, sy, kx, ky (see Raster).
inline Input input(const Floats& xyz, const Floats& normals, std::array<float, 3> spacing, std::array<double, 6> raster,
                   int width, int height)
{
    if (xyz.ndim() != 3 || xyz.shape(2) != 3 || xyz.shape(0) > (1u << 30) || xyz.shape(1) > (1u << 30))
        throw nb::value_error("surface vertices must be (rows, cols, 3) float32");
    if (normals.size() && (normals.ndim() != 3 || normals.shape(0) != xyz.shape(0) || normals.shape(1) != xyz.shape(1) ||
                           normals.shape(2) != 3))
        throw nb::value_error("vertex normals must be empty or match the vertices");
    if (width < 0 || height < 0)
        throw nb::value_error("invalid raster size");
    for (double v : raster)
        if (!std::isfinite(v))
            throw nb::value_error("nonfinite raster");
    for (float v : spacing)
        if (!std::isfinite(v) || !(v > 0))
            throw nb::value_error("spacing must be positive and finite");
    SurfaceOptions options;
    options.spacing = spacing;
    const Vertices v{xyz.data(), int(xyz.shape(0)), int(xyz.shape(1)), std::size_t(xyz.shape(1)) * 3,
                     normals.size() ? normals.data() : nullptr};
    return {std::make_unique<Surface>(v, options), Raster{raster[0], raster[1], raster[2], raster[3], width, height,
                                                          raster[4], raster[5]}};
}

inline void store(const Result& result, Floats& values, Bytes& level)
{
    std::copy(result.values.begin(), result.values.end(), values.data());
    std::copy(result.level.begin(), result.level.end(), reinterpret_cast<std::int8_t*>(level.data()));
}

inline void bind(nb::module_& m)
{
    m.def("_grid_resample",
          [](Floats xyz, Floats normals, std::array<float, 3> spacing, std::array<double, 6> raster, int width, int height,
             Floats points, Floats steps) {
              auto in = input(xyz, normals, spacing, raster, width, height);
              const std::size_t n = std::size_t(width) * std::size_t(height) * 3;
              if (points.size() != n || (steps.size() && steps.size() != n))
                  throw nb::value_error("resample outputs do not match the raster");
              nb::gil_scoped_release release;
              in.surface->resample(in.raster, points.data(), steps.size() ? steps.data() : nullptr);
          },
          "Positions (and normal steps: vertex-coordinate moves per physical unit along the unit normal) of a raster "
          "over a vertex grid; NaN where the surface has no point.");

    using Offsets = std::vector<float>;
    m.def("_grid_sample",
          [](Floats xyz, Floats normals, std::array<float, 3> spacing, std::array<double, 6> raster, int width, int height,
             Array& array, const std::string& kind, std::size_t item, bool swap, std::array<double, 12> map, int order,
             Offsets offsets, Floats values, Bytes level) {
              auto in = input(xyz, normals, spacing, raster, width, height);
              Request request;
              request.levels = {levelFromMap(array, sample_type(kind, item, swap), map)};
              request.order = order;
              request.offsets = std::move(offsets);
              const std::size_t pixels = std::size_t(width) * std::size_t(height);
              if (values.size() != pixels * request.offsets.size() || level.size() != pixels)
                  throw nb::value_error("sample outputs do not match the raster");
              nb::gil_scoped_release release;
              store(sample(*in.surface, in.raster, request), values, level);
          },
          "Sample one volume level at layer offsets along a vertex-grid surface (float32 values, layer-major; per-pixel "
          "level, -2 where the surface has no point).");
#ifdef VC_READER_CUDA
    m.def("_grid_sample_cuda",
          [](Floats xyz, Floats normals, std::array<float, 3> spacing, std::array<double, 6> raster, int width, int height,
             Array& array, const std::string& kind, std::size_t item, bool swap, std::array<double, 12> map, int order,
             Offsets offsets, Floats values, Bytes level, reader_cuda::Reader& reader) {
              auto in = input(xyz, normals, spacing, raster, width, height);
              Request request;
              request.levels = {levelFromMap(array, sample_type(kind, item, swap), map)};
              request.order = order;
              request.offsets = std::move(offsets);
              const std::size_t pixels = std::size_t(width) * std::size_t(height);
              if (values.size() != pixels * request.offsets.size() || level.size() != pixels)
                  throw nb::value_error("sample outputs do not match the raster");
              nb::gil_scoped_release release;
              store(cuda::sample(*in.surface, in.raster, request, {&reader}), values, level);
          },
          "_grid_sample on the CUDA reader of `array` (uint8 3-D).");
#endif
}

}  // namespace vcr::grid::python
