#pragma once
// Python bindings of vcr::grid requests and views: levels with their placement and
// optional CUDA reader, planes, tifxyz surfaces and prepared points, layers,
// composites, cache-only reads, pixel masks, and progressive frames.
#include "vc_reader/grid.hpp"
#include "vc_reader/tifxyz.hpp"
#include "vc_reader/zarr.hpp"
#ifdef VC_READER_CUDA
#include "../reader_cuda/reader.hpp"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>

namespace vcr::grid::python {

namespace nb = nanobind;
using namespace nb::literals;

using PlaneArgs = std::tuple<std::array<float, 3>, std::array<float, 3>, std::array<float, 3>, std::array<float, 3>, int, int>;
using RasterArgs = std::array<double, 8>;  // x0, y0, sx, sy, width, height, kx, ky
using HW3 = nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using HW = nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

inline Raster rasterOf(const RasterArgs& r)
{
    if (r[4] < 0 || r[5] < 0 || r[4] > INT_MAX || r[5] > INT_MAX)
        throw nb::value_error("raster width and height must be nonnegative");
    return {r[0], r[1], r[2], r[3], int(r[4]), int(r[5]), r[6], r[7]};
}

inline Plane planeOf(const PlaneArgs& p)
{
    Plane plane{std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<3>(p), std::get<4>(p), std::get<5>(p)};
    if (plane.width < 0 || plane.height < 0)
        throw nb::value_error("plane width and height must be nonnegative");
    return plane;
}

// A plane, or a raster over a tifxyz surface (kept alive by the caller: the frame does
// not own it).
inline Frame frameOf(const std::optional<PlaneArgs>& plane, const tifxyz::Surface* surface,
                     const std::optional<RasterArgs>& raster, const std::string& interpolation)
{
    Frame frame;
    if (plane) {
        if (surface)
            throw nb::value_error("give a plane or a surface, not both");
        frame.plane = planeOf(*plane);
        return frame;
    }
    if (!surface || !raster)
        throw nb::value_error("a frame needs a plane, or a surface and a raster");
    frame.surface = std::shared_ptr<const Surface>(&surface->interpolator(interpolationNamed(interpolation)),
                                                   [](const Surface*) {});
    frame.raster = rasterOf(*raster);
    return frame;
}

inline std::vector<Level> levelsOf(const std::vector<std::optional<Level>>& levels)
{
    std::vector<Level> out;
    out.reserve(levels.size());
    for (const auto& level : levels)
        out.push_back(level ? *level : Level{});
    return out;
}

inline Request requestOf(const std::vector<Level>& levels, int order, std::vector<float> offsets, bool cachedOnly,
                         const std::optional<composite::Options>& composite)
{
    if (levels.empty())
        throw nb::value_error("a request needs at least one level");
    for (const auto& level : levels)
        if (!level.array)
            throw nb::value_error("a request's levels must all be present");
    if (order != 0 && order != 1)
        throw nb::value_error("order must be 0 (nearest) or 1 (linear)");
    Request request;
    request.levels = levels;
    request.order = order;
    request.offsets = offsets.empty() ? std::vector<float>{0.0f} : std::move(offsets);
    request.cachedOnly = cachedOnly;
    request.composite = composite;
    return request;
}

// (values (layers, height, width) float32, level (height, width) int8, GPU failure).
inline nb::tuple resultOf(Result&& result)
{
    auto owned = std::make_unique<Result>(std::move(result));
    const std::size_t h = std::size_t(owned->height), w = std::size_t(owned->width);
    const std::size_t layers = h * w ? owned->values.size() / (h * w) : 0;
    auto* values = owned->values.data();
    auto* level = owned->level.data();
    const std::string failure = owned->cudaFailure;
    nb::capsule capsule(owned.release(), [](void* p) noexcept { delete static_cast<Result*>(p); });
    return nb::make_tuple(nb::ndarray<nb::numpy, float, nb::ndim<3>>(values, {layers, h, w}, capsule),
                          nb::ndarray<nb::numpy, std::int8_t, nb::ndim<2>>(level, {h, w}, capsule), failure);
}

inline void bind(nb::module_& m)
{
    nb::class_<Level>(m, "GridLevel", "One pyramid level as vcr::grid samples it: its array, placement and optional CUDA reader.")
        .def_prop_ro("scale", [](const Level& l) { return l.scale; })
        .def_prop_ro("offset", [](const Level& l) { return l.offset; })
        .def_prop_ro("linear", [](const Level& l) { return l.linear; })
        .def_prop_ro("gpu", [](const Level& l) { return l.cuda != nullptr; });

    const auto withCuda = [](Level level, nb::handle cuda) {
        if (!cuda.is_none()) {
#ifdef VC_READER_CUDA
            level.cuda = &nb::cast<reader_cuda::Reader&>(cuda);
#else
            throw nb::value_error("CUDA support is not built");
#endif
        }
        return level;
    };
    m.def("grid_level", [withCuda](zarr::OpenedArray& opened, std::array<double, 3> workingScale,
                                   std::array<double, 3> workingOffset, nb::handle cuda) {
        return withCuda(zarr::gridLevel(opened, {workingScale, workingOffset}), cuda);
    }, "opened"_a, "working_scale"_a, "working_offset"_a, "cuda"_a.none(),
       nb::keep_alive<0, 1>(), nb::keep_alive<0, 4>(),
       "The level for points in the voxels of the working placement (ZYX level-0 scale and offset).");
    m.def("grid_level_from_map", [withCuda](zarr::OpenedArray& opened, const std::array<double, 12>& map, nb::handle cuda) {
        return withCuda(levelFromMap(*opened.array, opened.meta.type, map), cuda);
    }, "opened"_a, "map"_a, "cuda"_a.none(), nb::keep_alive<0, 1>(), nb::keep_alive<0, 3>(),
       "The level placed by a row-major 3x4 map from XYZ points to ZYX indices.");
    m.def("grid_voxel_extent", &voxelExtent, "level"_a);
    m.def("level_for_view", [](const std::vector<std::optional<Level>>& levels, float pixelsPerVoxel, float qualityBias) {
        return levelForView(levelsOf(levels), pixelsPerVoxel, qualityBias);
    }, "levels"_a, "pixels_per_voxel"_a, "quality_bias"_a = 1.0f);
    m.def("fallback_levels", [](const std::vector<std::optional<Level>>& levels, int start, int width, int height,
                                float pixelsPerVoxel, int maximum) {
        return fallbackLevels(levelsOf(levels), start, width, height, pixelsPerVoxel, maximum);
    }, "levels"_a, "start"_a, "width"_a, "height"_a, "pixels_per_voxel"_a, "maximum"_a = 5);
    m.def("pixels_per_voxel", [](const tifxyz::Surface& surface, const RasterArgs& raster, const std::string& interpolation) {
        nb::gil_scoped_release release;
        return pixelsPerVoxel(surface.interpolator(interpolationNamed(interpolation)), rasterOf(raster));
    }, "surface"_a, "raster"_a, "interpolation"_a = "bicubic");

    static const std::map<std::string, composite::Method> methods{
        {"mean", composite::Method::Mean}, {"max", composite::Method::Max}, {"min", composite::Method::Min},
        {"median", composite::Method::Median}, {"min_abs", composite::Method::MinAbs},
        {"alpha", composite::Method::Alpha}, {"beer_lambert", composite::Method::BeerLambert}};
    nb::class_<composite::Options>(m, "CompositeOptions")
        .def("__init__", [](composite::Options* self, const std::string& method) {
            const auto found = methods.find(method);
            if (found == methods.end())
                throw nb::value_error(("unknown composite method '" + method + "'").c_str());
            new (self) composite::Options{};
            self->method = found->second;
        }, "method"_a)
        .def_prop_ro("method", [](const composite::Options& o) {
            for (const auto& [name, method] : methods)
                if (method == o.method)
                    return name;
            return std::string();
        })
        .def_rw("iso_cutoff", &composite::Options::isoCutoff)
        .def_rw("alpha_min", &composite::Options::alphaMin)
        .def_rw("alpha_max", &composite::Options::alphaMax)
        .def_rw("alpha_opacity", &composite::Options::alphaOpacity)
        .def_rw("alpha_cutoff", &composite::Options::alphaCutoff)
        .def_rw("extinction", &composite::Options::extinction)
        .def_rw("emission", &composite::Options::emission)
        .def_rw("ambient", &composite::Options::ambient);

    nb::class_<Pending>(m, "GridPending", "Blocking reads progressive frames could not sample from the caches.")
        .def(nb::init<>())
        .def_prop_ro("empty", &Pending::empty)
        .def_prop_ro("coarse_reads", [](const Pending& p) { return p.coarse.size(); })
        .def_prop_ro("target_reads", [](const Pending& p) { return p.target.size(); })
        .def("clear", [](Pending& p) { p = {}; });
    m.def("load_grid", [](const Pending& pending, nb::handle afterPart) {
        std::function<bool()> after;
        if (!afterPart.is_none())
            after = [&afterPart] {
                nb::gil_scoped_acquire acquire;
                return nb::cast<bool>(afterPart());
            };
        nb::gil_scoped_release release;
        return load(pending, after);
    }, "pending"_a, "after_part"_a.none() = nb::none(),
       "Runs the reads, coarse then target; after_part() after each part, False stops. Returns the first GPU failure.");

    m.def("sample_grid", [](const std::vector<std::optional<Level>>& levels, std::optional<PlaneArgs> plane,
                            const tifxyz::Surface* surface, std::optional<RasterArgs> raster, const std::string& interpolation,
                            std::optional<HW3> xyz, std::optional<HW3> normals, int order, std::vector<float> offsets,
                            bool cachedOnly, std::optional<HW> pixels, std::optional<composite::Options> composite,
                            Pending* pending) {
        auto request = requestOf(levelsOf(levels), order, std::move(offsets), cachedOnly, composite);
        std::optional<Points> points;
        Frame frame;
        if (xyz) {
            if (plane || surface || pending)
                throw nb::value_error("prepared points are sampled alone, without a plane, surface or pending reads");
            const auto h = xyz->shape(0), w = xyz->shape(1);
            if (xyz->shape(2) != 3 || h > INT_MAX || w > INT_MAX ||
                (normals && (normals->shape(0) != h || normals->shape(1) != w || normals->shape(2) != 3)))
                throw nb::value_error("expected H,W,3 positions and directions");
            points = Points{xyz->data(), normals ? normals->data() : nullptr, int(w), int(h)};
        } else {
            frame = frameOf(plane, surface, raster, interpolation);
        }
        const std::size_t h = points ? std::size_t(points->height) : std::size_t(frame.height());
        const std::size_t w = points ? std::size_t(points->width) : std::size_t(frame.width());
        if (pixels) {
            if (pixels->shape(0) != h || pixels->shape(1) != w)
                throw nb::value_error("pixels must have the frame's (height, width)");
            request.pixels = pixels->data();
        }
        Result result;
        {
            nb::gil_scoped_release release;
            if (points)
                result = sample(*points, request);
            else if (pending)
                result = sampleCached(frame, std::move(request), pending);
            else
                result = sample(frame, request);
        }
        return resultOf(std::move(result));
    }, "levels"_a, "plane"_a.none() = nb::none(), "surface"_a.none() = nb::none(), "raster"_a.none() = nb::none(),
       "interpolation"_a = "bicubic", "xyz"_a.none() = nb::none(), "normals"_a.none() = nb::none(), "order"_a = 1,
       "offsets"_a = std::vector<float>{0.0f}, "cached_only"_a = false, "pixels"_a.none() = nb::none(),
       "composite"_a.none() = nb::none(), "pending"_a.none() = nb::none(),
       "Samples levels (first preferred) over a plane, a tifxyz surface raster or prepared points. "
       "With pending, samples the caches alone and adds the reads the frame lacked.");
}

}  // namespace vcr::grid::python
