#pragma once
// Python access to vcr::tifxyz and vcr::sources::writeTiff.
#include "vc_reader/image_write.hpp"
#include "vc_reader/tifxyz.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace vcr::tifxyz::python {

using FloatsIn = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using BytesIn = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;

inline grid::Interpolation interpolationOf(const std::string& name)
{
    if (name == "linear") return grid::Interpolation::Linear;
    if (name == "smooth") return grid::Interpolation::Smooth;
    throw nb::value_error("interpolation must be 'linear' or 'smooth'");
}
inline nb::object fromJson(const nlohmann::json& value) { return nb::module_::import_("json").attr("loads")(value.dump()); }
inline nlohmann::json toJson(nb::handle value) { return nlohmann::json::parse(nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(value))); }

// A new float32 array of this shape with its owner.
template <std::size_t N>
inline nb::ndarray<nb::numpy, float, nb::ndim<N>> floats(std::array<std::size_t, N> shape)
{
    std::size_t total = 1;
    for (const auto n : shape) total *= n;
    auto* data = new float[total];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    return nb::ndarray<nb::numpy, float, nb::ndim<N>>(data, N, shape.data(), owner);
}
inline nb::ndarray<nb::numpy, bool, nb::ndim<2>> finiteOf(const float* xyz, std::size_t rows, std::size_t cols)
{
    const std::size_t n = rows * cols;
    auto* data = new bool[n];
    for (std::size_t i = 0; i < n; ++i) data[i] = xyz[i * 3] == xyz[i * 3];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
    const std::size_t shape[2]{rows, cols};
    return nb::ndarray<nb::numpy, bool, nb::ndim<2>>(data, 2, shape, owner);
}

inline nb::tuple pointsOut(const Surface& s, const float* xy, std::size_t n, std::size_t rows, std::size_t cols, grid::Interpolation mode, bool normals)
{
    auto xyz = floats<3>({rows, cols, 3});
    auto valid = nb::ndarray<nb::numpy, float, nb::ndim<3>>();
    if (normals) valid = floats<3>({rows, cols, 3});
    {
        nb::gil_scoped_release release;
        s.pointsAt(xy, n, mode, xyz.data(), normals ? valid.data() : nullptr);
    }
    return normals ? nb::make_tuple(xyz, finiteOf(xyz.data(), rows, cols), valid) : nb::make_tuple(xyz, finiteOf(xyz.data(), rows, cols));
}

inline nb::tuple rasterOut(const Surface& s, const grid::Raster& raster, grid::Interpolation mode, bool normals)
{
    const std::size_t r = std::size_t(raster.height), c = std::size_t(raster.width);
    auto xyz = floats<3>({r, c, 3});
    auto nrm = nb::ndarray<nb::numpy, float, nb::ndim<3>>();
    if (normals) nrm = floats<3>({r, c, 3});
    {
        nb::gil_scoped_release release;
        s.resample(raster, mode, xyz.data(), normals ? nrm.data() : nullptr);
    }
    return normals ? nb::make_tuple(xyz, finiteOf(xyz.data(), r, c), nrm) : nb::make_tuple(xyz, finiteOf(xyz.data(), r, c));
}

inline void bind(nb::module_& m)
{
    m.def("write_tiff", [](const std::filesystem::path& path, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> image, const std::string& dtype, const std::string& compression, std::optional<std::array<int, 2>> tile) {
        if (image.ndim() != 2 && image.ndim() != 3) throw nb::value_error("write_tiff takes (rows, cols) or (rows, cols, channels)");
        sources::TiffOptions options;
        options.compression = compression;
        options.tile = tile;
        const auto channels = image.ndim() == 3 ? std::int64_t(image.shape(2)) : 1;
        nb::gil_scoped_release release;
        sources::writeTiff(path, image.data(), dtype, std::int64_t(image.shape(0)), std::int64_t(image.shape(1)), channels, options);
    }, "path"_a, "image"_a, "dtype"_a, "compression"_a = "none", "tile"_a = nb::none());

    m.def("_tifxyz_full_extent", &fullExtent, "stored"_a, "scale"_a);

    nb::class_<Surface>(m, "TifxyzSurface")
        .def("__init__", [](Surface* self, FloatsIn xyz, nb::object mask, std::array<double, 2> scale, const std::string& uuid, std::optional<std::array<double, 6>> bbox, std::optional<double> area, nb::object extra, const std::string& location) {
            if (xyz.ndim() != 3 || xyz.shape(2) != 3) throw nb::value_error("tifxyz points must be (rows, cols, 3) float32 XYZ");
            std::vector<std::uint8_t> flags;
            if (!mask.is_none()) {
                const auto values = nb::cast<BytesIn>(mask);
                if (values.ndim() != 2 || values.shape(0) != xyz.shape(0) || values.shape(1) != xyz.shape(1)) throw nb::value_error("tifxyz mask shape does not match the coordinate grid");
                flags.assign(values.data(), values.data() + values.size());
            }
            Metadata metadata;
            metadata.uuid = uuid;
            metadata.scale = scale;
            metadata.bbox = bbox;
            metadata.area = area;
            if (!extra.is_none()) metadata.extra = toJson(extra);
            new (self) Surface(std::int64_t(xyz.shape(0)), std::int64_t(xyz.shape(1)), std::vector<float>(xyz.data(), xyz.data() + xyz.size()), std::move(flags), std::move(metadata), location);
        }, "xyz"_a, "mask"_a = nb::none(), "scale"_a = std::array<double, 2>{1, 1}, "uuid"_a = "", "bbox"_a = nb::none(), "area"_a = nb::none(), "extra"_a = nb::none(), "location"_a = "")
        .def_prop_ro("shape", [](const Surface& s) { return nb::make_tuple(s.rows(), s.cols()); })
        .def_prop_ro("full_shape", [](const Surface& s) { const auto f = s.fullShape(); return nb::make_tuple(f[0], f[1]); })
        .def_prop_ro("xyz", [](nb::handle_t<Surface> self) {
            const auto& s = nb::cast<const Surface&>(self);
            const std::size_t shape[3]{std::size_t(s.rows()), std::size_t(s.cols()), 3};
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(s.xyz().data(), 3, shape, self);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("valid", [](nb::handle_t<Surface> self) {
            const auto& s = nb::cast<const Surface&>(self);
            const std::size_t shape[2]{std::size_t(s.rows()), std::size_t(s.cols())};
            return nb::ndarray<nb::numpy, const bool, nb::ndim<2>>(reinterpret_cast<const bool*>(s.valid().data()), 2, shape, self);
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("scale", [](const Surface& s) { return nb::make_tuple(s.metadata().scale[0], s.metadata().scale[1]); })
        .def_prop_rw("uuid", [](const Surface& s) { return s.metadata().uuid; }, [](Surface& s, const std::string& v) { s.metadata().uuid = v; })
        .def_prop_ro("bbox", [](const Surface& s) { return s.metadata().bbox; })
        .def_prop_ro("area", [](const Surface& s) { return s.metadata().area; })
        .def_prop_ro("extra", [](const Surface& s) { return fromJson(s.metadata().extra); })
        .def_prop_ro("location", &Surface::location)
        .def("__getstate__", [](const Surface& s) {
            const auto& m = s.metadata();
            return nb::make_tuple(nb::bytes(s.xyz().data(), s.xyz().size() * sizeof(float)), s.rows(), s.cols(), nb::make_tuple(m.scale[0], m.scale[1]), m.uuid, m.bbox, m.area, fromJson(m.extra), s.location());
        })
        .def("__setstate__", [](Surface& self, nb::tuple state) {
            const auto bytes = nb::cast<nb::bytes>(state[0]);
            std::vector<float> xyz(bytes.size() / sizeof(float));
            std::memcpy(xyz.data(), bytes.c_str(), bytes.size());
            Metadata metadata;
            metadata.scale = nb::cast<std::array<double, 2>>(state[3]);
            metadata.uuid = nb::cast<std::string>(state[4]);
            metadata.bbox = nb::cast<std::optional<Bounds>>(state[5]);
            metadata.area = nb::cast<std::optional<double>>(state[6]);
            metadata.extra = toJson(state[7]);
            new (&self) Surface(nb::cast<std::int64_t>(state[1]), nb::cast<std::int64_t>(state[2]), std::move(xyz), {}, std::move(metadata), nb::cast<std::string>(state[8]));
        })
        .def("bounds", &Surface::bounds)
        .def("retarget", &Surface::retargeted, "factor"_a)
        .def("points_at", [](const Surface& s, FloatsIn rows, FloatsIn cols, const std::string& interpolation, bool normals) {
            if (rows.ndim() != 2 || cols.ndim() != 2 || rows.shape(0) != cols.shape(0) || rows.shape(1) != cols.shape(1)) throw nb::value_error("points_at takes 2-D rows and columns of one shape");
            const std::size_t r = rows.shape(0), c = rows.shape(1), n = r * c;
            std::vector<float> xy(n * 2);
            for (std::size_t i = 0; i < n; ++i) {
                xy[i * 2] = cols.data()[i];
                xy[i * 2 + 1] = rows.data()[i];
            }
            return pointsOut(s, xy.data(), n, r, c, interpolationOf(interpolation), normals);
        }, "rows"_a, "cols"_a, "interpolation"_a = "linear", "normals"_a = false)
        .def("resample", [](const Surface& s, std::array<double, 2> origin, std::array<double, 2> step, std::array<int, 2> size, const std::string& interpolation, bool normals) {
            if (size[0] <= 0 || size[1] <= 0) throw nb::value_error("resample size must be positive");
            return rasterOut(s, grid::Raster{origin[1], origin[0], step[1], step[0], size[1], size[0]}, interpolationOf(interpolation), normals);
        }, "origin"_a, "step"_a, "size"_a, "interpolation"_a = "linear", "normals"_a = false,
           "Positions at grid positions origin + (j, i) * step, in stored cells (row, column); NaN where no patch covers one.")
        .def("resample_full", [](const Surface& s, std::array<std::int64_t, 2> start, std::array<std::int64_t, 2> size, const std::string& interpolation, bool normals) {
            return rasterOut(s, s.fullRaster(start, size), interpolationOf(interpolation), normals);
        }, "start"_a, "size"_a, "interpolation"_a = "linear", "normals"_a = false,
           "Positions of a window at full resolution: start and size in rows and columns of full_shape.");

    m.def("_tifxyz_read", [](const std::string& location, bool mask, nb::object options) {
        auto optionsFor = sources::python::optionsFrom(std::move(options));
        nb::gil_scoped_release release;
        return std::make_shared<Surface>(read(location, mask, optionsFor));
    }, "location"_a, "mask"_a, "options"_a);
    m.def("_tifxyz_read_mask", [](const std::string& location, std::int64_t rows, std::int64_t cols, nb::object options) -> nb::object {
        auto optionsFor = sources::python::optionsFrom(std::move(options));
        std::vector<std::uint8_t> mask;
        {
            nb::gil_scoped_release release;
            mask = readMask(location, rows, cols, optionsFor);
        }
        if (mask.empty()) return nb::none();
        auto* data = new bool[mask.size()];
        std::copy(mask.begin(), mask.end(), data);
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
        const std::size_t shape[2]{std::size_t(rows), std::size_t(cols)};
        return nb::cast(nb::ndarray<nb::numpy, bool, nb::ndim<2>>(data, 2, shape, owner));
    }, "location"_a, "rows"_a, "cols"_a, "options"_a);
    m.def("_tifxyz_read_metadata", [](const std::string& location, nb::object options) {
        auto optionsFor = sources::python::optionsFrom(std::move(options));
        Metadata metadata;
        {
            nb::gil_scoped_release release;
            metadata = readMetadata(location, optionsFor);
        }
        return nb::make_tuple(metadata.uuid, nb::make_tuple(metadata.scale[0], metadata.scale[1]), metadata.bbox, metadata.area, fromJson(metadata.extra));
    }, "location"_a, "options"_a);
    m.def("_tifxyz_list", [](const std::string& folder, bool recursive, std::optional<std::array<double, 2>> zRange, nb::object options) {
        auto optionsFor = sources::python::optionsFrom(std::move(options));
        std::vector<Entry> entries;
        {
            nb::gil_scoped_release release;
            entries = list(folder, recursive, zRange, optionsFor);
        }
        nb::list out;
        for (const auto& e : entries)
            out.append(nb::make_tuple(e.location, e.metadata.uuid, nb::make_tuple(e.metadata.scale[0], e.metadata.scale[1]), e.metadata.bbox, e.metadata.area, fromJson(e.metadata.extra), e.error));
        return out;
    }, "folder"_a, "recursive"_a, "z_range"_a, "options"_a);
    m.def("_tifxyz_write", [](const std::filesystem::path& path, Surface& surface, const std::string& compression, std::optional<int> tile, bool mask, bool overwrite) {
        WriteOptions options;
        options.compression = compression;
        options.tile = tile;
        options.mask = mask;
        options.overwrite = overwrite;
        nb::gil_scoped_release release;
        write(path, surface, options);
    }, "path"_a, "surface"_a, "compression"_a, "tile"_a, "mask"_a, "overwrite"_a);
}

}  // namespace vcr::tifxyz::python
