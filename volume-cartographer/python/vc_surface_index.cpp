#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

namespace nb = nanobind;

namespace {

struct PyQuadSurface {
    using ZyxArray = nb::ndarray<nb::numpy, const float, nb::shape<-1, -1, 3>, nb::c_contig>;

    std::string id;
    SurfacePatchIndex::SurfacePtr surface;

    PyQuadSurface(const std::string& surface_id, ZyxArray zyx, float scale_i, float scale_j)
        : id(surface_id)
    {
        const int rows = static_cast<int>(zyx.shape(0));
        const int cols = static_cast<int>(zyx.shape(1));
        auto points = std::make_unique<cv::Mat_<cv::Vec3f>>(rows, cols);

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const float z = zyx(row, col, 0);
                const float y = zyx(row, col, 1);
                const float x = zyx(row, col, 2);
                (*points)(row, col) = cv::Vec3f(x, y, z);
            }
        }

        surface = std::make_shared<QuadSurface>(points.release(), cv::Vec2f(scale_j, scale_i));
        surface->id = id;
    }

    std::vector<float> ptr_to_grid(const std::array<float, 3>& ptr) const
    {
        const cv::Vec2f grid = surface->ptrToGrid(cv::Vec3f(ptr[0], ptr[1], ptr[2]));
        return {grid[0], grid[1]};
    }

    int valid_quad_count() const
    {
        return surface->countValidQuads();
    }
};

std::array<float, 3> vec3_from_object(const nb::handle& obj)
{
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    if (nb::len(seq) != 3) {
        throw std::runtime_error("expected a length-3 coordinate");
    }
    return {
        nb::cast<float>(seq[0]),
        nb::cast<float>(seq[1]),
        nb::cast<float>(seq[2]),
    };
}

struct PySurfacePatchIndex {
    SurfacePatchIndex index;
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    std::unordered_map<const QuadSurface*, std::string> ids_by_surface;
    std::unordered_map<const QuadSurface*, std::string> paths_by_surface;

    void rebuild(const nb::iterable& py_surfaces, float bbox_padding = 0.0f, int sampling_stride = 1)
    {
        surfaces.clear();
        ids_by_surface.clear();
        paths_by_surface.clear();

        if (sampling_stride < 1) {
            throw std::runtime_error("sampling_stride must be >= 1");
        }
        if (sampling_stride != 1) {
            index.setSamplingStride(sampling_stride);
        }

        for (nb::handle item : py_surfaces) {
            auto& py_surface = nb::cast<PyQuadSurface&>(item);
            surfaces.push_back(py_surface.surface);
            ids_by_surface[py_surface.surface.get()] = py_surface.id;
        }

        index.rebuild(surfaces, bbox_padding);
    }

    nb::object locate_xyz(const std::array<float, 3>& xyz, float tolerance) const
    {
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = cv::Vec3f(xyz[0], xyz[1], xyz[2]);
        query.tolerance = tolerance;

        auto hit = index.locate(query);
        if (!hit) {
            return nb::none();
        }

        const QuadSurface* raw = hit->surface.get();
        const cv::Vec2f grid = hit->surface->ptrToGrid(hit->ptr);

        nb::dict out;
        if (auto it = ids_by_surface.find(raw); it != ids_by_surface.end()) {
            out["id"] = it->second;
        } else {
            out["id"] = nb::none();
        }
        out["path"] = nb::none();
        out["distance"] = hit->distance;
        out["ptr"] = std::vector<float>{hit->ptr[0], hit->ptr[1], hit->ptr[2]};
        out["grid_xy"] = std::vector<float>{grid[0], grid[1]};
        out["ij"] = std::vector<float>{grid[1], grid[0]};
        return nb::object(out);
    }

    nb::list locate_xyz_batch(const nb::iterable& xyzs, float tolerance) const
    {
        nb::list out;
        for (nb::handle xyz : xyzs) {
            out.append(locate_xyz(vec3_from_object(xyz), tolerance));
        }
        return out;
    }

    bool empty() const { return index.empty(); }
    size_t surface_count() const { return index.surfaceCount(); }
    size_t patch_count() const { return index.patchCount(); }
};

}  // namespace

NB_MODULE(vc_surface_index, m) {
    m.doc() = "Bindings for QuadSurface and SurfacePatchIndex.";

    nb::class_<PyQuadSurface>(m, "QuadSurface")
        .def(nb::init<const std::string&, PyQuadSurface::ZyxArray, float, float>(),
             nb::arg("id"), nb::arg("zyx"), nb::arg("scale_i"), nb::arg("scale_j"))
        .def_ro("id", &PyQuadSurface::id)
        .def("ptr_to_grid", &PyQuadSurface::ptr_to_grid, nb::arg("ptr"))
        .def("valid_quad_count", &PyQuadSurface::valid_quad_count);

    nb::class_<PySurfacePatchIndex>(m, "SurfacePatchIndex")
        .def(nb::init<>())
        .def("rebuild", &PySurfacePatchIndex::rebuild,
             nb::arg("surfaces"), nb::arg("bbox_padding") = 0.0f, nb::arg("sampling_stride") = 1)
        .def("locate_xyz", &PySurfacePatchIndex::locate_xyz, nb::arg("xyz"), nb::arg("tolerance"))
        .def("locate_xyz_batch", &PySurfacePatchIndex::locate_xyz_batch, nb::arg("xyzs"), nb::arg("tolerance"))
        .def("empty", &PySurfacePatchIndex::empty)
        .def("surface_count", &PySurfacePatchIndex::surface_count)
        .def("patch_count", &PySurfacePatchIndex::patch_count);
}
