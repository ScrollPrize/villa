#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <spiral_graph/surface_index.hpp>

namespace nb = nanobind;

namespace {

using surfcore::SurfaceData;
using surfcore::SurfaceHit;
using surfcore::Vec3;

struct PyQuadSurface {
    using ZyxArray = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, -1, 3>, nb::c_contig>;

    std::shared_ptr<SurfaceData> data;

    PyQuadSurface(
        const std::string& id, ZyxArray zyx, float scale_i, float scale_j)
        : data(std::make_shared<SurfaceData>())
    {
        data->id = id;
        data->rows = zyx.shape(0);
        data->cols = zyx.shape(1);
        data->scale_i = scale_i;
        data->scale_j = scale_j;
        data->xyz.resize(data->rows * data->cols);

        for (size_t row = 0; row < data->rows; ++row) {
            for (size_t col = 0; col < data->cols; ++col) {
                data->xyz[row * data->cols + col] = {
                    zyx(row, col, 2),
                    zyx(row, col, 1),
                    zyx(row, col, 0),
                };
            }
        }
    }
};

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d(std::vector<T>&& values)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        held->data(), {held->size()}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d(
    std::vector<T>&& values, size_t columns)
{
    auto* held = new std::vector<T>(std::move(values));
    const size_t rows = columns == 0 ? 0 : held->size() / columns;
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        held->data(), {rows, columns}, owner);
}

class PySurfacePatchIndex {
public:
    using XyzBatch = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig>;
    using Subset = nb::ndarray<
        nb::numpy, const int32_t, nb::shape<-1>, nb::c_contig>;

    void rebuild(
        const nb::iterable& py_surfaces,
        float bbox_padding = 0.0f,
        int sampling_stride = 1)
    {
        std::vector<std::shared_ptr<SurfaceData>> new_surfaces;
        for (nb::handle item : py_surfaces) {
            new_surfaces.push_back(nb::cast<PyQuadSurface&>(item).data);
        }
        index_.rebuild(std::move(new_surfaces), bbox_padding, sampling_stride);
    }

    std::vector<std::string> surface_ids() const
    {
        std::vector<std::string> ids;
        ids.reserve(index_.surfaces().size());
        for (const auto& surface : index_.surfaces()) {
            ids.push_back(surface->id);
        }
        return ids;
    }

    nb::object locate_all_xyz_batch(XyzBatch xyzs, float tolerance) const
    {
        return query_batch(xyzs, nullptr, tolerance);
    }

    nb::object locate_all_xyz_batch_in(
        XyzBatch xyzs, Subset subset, float tolerance) const
    {
        std::vector<uint8_t> included(index_.surfaces().size(), 0);
        for (size_t index = 0; index < subset.shape(0); ++index) {
            const int32_t surface = subset(index);
            if (surface >= 0 && static_cast<size_t>(surface) < included.size()) {
                included[static_cast<size_t>(surface)] = 1;
            }
        }
        return query_batch(xyzs, &included, tolerance);
    }

private:
    nb::object query_batch(
        XyzBatch xyzs,
        const std::vector<uint8_t>* included,
        float tolerance) const
    {
        const size_t count = xyzs.shape(0);
        const float* coordinates = xyzs.data();
        std::vector<int64_t> offsets(count + 1, 0);
        std::vector<int32_t> surface_indices;
        std::vector<float> distances;
        std::vector<float> ij;

        {
            nb::gil_scoped_release release;
            if (tolerance >= 0.0f) {
                std::vector<SurfaceHit> hits;
                surfcore::QueryScratch scratch;
                for (size_t point_index = 0; point_index < count; ++point_index) {
                    const Vec3 point{
                        coordinates[3 * point_index],
                        coordinates[3 * point_index + 1],
                        coordinates[3 * point_index + 2],
                    };
                    hits.clear();
                    index_.query_point(point, tolerance, hits, scratch, included);
                    for (const SurfaceHit& hit : hits) {
                        surface_indices.push_back(hit.surface);
                        distances.push_back(hit.distance);
                        ij.push_back(hit.j);
                        ij.push_back(hit.i);
                    }
                    offsets[point_index + 1] = static_cast<int64_t>(surface_indices.size());
                }
            }
        }

        return nb::make_tuple(
            own_1d(std::move(offsets)),
            own_1d(std::move(surface_indices)),
            own_1d(std::move(distances)),
            own_2d(std::move(ij), 2));
    }

    surfcore::SurfacePatchIndex index_;
};

}  // namespace

NB_MODULE(surface_index, module)
{
    module.doc() = "Dependency-free surface index for Spiral fitting.";

    nb::class_<PyQuadSurface>(module, "QuadSurface")
        .def(
            nb::init<const std::string&, PyQuadSurface::ZyxArray, float, float>(),
            nb::arg("id"),
            nb::arg("zyx"),
            nb::arg("scale_i"),
            nb::arg("scale_j"));

    nb::class_<PySurfacePatchIndex>(module, "SurfacePatchIndex")
        .def(nb::init<>())
        .def(
            "rebuild",
            &PySurfacePatchIndex::rebuild,
            nb::arg("surfaces"),
            nb::arg("bbox_padding") = 0.0f,
            nb::arg("sampling_stride") = 1)
        .def("surface_ids", &PySurfacePatchIndex::surface_ids)
        .def(
            "locate_all_xyz_batch",
            &PySurfacePatchIndex::locate_all_xyz_batch,
            nb::arg("xyzs"),
            nb::arg("tolerance"))
        .def(
            "locate_all_xyz_batch_in",
            &PySurfacePatchIndex::locate_all_xyz_batch_in,
            nb::arg("xyzs"),
            nb::arg("subset"),
            nb::arg("tolerance"));
}
