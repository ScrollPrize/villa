#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

// Hand ownership of a std::vector to a 1-D numpy array (zero-copy): the array
// keeps the vector alive via a capsule and frees it when garbage-collected.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d(std::vector<T>&& vec)
{
    auto* held = new std::vector<T>(std::move(vec));
    nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(held->data(), {held->size()}, owner);
}

// As own_1d, but views the buffer as a 2-D (rows, cols) numpy array.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d(std::vector<T>&& vec, size_t cols)
{
    auto* held = new std::vector<T>(std::move(vec));
    const size_t rows = cols ? held->size() / cols : 0;
    nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(held->data(), {rows, cols}, owner);
}

struct PySurfacePatchIndex {
    using XyzBatch = nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>, nb::c_contig>;

    SurfacePatchIndex index;
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    std::unordered_map<const QuadSurface*, std::string> ids_by_surface;
    std::unordered_map<const QuadSurface*, int32_t> idx_by_surface;

    void rebuild(const nb::iterable& py_surfaces, float bbox_padding = 0.0f, int sampling_stride = 1)
    {
        surfaces.clear();
        ids_by_surface.clear();
        idx_by_surface.clear();

        if (sampling_stride < 1) {
            throw std::runtime_error("sampling_stride must be >= 1");
        }
        index.setSamplingStride(sampling_stride);

        for (nb::handle item : py_surfaces) {
            auto& py_surface = nb::cast<PyQuadSurface&>(item);
            idx_by_surface[py_surface.surface.get()] = static_cast<int32_t>(surfaces.size());
            surfaces.push_back(py_surface.surface);
            ids_by_surface[py_surface.surface.get()] = py_surface.id;
        }

        index.rebuild(surfaces, bbox_padding);
    }

    // Surface ids in index order; surface_ids()[i] is the id whose integer index
    // is i in the arrays returned by locate_all_xyz_batch.
    std::vector<std::string> surface_ids() const
    {
        std::vector<std::string> out;
        out.reserve(surfaces.size());
        for (const auto& s : surfaces) {
            auto it = ids_by_surface.find(s.get());
            out.push_back(it != ids_by_surface.end() ? it->second : std::string());
        }
        return out;
    }

    nb::dict result_to_dict(const SurfacePatchIndex::LookupResult& hit) const
    {
        const QuadSurface* raw = hit.surface.get();
        const cv::Vec2f grid = hit.surface->ptrToGrid(hit.ptr);

        nb::dict out;
        if (auto it = ids_by_surface.find(raw); it != ids_by_surface.end()) {
            out["id"] = it->second;
        } else {
            out["id"] = nb::none();
        }
        out["path"] = nb::none();
        out["distance"] = hit.distance;
        out["ptr"] = std::vector<float>{hit.ptr[0], hit.ptr[1], hit.ptr[2]};
        out["grid_xy"] = std::vector<float>{grid[0], grid[1]};
        out["ij"] = std::vector<float>{grid[1], grid[0]};
        return out;
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
        return nb::object(result_to_dict(*hit));
    }

    nb::list locate_xyz_batch(const nb::iterable& xyzs, float tolerance) const
    {
        nb::list out;
        for (nb::handle xyz : xyzs) {
            out.append(locate_xyz(vec3_from_object(xyz), tolerance));
        }
        return out;
    }

    // Unlike locate_xyz (single nearest surface), returns every surface within
    // `tolerance` of the point, with the closest point on each. Each list
    // element is a dict shaped exactly like locate_xyz's result.
    nb::list locate_all_xyz(const std::array<float, 3>& xyz, float tolerance) const
    {
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = cv::Vec3f(xyz[0], xyz[1], xyz[2]);
        query.tolerance = tolerance;

        nb::list out;
        for (const auto& hit : index.locateAll(query)) {
            out.append(result_to_dict(hit));
        }
        return out;
    }

    // Batched locate_all over an (N, 3) xyz array, returned as parallel numpy
    // arrays rather than nested lists of dicts. The hits are ragged, so they are
    // packed CSR-style: hits for point k occupy [offsets[k], offsets[k+1]).
    // Returns (offsets[int64, N+1], surf_idx[int32, M], distance[float32, M],
    // ij[float32, M, 2]) where surf_idx indexes surface_ids() (-1 if unknown)
    // and ij is (grid_y, grid_x) per hit. The query runs with the GIL released
    // -- it touches only C++ state -- so callers can parallelise across threads.
    nb::object locate_all_xyz_batch(XyzBatch xyzs, float tolerance) const
    {
        const size_t n = xyzs.shape(0);
        const float* xyz = xyzs.data();  // the ndarray arg keeps this alive across the release

        std::vector<int64_t> offsets(n + 1, 0);
        std::vector<int32_t> surf_idx;
        std::vector<float> distance;
        std::vector<float> ij;  // flattened (M, 2)

        {
            nb::gil_scoped_release release;
            for (size_t k = 0; k < n; ++k) {
                SurfacePatchIndex::PointQuery query;
                query.worldPoint = cv::Vec3f(xyz[3 * k + 0], xyz[3 * k + 1], xyz[3 * k + 2]);
                query.tolerance = tolerance;
                for (const auto& hit : index.locateAll(query)) {
                    auto it = idx_by_surface.find(hit.surface.get());
                    surf_idx.push_back(it != idx_by_surface.end() ? it->second : -1);
                    distance.push_back(hit.distance);
                    const cv::Vec2f grid = hit.surface->ptrToGrid(hit.ptr);
                    ij.push_back(grid[1]);
                    ij.push_back(grid[0]);
                }
                offsets[k + 1] = static_cast<int64_t>(surf_idx.size());
            }
        }

        return nb::make_tuple(own_1d(std::move(offsets)), own_1d(std::move(surf_idx)),
                              own_1d(std::move(distance)), own_2d(std::move(ij), 2));
    }

    // Single-nearest batched locate over an (N, 3) xyz array, parallelised across
    // threads with the GIL released. Returns compact parallel arrays (not the ragged
    // CSR of locate_all_xyz_batch):
    //   (surf_idx[int32, N], distance[float32, N], ij[float32, N, 2])
    // surf_idx[k] indexes surface_ids() (-1 if no surface within tolerance),
    // distance[k] is +inf when there is no hit, and ij is (grid_y, grid_x). Uses the
    // single-nearest locate() (cheaper than locateAll, which collects every hit);
    // locate() takes only a shared read lock, so disjoint-index workers are safe.
    nb::object locate_xyz_nearest_batch(XyzBatch xyzs, float tolerance) const
    {
        const size_t n = xyzs.shape(0);
        const float* xyz = xyzs.data();  // the ndarray arg keeps this alive across the release

        std::vector<int32_t> surf_idx(n, -1);
        std::vector<float> distance(n, std::numeric_limits<float>::infinity());
        std::vector<float> ij(2 * n, 0.0f);

        {
            nb::gil_scoped_release release;
            unsigned hw = std::thread::hardware_concurrency();
            if (hw == 0) { hw = 1; }
            const size_t nthreads = std::min<size_t>(hw, std::max<size_t>(size_t{1}, n));

            auto worker = [&](size_t lo, size_t hi) {
                for (size_t k = lo; k < hi; ++k) {
                    SurfacePatchIndex::PointQuery query;
                    query.worldPoint = cv::Vec3f(xyz[3 * k + 0], xyz[3 * k + 1], xyz[3 * k + 2]);
                    query.tolerance = tolerance;
                    auto hit = index.locate(query);
                    if (!hit) { continue; }
                    auto it = idx_by_surface.find(hit->surface.get());
                    surf_idx[k] = (it != idx_by_surface.end()) ? it->second : -1;
                    distance[k] = hit->distance;
                    const cv::Vec2f grid = hit->surface->ptrToGrid(hit->ptr);
                    ij[2 * k + 0] = grid[1];
                    ij[2 * k + 1] = grid[0];
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(nthreads);
            const size_t chunk = (n + nthreads - 1) / nthreads;
            for (size_t t = 0; t < nthreads; ++t) {
                const size_t lo = t * chunk;
                const size_t hi = std::min(n, lo + chunk);
                if (lo >= hi) { break; }
                threads.emplace_back(worker, lo, hi);
            }
            for (auto& th : threads) { th.join(); }
        }

        return nb::make_tuple(own_1d(std::move(surf_idx)), own_1d(std::move(distance)),
                              own_2d(std::move(ij), 2));
    }

    // Like locate_all_xyz_batch, but evaluate only the surfaces whose indices
    // into surface_ids() are present in subset. This avoids projecting against
    // unrelated overlapping surfaces for between-patch point collections.
    nb::object locate_all_xyz_batch_in(
        XyzBatch xyzs,
        nb::ndarray<nb::numpy, const int32_t, nb::shape<-1>, nb::c_contig> subset,
        float tolerance) const
    {
        const size_t n = xyzs.shape(0);
        const float* xyz = xyzs.data();

        std::unordered_set<SurfacePatchIndex::SurfacePtr> include;
        const size_t num_subset = subset.shape(0);
        const int32_t* sub = subset.data();
        include.reserve(num_subset);
        for (size_t t = 0; t < num_subset; ++t) {
            const int32_t surface_index = sub[t];
            if (surface_index >= 0 && static_cast<size_t>(surface_index) < surfaces.size()) {
                include.insert(surfaces[surface_index]);
            }
        }

        std::vector<int64_t> offsets(n + 1, 0);
        std::vector<int32_t> surf_idx;
        std::vector<float> distance;
        std::vector<float> ij;

        {
            nb::gil_scoped_release release;
            for (size_t k = 0; k < n; ++k) {
                SurfacePatchIndex::PointQuery query;
                query.worldPoint = cv::Vec3f(xyz[3 * k + 0], xyz[3 * k + 1], xyz[3 * k + 2]);
                query.tolerance = tolerance;
                query.surfaces.include = &include;
                for (const auto& hit : index.locateAll(query)) {
                    auto it = idx_by_surface.find(hit.surface.get());
                    surf_idx.push_back(it != idx_by_surface.end() ? it->second : -1);
                    distance.push_back(hit.distance);
                    const cv::Vec2f grid = hit.surface->ptrToGrid(hit.ptr);
                    ij.push_back(grid[1]);
                    ij.push_back(grid[0]);
                }
                offsets[k + 1] = static_cast<int64_t>(surf_idx.size());
            }
        }

        return nb::make_tuple(own_1d(std::move(offsets)), own_1d(std::move(surf_idx)),
                              own_1d(std::move(distance)), own_2d(std::move(ij), 2));
    }

    bool empty() const { return index.empty(); }
    size_t surface_count() const { return index.surfaceCount(); }
    size_t patch_count() const { return index.patchCount(); }
};

}  // namespace

NB_MODULE(surface_index, m) {
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
        .def("locate_all_xyz", &PySurfacePatchIndex::locate_all_xyz, nb::arg("xyz"), nb::arg("tolerance"))
        .def("locate_all_xyz_batch", &PySurfacePatchIndex::locate_all_xyz_batch, nb::arg("xyzs"), nb::arg("tolerance"))
        .def("locate_xyz_nearest_batch", &PySurfacePatchIndex::locate_xyz_nearest_batch, nb::arg("xyzs"), nb::arg("tolerance"))
        .def("locate_all_xyz_batch_in", &PySurfacePatchIndex::locate_all_xyz_batch_in,
             nb::arg("xyzs"), nb::arg("subset"), nb::arg("tolerance"))
        .def("surface_ids", &PySurfacePatchIndex::surface_ids)
        .def("empty", &PySurfacePatchIndex::empty)
        .def("surface_count", &PySurfacePatchIndex::surface_count)
        .def("patch_count", &PySurfacePatchIndex::patch_count);
}
