#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3& a, float scalar)
{
    return {a.x * scalar, a.y * scalar, a.z * scalar};
}

float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float distance_squared(const Vec3& a, const Vec3& b)
{
    const Vec3 d = a - b;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

bool finite(const Vec3& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

// QuadSurface uses x == -1 as the inexpensive validity sentinel. Tifxyz's
// invalid vertices are [-1, -1, -1], so retain the same test here.
bool valid_surface_point(const Vec3& point)
{
    return point.x != -1.0f && finite(point);
}

struct TriangleHit {
    Vec3 bary{};
    float distance_squared = std::numeric_limits<float>::max();
};

TriangleHit closest_point_on_triangle(
    const Vec3& point, const Vec3& a, const Vec3& b, const Vec3& c)
{
    TriangleHit hit;
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        hit.bary = {1.0f, 0.0f, 0.0f};
        hit.distance_squared = distance_squared(point, a);
        return hit;
    }

    const Vec3 bp = point - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        hit.bary = {0.0f, 1.0f, 0.0f};
        hit.distance_squared = distance_squared(point, b);
        return hit;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        hit.bary = {1.0f - v, v, 0.0f};
        hit.distance_squared = distance_squared(point, a + ab * v);
        return hit;
    }

    const Vec3 cp = point - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        hit.bary = {0.0f, 0.0f, 1.0f};
        hit.distance_squared = distance_squared(point, c);
        return hit;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        hit.bary = {1.0f - w, 0.0f, w};
        hit.distance_squared = distance_squared(point, a + ac * w);
        return hit;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.bary = {0.0f, 1.0f - w, w};
        hit.distance_squared = distance_squared(point, b + (c - b) * w);
        return hit;
    }

    const float inverse_denom = 1.0f / (va + vb + vc);
    const float v = vb * inverse_denom;
    const float w = vc * inverse_denom;
    hit.bary = {1.0f - v - w, v, w};
    hit.distance_squared = distance_squared(point, a + ab * v + ac * w);
    return hit;
}

float clamp01(float value)
{
    return std::max(0.0f, std::min(1.0f, value));
}

struct Aabb {
    Vec3 low{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    Vec3 high{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };

    void extend(const Vec3& point)
    {
        low.x = std::min(low.x, point.x);
        low.y = std::min(low.y, point.y);
        low.z = std::min(low.z, point.z);
        high.x = std::max(high.x, point.x);
        high.y = std::max(high.y, point.y);
        high.z = std::max(high.z, point.z);
    }

    void extend(const Aabb& other)
    {
        extend(other.low);
        extend(other.high);
    }

    void pad(float padding)
    {
        if (padding <= 0.0f) {
            return;
        }
        low.x -= padding;
        low.y -= padding;
        low.z -= padding;
        high.x += padding;
        high.y += padding;
        high.z += padding;
    }

    bool intersects(const Aabb& other) const
    {
        return low.x <= other.high.x && high.x >= other.low.x
            && low.y <= other.high.y && high.y >= other.low.y
            && low.z <= other.high.z && high.z >= other.low.z;
    }

    float centroid(int axis) const
    {
        if (axis == 0) {
            return (low.x + high.x) * 0.5f;
        }
        if (axis == 1) {
            return (low.y + high.y) * 0.5f;
        }
        return (low.z + high.z) * 0.5f;
    }
};

struct SurfaceData {
    std::string id;
    size_t rows = 0;
    size_t cols = 0;
    float scale_i = 1.0f;
    float scale_j = 1.0f;
    std::vector<Vec3> xyz;

    const Vec3& at(size_t row, size_t col) const
    {
        return xyz[row * cols + col];
    }
};

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

struct Tile {
    Aabb bounds;
    int32_t surface = -1;
    int row = 0;
    int col = 0;
};

struct BvhNode {
    Aabb bounds;
    uint32_t first = 0;
    uint32_t count = 0;
    uint32_t left = 0;
    uint32_t right = 0;
};

struct BestHit {
    bool valid = false;
    float distance_squared = std::numeric_limits<float>::infinity();
    float i = 0.0f;
    float j = 0.0f;
};

// Bool numpy array over a uint8 vector (nanobind maps bool ndarrays to bytes).
nb::ndarray<nb::numpy, bool, nb::ndim<1>> own_bool_1d(std::vector<uint8_t>&& values)
{
    auto* held = new std::vector<uint8_t>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<uint8_t>*>(pointer);
    });
    return nb::ndarray<nb::numpy, bool, nb::ndim<1>>(
        reinterpret_cast<bool*>(held->data()), {held->size()}, owner);
}

// tifxyz.Patch.valid_vertex_mask: torch.any(zyxs != -1, dim=-1).
bool tifxyz_vertex_valid(const Vec3& point)
{
    return point.x != -1.0f || point.y != -1.0f || point.z != -1.0f;
}

// One sample of tifxyz.Patch.ij_to_zyx (see PySurfacePatchIndex::interpolate_ij).
// Each arithmetic step mirrors one torch op on float32 tensors, so this
// function must not be compiled with floating-point contraction (a fused
// multiply-add rounds once where torch rounds twice). The rest of this file
// keeps the toolchain's default, so the projection queries are unchanged.
#if defined(__clang__)
#define VC_NO_FP_CONTRACT_ATTR
#define VC_NO_FP_CONTRACT_BODY _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define VC_NO_FP_CONTRACT_ATTR __attribute__((optimize("fp-contract=off")))
#define VC_NO_FP_CONTRACT_BODY
#else
// MSVC's default /fp:precise does not contract.
#define VC_NO_FP_CONTRACT_ATTR
#define VC_NO_FP_CONTRACT_BODY
#endif

VC_NO_FP_CONTRACT_ATTR
bool interpolate_one(
    const SurfaceData& data, float i, float j, bool triangulated, float out[3])
{
    VC_NO_FP_CONTRACT_BODY
    if (!std::isfinite(i) || !std::isfinite(j)) {
        return false;
    }
    const int64_t rows = static_cast<int64_t>(data.rows);
    const int64_t cols = static_cast<int64_t>(data.cols);
    int64_t i0 = static_cast<int64_t>(std::floor(i));
    int64_t j0 = static_cast<int64_t>(std::floor(j));
    if (triangulated) {
        // Projection may land exactly on the outer edge of the last quad.
        if (i == static_cast<float>(rows - 1)) {
            i0 = rows - 2;
        }
        if (j == static_cast<float>(cols - 1)) {
            j0 = cols - 2;
        }
    }
    if (!(i0 >= 0 && j0 >= 0 && i0 < rows - 1 && j0 < cols - 1)) {
        return false;
    }
    const Vec3& tl = data.at(static_cast<size_t>(i0), static_cast<size_t>(j0));
    const Vec3& tr = data.at(static_cast<size_t>(i0), static_cast<size_t>(j0 + 1));
    const Vec3& bl = data.at(static_cast<size_t>(i0 + 1), static_cast<size_t>(j0));
    const Vec3& br = data.at(static_cast<size_t>(i0 + 1), static_cast<size_t>(j0 + 1));
    if (!(tifxyz_vertex_valid(tl) && tifxyz_vertex_valid(tr)
          && tifxyz_vertex_valid(bl) && tifxyz_vertex_valid(br))) {
        return false;
    }
    const float di = i - static_cast<float>(i0);
    const float dj = j - static_cast<float>(j0);
    // SurfaceData stores x, y, z; emit z, y, x like the patch grid.
    const float c_tl[3] = {tl.z, tl.y, tl.x};
    const float c_tr[3] = {tr.z, tr.y, tr.x};
    const float c_bl[3] = {bl.z, bl.y, bl.x};
    const float c_br[3] = {br.z, br.y, br.x};
    if (triangulated) {
        const bool lower = (di + dj) <= 1.0f;
        const float one_minus_di = 1.0f - di;
        const float one_minus_dj = 1.0f - dj;
        for (int c = 0; c < 3; ++c) {
            if (lower) {
                // tl + di * (bl - tl) + dj * (tr - tl)
                const float t1 = c_bl[c] - c_tl[c];
                const float t2 = di * t1;
                const float t3 = c_tl[c] + t2;
                const float t4 = c_tr[c] - c_tl[c];
                const float t5 = dj * t4;
                out[c] = t3 + t5;
            } else {
                // br + (1 - di) * (tr - br) + (1 - dj) * (bl - br)
                const float t1 = c_tr[c] - c_br[c];
                const float t2 = one_minus_di * t1;
                const float t3 = c_br[c] + t2;
                const float t4 = c_bl[c] - c_br[c];
                const float t5 = one_minus_dj * t4;
                out[c] = t3 + t5;
            }
        }
    } else {
        for (int c = 0; c < 3; ++c) {
            // top = tl + (tr - tl) * dj; bottom = bl + (br - bl) * dj;
            // top + (bottom - top) * di
            const float top = c_tl[c] + (c_tr[c] - c_tl[c]) * dj;
            const float bottom = c_bl[c] + (c_br[c] - c_bl[c]) * dj;
            out[c] = top + (bottom - top) * di;
        }
    }
    return true;
}

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
        rebuild_from_surfaces(std::move(new_surfaces), bbox_padding, sampling_stride);
    }

    using GridArray = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, -1, 3>, nb::c_contig>;
    using ScaleBatch = nb::ndarray<
        nb::numpy, const float, nb::shape<-1>, nb::c_contig>;

    // rebuild() over (id, zyx grid, scale_i, scale_j) columns: the same index
    // as building one QuadSurface per grid, but the geometry copies run in
    // parallel and no per-surface Python objects are created.
    void rebuild_grids(
        const std::vector<std::string>& ids,
        const nb::sequence& grids,
        ScaleBatch scales_i,
        ScaleBatch scales_j,
        float bbox_padding = 0.0f,
        int sampling_stride = 1)
    {
        const size_t count = ids.size();
        if (static_cast<size_t>(nb::len(grids)) != count
            || scales_i.shape(0) != count || scales_j.shape(0) != count) {
            throw std::runtime_error("ids, grids and scales must have the same length");
        }
        std::vector<GridArray> arrays;
        arrays.reserve(count);
        for (nb::handle item : grids) {
            arrays.push_back(nb::cast<GridArray>(item));
        }
        std::vector<std::shared_ptr<SurfaceData>> new_surfaces(count);
        for (size_t index = 0; index < count; ++index) {
            auto data = std::make_shared<SurfaceData>();
            data->id = ids[index];
            data->rows = arrays[index].shape(0);
            data->cols = arrays[index].shape(1);
            data->scale_i = scales_i(index);
            data->scale_j = scales_j(index);
            new_surfaces[index] = std::move(data);
        }
        {
            nb::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
            for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count); ++index) {
                SurfaceData& data = *new_surfaces[static_cast<size_t>(index)];
                const float* zyx = arrays[static_cast<size_t>(index)].data();
                data.xyz.resize(data.rows * data.cols);
                for (size_t vertex = 0; vertex < data.rows * data.cols; ++vertex) {
                    data.xyz[vertex] = {zyx[3 * vertex + 2], zyx[3 * vertex + 1], zyx[3 * vertex]};
                }
            }
        }
        rebuild_from_surfaces(std::move(new_surfaces), bbox_padding, sampling_stride);
    }

    std::vector<std::string> surface_ids() const
    {
        std::vector<std::string> ids;
        ids.reserve(surfaces_.size());
        for (const auto& surface : surfaces_) {
            ids.push_back(surface->id);
        }
        return ids;
    }

    auto sample_xyz(int32_t surface_index, int stride) const
    {
        if (surface_index < 0 || static_cast<size_t>(surface_index) >= surfaces_.size()) {
            throw std::runtime_error("surface_index out of range");
        }
        if (stride < 1) {
            throw std::runtime_error("stride must be >= 1");
        }
        const SurfaceData& surface = *surfaces_[static_cast<size_t>(surface_index)];
        std::vector<float> xyz;
        for (size_t row = 0; row < surface.rows; row += static_cast<size_t>(stride)) {
            for (size_t col = 0; col < surface.cols; col += static_cast<size_t>(stride)) {
                const Vec3& point = surface.at(row, col);
                // Match patch_query_points: discard only the [-1, -1, -1] sentinel.
                if (point.x == -1.0f && point.y == -1.0f && point.z == -1.0f) {
                    continue;
                }
                xyz.insert(xyz.end(), {point.x, point.y, point.z});
            }
        }
        return own_2d(std::move(xyz), 3);
    }

    auto nearby_surfaces(int32_t surface_index) const
    {
        if (surface_index < 0 || static_cast<size_t>(surface_index) >= surfaces_.size()) {
            throw std::runtime_error("surface_index out of range");
        }
        std::vector<int32_t> result;
        {
            nb::gil_scoped_release release;
            std::vector<uint8_t> seen(surfaces_.size(), 0);
            std::vector<uint32_t> stack;
            for (size_t source_tile = surface_tile_offsets_[static_cast<size_t>(surface_index)];
                 source_tile < surface_tile_offsets_[static_cast<size_t>(surface_index) + 1];
                 ++source_tile) {
                const Aabb& query = tiles_[source_tile].bounds;
                if (nodes_.empty()) break;
                stack.clear();
                stack.push_back(0);
                while (!stack.empty()) {
                    const BvhNode& node = nodes_[stack.back()];
                    stack.pop_back();
                    if (!node.bounds.intersects(query)) continue;
                    if (node.count != 0) {
                        for (uint32_t offset = 0; offset < node.count; ++offset) {
                            const Tile& tile = tiles_[tile_order_[node.first + offset]];
                            const size_t other = static_cast<size_t>(tile.surface);
                            if (!seen[other] && tile.bounds.intersects(query)) {
                                seen[other] = 1;
                                result.push_back(tile.surface);
                            }
                        }
                    } else {
                        stack.push_back(node.right);
                        stack.push_back(node.left);
                    }
                }
            }
            std::sort(result.begin(), result.end());
        }
        return own_1d(std::move(result));
    }

    nb::object locate_all_xyz_batch(XyzBatch xyzs, float tolerance) const
    {
        return query_batch(xyzs, nullptr, tolerance);
    }

    using SurfaceBatch = nb::ndarray<
        nb::numpy, const int32_t, nb::shape<-1>, nb::c_contig>;
    using IjBatch = nb::ndarray<
        nb::numpy, const float, nb::shape<-1, 2>, nb::c_contig>;

    // tifxyz.Patch.ij_to_zyx over many (surface, ij) pairs at once, with the
    // out-of-range nudge point_collection._projection_feet applies on top.
    //
    // Reproduces the torch implementation's float32 arithmetic operation for
    // operation (interpolate_one is compiled without floating-point
    // contraction), so a foot computed here is bit-identical to one computed
    // by patch.ij_to_zyx(ij, triangulated=...) on the same patch: bilinear
    // interpolation of the quad's four corners, or the two-triangle
    // interpolation of interpolate_quad_triangles when `triangulated`. A hit
    // whose ij lands outside the grid or on an invalid quad is retried with ij
    // clamped into the grid's last quad (the "nudge"); returns (zyx, valid)
    // with zyx == -1 where still invalid.
    nb::object interpolate_ij(
        SurfaceBatch surface_indices, IjBatch ijs, bool triangulated) const
    {
        const size_t count = surface_indices.shape(0);
        if (ijs.shape(0) != count) {
            throw std::runtime_error("surface_indices and ijs must have the same length");
        }
        std::vector<float> zyx(count * 3, -1.0f);
        std::vector<uint8_t> valid(count, 0);
        const int32_t* surfaces = surface_indices.data();
        const float* ij = ijs.data();
        {
            nb::gil_scoped_release release;
            for (size_t k = 0; k < count; ++k) {
                const int32_t surface = surfaces[k];
                if (surface < 0 || static_cast<size_t>(surface) >= surfaces_.size()) {
                    continue;
                }
                const SurfaceData& data = *surfaces_[static_cast<size_t>(surface)];
                float out[3];
                if (interpolate_one(data, ij[2 * k], ij[2 * k + 1], triangulated, out)) {
                    valid[k] = 1;
                } else {
                    // Nudge: clamp into [0, extent - 1 - 1e-3] (float32 of the
                    // python double, as torch's clamp does) and retry.
                    const float max_i = static_cast<float>(
                        static_cast<double>(data.rows) - 1.0 - 1e-3);
                    const float max_j = static_cast<float>(
                        static_cast<double>(data.cols) - 1.0 - 1e-3);
                    const float ni = std::min(std::max(ij[2 * k], 0.0f), max_i);
                    const float nj = std::min(std::max(ij[2 * k + 1], 0.0f), max_j);
                    if (interpolate_one(data, ni, nj, triangulated, out)) {
                        valid[k] = 1;
                    } else {
                        continue;
                    }
                }
                zyx[3 * k] = out[0];
                zyx[3 * k + 1] = out[1];
                zyx[3 * k + 2] = out[2];
            }
        }
        return nb::make_tuple(own_2d(std::move(zyx), 3), own_bool_1d(std::move(valid)));
    }

    nb::object locate_all_xyz_batch_in(
        XyzBatch xyzs, Subset subset, float tolerance) const
    {
        std::vector<uint8_t> included(surfaces_.size(), 0);
        for (size_t index = 0; index < subset.shape(0); ++index) {
            const int32_t surface = subset(index);
            if (surface >= 0 && static_cast<size_t>(surface) < included.size()) {
                included[static_cast<size_t>(surface)] = 1;
            }
        }
        return query_batch(xyzs, &included, tolerance);
    }

private:
    static constexpr size_t leaf_size = 8;

    static int compute_tile_stride(int sampling_stride)
    {
        if (sampling_stride >= 8) {
            return sampling_stride;
        }
        return ((8 + sampling_stride - 1) / sampling_stride) * sampling_stride;
    }

    void rebuild_from_surfaces(
        std::vector<std::shared_ptr<SurfaceData>> new_surfaces,
        float bbox_padding,
        int sampling_stride)
    {
        if (sampling_stride < 1) {
            throw std::runtime_error("sampling_stride must be >= 1");
        }
        const int new_tile_stride = compute_tile_stride(sampling_stride);

        // Tiles per surface, computed independently (parallel) and concatenated
        // in surface order, so the tile list is the one a serial pass produces.
        std::vector<std::vector<Tile>> per_surface(new_surfaces.size());
        {
            nb::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (std::ptrdiff_t surface_index = 0;
                 surface_index < static_cast<std::ptrdiff_t>(new_surfaces.size());
                 ++surface_index) {
                per_surface[static_cast<size_t>(surface_index)] = surface_tiles(
                    *new_surfaces[static_cast<size_t>(surface_index)],
                    static_cast<int32_t>(surface_index), new_tile_stride, bbox_padding);
            }
        }
        std::vector<Tile> new_tiles;
        std::vector<size_t> new_surface_tile_offsets;
        new_surface_tile_offsets.reserve(per_surface.size() + 1);
        new_surface_tile_offsets.push_back(0);
        size_t total = 0;
        for (const auto& tiles : per_surface) {
            total += tiles.size();
        }
        new_tiles.reserve(total);
        for (auto& tiles : per_surface) {
            new_tiles.insert(new_tiles.end(), tiles.begin(), tiles.end());
            new_surface_tile_offsets.push_back(new_tiles.size());
        }
        per_surface.clear();

        std::vector<uint32_t> new_order(new_tiles.size());
        std::iota(new_order.begin(), new_order.end(), uint32_t{0});
        std::vector<BvhNode> new_nodes;
        if (!new_tiles.empty()) {
            nb::gil_scoped_release release;
            build_bvh_levels(new_tiles, new_order, new_nodes);
        }

        surfaces_ = std::move(new_surfaces);
        surface_tile_offsets_ = std::move(new_surface_tile_offsets);
        tiles_ = std::move(new_tiles);
        tile_order_ = std::move(new_order);
        nodes_ = std::move(new_nodes);
        sampling_stride_ = sampling_stride;
        tile_stride_ = new_tile_stride;
    }

    static std::vector<Tile> surface_tiles(
        const SurfaceData& surface, int32_t surface_index, int tile_stride, float bbox_padding)
    {
        std::vector<Tile> tiles;
        if (surface.rows < 2 || surface.cols < 2) {
            return tiles;
        }
        for (size_t row = 0; row + 1 < surface.rows; row += tile_stride) {
            for (size_t col = 0; col + 1 < surface.cols; col += tile_stride) {
                const size_t row_end = std::min(
                    surface.rows - 1, row + static_cast<size_t>(tile_stride));
                const size_t col_end = std::min(
                    surface.cols - 1, col + static_cast<size_t>(tile_stride));
                Aabb bounds;
                bool any_valid = false;
                for (size_t r = row; r <= row_end; ++r) {
                    for (size_t c = col; c <= col_end; ++c) {
                        const Vec3& point = surface.at(r, c);
                        if (valid_surface_point(point)) {
                            bounds.extend(point);
                            any_valid = true;
                        }
                    }
                }
                if (any_valid) {
                    bounds.pad(bbox_padding);
                    tiles.push_back({
                        bounds,
                        surface_index,
                        static_cast<int>(row),
                        static_cast<int>(col),
                    });
                }
            }
        }
        return tiles;
    }

    // Median-split BVH built one depth level at a time: every node of a level
    // is bounded, split-sorted and (if not a leaf) split in parallel, then its
    // children are allocated serially in node order. The tree -- each node's
    // tile range, split axis and stable centroid order -- is exactly the one the
    // depth-first recursion built; only node numbering differs, and queries
    // gather then sort candidate tiles, so their results do not depend on it.
    static void build_bvh_levels(
        const std::vector<Tile>& tiles,
        std::vector<uint32_t>& order,
        std::vector<BvhNode>& nodes)
    {
        struct Work {
            uint32_t node;
            size_t first;
            size_t last;
        };
        nodes.reserve(tiles.size() * 2);
        nodes.emplace_back();
        std::vector<Work> level{{0, 0, order.size()}};
        std::vector<uint8_t> is_leaf;
        while (!level.empty()) {
            is_leaf.assign(level.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::ptrdiff_t w = 0; w < static_cast<std::ptrdiff_t>(level.size()); ++w) {
                const Work& work = level[static_cast<size_t>(w)];
                is_leaf[static_cast<size_t>(w)] = split_node(
                    tiles, order, nodes[work.node], work.first, work.last) ? 0 : 1;
            }
            std::vector<Work> next;
            for (size_t w = 0; w < level.size(); ++w) {
                if (is_leaf[w]) {
                    continue;
                }
                const Work& work = level[w];
                const size_t middle = work.first + (work.last - work.first) / 2;
                const uint32_t left = static_cast<uint32_t>(nodes.size());
                nodes.emplace_back();
                const uint32_t right = static_cast<uint32_t>(nodes.size());
                nodes.emplace_back();
                nodes[work.node].left = left;
                nodes[work.node].right = right;
                next.push_back({left, work.first, middle});
                next.push_back({right, middle, work.last});
            }
            level = std::move(next);
        }
    }

    // Bounds `node`; for an interior node also stable-sorts its tile range by
    // split-axis centroid (the recursion's exact key) and returns true.
    static bool split_node(
        const std::vector<Tile>& tiles,
        std::vector<uint32_t>& order,
        BvhNode& node,
        size_t first,
        size_t last)
    {
        Aabb bounds;
        for (size_t index = first; index < last; ++index) {
            bounds.extend(tiles[order[index]].bounds);
        }
        node.bounds = bounds;

        const size_t count = last - first;
        if (count <= leaf_size) {
            node.first = static_cast<uint32_t>(first);
            node.count = static_cast<uint32_t>(count);
            return false;
        }

        Aabb centroids;
        for (size_t index = first; index < last; ++index) {
            const Aabb& tile_bounds = tiles[order[index]].bounds;
            centroids.extend(Vec3{
                tile_bounds.centroid(0),
                tile_bounds.centroid(1),
                tile_bounds.centroid(2),
            });
        }
        const Vec3 extent = centroids.high - centroids.low;
        int axis = 0;
        if (extent.y > extent.x) {
            axis = 1;
        }
        const float selected_extent = axis == 0 ? extent.x : extent.y;
        if (extent.z > selected_extent) {
            axis = 2;
        }

        std::stable_sort(
            order.begin() + static_cast<std::ptrdiff_t>(first),
            order.begin() + static_cast<std::ptrdiff_t>(last),
            [&](uint32_t a, uint32_t b) {
                const float ca = tiles[a].bounds.centroid(axis);
                const float cb = tiles[b].bounds.centroid(axis);
                return ca < cb || (ca == cb && a < b);
            });
        return true;
    }

    void find_candidate_tiles(const Vec3& point, float tolerance, std::vector<uint32_t>& out) const
    {
        if (nodes_.empty()) {
            return;
        }
        Aabb query;
        query.low = {point.x - tolerance, point.y - tolerance, point.z - tolerance};
        query.high = {point.x + tolerance, point.y + tolerance, point.z + tolerance};

        std::vector<uint32_t> stack{0};
        while (!stack.empty()) {
            const uint32_t node_index = stack.back();
            stack.pop_back();
            const BvhNode& node = nodes_[node_index];
            if (!node.bounds.intersects(query)) {
                continue;
            }
            if (node.count != 0) {
                for (uint32_t offset = 0; offset < node.count; ++offset) {
                    const uint32_t tile_index = tile_order_[node.first + offset];
                    if (tiles_[tile_index].bounds.intersects(query)) {
                        out.push_back(tile_index);
                    }
                }
            } else {
                stack.push_back(node.right);
                stack.push_back(node.left);
            }
        }
        std::sort(out.begin(), out.end());
    }

    BestHit evaluate_tile(const Tile& tile, const Vec3& point) const
    {
        BestHit best;
        const SurfaceData& surface = *surfaces_[static_cast<size_t>(tile.surface)];
        const int row_limit = std::min(
            tile.row + tile_stride_, static_cast<int>(surface.rows) - 1);
        const int col_limit = std::min(
            tile.col + tile_stride_, static_cast<int>(surface.cols) - 1);

        for (int row = tile.row; row < row_limit; row += sampling_stride_) {
            for (int col = tile.col; col < col_limit; col += sampling_stride_) {
                const int row_step = std::min(
                    sampling_stride_, static_cast<int>(surface.rows) - 1 - row);
                const int col_step = std::min(
                    sampling_stride_, static_cast<int>(surface.cols) - 1 - col);
                if (row_step <= 0 || col_step <= 0) {
                    continue;
                }

                const Vec3& p00 = surface.at(row, col);
                const Vec3& p10 = surface.at(row, col + col_step);
                const Vec3& p01 = surface.at(row + row_step, col);
                const Vec3& p11 = surface.at(row + row_step, col + col_step);
                if (!valid_surface_point(p00) || !valid_surface_point(p10)
                    || !valid_surface_point(p01) || !valid_surface_point(p11)) {
                    continue;
                }

                auto record = [&](float u, float v, float distance_sq) {
                    if (distance_sq >= best.distance_squared) {
                        return;
                    }
                    best.valid = true;
                    best.distance_squared = distance_sq;
                    // Match SurfacePatchIndex: the parameter span is the requested
                    // stride, including its clamped final boundary cell.
                    best.i = static_cast<float>(col)
                        + u * static_cast<float>(sampling_stride_);
                    best.j = static_cast<float>(row)
                        + v * static_cast<float>(sampling_stride_);
                };

                const TriangleHit first = closest_point_on_triangle(point, p00, p10, p01);
                record(clamp01(first.bary.y), clamp01(first.bary.z), first.distance_squared);

                const TriangleHit second = closest_point_on_triangle(point, p10, p11, p01);
                record(
                    clamp01(second.bary.x + second.bary.y),
                    clamp01(second.bary.y + second.bary.z),
                    second.distance_squared);
            }
        }
        return best;
    }

    struct QueryResult {
        std::vector<int64_t> offsets;
        std::vector<int32_t> surfaces;
        std::vector<float> distances;
        std::vector<float> ij;
    };

    QueryResult query_points(
        const float* coordinates, size_t count,
        const std::vector<uint8_t>* included,
        float tolerance) const
    {
        std::vector<int64_t> offsets(count + 1, 0);
        std::vector<int32_t> surface_indices;
        std::vector<float> distances;
        std::vector<float> ij;

        {
            if (tolerance >= 0.0f) {
                const float tolerance_squared = tolerance * tolerance;
                for (size_t point_index = 0; point_index < count; ++point_index) {
                    const Vec3 point{
                        coordinates[3 * point_index],
                        coordinates[3 * point_index + 1],
                        coordinates[3 * point_index + 2],
                    };
                    if (!finite(point)) {
                        offsets[point_index + 1] = static_cast<int64_t>(surface_indices.size());
                        continue;
                    }

                    std::vector<uint32_t> candidates;
                    find_candidate_tiles(point, tolerance, candidates);
                    // Keep per-point storage proportional to BVH candidates while
                    // retaining the API's ascending surface-index order.
                    std::vector<int32_t> candidate_surfaces;
                    candidate_surfaces.reserve(candidates.size());
                    for (const uint32_t tile_index : candidates) {
                        const int32_t surface_index = tiles_[tile_index].surface;
                        if (included == nullptr
                            || (*included)[static_cast<size_t>(surface_index)]) {
                            candidate_surfaces.push_back(surface_index);
                        }
                    }
                    std::sort(candidate_surfaces.begin(), candidate_surfaces.end());
                    candidate_surfaces.erase(
                        std::unique(candidate_surfaces.begin(), candidate_surfaces.end()),
                        candidate_surfaces.end());

                    std::vector<BestHit> best_hits(candidate_surfaces.size());
                    for (const uint32_t tile_index : candidates) {
                        const Tile& tile = tiles_[tile_index];
                        const size_t surface_index = static_cast<size_t>(tile.surface);
                        if (included != nullptr && !(*included)[surface_index]) {
                            continue;
                        }
                        const BestHit tile_hit = evaluate_tile(tile, point);
                        const auto surface = std::lower_bound(
                            candidate_surfaces.begin(), candidate_surfaces.end(), tile.surface);
                        BestHit& best = best_hits[static_cast<size_t>(
                            surface - candidate_surfaces.begin())];
                        if (tile_hit.valid && tile_hit.distance_squared < best.distance_squared) {
                            best = tile_hit;
                        }
                    }

                    for (size_t candidate_index = 0;
                         candidate_index < candidate_surfaces.size();
                         ++candidate_index) {
                        const BestHit& hit = best_hits[candidate_index];
                        if (!hit.valid || hit.distance_squared > tolerance_squared) {
                            continue;
                        }
                        surface_indices.push_back(candidate_surfaces[candidate_index]);
                        distances.push_back(std::sqrt(hit.distance_squared));
                        ij.push_back(hit.j);
                        ij.push_back(hit.i);
                    }
                    offsets[point_index + 1] = static_cast<int64_t>(surface_indices.size());
                }
            }
        }

        return {std::move(offsets), std::move(surface_indices), std::move(distances), std::move(ij)};
    }

    nb::object query_batch(
        XyzBatch xyzs,
        const std::vector<uint8_t>* included,
        float tolerance) const
    {
        const size_t count = xyzs.shape(0);
        QueryResult result;
        {
            nb::gil_scoped_release release;
            // Each point executes the identical serial tile traversal. Work is
            // partitioned into contiguous batches and concatenated in input
            // order, preserving nearest-hit ties and every output bit.
            constexpr size_t chunk_size = 1024;
            const size_t chunks = (count + chunk_size - 1) / chunk_size;
            if (chunks <= 1) {
                result = query_points(xyzs.data(), count, included, tolerance);
            } else {
                std::vector<QueryResult> pieces(chunks);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int64_t k = 0; k < static_cast<int64_t>(chunks); ++k) {
                    const size_t start = static_cast<size_t>(k) * chunk_size;
                    pieces[static_cast<size_t>(k)] = query_points(
                        xyzs.data() + 3 * start, std::min(chunk_size, count - start), included, tolerance);
                }
                size_t hits = 0;
                for (const auto& piece : pieces) hits += piece.surfaces.size();
                result.offsets.resize(count + 1, 0);
                result.surfaces.reserve(hits); result.distances.reserve(hits); result.ij.reserve(2 * hits);
                for (size_t k = 0; k < chunks; ++k) {
                    const auto& piece = pieces[k];
                    const int64_t base = static_cast<int64_t>(result.surfaces.size());
                    for (size_t i = 1; i < piece.offsets.size(); ++i)
                        result.offsets[k * chunk_size + i] = base + piece.offsets[i];
                    result.surfaces.insert(result.surfaces.end(), piece.surfaces.begin(), piece.surfaces.end());
                    result.distances.insert(result.distances.end(), piece.distances.begin(), piece.distances.end());
                    result.ij.insert(result.ij.end(), piece.ij.begin(), piece.ij.end());
                }
            }
        }
        return nb::make_tuple(
            own_1d(std::move(result.offsets)),
            own_1d(std::move(result.surfaces)),
            own_1d(std::move(result.distances)),
            own_2d(std::move(result.ij), 2));
    }

    std::vector<std::shared_ptr<SurfaceData>> surfaces_;
    std::vector<size_t> surface_tile_offsets_;
    std::vector<Tile> tiles_;
    std::vector<uint32_t> tile_order_;
    std::vector<BvhNode> nodes_;
    int sampling_stride_ = 1;
    int tile_stride_ = 8;
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
        .def(
            "rebuild_grids",
            &PySurfacePatchIndex::rebuild_grids,
            nb::arg("ids"),
            nb::arg("grids"),
            nb::arg("scales_i"),
            nb::arg("scales_j"),
            nb::arg("bbox_padding") = 0.0f,
            nb::arg("sampling_stride") = 1)
        .def("surface_ids", &PySurfacePatchIndex::surface_ids)
        .def("sample_xyz", &PySurfacePatchIndex::sample_xyz,
             nb::arg("surface_index"), nb::arg("stride") = 1)
        .def("nearby_surfaces", &PySurfacePatchIndex::nearby_surfaces,
             nb::arg("surface_index"))
        .def(
            "locate_all_xyz_batch",
            &PySurfacePatchIndex::locate_all_xyz_batch,
            nb::arg("xyzs"),
            nb::arg("tolerance"))
        .def(
            "interpolate_ij",
            &PySurfacePatchIndex::interpolate_ij,
            nb::arg("surface_indices"),
            nb::arg("ijs"),
            nb::arg("triangulated") = false)
        .def(
            "locate_all_xyz_batch_in",
            &PySurfacePatchIndex::locate_all_xyz_batch_in,
            nb::arg("xyzs"),
            nb::arg("subset"),
            nb::arg("tolerance"));
}
