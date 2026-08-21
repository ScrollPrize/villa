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
        if (sampling_stride < 1) {
            throw std::runtime_error("sampling_stride must be >= 1");
        }

        std::vector<std::shared_ptr<SurfaceData>> new_surfaces;
        for (nb::handle item : py_surfaces) {
            new_surfaces.push_back(nb::cast<PyQuadSurface&>(item).data);
        }

        const int new_tile_stride = compute_tile_stride(sampling_stride);
        std::vector<Tile> new_tiles;
        for (size_t surface_index = 0; surface_index < new_surfaces.size(); ++surface_index) {
            const SurfaceData& surface = *new_surfaces[surface_index];
            if (surface.rows < 2 || surface.cols < 2) {
                continue;
            }
            for (size_t row = 0; row + 1 < surface.rows; row += new_tile_stride) {
                for (size_t col = 0; col + 1 < surface.cols; col += new_tile_stride) {
                    const size_t row_end = std::min(
                        surface.rows - 1, row + static_cast<size_t>(new_tile_stride));
                    const size_t col_end = std::min(
                        surface.cols - 1, col + static_cast<size_t>(new_tile_stride));
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
                        new_tiles.push_back({
                            bounds,
                            static_cast<int32_t>(surface_index),
                            static_cast<int>(row),
                            static_cast<int>(col),
                        });
                    }
                }
            }
        }

        std::vector<uint32_t> new_order(new_tiles.size());
        std::iota(new_order.begin(), new_order.end(), uint32_t{0});
        std::vector<BvhNode> new_nodes;
        if (!new_tiles.empty()) {
            new_nodes.reserve(new_tiles.size() * 2);
            build_bvh(new_tiles, new_order, new_nodes, 0, new_order.size());
        }

        surfaces_ = std::move(new_surfaces);
        tiles_ = std::move(new_tiles);
        tile_order_ = std::move(new_order);
        nodes_ = std::move(new_nodes);
        sampling_stride_ = sampling_stride;
        tile_stride_ = new_tile_stride;
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

    nb::object locate_all_xyz_batch(XyzBatch xyzs, float tolerance) const
    {
        return query_batch(xyzs, nullptr, tolerance);
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

    static uint32_t build_bvh(
        const std::vector<Tile>& tiles,
        std::vector<uint32_t>& order,
        std::vector<BvhNode>& nodes,
        size_t first,
        size_t last)
    {
        const uint32_t node_index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();

        Aabb bounds;
        for (size_t index = first; index < last; ++index) {
            bounds.extend(tiles[order[index]].bounds);
        }
        nodes[node_index].bounds = bounds;

        const size_t count = last - first;
        if (count <= leaf_size) {
            nodes[node_index].first = static_cast<uint32_t>(first);
            nodes[node_index].count = static_cast<uint32_t>(count);
            return node_index;
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
        const size_t middle = first + count / 2;
        const uint32_t left = build_bvh(tiles, order, nodes, first, middle);
        const uint32_t right = build_bvh(tiles, order, nodes, middle, last);
        nodes[node_index].left = left;
        nodes[node_index].right = right;
        return node_index;
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
            if (tolerance > 0.0f) {
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
                    std::vector<BestHit> best_by_surface(surfaces_.size());
                    for (const uint32_t tile_index : candidates) {
                        const Tile& tile = tiles_[tile_index];
                        const size_t surface_index = static_cast<size_t>(tile.surface);
                        if (included != nullptr && !(*included)[surface_index]) {
                            continue;
                        }
                        const BestHit tile_hit = evaluate_tile(tile, point);
                        BestHit& best = best_by_surface[surface_index];
                        if (tile_hit.valid && tile_hit.distance_squared < best.distance_squared) {
                            best = tile_hit;
                        }
                    }

                    for (size_t surface_index = 0; surface_index < best_by_surface.size(); ++surface_index) {
                        const BestHit& hit = best_by_surface[surface_index];
                        if (!hit.valid || hit.distance_squared > tolerance_squared) {
                            continue;
                        }
                        surface_indices.push_back(static_cast<int32_t>(surface_index));
                        distances.push_back(std::sqrt(hit.distance_squared));
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

    std::vector<std::shared_ptr<SurfaceData>> surfaces_;
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
