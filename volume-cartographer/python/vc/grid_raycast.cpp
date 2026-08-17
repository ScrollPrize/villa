// Ray casting against structured tifxyz-style quad grids.
//
// A segment surface is a [H, W, 3] grid of XYZ vertices where each valid
// quad (r, c) spans vertices (r..r+1, c..c+1) and is split into the same two
// triangles the winding dataset historically fed to trimesh:
//   triangle 0: (r, c), (r+1, c), (r, c+1)
//   triangle 1: (r, c+1), (r+1, c), (r+1, c+1)
// An implicit AABB quadtree over fixed-size leaf blocks of quads makes a
// full multi-hit ray query run in microseconds even on ~60M-quad grids,
// where a generic BVH build alone would take minutes.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr int kLeafBlock = 8;

struct Box {
    float lo[3];
    float hi[3];
};

constexpr Box emptyBox()
{
    return Box{
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()}};
}

struct Level {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<Box> boxes;

    Box& at(int64_t i, int64_t j) { return boxes[i * cols + j]; }
    const Box& at(int64_t i, int64_t j) const { return boxes[i * cols + j]; }
};

struct Hit {
    double t;
    double point[3];
    int32_t row;
    int32_t col;
    int32_t triangle;
};

template <typename T, size_t N>
nb::ndarray<T, nb::numpy, nb::c_contig> makeArray(std::vector<T>&& data,
                                                  std::array<size_t, N> shape)
{
    auto* heap = new std::vector<T>(std::move(data));
    nb::capsule owner(heap, [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return nb::ndarray<T, nb::numpy, nb::c_contig>(heap->data(), N, shape.data(), owner);
}

class GridRaycaster {
public:
    using XyzArray =
        nb::ndarray<const float, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu>;
    using QuadMask =
        nb::ndarray<const bool, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

    GridRaycaster(XyzArray xyz, QuadMask validQuads)
        : xyz_(std::move(xyz)), validQuads_(std::move(validQuads))
    {
        rows_ = static_cast<int64_t>(xyz_.shape(0));
        cols_ = static_cast<int64_t>(xyz_.shape(1));
        quadRows_ = rows_ - 1;
        quadCols_ = cols_ - 1;
        if (validQuads_.shape(0) != static_cast<size_t>(std::max<int64_t>(quadRows_, 0)) ||
            validQuads_.shape(1) != static_cast<size_t>(std::max<int64_t>(quadCols_, 0)))
            throw nb::value_error("valid_quads must have shape [H-1, W-1]");

        nb::gil_scoped_release release;
        build();
    }

    bool empty() const { return quadCount_ == 0; }
    int64_t valid_quad_count() const { return quadCount_; }

    nb::tuple hits(std::array<double, 3> origin,
                   std::array<double, 3> direction,
                   double tMin,
                   double tMax) const
    {
        std::vector<Hit> found;
        {
            nb::gil_scoped_release release;
            found = query(origin, direction, tMin, tMax);
        }

        const size_t count = found.size();
        std::vector<double> ts(count);
        std::vector<double> points(count * 3);
        std::vector<int32_t> quadRows(count);
        std::vector<int32_t> quadCols(count);
        std::vector<int32_t> triangles(count);
        for (size_t i = 0; i < count; ++i) {
            ts[i] = found[i].t;
            points[3 * i] = found[i].point[0];
            points[3 * i + 1] = found[i].point[1];
            points[3 * i + 2] = found[i].point[2];
            quadRows[i] = found[i].row;
            quadCols[i] = found[i].col;
            triangles[i] = found[i].triangle;
        }
        return nb::make_tuple(
            makeArray<double, 1>(std::move(ts), {count}),
            makeArray<double, 2>(std::move(points), {count, 3}),
            makeArray<int32_t, 1>(std::move(quadRows), {count}),
            makeArray<int32_t, 1>(std::move(quadCols), {count}),
            makeArray<int32_t, 1>(std::move(triangles), {count}));
    }

private:
    const float* vertex(int64_t r, int64_t c) const
    {
        return xyz_.data() + 3 * (r * cols_ + c);
    }

    bool quadValid(int64_t r, int64_t c) const
    {
        return validQuads_.data()[r * quadCols_ + c];
    }

    void build()
    {
        quadCount_ = 0;
        if (quadRows_ <= 0 || quadCols_ <= 0)
            return;

        Level leaves;
        leaves.rows = (quadRows_ + kLeafBlock - 1) / kLeafBlock;
        leaves.cols = (quadCols_ + kLeafBlock - 1) / kLeafBlock;
        leaves.boxes.assign(static_cast<size_t>(leaves.rows * leaves.cols), emptyBox());

        int64_t quadCount = 0;
#pragma omp parallel for reduction(+ : quadCount) schedule(static)
        for (int64_t bi = 0; bi < leaves.rows; ++bi) {
            for (int64_t bj = 0; bj < leaves.cols; ++bj) {
                Box box = emptyBox();
                const int64_t rEnd = std::min(quadRows_, (bi + 1) * kLeafBlock);
                const int64_t cEnd = std::min(quadCols_, (bj + 1) * kLeafBlock);
                for (int64_t r = bi * kLeafBlock; r < rEnd; ++r) {
                    for (int64_t c = bj * kLeafBlock; c < cEnd; ++c) {
                        if (!quadValid(r, c))
                            continue;
                        bool finite = true;
                        float lo[3];
                        float hi[3];
                        std::copy_n(box.lo, 3, lo);
                        std::copy_n(box.hi, 3, hi);
                        for (int dr = 0; dr < 2 && finite; ++dr) {
                            for (int dc = 0; dc < 2 && finite; ++dc) {
                                const float* v = vertex(r + dr, c + dc);
                                for (int axis = 0; axis < 3; ++axis) {
                                    if (!std::isfinite(v[axis])) {
                                        finite = false;
                                        break;
                                    }
                                    lo[axis] = std::min(lo[axis], v[axis]);
                                    hi[axis] = std::max(hi[axis], v[axis]);
                                }
                            }
                        }
                        if (!finite)
                            continue;
                        std::copy_n(lo, 3, box.lo);
                        std::copy_n(hi, 3, box.hi);
                        ++quadCount;
                    }
                }
                leaves.at(bi, bj) = box;
            }
        }
        quadCount_ = quadCount;

        levels_.clear();
        levels_.push_back(std::move(leaves));
        while (levels_.back().rows > 1 || levels_.back().cols > 1) {
            const Level& child = levels_.back();
            Level parent;
            parent.rows = (child.rows + 1) / 2;
            parent.cols = (child.cols + 1) / 2;
            parent.boxes.assign(static_cast<size_t>(parent.rows * parent.cols),
                                emptyBox());
            for (int64_t i = 0; i < child.rows; ++i) {
                for (int64_t j = 0; j < child.cols; ++j) {
                    const Box& box = child.at(i, j);
                    Box& target = parent.at(i / 2, j / 2);
                    for (int axis = 0; axis < 3; ++axis) {
                        target.lo[axis] = std::min(target.lo[axis], box.lo[axis]);
                        target.hi[axis] = std::max(target.hi[axis], box.hi[axis]);
                    }
                }
            }
            levels_.push_back(std::move(parent));
        }
    }

    static bool intersectsBox(const Box& box,
                              const std::array<double, 3>& origin,
                              const std::array<double, 3>& invDir,
                              double tMin,
                              double tMax)
    {
        double enter = tMin;
        double exit = tMax;
        for (int axis = 0; axis < 3; ++axis) {
            const double near = (box.lo[axis] - origin[axis]) * invDir[axis];
            const double far = (box.hi[axis] - origin[axis]) * invDir[axis];
            // NaN (0 * inf when the origin sits on a slab of a zero-width
            // box) must not prune: std::min/max keep the first argument on
            // NaN, so order the arguments to preserve enter/exit instead.
            enter = std::max(enter, std::min(near, far));
            exit = std::min(exit, std::max(near, far));
        }
        return enter <= exit;
    }

    void intersectTriangle(const float* a,
                           const float* b,
                           const float* c,
                           const std::array<double, 3>& origin,
                           const std::array<double, 3>& direction,
                           double tMin,
                           double tMax,
                           int64_t row,
                           int64_t col,
                           int triangle,
                           std::vector<Hit>& out) const
    {
        const double e1[3] = {double(b[0]) - a[0], double(b[1]) - a[1],
                              double(b[2]) - a[2]};
        const double e2[3] = {double(c[0]) - a[0], double(c[1]) - a[1],
                              double(c[2]) - a[2]};
        const double p[3] = {direction[1] * e2[2] - direction[2] * e2[1],
                             direction[2] * e2[0] - direction[0] * e2[2],
                             direction[0] * e2[1] - direction[1] * e2[0]};
        const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (std::abs(det) < 1e-14)
            return;
        const double invDet = 1.0 / det;
        const double s[3] = {origin[0] - a[0], origin[1] - a[1], origin[2] - a[2]};
        const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * invDet;
        // Slack keeps shared-edge hits from slipping between adjacent
        // triangles; duplicates merge downstream within hit tolerance.
        constexpr double kBaryEps = 1e-9;
        if (u < -kBaryEps || u > 1.0 + kBaryEps)
            return;
        const double q[3] = {s[1] * e1[2] - s[2] * e1[1],
                             s[2] * e1[0] - s[0] * e1[2],
                             s[0] * e1[1] - s[1] * e1[0]};
        const double v =
            (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * invDet;
        if (v < -kBaryEps || u + v > 1.0 + kBaryEps)
            return;
        const double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * invDet;
        if (t < tMin || t > tMax)
            return;
        out.push_back(Hit{
            t,
            {origin[0] + t * direction[0], origin[1] + t * direction[1],
             origin[2] + t * direction[2]},
            static_cast<int32_t>(row),
            static_cast<int32_t>(col),
            triangle});
    }

    void intersectLeaf(int64_t bi,
                       int64_t bj,
                       const std::array<double, 3>& origin,
                       const std::array<double, 3>& direction,
                       double tMin,
                       double tMax,
                       std::vector<Hit>& out) const
    {
        const int64_t rEnd = std::min(quadRows_, (bi + 1) * kLeafBlock);
        const int64_t cEnd = std::min(quadCols_, (bj + 1) * kLeafBlock);
        for (int64_t r = bi * kLeafBlock; r < rEnd; ++r) {
            for (int64_t c = bj * kLeafBlock; c < cEnd; ++c) {
                if (!quadValid(r, c))
                    continue;
                const float* v00 = vertex(r, c);
                const float* v10 = vertex(r + 1, c);
                const float* v01 = vertex(r, c + 1);
                const float* v11 = vertex(r + 1, c + 1);
                intersectTriangle(v00, v10, v01, origin, direction, tMin, tMax,
                                  r, c, 0, out);
                intersectTriangle(v01, v10, v11, origin, direction, tMin, tMax,
                                  r, c, 1, out);
            }
        }
    }

    std::vector<Hit> query(const std::array<double, 3>& origin,
                           const std::array<double, 3>& direction,
                           double tMin,
                           double tMax) const
    {
        std::vector<Hit> out;
        if (levels_.empty() || quadCount_ == 0)
            return out;

        std::array<double, 3> invDir;
        for (int axis = 0; axis < 3; ++axis)
            invDir[axis] = 1.0 / direction[axis];

        struct Node {
            int level;
            int64_t i;
            int64_t j;
        };
        std::vector<Node> stack;
        stack.push_back(Node{static_cast<int>(levels_.size()) - 1, 0, 0});
        while (!stack.empty()) {
            const Node node = stack.back();
            stack.pop_back();
            const Level& level = levels_[node.level];
            if (!intersectsBox(level.at(node.i, node.j), origin, invDir, tMin, tMax))
                continue;
            if (node.level == 0) {
                intersectLeaf(node.i, node.j, origin, direction, tMin, tMax, out);
                continue;
            }
            const Level& child = levels_[node.level - 1];
            for (int64_t di = 0; di < 2; ++di) {
                for (int64_t dj = 0; dj < 2; ++dj) {
                    const int64_t ci = node.i * 2 + di;
                    const int64_t cj = node.j * 2 + dj;
                    if (ci < child.rows && cj < child.cols)
                        stack.push_back(Node{node.level - 1, ci, cj});
                }
            }
        }

        std::sort(out.begin(), out.end(),
                  [](const Hit& lhs, const Hit& rhs) { return lhs.t < rhs.t; });
        return out;
    }

    XyzArray xyz_;
    QuadMask validQuads_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t quadRows_ = 0;
    int64_t quadCols_ = 0;
    int64_t quadCount_ = 0;
    std::vector<Level> levels_;
};

} // namespace

NB_MODULE(grid_raycast, m)
{
    m.doc() = "Multi-hit ray casting against structured quad-grid surfaces";

    nb::class_<GridRaycaster>(m, "GridRaycaster")
        .def(nb::init<GridRaycaster::XyzArray, GridRaycaster::QuadMask>(),
             "xyz"_a,
             "valid_quads"_a,
             "Build an AABB hierarchy over a [H, W, 3] float32 vertex grid; "
             "valid_quads is a [H-1, W-1] bool mask of usable quads. The "
             "arrays are referenced, not copied, and must stay unchanged.")
        .def("hits", &GridRaycaster::hits,
             "origin"_a,
             "direction"_a,
             "t_min"_a = 0.0,
             "t_max"_a = std::numeric_limits<double>::infinity(),
             "Return (ts, points_xyz, quad_rows, quad_cols, triangles) for "
             "every ray/surface intersection with t in [t_min, t_max], "
             "sorted by ascending t. Triangle 0 covers vertices "
             "(r,c),(r+1,c),(r,c+1); triangle 1 covers (r,c+1),(r+1,c),(r+1,c+1).")
        .def_prop_ro("empty", &GridRaycaster::empty)
        .def_prop_ro("valid_quad_count", &GridRaycaster::valid_quad_count);
}
