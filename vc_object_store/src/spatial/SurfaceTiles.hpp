#pragma once
#include "geometry/GeometryReader.hpp"
#include "spatial/TriangleDistance.hpp"
#include "vc_object_store/objects/GeometryRegion.hpp"
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

namespace vc::objects::detail {
// A surface's spatial tiles: 32 cells a side, row-major, each sharing its last
// row and column of vertices with its neighbours. Stored surfaces index them in
// spatial_regions; drafts keep them in memory. One layout, so both answer alike.
inline constexpr int64_t surfaceTileCells = 32;
struct SurfaceTileRect { int64_t row = 0, column = 0, rows = 0, columns = 0; };
std::vector<SurfaceTileRect> surfaceTiles(int64_t rows, int64_t columns);

// Where a surface's triangle mesh crosses a slab's mid-plane: cells whose four
// vertices are valid split along VC's diagonal into (r,c),(r,c+1),(r+1,c) and
// (r,c+1),(r+1,c+1),(r+1,c). Crossings are computed once per grid edge, so
// neighbouring triangles share exact points and chain into polylines across
// tiles. Add the tiles in row/column order, then take the pieces.
class SurfaceCut {
public:
    using Point = std::array<double, 3>;
    SurfaceCut(const RegionSlice& view, const std::optional<VolumeAffine>& toTarget, int64_t columns);
    // Returns the triangles tested.
    uint64_t add(const SurfaceTile&, int64_t row, int64_t column, int64_t rows, int64_t columns);
    // The chained polylines clipped to the slice's u/v extent, one segment at a
    // time in run order: `piece(segment, continues, start, end)`, where
    // `continues` means the piece extends the previous run (only `end` is new)
    // and otherwise starts a run at `start`.
    template <class Piece> void pieces(Piece&& piece) const;

private:
    using GridEdge = std::pair<int64_t, int64_t>;
    struct Slot { GridEdge edge; Point point{}; std::array<size_t, 2> segments{none, none}; };
    static constexpr size_t none = SIZE_MAX;
    std::pair<size_t, bool> insert(const GridEdge&);
    void grow();
    std::vector<std::deque<size_t>> chains() const;
    bool clip(const Point& a, const Point& b, double& first, double& last) const;

    RegionSlice view_;
    std::optional<VolumeAffine> toTarget_;
    int64_t columns_;
    double planeOffset_;
    std::vector<size_t> table_;
    std::vector<Slot> slots_;
    std::vector<std::pair<size_t, size_t>> segments_;
    std::vector<Point> world_;
    std::vector<double> distance_;
};

template <class Piece> void SurfaceCut::pieces(Piece&& piece) const
{
    for (const auto& chain : chains()) {
        bool open = false;
        for (size_t i = 0; i + 1 < chain.size(); ++i) {
            const auto& a = slots_[chain[i]].point;
            const auto& b = slots_[chain[i + 1]].point;
            double first, last;
            if (!clip(a, b, first, last)) { open = false; continue; }
            auto at = [&](double t) {
                if (t <= 0) return a;
                if (t >= 1) return b;
                // a and b are crossings of target-space vertices.
                Point p; for (int k = 0; k < 3; ++k) p[k] = a[k] + t * (b[k] - a[k]);
                return p;
            };
            piece(int64_t(i), open && first <= 0, at(first), at(last));
            open = last >= 1;
        }
    }
}

// The nearest point of a surface's valid triangles to a point, over its tiles.
class SurfaceNearest {
public:
    struct Hit { std::array<double, 2> grid{0, 0}; std::array<double, 3> point{0, 0, 0}; double distance = 0; };
    // `point` and `tolerance` are in target space; `toTarget` maps the surface there.
    SurfaceNearest(const std::array<double, 3>& point, double tolerance, const std::optional<VolumeAffine>& toTarget);
    void add(const SurfaceTile&, int64_t row, int64_t column, int64_t rows, int64_t columns);
    std::optional<Hit> take();
    // The tolerance box around the point, in target space.
    const std::array<double, 6>& box() const { return box_; }

private:
    struct Vec3 {
        std::array<double, 3> v;
        double operator[](size_t i) const { return v[i]; }
        double dot(const Vec3& o) const { return v[0] * o.v[0] + v[1] * o.v[1] + v[2] * o.v[2]; }
        friend Vec3 operator-(const Vec3& a, const Vec3& b) { return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}}; }
        friend Vec3 operator+(const Vec3& a, const Vec3& b) { return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}}; }
        friend Vec3 operator*(double s, const Vec3& a) { return {{s * a[0], s * a[1], s * a[2]}}; }
        friend Vec3 operator*(const Vec3& a, double s) { return s * a; }
    };
    Vec3 point_;
    std::array<double, 6> box_;
    double toleranceSq_;
    std::optional<VolumeAffine> toTarget_;
    std::optional<Hit> best_;
    double bestSq_ = INFINITY;
    std::vector<Vec3> world_;
};
}
