#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Read-only reader for an Automated Sheet Volume (.asv): the sheet-mesh sibling of
// vc::fibers::FiberCollection (.afv). One SQLite file holds, per sheet instance, a
// grid-implicit quantised quad mesh and a bounding box, indexed by an RTree. The
// viewer loads only the sheets crossing the visible slice and draws their
// intersection with the slice plane as polylines, so a file with millions of sheets
// renders slice by slice. Coordinates are native L0 XYZ doubles. See
// hercunet/meshes/asv.py for the writer and the on-disk format (format version 1).
namespace vc::sheets
{
using Point = std::array<double, 3>;

// 'R' recto, 'V' verso, empty = unclassified. Mirrors the fibre H/V family filter;
// our sheets are currently written unclassified, so None keeps unclassified sheets.
enum class FamilyFilter { All, Recto, Verso, None };

// Optional per-sheet quality thresholds, applied on the .asv's (nullable) quality
// columns. A field at its disabled default imposes no constraint, and a sheet whose
// column is NULL -- or a file that predates the column entirely -- always passes, so
// a filter never empties an older .asv. See hercunet/meshes/asv.py for the columns.
struct QualityFilter {
    double minConfidence{0.0};  // confidence >= this  (mean own-membership, 0..1)
    double minCoverage{0.0};    // coverage   >= this  (mask voxels within a grid step, 0..1)
    double minMassFrac{0.0};    // mass_frac  >= this  (self_mass / window max, 0..1)
    double maxFitRmsUm{0.0};    // fit_rms_um <= this  (mesh-vs-mask RMS, um); <= 0 disables
    bool active() const
    {
        return minConfidence > 0.0 || minCoverage > 0.0 || minMassFrac > 0.0 || maxFitRmsUm > 0.0;
    }
};

struct Bounds {
    Point lower{}, upper{};
    bool operator==(const Bounds&) const = default;
};

struct Summary {
    int64_t id{};
    std::string name, window, family;
    int64_t nu{}, nv{}, vertCount{};
    Bounds bounds;
};

// One contiguous curve of a sheet's intersection with the slice, in native L0 coords.
struct Polyline {
    std::vector<Point> points;
};

// A sheet crossing the slice: the set of polylines where its quad mesh meets the
// slice plane within the slab, kept together so the overlay can colour them per sheet.
struct SheetRuns {
    int64_t sheetId{};
    std::string family;
    std::vector<Polyline> curves;
};

struct RegionPage {
    std::vector<SheetRuns> sheets;
    bool complete{true};        // false if byteBudget stopped the query early
    size_t decodedBytes{};
};

// An oriented slice: the plane through `origin` with normal `n` and in-plane axes
// (u, v). `box` is the axis-aligned bounding box of the slab in world (native L0)
// coordinates, used for the RTree candidate query. `local` is the slab expressed in
// the (u, v, n) frame relative to `origin`: lower/upper[0,1] bound the visible
// extent in u, v; lower/upper[2] bound the half-thickness along n.
struct ViewRegion {
    Point origin{}, u{1, 0, 0}, v{0, 1, 0}, n{0, 0, 1};
    Bounds box, local;
    bool operator==(const ViewRegion&) const = default;
    // Project a world point into the (u, v, n) frame relative to origin.
    Point project(const Point& p) const;
};

// Full geometry of one sheet: footprint-only vertices, triangles indexing them, and
// the nu*nv footprint mask (row-major, 1 byte per cell). Triangles are reconstructed
// exactly from the footprint, matching hercunet.meshes.fit.grid_triangles.
struct Mesh {
    int64_t nu{}, nv{};
    std::vector<Point> verts;
    std::vector<std::array<int64_t, 3>> tris;
    std::vector<uint8_t> foot;
};

// A decoded sheet with its id, family and bounding box. region() returns these in
// bulk so a caller can cache the expensive decode once per region and intersect the
// meshes with a moving slice cheaply, per frame.
struct SheetMesh {
    int64_t id{};
    std::string family;
    Bounds bounds;
    Mesh mesh;
    // Per-sheet quality values from the .asv (see hercunet/meshes/asv.py), for
    // colour-by-metric. NaN when the column is NULL or the file predates it.
    double confidence{std::numeric_limits<double>::quiet_NaN()};
    double coverage{std::numeric_limits<double>::quiet_NaN()};
    double massFrac{std::numeric_limits<double>::quiet_NaN()};
    double fitRmsUm{std::numeric_limits<double>::quiet_NaN()};
};

class SheetCollection
{
public:
    explicit SheetCollection(const std::filesystem::path& path);
    ~SheetCollection();
    SheetCollection(const SheetCollection&) = delete;
    SheetCollection& operator=(const SheetCollection&) = delete;

    std::string metadata(const std::string& key) const;
    int64_t sheetCount() const;
    Summary summary(int64_t sheetId) const;

    // Descending vert_count (a size proxy), then ascending id. Start with
    // beforeVerts = INT64_MAX / afterId = 0 and continue from the last returned row.
    // minVerts is inclusive.
    std::vector<Summary> catalogBySize(
        int64_t beforeVerts, int64_t afterId = 0, int limit = 200,
        int64_t minVerts = 0, FamilyFilter family = FamilyFilter::All) const;

    // Sheets whose quad mesh crosses the oriented slice, each as the polyline curves
    // of the intersection (native L0 coords). maximum = 0 returns every crossing
    // sheet; otherwise at most `maximum` sheets by stable per-sheet priority, with
    // `selected` first when it crosses. complete = false reports that byteBudget
    // stopped the query.
    RegionPage viewRegion(
        const ViewRegion& view, size_t maximum = 0, int64_t selected = 0,
        size_t byteBudget = 8 * 1024 * 1024, const std::atomic_bool* cancelled = nullptr,
        int64_t minVerts = 0, FamilyFilter family = FamilyFilter::All,
        const QualityFilter& quality = {}) const;

    // Decoded meshes for the sheets whose bounding box intersects `box`, largest
    // first, up to `maxSheets` (0 = all) and `byteBudget` of decoded payload. This is
    // the expensive step (SQLite + decode); callers cache the result over a region
    // and intersect the meshes with the slice themselves. Safe to call from a worker
    // thread on its own SheetCollection instance.
    std::vector<SheetMesh> region(
        const Bounds& box, size_t maxSheets = 0, int64_t minVerts = 0,
        FamilyFilter family = FamilyFilter::All, const std::atomic_bool* cancelled = nullptr,
        size_t byteBudget = 64 * 1024 * 1024, const QualityFilter& quality = {}) const;

    Mesh mesh(int64_t sheetId) const;
    std::string annotation(int64_t sheetId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Decode one .asv payload blob (codec byte + compressed "ASV1" record) into a Mesh.
// Exposed for testing and reuse; throws std::runtime_error on a malformed blob or a
// codec that is not compiled in.
Mesh decodeSheetBlob(const unsigned char* data, size_t size);

// Intersect triangle (a, b, c) with the plane through `origin` with unit normal `n`.
// If the triangle straddles the plane, appends the crossing segment's two endpoints
// to `out` and returns true; otherwise returns false and leaves `out` unchanged.
bool intersectTrianglePlane(
    const Point& a, const Point& b, const Point& c,
    const Point& origin, const Point& n, std::array<Point, 2>& out);

}  // namespace vc::sheets
