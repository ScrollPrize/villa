#pragma once

namespace vc::objects { class TransformedGeometryView; }

#include <atomic>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#include "Surface.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "Rect3D.hpp"
#include <vc_reader/grid.hpp>

// Surface loading and channel flags
#define SURF_LOAD_IGNORE_MASK 1
#define SURF_CHANNEL_NORESIZE 1

// Debug prefix for auto-generated surfaces
#define Z_DBG_GEN_PREFIX "auto_grown_"

// Forward declarations
class QuadSurface;
class SurfaceQuerySet;

// Reference to a valid point in the grid (for iteration)
template<typename PointType>
struct PointRef {
    int row;
    int col;
    PointType& point;

    // Enable structured bindings - return by value for int, by ref for point
    template<std::size_t I>
    auto get() const -> decltype(auto) {
        if constexpr (I == 0) return row;
        else if constexpr (I == 1) return col;
        else return (point);  // parentheses ensure lvalue reference return
    }
};

// Reference to a valid quad (2x2 cell) in the grid
template<typename PointType>
struct QuadRef {
    int row;
    int col;
    PointType& p00;  // (row, col)
    PointType& p01;  // (row, col+1)
    PointType& p10;  // (row+1, col)
    PointType& p11;  // (row+1, col+1)

    template<std::size_t I>
    auto get() const -> decltype(auto) {
        if constexpr (I == 0) return row;
        else if constexpr (I == 1) return col;
        else if constexpr (I == 2) return (p00);
        else if constexpr (I == 3) return (p01);
        else if constexpr (I == 4) return (p10);
        else return (p11);
    }
};

// Structured binding support for PointRef
namespace std {
template<typename T>
struct tuple_size<PointRef<T>> : std::integral_constant<std::size_t, 3> {};

template<typename T>
struct tuple_element<0, PointRef<T>> { using type = int; };
template<typename T>
struct tuple_element<1, PointRef<T>> { using type = int; };
template<typename T>
struct tuple_element<2, PointRef<T>> { using type = T&; };

template<typename T>
struct tuple_size<QuadRef<T>> : std::integral_constant<std::size_t, 6> {};

template<typename T>
struct tuple_element<0, QuadRef<T>> { using type = int; };
template<typename T>
struct tuple_element<1, QuadRef<T>> { using type = int; };
template<typename T>
struct tuple_element<2, QuadRef<T>> { using type = T&; };
template<typename T>
struct tuple_element<3, QuadRef<T>> { using type = T&; };
template<typename T>
struct tuple_element<4, QuadRef<T>> { using type = T&; };
template<typename T>
struct tuple_element<5, QuadRef<T>> { using type = T&; };
}

// Range for iterating over valid points
template<typename PointType>
class ValidPointRange {
    // Use const Mat* when PointType is const, non-const Mat* otherwise
    using MatPtr = std::conditional_t<std::is_const_v<PointType>,
                                      const cv::Mat_<cv::Vec3f>*,
                                      cv::Mat_<cv::Vec3f>*>;
public:
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = PointRef<PointType>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        Iterator(MatPtr points, int row, int col)
            : _points(points), _row(row), _col(col) {
            advanceToValid();
        }

        reference operator*() const {
            return PointRef<PointType>{_row, _col, (*_points)(_row, _col)};
        }

        Iterator& operator++() {
            advance();
            advanceToValid();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return _row == other._row && _col == other._col;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void advance() {
            ++_col;
            if (_col >= _points->cols) {
                _col = 0;
                ++_row;
            }
        }

        void advanceToValid() {
            while (_row < _points->rows) {
                if (vc::geometry::validSurfacePoint((*_points)(_row, _col))) {
                    return;
                }
                advance();
            }
        }

        MatPtr _points;
        int _row;
        int _col;
    };

    ValidPointRange(MatPtr points) : _points(points) {}

    Iterator begin() { return Iterator(_points, 0, 0); }
    Iterator end() { return Iterator(_points, _points->rows, 0); }

private:
    MatPtr _points;
};

// Range for iterating over valid quads (2x2 cells where all 4 corners are valid)
template<typename PointType>
class ValidQuadRange {
    // Use const Mat* when PointType is const, non-const Mat* otherwise
    using MatPtr = std::conditional_t<std::is_const_v<PointType>,
                                      const cv::Mat_<cv::Vec3f>*,
                                      cv::Mat_<cv::Vec3f>*>;
public:
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = QuadRef<PointType>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        Iterator(MatPtr points, int row, int col)
            : _points(points), _row(row), _col(col) {
            advanceToValid();
        }

        reference operator*() const {
            return QuadRef<PointType>{
                _row, _col,
                (*_points)(_row, _col),
                (*_points)(_row, _col + 1),
                (*_points)(_row + 1, _col),
                (*_points)(_row + 1, _col + 1)
            };
        }

        Iterator& operator++() {
            advance();
            advanceToValid();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return _row == other._row && _col == other._col;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void advance() {
            ++_col;
            if (_col >= _points->cols - 1) {
                _col = 0;
                ++_row;
            }
        }

        void advanceToValid() {
            while (_row < _points->rows - 1) {
                if (isQuadValid()) {
                    return;
                }
                advance();
            }
        }

        bool isQuadValid() const {
            return vc::geometry::validSurfacePoint((*_points)(_row, _col)) &&
                   vc::geometry::validSurfacePoint((*_points)(_row, _col + 1)) &&
                   vc::geometry::validSurfacePoint((*_points)(_row + 1, _col)) &&
                   vc::geometry::validSurfacePoint((*_points)(_row + 1, _col + 1));
        }

        MatPtr _points;
        int _row;
        int _col;
    };

    ValidQuadRange(MatPtr points) : _points(points) {}

    Iterator begin() { return Iterator(_points, 0, 0); }
    Iterator end() { return Iterator(_points, _points->rows - 1, 0); }

private:
    MatPtr _points;
};

// Quad-grid surface. Renderable surface coordinates are level-0/base-volume
// voxel units. `scale()` is grid samples per surface unit (x=column, y=row),
// so a grid spaced 50 base voxels apart declares scale 1/50. This is
// parameterization metadata; the source Zarr level is the renderer's only LOD.
class QuadSurface : public Surface
{
public:
    QuadSurface() = default;
    // points will be cloned in constructor
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    // points will not be cloned in constructor, but pointer stored
    QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale);
    // Load from path with meta.json - lazy loading (only loads meta, loads points on first access)
    explicit QuadSurface(const std::filesystem::path &path_);
    // Load from path with provided meta json - lazy loading
    QuadSurface(const std::filesystem::path &path_, const nlohmann::json &json);
    ~QuadSurface() override;

    // Ensure points are loaded (for lazy loading constructors)
    void ensureLoaded();
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc_raw(const cv::Vec3f &ptr);
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) const override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    // Get normal directly from grid coordinates (avoids expensive pointTo lookup)
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000,
                  class SurfaceQuerySet* surfaceIndex = nullptr) override;
    cv::Size size();
    // Raw grid dimensions without forcing a lazy surface to load its XYZ data.
    [[nodiscard]] cv::Size gridSize() const;
    // Grid samples per level-0/base-volume voxel in each parameter direction.
    [[nodiscard]] cv::Vec2f scale() const;
    [[nodiscard]] cv::Vec3f center() const;
    [[nodiscard]] cv::Vec2d surfaceToGrid(const cv::Vec2d& surface) const;
    [[nodiscard]] cv::Vec2d gridToSurface(const cv::Vec2d& grid) const;
    // Shift the surface/ptr-space origin (i.e. _center): content previously
    // addressed by surface coordinate s is afterwards addressed by s - delta.
    // Lets a regenerated derived surface (e.g. a line-annotation strip, whose
    // arc-length parameterization shifts with every re-optimization) stay
    // addressable by camera coordinates that survive the surface swap. Call
    // only on a surface no viewer is rendering yet — the shift is not
    // synchronized against concurrent readers. Session-local by design: the
    // shift is not persisted by save() and a lazy surface's cache
    // invalidation recomputes _center, so use it only on in-memory derived
    // surfaces, never on disk-backed ones.
    void shiftSurfaceOrigin(const cv::Vec2d& delta);
    // The vertex grid as vcr::grid sees it: the options of a vcr::grid::Surface over
    // rawPointsPtr() (its pieces; linear patches), and the raster of `size` pixels whose
    // top-left pixel sits at surface coordinate (u0, v0), `scale` pixels per unit.
    [[nodiscard]] vcr::grid::SurfaceOptions gridOptions() const;
    [[nodiscard]] vcr::grid::Raster gridRaster(double u0, double v0, double scale, cv::Size size) const;
    // vcr::grid::Vertices over `points` (rawPointsPtr() by default), with `valid`
    // filled in: VC marks an invalid vertex (-1, -1, -1); a non-finite vertex is invalid too.
    [[nodiscard]] vcr::grid::Vertices gridVertices(std::vector<std::uint8_t>& valid, const cv::Mat_<cv::Vec3f>* points = nullptr) const;

    // Convert ptr-space coordinates to absolute grid row/col.
    // ptr-space stores (col - center.x*scale.x, row - center.y*scale.y, 0).
    [[nodiscard]] cv::Vec2f ptrToGrid(const cv::Vec3f& ptr) const;
    // The pointer at a grid position (column, row): ptrToGrid's inverse.
    [[nodiscard]] cv::Vec3f gridToPtr(const cv::Vec2f& grid) const;

    // Save under this surface's persistent object UUID (meta["uuid"]),
    // minting one when meta has none. Use for in-place edits, renames and
    // replicas of the same object. The directory name is never the identity.
    void save(const std::filesystem::path &path, bool force_overwrite = false);
    // Save as a new object: always assigns a fresh object UUID first. Use for
    // copies and derived outputs whose meta was taken from another surface.
    // Removes catalog representation lookup metadata from the local copy.
    void saveAsNew(const std::filesystem::path &path, bool force_overwrite = false);
    // Save an edit of the surface read from `source`: writing back over
    // `source` is an in-place edit (save), any other destination is a new
    // object (saveAsNew).
    void saveEdited(const std::filesystem::path &source,
                    const std::filesystem::path &dest,
                    bool force_overwrite = false);
    // The object UUID is never passed alongside a path: set meta["uuid"]
    // explicitly for a known identity. Deleted so a name argument cannot
    // silently convert to force_overwrite.
    void save(const std::filesystem::path&, const char*, bool = false) = delete;
    void save(const std::filesystem::path&, const std::string&, bool = false) = delete;
    // meta["uuid"], or empty when the surface has no persistent identity yet.
    [[nodiscard]] std::string objectUuid() const;
    Rect3D bbox();

    bool isLoaded() const { return !_needsLoad; }

    // Drop derived caches (validity mask, etc.) without unloading _points.
    // Called when this surface is no longer the active editing target so
    // RAM is reserved for the segment the user is currently working on.
    // In-flight validMask() calls keep the mask they returned.
    // Concurrent channel access, point unloading and geometry edits still
    // require external synchronization.
    void unloadCaches();

    // The surface's tags: its store record's, or a tifxyz meta.json's "tags" keys.
    std::vector<std::string> tags;

    // Column ranges [first, last) of disconnected components; empty: one component.
    const std::vector<std::pair<int, int>>& components() const { return _components; }
    // Override disconnected component column ranges for an in-memory view.
    void setComponents(std::vector<std::pair<int, int>> components)
    {
        _components = std::move(components);
        dropPointSurface();
    }

    // The non-const accessors hand out the points to be edited, so they drop the
    // vcr::grid::Surface that coord() and normal() use; it is rebuilt on the next call.
    // Fetch the pointer again for each edit rather than keeping it across calls to them.
    virtual cv::Mat_<cv::Vec3f> rawPoints() { ensureLoaded(); dropPointSurface(); return *_points; }
    virtual cv::Mat_<cv::Vec3f> *rawPointsPtr() { ensureLoaded(); dropPointSurface(); return _points.get(); }
    virtual const cv::Mat_<cv::Vec3f> *rawPointsPtr() const { const_cast<QuadSurface*>(this)->ensureLoaded(); return _points.get(); }

    // Grid iteration helpers
    ValidPointRange<cv::Vec3f> validPoints() { ensureLoaded(); return ValidPointRange<cv::Vec3f>(_points.get()); }
    ValidPointRange<const cv::Vec3f> validPoints() const {
        const_cast<QuadSurface*>(this)->ensureLoaded();
        return ValidPointRange<const cv::Vec3f>(_points.get());
    }
    ValidQuadRange<cv::Vec3f> validQuads() { ensureLoaded(); return ValidQuadRange<cv::Vec3f>(_points.get()); }
    ValidQuadRange<const cv::Vec3f> validQuads() const {
        const_cast<QuadSurface*>(this)->ensureLoaded();
        return ValidQuadRange<const cv::Vec3f>(_points.get());
    }

    // Single-point validity checks
    bool isPointValid(int row, int col) const {
        const_cast<QuadSurface*>(this)->ensureLoaded();
        if (!_points || row < 0 || row >= _points->rows || col < 0 || col >= _points->cols)
            return false;
        return vc::geometry::validSurfacePoint((*_points)(row, col));
    }
    bool isQuadValid(int row, int col) const {
        const_cast<QuadSurface*>(this)->ensureLoaded();
        if (!_points || row < 0 || row >= _points->rows - 1 || col < 0 || col >= _points->cols - 1)
            return false;
        return vc::geometry::validSurfacePoint((*_points)(row, col)) &&
               vc::geometry::validSurfacePoint((*_points)(row, col + 1)) &&
               vc::geometry::validSurfacePoint((*_points)(row + 1, col)) &&
               vc::geometry::validSurfacePoint((*_points)(row + 1, col + 1));
    }

    // Counting helpers
    int countValidPoints() const;
    int countValidQuads() const;

    // Generate validity mask at native resolution (255=valid, 0=invalid)
    cv::Mat_<uint8_t> validMask() const;

    // Write validity mask to path/mask.tif. If img is provided, writes multi-layer TIFF.
    void writeValidMask();

    mutable cv::Mat_<uint8_t> _validMaskCache;
    // Immutable geometry retained by sampling consumers; built on first use.
    mutable std::shared_ptr<const vcr::grid::Surface> _pointSurface;
    [[nodiscard]] std::shared_ptr<const vcr::grid::Surface> pointSurface() const;
    void dropPointSurface() const;
    // Guards the validity mask's lazy construction and invalidation.
    mutable std::mutex _cacheMutex;
    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name, int flags = 0);
    // Loads a registered channel that has no data yet (e.g. from its stored
    // geometry revision) on first access.
    void setChannelLoader(std::function<cv::Mat(const std::string&)> loader) { _channelLoader = std::move(loader); }
    void invalidateCache();
    void saveOverwrite();
    // Rotating backup under <parent>/backups/<seg>/. Throttled to at most one
    // snapshot per segment every couple minutes; pass force=true to bypass.
    // maxBackups < 0 means the default (10).
    void saveSnapshot(int maxBackups = -1, bool force = false);
    void invalidateMask();
    std::vector<std::string> channelNames() const;
    // Whether a registered channel holds data (a store channel loads on first use).
    bool channelLoaded(const std::string& name) const;

    /** Rotate the surface by arbitrary angle (degrees). Expands canvas to fit. */
    void rotate(float angleDeg);

    /** Flip the surface over the U axis (reverses rows/V direction) */
    void flipU();

    /** Flip the surface over the V axis (reverses columns/U direction) */
    void flipV();

    // DPI for TIFF output (0 = don't set). Set via setDpi() or voxelSizeToDpi().
    float dpi() const { return dpi_; }
    void setDpi(float d) { dpi_ = d; }

protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    std::function<cv::Mat(const std::string&)> _channelLoader;
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _points;
    cv::Rect _bounds;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
    // Column ranges of disconnected surface components (meta "components").
    // Each pair is [col_start, col_end). Empty = single contiguous surface.
    std::vector<std::pair<int,int>> _components;
    float dpi_ = 0.f;

private:

    void saveWithUuid(const std::filesystem::path& path, const std::string& uuid, bool force_overwrite);
    std::string objectUuidOrNew() const;
    // Flag for lazy loading - true if points need to be loaded from path
    bool _needsLoad = false;
    // Mutex to protect lazy loading from concurrent access
    mutable std::mutex _loadMutex;

    // Serializes all on-disk writes targeting a given segment directory, keyed by
    // the directory path so it works across separate QuadSurface objects pointing
    // at the same dir. Recursive so a public entry point (saveOverwrite) can hold
    // it across nested saveSnapshot()/save() calls on the same thread without
    // self-deadlock.
    static std::recursive_mutex& dirWriteMutex(const std::filesystem::path& dir);
};

// --- vc_object_store surfaces ------------------------------------------------
// The store equivalents of load_quad_from_tifxyz and save(). Callers own the
// transactions and revision pins. Stored vertices whose `valid` channel is 0 load
// as the (-1,-1,-1) sentinel and `valid` is exposed as the "mask" channel (255/0);
// record tags appear as meta["tags"] keys;
// writes touch only vertices that changed, so masked coordinates stay stored.

// Column ranges [c0, c1) of disconnected components recorded as meta
// "components" (Lasagna multi-component fits); empty for one component.
std::vector<std::pair<int, int>> surface_components(const nlohmann::json& meta);
// The tags a tifxyz meta.json records: the keys of its "tags" object.
std::vector<std::string> tifxyz_tags(const nlohmann::json& meta);

// A surface object's geometry revision (default: its current one). Other
// channels are registered empty; set a loader (store_channel_loader) to read them.
// Without applyValidity, masked vertices keep their coordinates (like SURF_LOAD_IGNORE_MASK).
// Through `transformed` (a view of that revision), points are in the view's volume voxels.
std::unique_ptr<QuadSurface> load_quad_from_store(const vc::objects::ObjectStore&, const vc::objects::Txn&,
    const std::string& objectUuid, const std::optional<std::string>& geometryRevision = std::nullopt,
    bool applyValidity = true, const vc::objects::TransformedGeometryView* transformed = nullptr);
// Reads channels of an immutable geometry revision through a session on demand.
std::function<cv::Mat(const std::string&)> store_channel_loader(
    std::shared_ptr<vc::objects::ObjectStoreSession>, vc::objects::AttachmentIdentity, std::string geometryRevision);
// Create surface object `uuid` holding `surface` (the caller's transaction
// expects it absent) through the store's createSurface: bound by
// `coordinates`, with `provenance`, keeping the surface's other meta.
void create_quad_in_store(vc::objects::ObjectStore&, vc::objects::WriteTxn&, const std::string& uuid, QuadSurface&,
    const std::string& name, const std::vector<std::string>& tags, const nlohmann::json& coordinates,
    const vc::objects::SurfaceProvenance& provenance);
// The surface's grid as stored: points (missing vertices as the sentinel) and
// validity from the "mask" channel (reduced to the grid: a vertex is valid only
// if every mask pixel it covers is 255); no other channels. A store draft of it.
vc::geometry::Surface quad_surface_grid(QuadSurface&);
// Geometry for putGeometry: quad_surface_grid and the other channels.
vc::objects::GeometryData quad_surface_geometry(QuadSurface&);
// Publish the vertices of `rect` (grid rows/columns) that differ from the stored
// revision `expected` in one sparse edit: new coordinates also mark the vertex
// valid; a vertex cleared to the sentinel loses its coordinates. `decide` answers
// for relationships the edit affects. Returns the new (or unchanged) geometry revision.
std::string write_quad_rect_to_store(vc::objects::ObjectStore&, vc::objects::WriteTxn&, const std::string& objectUuid,
    QuadSurface&, const cv::Rect& rect, const std::string& expected,
    const vc::objects::ObjectStoreSession::KeepRecorded& decide = {},
    const vc::objects::TransformedGeometryView* transformed = nullptr);
// Replace the named channels ("mask" is `valid`) with the surface's and remove `remove`.
std::string write_quad_channels_to_store(vc::objects::ObjectStore&, vc::objects::WriteTxn&, const std::string& objectUuid,
    QuadSurface&, const std::vector<std::string>& names, const std::vector<std::string>& remove,
    const std::string& expected, const std::vector<vc::objects::GeometryEdgeEdit>& edgeEdits = {});

// Serialize a bbox as [[minX,minY,minZ],[maxX,maxY,maxZ]], the layout
// QuadSurface::save() writes for its own bbox key.
nlohmann::json bbox_to_json(const Rect3D& bbox);

std::unique_ptr<QuadSurface> load_quad_from_tifxyz(const std::filesystem::path &path, int flags = 0);
std::unique_ptr<QuadSurface> load_quad_from_tifxyz_region(
    const std::filesystem::path &path, const cv::Rect &region, int flags = 0);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

// Look up winding depth index from the "d" channel at grid (row, col).
// Returns NAN if surface is null, "d" channel is missing, or coords are out of bounds.
float lookupDepthIndex(QuadSurface* surface, int row, int col);

// Control CUDA usage in GrowPatch (space_tracing_quad_phys). Default is true.

// Surface containment tests
bool contains(QuadSurface& a, const cv::Vec3f& loc, int max_iters = 1000);
bool contains(QuadSurface& a, const std::vector<cv::Vec3f>& locs);
