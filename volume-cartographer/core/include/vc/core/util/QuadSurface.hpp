#pragma once

#include <atomic>
#include <filesystem>
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
#include "Rect3D.hpp"

// Surface loading and channel flags
#define SURF_LOAD_IGNORE_MASK 1
#define SURF_CHANNEL_NORESIZE 1

// Debug prefix for auto-generated surfaces
#define Z_DBG_GEN_PREFIX "auto_grown_"

// Forward declarations
class QuadSurface;
class SurfacePatchIndex;

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
                if ((*_points)(_row, _col)[0] != -1.f) {
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
            return (*_points)(_row, _col)[0] != -1.f &&
                   (*_points)(_row, _col + 1)[0] != -1.f &&
                   (*_points)(_row + 1, _col)[0] != -1.f &&
                   (*_points)(_row + 1, _col + 1)[0] != -1.f;
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
    struct SurfaceSample {
        enum class Status {
            Valid,
            NonFinite,
            InvalidScale,
            MissingPoints,
            OutsideGrid,
            InvalidQuad,
        };

        Status status = Status::MissingPoints;
        cv::Vec2d grid{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
        cv::Vec3f volume{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};

        [[nodiscard]] bool valid() const { return status == Status::Valid; }
        explicit operator bool() const { return valid(); }
    };

    QuadSurface() = default;
    // points will be cloned in constructor
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    // points will not be cloned in constructor, but pointer stored
    QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale);
    // Load from path with meta.json - lazy loading (only loads meta, loads points on first access)
    explicit QuadSurface(const std::filesystem::path &path_);
    // Load from path with provided meta json - lazy loading
    QuadSurface(const std::filesystem::path &path_, const utils::Json &json);
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
    cv::Vec3f gridNormal(int row, int col) const;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) const override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000,
                  class SurfacePatchIndex* surfaceIndex = nullptr, class PointIndex* pointIndex = nullptr) override;
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
    [[nodiscard]] SurfaceSample sampleAtSurface(const cv::Vec2d& surface) const;

    // The legacy renderer treats native vertex validity as pixel coverage.
    // Spiral's drawn-input view opts into the stricter contract that a rendered
    // location must be backed by a complete bilinear quad. The default remains
    // false so existing QuadSurface consumers retain their current behavior.
    void setStrictQuadRenderValidity(bool enabled) { _strictQuadRenderValidity = enabled; }
    [[nodiscard]] bool strictQuadRenderValidity() const { return _strictQuadRenderValidity; }

    // Convert ptr-space coordinates to absolute grid row/col.
    // ptr-space stores (col - center.x*scale.x, row - center.y*scale.y, 0).
    [[nodiscard]] cv::Vec2f ptrToGrid(const cv::Vec3f& ptr) const;

    void save(const std::filesystem::path &path, const std::string &uuid, bool force_overwrite = false);
    void save(const std::filesystem::path &path, bool force_overwrite = false);
    void save_meta();
    Rect3D bbox();

    bool isLoaded() const { return !_needsLoad; }

    // Drop derived caches (validity mask, etc.) without unloading _points.
    // Called when this surface is no longer the active editing target so
    // RAM is reserved for the segment the user is currently working on.
    void unloadCaches();

    // True iff this surface was loaded from disk and can be safely unloaded.
    bool canUnload() const { return !path.empty(); }

    // Override disconnected component column ranges for an in-memory view.
    void setComponents(std::vector<std::pair<int, int>> components)
    {
        _components = std::move(components);
    }

    // Drop _points and all derived caches; ensureLoaded() will re-read from
    // disk on next access. No-op for in-memory-only surfaces.
    void unloadPoints();

    virtual cv::Mat_<cv::Vec3f> rawPoints() { ensureLoaded(); return *_points; }
    virtual cv::Mat_<cv::Vec3f> *rawPointsPtr() { ensureLoaded(); return _points.get(); }
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
        return (*_points)(row, col)[0] != -1.f;
    }
    bool isQuadValid(int row, int col) const {
        const_cast<QuadSurface*>(this)->ensureLoaded();
        if (!_points || row < 0 || row >= _points->rows - 1 || col < 0 || col >= _points->cols - 1)
            return false;
        return (*_points)(row, col)[0] != -1.f &&
               (*_points)(row, col + 1)[0] != -1.f &&
               (*_points)(row + 1, col)[0] != -1.f &&
               (*_points)(row + 1, col + 1)[0] != -1.f;
    }

    // Counting helpers
    int countValidPoints() const;
    int countValidQuads() const;

    // Generate validity mask at native resolution (255=valid, 0=invalid)
    cv::Mat_<uint8_t> validMask() const;

    // Write validity mask to path/mask.tif. If img is provided, writes multi-layer TIFF.
    void writeValidMask(const cv::Mat& img = cv::Mat());

    mutable cv::Mat_<uint8_t> _validMaskCache;
    // Set when _validMaskCache contains no 0s — gen() can skip the
    // validity warp + per-pixel invalidation pass entirely. Atomic
    // because gen() can be called from concurrent OMP threads.
    mutable std::atomic<bool> _validMaskAllValid{false};
    mutable cv::Mat_<cv::Vec3f> _normalCache;
    // Guards the lazy one-time build of the shared derived caches
    // (_normalCache, _validMaskCache) so concurrent gen()/validMask() calls
    // from the batch renderer's OMP tile loop don't race on construction.
    // The caches are read-only once built (until unloadCaches() on surface
    // switch), so reads after the build run lock-free.
    mutable std::mutex _cacheMutex;
    // NOTE: gen()'s per-call coords/normals/valid scratch buffers used to live
    // here as members and were reused across render ticks to avoid per-frame
    // reallocation (at 1920×1080 each coords/normals Mat is ~170 MiB of
    // cv::Vec3f). But gen() is called concurrently per-tile from the renderer's
    // OMP loop, and sharing those buffers (plus returning views into them)
    // raced -> SIGSEGV. They are now thread_local inside gen(): each thread
    // keeps its own buffers, so the per-frame reuse is preserved per-thread
    // while staying race-free.
    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name, int flags = 0);
    void invalidateCache();
    void saveOverwrite();
    // Write a single ancillary channel to path/<name>.tif in place, without
    // snapshotting or rewriting x/y/z. No-op if the channel is absent/empty.
    void saveChannel(const std::string& name);
    // Rotating backup under <backupRoot>/backups/<seg>/ (backupRoot defaults to
    // the segment's parent dir; VolumePkg sets it to the volpkg.json's dir).
    // Throttled to at most one snapshot per segment every couple minutes; pass
    // force=true to bypass. maxBackups < 0 means "use the configured default".
    void saveSnapshot(int maxBackups = -1, bool force = false);

    // App-configurable number of rotating snapshots kept per segment. VC3D
    // wires this to a user setting; defaults to 10 so non-GUI tools behave as
    // before. Used by saveOverwrite() and saveSnapshot()'s default.
    static void setBackupCount(int count);
    static int backupCount();
    void invalidateMask();
    std::vector<std::string> channelNames() const;

    /** Rotate the surface by arbitrary angle (degrees). Expands canvas to fit. */
    void rotate(float angleDeg);

    /** Resample the surface by a scale factor. factor > 1 increases density, < 1 decreases.
     *  Uses bilinear interpolation by default (cv::INTER_LINEAR), matching rotate(). */
    void resample(float factor, int interpolation = 1);  // 1 = cv::INTER_LINEAR
    /** Resample the surface independently in X and Y. */
    void resample(float factor_x, float factor_y, int interpolation);

    /** Compute optimal rotation angle to place highest Z values at row 0 */
    float computeZOrientationAngle() const;

    /** Rotate to place highest Z values at top (row 0) */
    void orientZUp();

    /** Flip the surface over the U axis (reverses rows/V direction) */
    void flipU();

    /** Flip the surface over the V axis (reverses columns/U direction) */
    void flipV();

    // Overlapping surfaces management (by ID/name)
    const std::set<std::string>& overlappingIds() const { return _overlappingIds; }
    void setOverlappingIds(const std::set<std::string>& ids) { _overlappingIds = ids; }
    void addOverlappingId(const std::string& id) { _overlappingIds.insert(id); }
    void removeOverlappingId(const std::string& id) { _overlappingIds.erase(id); }
    void readOverlappingJson();   // Load from path/overlapping.json
    void writeOverlappingJson() const;

    // Mask timestamp caching
    std::optional<std::filesystem::file_time_type> maskTimestamp() const { return _maskTimestamp; }
    void refreshMaskTimestamp();
    static std::optional<std::filesystem::file_time_type> readMaskTimestamp(const std::filesystem::path& dir);

    // DPI for TIFF output (0 = don't set). Set via setDpi() or voxelSizeToDpi().
    float dpi() const { return dpi_; }
    void setDpi(float d) { dpi_ = d; }

protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _points;
    cv::Rect _bounds;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
    std::set<std::string> _overlappingIds;
    std::optional<std::filesystem::file_time_type> _maskTimestamp;
    // Column ranges of disconnected surface components (from meta.json "components").
    // Each pair is [col_start, col_end). Empty = single contiguous surface.
    std::vector<std::pair<int,int>> _components;
    bool _strictQuadRenderValidity = false;
    float dpi_ = 0.f;

private:
    // Write surface data to directory without modifying state. skipChannel can be used to exclude a channel.
    void writeDataToDirectory(const std::filesystem::path& dir, const std::string& skipChannel = "");
    // Write a single ancillary channel as dir/<name>.tif.
    void writeChannelFile(const std::filesystem::path& dir, const std::string& name, const cv::Mat& mat);
    // Flag for lazy loading - true if points need to be loaded from path
    bool _needsLoad = false;
    // Mutex to protect lazy loading from concurrent access
    mutable std::mutex _loadMutex;

    // Serializes all on-disk writes targeting a given segment directory, keyed by
    // the directory path so it works across separate QuadSurface objects pointing
    // at the same dir (e.g. the autosave worker's snapshot vs. the live surface).
    // Without it, a worker-thread saveOverwrite() directory swap can delete a tmp
    // file out from under a main-thread saveChannel() rename -> uncaught
    // filesystem_error -> std::terminate. Recursive so a public entry point
    // (saveOverwrite) can hold it across nested saveSnapshot()/save() calls on
    // the same thread without self-deadlock.
    static std::recursive_mutex& dirWriteMutex(const std::filesystem::path& dir);
};

// Serialize a bbox as [[minX,minY,minZ],[maxX,maxY,maxZ]], the layout
// QuadSurface::save() writes for its own bbox key.
utils::Json bbox_to_json(const Rect3D& bbox);

std::unique_ptr<QuadSurface> load_quad_from_tifxyz(const std::filesystem::path &path, int flags = 0);
std::unique_ptr<QuadSurface> load_quad_from_tifxyz_region(
    const std::filesystem::path &path, const cv::Rect &region, int flags = 0);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

// Look up winding depth index from the "d" channel at grid (row, col).
// Returns NAN if surface is null, "d" channel is missing, or coords are out of bounds.
float lookupDepthIndex(QuadSurface* surface, int row, int col);

std::unique_ptr<QuadSurface> surface_diff(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
std::unique_ptr<QuadSurface> surface_union(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
std::unique_ptr<QuadSurface> surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);

// Control CUDA usage in GrowPatch (space_tracing_quad_phys). Default is true.
void set_space_tracing_use_cuda(bool enable);

// Overlapping JSON file utilities
void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);

// Surface overlap/containment tests
bool overlap(QuadSurface& a, QuadSurface& b, int max_iters = 1000);
bool contains(QuadSurface& a, const cv::Vec3f& loc, int max_iters = 1000);
bool contains(QuadSurface& a, const std::vector<cv::Vec3f>& locs);
bool contains_any(QuadSurface& a, const std::vector<cv::Vec3f>& locs);
