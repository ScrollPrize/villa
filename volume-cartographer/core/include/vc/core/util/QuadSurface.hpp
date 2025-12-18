#pragma once

#include <filesystem>
#include <iterator>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <string>

#include "Surface.hpp"
#include "Tiff.hpp"

// Forward declarations
template<typename T>
class ChunkCache;

// Surface loading and channel flags
#define SURF_LOAD_IGNORE_MASK 1
#define SURF_CHANNEL_NORESIZE 1

// Debug prefix for auto-generated surfaces
#define Z_DBG_GEN_PREFIX "auto_grown_"

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);

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

//quads based surface class with a pointer implementing a nominal scale of 1 voxel
class QuadSurface : public Surface
{
public:
    cv::Vec3f pointer() override;
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
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    // Get normal directly from grid coordinates (avoids expensive pointTo lookup)
    cv::Vec3f gridNormal(int row, int col) const;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000,
                  class SurfacePatchIndex* surfaceIndex = nullptr, class PointIndex* pointIndex = nullptr) override;
    cv::Size size();
    [[nodiscard]] cv::Vec2f scale() const;
    [[nodiscard]] cv::Vec3f center() const;

    void save(const std::string &path, const std::string &uuid, bool force_overwrite = false);
    void save(const std::filesystem::path &path, bool force_overwrite = false);
    void save_meta();
    Rect3D bbox();

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

    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name, int flags = 0);
    void invalidateCache();
    void saveOverwrite();
    void saveSnapshot(int maxBackups = 10);
    void invalidateMask();
    std::vector<std::string> channelNames() const;

    /** Rotate the surface by arbitrary angle (degrees). Expands canvas to fit. */
    void rotate(float angleDeg);

    /** Compute optimal rotation angle to place highest Z values at row 0 */
    float computeZOrientationAngle() const;

    /** Rotate to place highest Z values at top (row 0) */
    void orientZUp();

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

protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    std::unique_ptr<cv::Mat_<cv::Vec3f>> _points;
    cv::Rect _bounds;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
    std::set<std::string> _overlappingIds;
    std::optional<std::filesystem::file_time_type> _maskTimestamp;

private:
    // Write surface data to directory without modifying state. skipChannel can be used to exclude a channel.
    void writeDataToDirectory(const std::filesystem::path& dir, const std::string& skipChannel = "");
    // Flag for lazy loading - true if points need to be loaded from path
    bool _needsLoad = false;
    // Mutex to protect lazy loading from concurrent access
    mutable std::mutex _loadMutex;
};

std::unique_ptr<QuadSurface> load_quad_from_tifxyz(const std::string &path, int flags = 0);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

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

