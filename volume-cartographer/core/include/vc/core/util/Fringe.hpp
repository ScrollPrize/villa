#pragma once

#include <opencv2/core.hpp>
#include <set>
#include <shared_mutex>
#include <functional>

namespace vc::surface_helpers {

namespace detail {
// Internal comparator to avoid dependency on SurfaceHelpers.hpp (which requires Ceres)
struct FringeVec2iLess {
    bool operator()(const cv::Vec2i& a, const cv::Vec2i& b) const {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
    }
};

// State flags (duplicated from SurfaceModeling.hpp to avoid Ceres dependency)
constexpr int kStateFringeLocValid = 1;     // STATE_LOC_VALID
constexpr int kStateFringeProcessing = 2;   // STATE_PROCESSING
}  // namespace detail

/**
 * Thread-safety: This class is NOT internally synchronized.
 *
 * For multi-threaded use, callers must either:
 * 1. Use the provided Lockable interface: std::lock_guard<Fringe> lock(fringe);
 * 2. Ensure all access is serialized via external synchronization
 *
 * Read operations (iteration, empty, size, getAttempts) require shared lock.
 * Write operations (insert, clear, incrementAttempts, etc.) require exclusive lock.
 */
class Fringe {
public:
    struct Config {
        int max_attempts = 0;       // 0 = unlimited attempts before pruning
        bool full_boundary = false; // true = rebuild as proper boundary, false = incremental
        int neighbor_connectivity = 4; // 4 or 8 connectivity for boundary detection
    };

    // External references needed for rebuild operations
    struct GridContext {
        const cv::Mat_<uint8_t>* state = nullptr;
        const cv::Rect* used_area = nullptr;
        const cv::Rect* active_bounds = nullptr;
        std::function<bool(const cv::Vec2i&)> is_savable;
    };

    Fringe();
    void init(cv::Size grid_size, const Config& config);
    void setContext(const GridContext& ctx);

    // ---- Core set operations ----
    void insert(const cv::Vec2i& p);
    void clear();
    bool empty() const;
    size_t size() const;

    // Range-based iteration
    auto begin() const { return _fringe.begin(); }
    auto end() const { return _fringe.end(); }

    // ---- Attempt tracking ----
    void incrementAttempts(const cv::Vec2i& p);
    void resetAttempts(const cv::Vec2i& p);
    void resetAllAttempts();
    bool shouldPrune(const cv::Vec2i& p) const;
    uint16_t getAttempts(const cv::Vec2i& p) const;

    // ---- Border tracking (for expansion triggers) ----
    void checkBorderContact(const cv::Vec2i& neighbor, cv::Size grid_size);
    void clearBorderFlags();
    bool atRightBorder() const { return _at_right_border; }
    bool atTopBorder() const { return _at_top_border; }
    bool atBottomBorder() const { return _at_bottom_border; }
    bool atAnyBorder() const { return _at_right_border || _at_top_border || _at_bottom_border; }

    // ---- Grid resize (called during expansion) ----
    void resize(cv::Size new_size, cv::Rect copy_roi);

    // ---- Rebuild operations ----
    void rebuildBoundary();
    void rebuildIncremental(int padding = 2);
    void rebuildIncrementalRect(const cv::Rect& rect, int padding = 2);

    // ---- Thread Safety: BasicLockable + SharedLockable ----
    void lock() { _mutex.lock(); }
    void unlock() { _mutex.unlock(); }
    bool try_lock() { return _mutex.try_lock(); }

    void lock_shared() { _mutex.lock_shared(); }
    void unlock_shared() { _mutex.unlock_shared(); }
    bool try_lock_shared() { return _mutex.try_lock_shared(); }

    std::shared_mutex& mutex() { return _mutex; }

private:
    std::set<cv::Vec2i, detail::FringeVec2iLess> _fringe;
    cv::Mat_<uint16_t> _attempts;
    cv::Size _size;
    Config _config;
    GridContext _ctx;
    mutable std::shared_mutex _mutex;

    bool _at_right_border = false;
    bool _at_top_border = false;
    bool _at_bottom_border = false;

    bool inBounds(const cv::Vec2i& p) const;
};

}  // namespace vc::surface_helpers
