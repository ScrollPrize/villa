#pragma once

#include <deque>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace segmentation
{

// Delta-based vertex edit for efficient undo
struct VertexDelta
{
    int row{0};
    int col{0};
    cv::Vec3f previousWorld{0.0f, 0.0f, 0.0f};  // Position before the edit
};

class UndoHistory
{
public:
    // Capture full surface state (legacy method - memory intensive)
    bool capture(const cv::Mat_<cv::Vec3f>& points)
    {
        if (points.empty()) {
            return false;
        }

        cv::Mat_<cv::Vec3f> clone = points.clone();
        if (clone.empty()) {
            return false;
        }

        if (_states.size() >= kMaxEntries) {
            _states.pop_front();
        }
        _states.push_back({std::move(clone), {}});
        return true;
    }

    // Capture only changed vertices (delta-based - memory efficient)
    bool captureDelta(const std::vector<VertexDelta>& deltas)
    {
        if (deltas.empty()) {
            return false;
        }

        if (_states.size() >= kMaxEntries) {
            _states.pop_front();
        }
        _states.push_back({cv::Mat_<cv::Vec3f>(), deltas});
        return true;
    }

    void discardLast()
    {
        if (!_states.empty()) {
            _states.pop_back();
        }
    }

    [[nodiscard]] std::optional<cv::Mat_<cv::Vec3f>> takeLast()
    {
        if (_states.empty()) {
            return std::nullopt;
        }
        Entry entry = std::move(_states.back());
        _states.pop_back();

        // Return full points if available (legacy)
        if (!entry.points.empty()) {
            return entry.points;
        }

        // Delta-based entry - return empty (caller should use takeLastDelta)
        return std::nullopt;
    }

    // Take the last entry, returning deltas if available
    [[nodiscard]] std::optional<std::vector<VertexDelta>> takeLastDelta()
    {
        if (_states.empty()) {
            return std::nullopt;
        }
        Entry entry = std::move(_states.back());
        _states.pop_back();

        // Return deltas if this was a delta-based entry
        if (!entry.deltas.empty()) {
            return entry.deltas;
        }

        return std::nullopt;
    }

    // Check if the last entry is delta-based (vs full snapshot)
    [[nodiscard]] bool lastIsDelta() const
    {
        if (_states.empty()) {
            return false;
        }
        return !_states.back().deltas.empty();
    }

    void pushBack(cv::Mat_<cv::Vec3f> points)
    {
        if (points.empty()) {
            return;
        }
        if (_states.size() >= kMaxEntries) {
            _states.pop_front();
        }
        _states.push_back({std::move(points), {}});
    }

    void clear()
    {
        _states.clear();
    }

    [[nodiscard]] bool empty() const
    {
        return _states.empty();
    }

    [[nodiscard]] size_t size() const
    {
        return _states.size();
    }

private:
    struct Entry
    {
        cv::Mat_<cv::Vec3f> points;        // Full snapshot (legacy, memory intensive)
        std::vector<VertexDelta> deltas;   // Delta-based (memory efficient)
    };

    static constexpr std::size_t kMaxEntries = 1000;

    std::deque<Entry> _states;
};

} // namespace segmentation

