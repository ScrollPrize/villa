#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <optional>
#include <string>

class FocusHistoryManager
{
public:
    struct Entry {
        cv::Vec3f position;
        cv::Vec3f normal;
        std::string surfaceId;
    };

    FocusHistoryManager() = default;

    // Record a new focus point (deduplicates if unchanged)
    void record(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& surfaceId);

    // Step forward or backward. Returns nullopt if can't step, or a copy of the entry.
    std::optional<Entry> step(int direction);

    // Clear all history
    void clear();

    bool isNavigating() const { return _navigating; }
    void setNavigating(bool v) { _navigating = v; }

    static constexpr size_t MAX_HISTORY = 10;

private:
    std::deque<Entry> _history;
    int _index{-1};
    bool _navigating{false};
};
