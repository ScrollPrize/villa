#pragma once

#include <deque>
#include <optional>

#include <opencv2/core.hpp>

namespace segmentation
{
class UndoHistory
{
public:
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
        _states.push_back({std::move(clone)});
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
        cv::Mat_<cv::Vec3f> points = std::move(_states.back().points);
        _states.pop_back();
        if (points.empty()) {
            return std::nullopt;
        }
        return points;
    }

    void pushBack(cv::Mat_<cv::Vec3f> points)
    {
        if (points.empty()) {
            return;
        }
        if (_states.size() >= kMaxEntries) {
            _states.pop_front();
        }
        _states.push_back({std::move(points)});
    }

    void clear()
    {
        _states.clear();
    }

    [[nodiscard]] bool empty() const
    {
        return _states.empty();
    }

private:
    struct Entry
    {
        cv::Mat_<cv::Vec3f> points;
    };

    static constexpr std::size_t kMaxEntries = 5;

    std::deque<Entry> _states;
};

} // namespace segmentation

