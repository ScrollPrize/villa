#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>
#include <mutex>

namespace vc {
namespace core {
namespace util {

class LineSegList {
public:
    LineSegList(const std::vector<cv::Point>& points) {
        if (points.empty()) {
            return;
        }
        start_point_ = points[0];
        compressed_data_.reserve(2 * (points.size() - 1));
        for (size_t i = 1; i < points.size(); ++i) {
            cv::Point offset = points[i] - points[i-1];
            if (offset.x > std::numeric_limits<int8_t>::max() || offset.x < std::numeric_limits<int8_t>::min() ||
                offset.y > std::numeric_limits<int8_t>::max() || offset.y < std::numeric_limits<int8_t>::min()) {
                throw std::runtime_error("Offset out of range for int8_t");
            }
            compressed_data_.push_back(static_cast<int8_t>(offset.x));
            compressed_data_.push_back(static_cast<int8_t>(offset.y));
        }
        compressed_data_ptr_ = compressed_data_.data();
        compressed_data_size_ = compressed_data_.size();
    }

    LineSegList(cv::Point start_point, const int8_t* data, size_t size)
        : start_point_(start_point), compressed_data_ptr_(data), compressed_data_size_(size) {}

    std::shared_ptr<std::vector<cv::Point>> get() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::shared_ptr<std::vector<cv::Point>> points_ptr = points_cache_.lock();
        if (!points_ptr) {
            points_ptr = std::make_shared<std::vector<cv::Point>>();
            points_ptr->push_back(start_point_);
            cv::Point current_point = start_point_;
            for (size_t i = 0; i < compressed_data_size_; i += 2) {
                current_point.x += compressed_data_ptr_[i];
                current_point.y += compressed_data_ptr_[i+1];
                points_ptr->push_back(current_point);
            }
            points_cache_ = points_ptr;
        }
        return points_ptr;
    }

    const int8_t* compressed_data() const { return compressed_data_ptr_; }
    size_t compressed_data_size() const { return compressed_data_size_; }
    cv::Point start_point() const { return start_point_; }
    size_t num_points() const { return 1 + compressed_data_size_ / 2; }

private:
    cv::Point start_point_;
    std::vector<int8_t> compressed_data_; // Owns the data for non-views
    const int8_t* compressed_data_ptr_ = nullptr;
    size_t compressed_data_size_ = 0;
    std::weak_ptr<std::vector<cv::Point>> points_cache_;
    mutable std::mutex mutex_;
};

}
}
}