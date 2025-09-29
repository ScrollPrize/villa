#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace vc::core::util {

class GridStore {
public:
    GridStore(const cv::Rect& bounds, int cell_size);
    explicit GridStore(const std::string& path);
    ~GridStore();

    void add(const std::vector<cv::Point>& points);
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Rect& query_rect) const;
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Point2f& center, float radius) const;
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get_all() const;
    cv::Size size() const;
    size_t get_memory_usage() const;
    size_t numSegments() const;
    size_t numNonEmptyBuckets() const;

    nlohmann::json meta;

    void save(const std::string& path) const;
    void load_mmap(const std::string& path);

private:
    class GridStoreImpl;
    std::unique_ptr<GridStoreImpl> pimpl_;
};

}
