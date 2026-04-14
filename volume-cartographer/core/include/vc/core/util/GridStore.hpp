#pragma once

#include <cstdint>
#include <functional>
#include <opencv2/core/types.hpp>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace vc::core::util {

class GridStore {
public:
    struct SaveOptions {
        bool verify_reload = true;
    };

    struct CacheStats {
        uint64_t decodedPathHits = 0;
        uint64_t decodedPathMisses = 0;
        uint64_t decodedPathEvictions = 0;
        size_t decodedPathEntries = 0;
        size_t decodedPathBytes = 0;
    };

    struct QueryScratch {
        std::vector<size_t> ids;
        std::vector<std::shared_ptr<std::vector<cv::Point>>> results;
    };

    GridStore(const cv::Rect& bounds, int cell_size);
    explicit GridStore(const std::string& path);
    ~GridStore();

    void add(const std::vector<cv::Point>& points);
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Rect& query_rect) const;
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Point2f& center, float radius) const;
    void forEach(const cv::Rect& query_rect,
                 QueryScratch& scratch,
                 const std::function<void(const std::shared_ptr<std::vector<cv::Point>>&)>& visitor) const;
    std::vector<std::shared_ptr<std::vector<cv::Point>>> get_all() const;
    cv::Size size() const;
    size_t get_memory_usage() const;
    size_t numSegments() const;
    size_t numNonEmptyBuckets() const;
    CacheStats cacheStats() const;
    void resetCacheStats() const;

    nlohmann::json meta;

    void save(const std::string& path, const SaveOptions& options) const;
    void save(const std::string& path) const;
    void load_mmap(const std::string& path);

private:
    friend class LineSegList;
    class GridStoreImpl;
    std::unique_ptr<GridStoreImpl> pimpl_;
};

}
