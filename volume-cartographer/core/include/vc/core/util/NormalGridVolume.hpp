#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <optional>

#include <opencv2/core/types.hpp>
#include "utils/Json.hpp"

#include "vc/core/util/GridStore.hpp"

namespace vc::core::util {

    class NormalGridVolume {
    public:
        struct CacheStats {
            uint64_t gridHits = 0;
            uint64_t gridMisses = 0;
            size_t liveGridEntries = 0;
            uint64_t decodedPathHits = 0;
            uint64_t decodedPathMisses = 0;
            uint64_t decodedPathEvictions = 0;
            size_t decodedPathEntries = 0;
            size_t decodedPathBytes = 0;
        };

        explicit NormalGridVolume(const std::string& path);
        ~NormalGridVolume();
        NormalGridVolume(NormalGridVolume&&) noexcept;
        NormalGridVolume& operator=(NormalGridVolume&&) noexcept;

        struct GridQueryResult {
            std::shared_ptr<const GridStore> grid1;
            std::shared_ptr<const GridStore> grid2;
            double weight;
        };

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const;
        std::shared_ptr<const GridStore> query_nearest(const cv::Point3f& point, int plane_idx) const;
        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx) const;
        CacheStats cacheStats() const;
        void resetCacheStats() const;

    public:
        const utils::Json& metadata() const;

    private:
        struct pimpl;
        std::unique_ptr<pimpl> pimpl_;
    };

} // namespace vc::core::util
