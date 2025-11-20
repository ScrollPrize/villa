#pragma once

#include <string>
#include <memory>
#include <optional>

#include <opencv2/core/types.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/GridStore.hpp"

namespace vc::core::util {

    class NormalGridVolume {
    public:
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

    public:
        const nlohmann::json& metadata() const;

    private:
        struct pimpl;
        std::unique_ptr<pimpl> pimpl_;
    };

} // namespace vc::core::util
