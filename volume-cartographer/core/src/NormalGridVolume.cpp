#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/HashFunctions.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace vc::core::util {

    struct NormalGridVolume::pimpl {
        std::string base_path;
        int sparse_volume;
        nlohmann::json metadata;
        mutable std::mutex mutex;
        mutable std::unordered_map<cv::Vec2i, std::unique_ptr<GridStore>> grid_cache;
        std::vector<std::string> plane_dirs = {"xy", "xz", "yz"};

        explicit pimpl(const std::string& path) : base_path(path) {
            std::ifstream metadata_file((fs::path(base_path) / "metadata.json").string());
            if (!metadata_file.is_open()) {
                throw std::runtime_error("Failed to open metadata.json in " + base_path);
            }
            metadata_file >> metadata;
            sparse_volume = metadata.value("sparse-volume", 1);
        }

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const {

            float coord;
            switch (plane_idx) {
                case 0: coord = point.z; break; // XY plane
                case 1: coord = point.y; break; // XZ plane
                case 2: coord = point.x; break; // YZ plane
                default: return std::nullopt;
            }

            int slice_idx1 = static_cast<int>(coord / sparse_volume) * sparse_volume;
            int slice_idx2 = slice_idx1 + sparse_volume;

            double weight = (coord - slice_idx1) / sparse_volume;

            const GridStore* grid1 = get_grid(plane_idx, slice_idx1);
            const GridStore* grid2 = get_grid(plane_idx, slice_idx2);

            if (!grid1 || !grid2) {
                return std::nullopt;
            }

            return GridQueryResult{grid1, grid2, weight};
        }

        const GridStore* query_nearest(const cv::Point3f& point, int plane_idx) const {

            float coord;
            switch (plane_idx) {
                case 0: coord = point.z; break; // XY plane
                case 1: coord = point.y; break; // XZ plane
                case 2: coord = point.x; break; // YZ plane
                default: return nullptr;
            }

            int slice_idx = static_cast<int>(std::round(coord / sparse_volume)) * sparse_volume;

            return get_grid(plane_idx, slice_idx);
        }

        const GridStore* get_grid(int plane_idx, int slice_idx) const {
            cv::Vec2i key(plane_idx, slice_idx);

            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    return it->second.get();
                }
            }

            const std::string& dir = plane_dirs[plane_idx];
            char filename[256];
            snprintf(filename, sizeof(filename), "%06d.grid", slice_idx);
            std::string grid_path = (fs::path(base_path) / dir / filename).string();

            if (!fs::exists(grid_path)) {
                std::lock_guard<std::mutex> lock(mutex);
                grid_cache[key] = nullptr;
                return nullptr;
            }

            auto grid_store = std::make_unique<GridStore>(grid_path);

            // if (plane_idx == 0) { // XY plane
            //     if (!grid_store->meta.contains("umbilicus_x") || !grid_store->meta.contains("umbilicus_y")) {
            //         throw std::runtime_error("Missing umbilicus metadata in " + grid_path);
            //     }
            //     if (std::isnan(grid_store->meta["umbilicus_x"].get<float>()) || std::isnan(grid_store->meta["umbilicus_y"].get<float>())) {
            //         throw std::runtime_error("NaN umbilicus metadata in " + grid_path);
            //     }
            // }

            GridStore* ptr;
            {
                std::lock_guard<std::mutex> lock(mutex);

                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    return it->second.get();
                }

                ptr = grid_store.get();
                grid_cache[key] = std::move(grid_store);
            }
            return ptr;
        }
    };

    NormalGridVolume::NormalGridVolume(const std::string& path)
        : pimpl_(std::make_unique<pimpl>(path)) {}

    std::optional<NormalGridVolume::GridQueryResult> NormalGridVolume::query(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query(point, plane_idx);
    }

    const GridStore* NormalGridVolume::query_nearest(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query_nearest(point, plane_idx);
    }

    NormalGridVolume::~NormalGridVolume() = default;
    NormalGridVolume::NormalGridVolume(NormalGridVolume&&) noexcept = default;
    NormalGridVolume& NormalGridVolume::operator=(NormalGridVolume&&) noexcept = default;
    const nlohmann::json& NormalGridVolume::metadata() const {
        return pimpl_->metadata;
    }
} // namespace vc::core::util
