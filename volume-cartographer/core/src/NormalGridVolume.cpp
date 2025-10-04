#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/HashFunctions.hpp"
 
 #include <filesystem>
 #include <fstream>
#include <iostream>
#include <unordered_map>
#include <random>
#include <chrono>
#include <atomic>
 
 namespace fs = std::filesystem;
 
 namespace vc::core::util {
 
    struct CacheEntry {
        std::shared_ptr<GridStore> grid_store;
        uint64_t generation;
    };

     struct NormalGridVolume::pimpl {
         std::string base_path;
         int sparse_volume;
         nlohmann::json metadata;
         mutable std::mutex mutex;
         mutable std::unordered_map<cv::Vec2i, CacheEntry> grid_cache;
         mutable uint64_t generation_counter = 0;
         size_t max_cache_size = 4096;
         size_t eviction_sample_size = 10;
        
         mutable std::atomic<uint64_t> cache_hits{0};
         mutable std::atomic<uint64_t> cache_misses{0};
         mutable std::chrono::steady_clock::time_point last_stat_time = std::chrono::steady_clock::now();

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

            auto grid1 = get_grid(plane_idx, slice_idx1);
            auto grid2 = get_grid(plane_idx, slice_idx2);

            if (!grid1 || !grid2) {
                return std::nullopt;
            }

            return GridQueryResult{grid1, grid2, weight};
        }

        std::shared_ptr<const GridStore> query_nearest(const cv::Point3f& point, int plane_idx) const {

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

        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx) const {
            cv::Vec2i key(plane_idx, slice_idx);

            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    cache_hits++;
                    it->second.generation = ++generation_counter;
                    check_print_stats();
                    return it->second.grid_store;
                }
            }
 
            cache_misses++;
             const std::string& dir = plane_dirs[plane_idx];
            char filename[256];
            snprintf(filename, sizeof(filename), "%06d.grid", slice_idx);
            std::string grid_path = (fs::path(base_path) / dir / filename).string();

            if (!fs::exists(grid_path)) {
                std::lock_guard<std::mutex> lock(mutex);
                grid_cache[key] = {nullptr, ++generation_counter};
                return nullptr;
            }
 
            auto grid_store = std::make_shared<GridStore>(grid_path);

            // if (plane_idx == 0) { // XY plane
            //     if (!grid_store->meta.contains("umbilicus_x") || !grid_store->meta.contains("umbilicus_y")) {
            //         throw std::runtime_error("Missing umbilicus metadata in " + grid_path);
            //     }
            //     if (std::isnan(grid_store->meta["umbilicus_x"].get<float>()) || std::isnan(grid_store->meta["umbilicus_y"].get<float>())) {
            //         throw std::runtime_error("NaN umbilicus metadata in " + grid_path);
            //     }
            // }

            {
                std::lock_guard<std::mutex> lock(mutex);

                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    // Another thread might have loaded it in the meantime
                    cache_hits++;
                    it->second.generation = ++generation_counter;
                    return it->second.grid_store;
                }
 
                grid_cache[key] = {grid_store, ++generation_counter};

                // Eviction logic
                if (grid_cache.size() > max_cache_size) {
                    std::vector<cv::Vec2i> keys;
                    keys.reserve(grid_cache.size());
                    for (const auto& pair : grid_cache) {
                        keys.push_back(pair.first);
                    }

                    std::mt19937 gen(std::random_device{}());
                    std::uniform_int_distribution<size_t> dist(0, keys.size() - 1);

                    cv::Vec2i key_to_evict;
                    uint64_t min_generation = std::numeric_limits<uint64_t>::max();

                    for (size_t i = 0; i < eviction_sample_size && !keys.empty(); ++i) {
                        size_t rand_idx = dist(gen);
                        const auto& key = keys[rand_idx];
                        const auto& entry = grid_cache.at(key);
                        if (entry.generation < min_generation) {
                            min_generation = entry.generation;
                            key_to_evict = key;
                        }
                    }

                    if (min_generation != std::numeric_limits<uint64_t>::max()) {
                        grid_cache.erase(key_to_evict);
                    }
                }

                check_print_stats();
            }
            return grid_store;
        }

        void check_print_stats() const {
            if (generation_counter % 1000 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - last_stat_time);
                if (diff.count() >= 1) {
                    uint64_t hits = cache_hits.load();
                    uint64_t misses = cache_misses.load();
                    uint64_t total = hits + misses;
                    double hit_rate = (total == 0) ? 0.0 : (static_cast<double>(hits) / total) * 100.0;
                    std::cout << "[GridStore Cache] Hits: " << hits << ", Misses: " << misses << ", Total: " << total << ", Hit Rate: " << std::fixed << std::setprecision(2) << hit_rate << "%" << std::endl;
                    last_stat_time = now;
                }
            }
        }
    };

    NormalGridVolume::NormalGridVolume(const std::string& path)
        : pimpl_(std::make_unique<pimpl>(path)) {}
 
    std::optional<NormalGridVolume::GridQueryResult> NormalGridVolume::query(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query(point, plane_idx);
    }

    std::shared_ptr<const GridStore> NormalGridVolume::query_nearest(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query_nearest(point, plane_idx);
    }

    NormalGridVolume::~NormalGridVolume() = default;
    NormalGridVolume::NormalGridVolume(NormalGridVolume&&) noexcept = default;
    NormalGridVolume& NormalGridVolume::operator=(NormalGridVolume&&) noexcept = default;
    const nlohmann::json& NormalGridVolume::metadata() const {
        return pimpl_->metadata;
    }
} // namespace vc::core::util
