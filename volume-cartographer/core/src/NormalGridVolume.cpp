#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/HashFunctions.hpp"

#include "utils/Json.hpp"
#include <utils/http_fetch.hpp>

 #include <filesystem>
 #include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>

 namespace fs = std::filesystem;
 
 namespace vc::core::util {
 
    struct CacheEntry {
        std::shared_ptr<GridStore> grid_store;
        uint64_t generation;
        // Index of this entry's key in pimpl::cache_keys. Keeping it here is
        // what lets eviction sample and erase in O(1) instead of rebuilding
        // the key list from the map every time.
        size_t key_slot = 0;
        // Bytes charged to the budget when this entry was last (re)counted,
        // held here so eviction subtracts exactly what was added.
        size_t bytes = 0;
    };

    // A positive integer from the environment, else `fallback`.
    static size_t env_size(const char* name, size_t fallback)
    {
        if (const char* e = std::getenv(name)) {
            const long long v = std::atoll(e);
            if (v > 0) {
                return static_cast<size_t>(v);
            }
        }
        return fallback;
    }

     struct NormalGridVolume::pimpl {
         std::string base_path;
         int sparse_volume;
         utils::Json metadata;
         double coordinate_scale = 1.0;
         double output_spiral_step = 20.0;
         int selected_level = 0;
         bool multiscale = false;
         mutable std::shared_mutex mutex;
         mutable std::unordered_map<cv::Vec2i, CacheEntry> grid_cache;
         // Dense, unordered view of the resident keys, kept in step with
         // grid_cache so eviction can draw its random sample by index. Erase
         // is the usual swap-with-last, which is why each entry remembers its
         // own slot.
         mutable std::vector<cv::Vec2i> cache_keys;
         mutable uint64_t generation_counter = 0;
         // Optional budget in BYTES (opt-in) instead of the entry cap.
         //
         // DEFAULT BEHAVIOUR IS UNCHANGED: with VC_GRID_CACHE_BYTES unset the
         // cache holds at most 512 entries, as it always has, and none of the
         // byte accounting below runs. Set VC_GRID_CACHE_BYTES=<bytes> to bound
         // the cache by size instead (the entry cap then relaxes to a 65,536
         // backstop unless VC_GRID_CACHE_ENTRIES says otherwise).
         //
         // An entry count is the wrong unit here. A cached GridStore costs the
         // .grid file it maps -- whose size varies by an order of magnitude
         // between slices -- plus up to 2 MiB of decoded seglists, so any
         // entry cap is simultaneously too loose (a worst case of entries x
         // 2 MiB) and too tight (it evicts small stores that cost almost
         // nothing to keep). A byte budget bounds the cache directly and does
         // not need retuning when the grids change.
         //
         // The old cap of 512 entries was far below a real tracing working
         // set. Measured on one resume round of a large patch, whose grid
         // working set was 7424 distinct slices totalling 4.0 GB: the tracer
         // opened .grid files 110631 times, i.e. it reopened, remapped and
         // re-parsed each file about 15 times over, and threw away each
         // store's decoded polylines with it. A budget that holds the working
         // set turns that into one construction per distinct slice.
         //
         // A budget of 4 GiB is the value that measured best. Sweeping the budget
         // on that same round gave:
         //
         //     budget    GridStore constructions   resident entries
         //     256 MiB   112447                      659
         //       1 GiB    50426                     2162
         //       2 GiB    32423                     3730
         //       4 GiB     7427                     7427
         //       8 GiB     7427                     7427
         //
         // 4 GiB is the knee (VC_GRID_CACHE_BYTES=4294967296): it holds the whole working set with no
         // evictions, and 8 GiB buys nothing. All budgets produced
         // byte-identical output -- the cache is transparent, so a store that
         // is evicted is simply reloaded from the same immutable file.
         //
         // The accounting charges each store its whole mapped file size while
         // only touched pages are actually resident, so the real RSS cost is
         // well under the budget (1.8 GB at a 4 GiB budget in that run).
         // The budget over-charges rather than under-charges, which is the
         // safe direction. Mappings are file-backed, so concurrent tracers
         // working the same grids share those pages through the page cache.
         // 0 = byte budget off (default).
         size_t max_cache_bytes = env_size("VC_GRID_CACHE_BYTES", 0);
         bool byte_budget = max_cache_bytes > 0;
         // Entry cap. 512 by default. With a byte budget it becomes a backstop so
         // a volume of unusually tiny slices cannot grow the map without bound.
         size_t max_cache_entries =
             env_size("VC_GRID_CACHE_ENTRIES", byte_budget ? 65536 : 512);
         mutable size_t cache_bytes = 0;
         // A store's decoded-polyline cache keeps growing AFTER it is inserted
         // (on hits), so the running total drifts low. It is therefore
         // re-summed from the stores themselves, both every kResyncEveryInserts
         // insertions and every kResyncEveryHits hits, so a working set that is
         // fully resident (no more insertions) is still held to the budget.
         static constexpr uint64_t kResyncEveryInserts = 1024;
         static constexpr uint64_t kResyncEveryHits = 1ull << 16;
         mutable uint64_t inserts_since_resync = 0;
         mutable std::atomic<uint64_t> hits_since_resync{0};
         size_t eviction_sample_size = 10;
         // Seeded once. This used to be constructed -- 2.5 KiB of state,
         // seeded from std::random_device -- on every single eviction, while
         // holding the exclusive cache lock.
         mutable std::mt19937 eviction_rng{std::random_device{}()};
        
         mutable std::atomic<uint64_t> cache_hits{0};
         mutable std::atomic<uint64_t> cache_misses{0};
         mutable std::chrono::steady_clock::time_point last_stat_time = std::chrono::steady_clock::now();

         std::vector<std::string> plane_dirs = {"xy", "xz", "yz"};

         // Remote streaming (enabled by kNormalGridsRemoteMarker in base_path):
         // grid files missing locally are fetched from remote_url on demand and
         // cached in base_path. 404s leave a "<file>.missing" marker so they
         // are not re-fetched across runs.
         bool remote = false;
         std::string remote_url;
         utils::HttpClient http_client;
         static constexpr int prefetch_radius = 4;
         static constexpr int prefetch_worker_count = 4;
         static constexpr size_t prefetch_queue_limit = 256;
         mutable std::mutex prefetch_mutex;
         mutable std::condition_variable prefetch_cv;
         mutable std::deque<std::pair<int, int>> prefetch_queue;
         mutable std::unordered_set<uint64_t> prefetch_pending;
         std::vector<std::thread> prefetch_threads;
         bool prefetch_stop = false;

         pimpl(const std::string& path, int requested_level) : base_path(path) {
            load_remote_marker();
            if (!ensure_local_file("metadata.json")) {
                throw std::runtime_error(missing_file_error("metadata.json"));
            }
            metadata = utils::Json::parse_file(fs::path(base_path) / "metadata.json");
            multiscale = metadata.value("format", std::string()) == "normal-grid-multiscale";
            if (multiscale) {
                const int min_level = metadata.value("min-level", 0);
                const int max_level = metadata.value("max-level", 0);
                selected_level = std::clamp(requested_level, min_level, max_level);
                if (selected_level != requested_level) {
                    std::cerr << "warning: normal_grid_level=" << requested_level
                              << " is outside the levels " << min_level << ".." << max_level
                              << " available in " << base_path
                              << "; using level " << selected_level << std::endl;
                }

                utils::Json level_metadata;
                const std::string level_metadata_name =
                    "metadata.level" + std::to_string(selected_level) + ".json";
                const fs::path level_metadata_path = fs::path(base_path) / level_metadata_name;
                const bool level_present = ensure_local_file(level_metadata_name);
                if (level_present && fs::exists(level_metadata_path)) {
                    level_metadata = utils::Json::parse_file(level_metadata_path);
                } else if (metadata.contains("source-metadata")) {
                    level_metadata = metadata["source-metadata"];
                } else if (remote && !level_present) {
                    throw std::runtime_error(missing_file_error(level_metadata_name));
                } else {
                    throw std::runtime_error("multiscale normal grids missing level metadata: " +
                                             level_metadata_path.string());
                }

                metadata = std::move(level_metadata);
                const int derived_scale = std::max(1, metadata.value("derived-scale", 1 << selected_level));
                coordinate_scale = 1.0 / static_cast<double>(derived_scale);
                sparse_volume = std::max(1, metadata.value("sparse-volume", 1));
                const double level_step = metadata.value("spiral-step", 20.0);
                output_spiral_step = level_step / coordinate_scale;
            } else {
                selected_level = 0;
                if (requested_level != 0) {
                    std::cerr << "warning: normal_grid_level=" << requested_level
                              << " ignored; store at " << base_path
                              << " is single-scale (no \"format\":\"normal-grid-multiscale\" in metadata.json)"
                              << "; using level 0. A multiscale store can be derived from a local copy with"
                              << " vc_gen_normalgrids pyramid -i <store> -o <multiscale store>" << std::endl;
                }
                sparse_volume = metadata.value("sparse-volume", 1);
                output_spiral_step = metadata.value("spiral-step", 20.0);
            }

            if (remote) {
                prefetch_threads.reserve(prefetch_worker_count);
                for (int i = 0; i < prefetch_worker_count; ++i) {
                    prefetch_threads.emplace_back([this]() { prefetch_loop(); });
                }
            }
        }

        ~pimpl() {
            {
                std::lock_guard<std::mutex> lock(prefetch_mutex);
                prefetch_stop = true;
            }
            prefetch_cv.notify_all();
            for (auto& thread : prefetch_threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

        void load_remote_marker() {
            const fs::path marker = fs::path(base_path) / kNormalGridsRemoteMarker;
            std::error_code ec;
            if (!fs::is_regular_file(marker, ec)) {
                return;
            }
            try {
                const auto json = utils::Json::parse_file(marker);
                remote_url = json.value("url", std::string());
            } catch (const std::exception& e) {
                std::cerr << "Ignoring unreadable " << marker << ": " << e.what() << std::endl;
                return;
            }
            while (!remote_url.empty() && remote_url.back() == '/') {
                remote_url.pop_back();
            }
            remote = !remote_url.empty();
        }

        // Build an intentional, actionable error message for a normal-grid file
        // that could not be made available locally. Distinguishes the remote-
        // streaming failure (fetch from remote_url failed) from a plain missing
        // local file so the caller can surface it instead of the generic
        // "Cannot open" thrown by Json::parse_file on a still-absent file.
        std::string missing_file_error(const std::string& rel) const {
            if (remote) {
                return "Failed to fetch remote normal-grid file '" + rel +
                       "' from " + remote_url +
                       " (cached under " + base_path +
                       "); see preceding stderr for the HTTP status or network error";
            }
            return "normal-grid file not found: " +
                   (fs::path(base_path) / rel).string();
        }

        std::string relative_grid_path(int plane_idx, int slice_idx) const {
            char filename[64];
            snprintf(filename, sizeof(filename), "%06d.grid", slice_idx);
            std::string rel = plane_dirs[plane_idx];
            if (multiscale) {
                rel += "/" + std::to_string(selected_level);
            }
            rel += "/";
            rel += filename;
            return rel;
        }

        // Make base_path/rel exist locally, fetching it from remote_url when
        // streaming. Returns false when the file is genuinely missing (locally
        // for plain stores, remotely for streaming stores) or the fetch failed.
        bool ensure_local_file(const std::string& rel) const {
            const fs::path local = fs::path(base_path) / rel;
            std::error_code ec;
            if (fs::exists(local, ec)) {
                return true;
            }
            if (!remote) {
                return false;
            }
            const fs::path missing_marker = local.string() + ".missing";
            if (fs::exists(missing_marker, ec)) {
                return false;
            }

            utils::HttpResponse resp;
            try {
                resp = http_client.get(remote_url + "/" + rel);
            } catch (const std::exception& e) {
                std::cerr << "normal-grid fetch failed for " << rel << ": " << e.what()
                          << std::endl;
                return false;
            }

            if (resp.ok()) {
                fs::create_directories(local.parent_path(), ec);
                // Unique temp + rename so concurrent readers (and concurrent
                // processes streaming into the same cache) never see partial files.
                const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
                const fs::path tmp = local.string() + ".tmp-" +
                    std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
                    "-" + std::to_string(tick);
                {
                    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                    if (!out) {
                        return false;
                    }
                    if (!resp.body.empty()) {
                        out.write(reinterpret_cast<const char*>(resp.body.data()),
                                  static_cast<std::streamsize>(resp.body.size()));
                    }
                    out.close();
                    if (!out) {
                        fs::remove(tmp, ec);
                        return false;
                    }
                }
                fs::rename(tmp, local, ec);
                if (ec) {
                    fs::remove(tmp, ec);
                    return fs::exists(local, ec);
                }
                return true;
            }

            if (resp.not_found()) {
                fs::create_directories(local.parent_path(), ec);
                std::ofstream(missing_marker.string()).flush();
                return false;
            }

            std::cerr << "normal-grid fetch for " << rel << " returned HTTP "
                      << resp.status_code << std::endl;
            return false;
        }

        void prefetch_loop() {
            while (true) {
                std::pair<int, int> item;
                {
                    std::unique_lock<std::mutex> lock(prefetch_mutex);
                    prefetch_cv.wait(lock, [&]() {
                        return prefetch_stop || !prefetch_queue.empty();
                    });
                    if (prefetch_stop) {
                        return;
                    }
                    item = prefetch_queue.front();
                    prefetch_queue.pop_front();
                }
                ensure_local_file(relative_grid_path(item.first, item.second));
                {
                    std::lock_guard<std::mutex> lock(prefetch_mutex);
                    prefetch_pending.erase(prefetch_key(item.first, item.second));
                }
            }
        }

        static uint64_t prefetch_key(int plane_idx, int slice_idx) {
            return (static_cast<uint64_t>(static_cast<uint32_t>(plane_idx)) << 32) |
                   static_cast<uint32_t>(slice_idx);
        }

        // Queue nearby slices of the same plane so the tracer's advancing
        // frontier mostly hits warm files.
        void schedule_prefetch(int plane_idx, int slice_idx) const {
            std::error_code ec;
            std::lock_guard<std::mutex> lock(prefetch_mutex);
            for (int distance = 1; distance <= prefetch_radius; ++distance) {
                for (const int sign : {1, -1}) {
                    const int slice = slice_idx + sign * distance * sparse_volume;
                    if (slice < 0 || prefetch_queue.size() >= prefetch_queue_limit) {
                        continue;
                    }
                    const uint64_t key = prefetch_key(plane_idx, slice);
                    if (prefetch_pending.count(key)) {
                        continue;
                    }
                    const fs::path local =
                        fs::path(base_path) / relative_grid_path(plane_idx, slice);
                    if (fs::exists(local, ec) ||
                        fs::exists(fs::path(local.string() + ".missing"), ec)) {
                        continue;
                    }
                    prefetch_pending.insert(key);
                    prefetch_queue.push_back({plane_idx, slice});
                }
            }
            prefetch_cv.notify_all();
        }

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const {

            float coord;
            switch (plane_idx) {
                case 0: coord = point.z * coordinate_scale; break; // XY plane
                case 1: coord = point.y * coordinate_scale; break; // XZ plane
                case 2: coord = point.x * coordinate_scale; break; // YZ plane
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
                case 0: coord = point.z * coordinate_scale; break; // XY plane
                case 1: coord = point.y * coordinate_scale; break; // XZ plane
                case 2: coord = point.x * coordinate_scale; break; // YZ plane
                default: return nullptr;
            }

            int slice_idx = static_cast<int>(std::round(coord / sparse_volume)) * sparse_volume;

            return get_grid(plane_idx, slice_idx);
        }

        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx) const {
            cv::Vec2i key(plane_idx, slice_idx);

            // Use shared_lock for read-only cache lookup (hot path)
            {
                std::shared_ptr<GridStore> hit;
                bool found = false;
                {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    auto it = grid_cache.find(key);
                    if (it != grid_cache.end()) {
                        cache_hits++;
                        // Note: Removed generation update from hot path to avoid write contention
                        // LRU eviction will still work reasonably well without per-access updates
                        hit = it->second.grid_store;
                        found = true;
                    }
                }
                if (found) {
                    if (byte_budget &&
                        hits_since_resync.fetch_add(1, std::memory_order_relaxed)
                            % kResyncEveryHits == kResyncEveryHits - 1) {
                        std::unique_lock<std::shared_mutex> lock(mutex);
                        resync_bytes_locked();
                        evict_to_budget_locked();
                    }
                    return hit;
                }
            }
 
            cache_misses++;
            const std::string rel = relative_grid_path(plane_idx, slice_idx);
            std::string grid_path = (fs::path(base_path) / rel).string();

            if (remote) {
                schedule_prefetch(plane_idx, slice_idx);
            }
            if (!ensure_local_file(rel)) {
                std::unique_lock<std::shared_mutex> lock(mutex);
                insert_locked(key, nullptr);
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
                std::unique_lock<std::shared_mutex> lock(mutex);

                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    // Another thread might have loaded it in the meantime
                    cache_hits++;
                    it->second.generation = ++generation_counter;
                    return it->second.grid_store;
                }
 
                insert_locked(key, grid_store);

                if (byte_budget && ++inserts_since_resync >= kResyncEveryInserts) {
                    resync_bytes_locked();
                }
                evict_to_budget_locked();

                check_print_stats();
            }
            return grid_store;
        }

        // Caller must hold the exclusive lock.
        void insert_locked(const cv::Vec2i& key, std::shared_ptr<GridStore> store) const
        {
            // Only measured when a byte budget is on: residentBytes() walks the
            // store's cells, and the default (entry-cap) mode does not need it.
            const size_t entry_bytes = (byte_budget && store) ? store->residentBytes() : 0;
            auto it = grid_cache.find(key);
            if (it != grid_cache.end()) {
                cache_bytes -= std::min(cache_bytes, it->second.bytes);
                it->second.grid_store = std::move(store);
                it->second.generation = ++generation_counter;
                it->second.bytes = entry_bytes;
                cache_bytes += entry_bytes;
                return;
            }
            cache_keys.push_back(key);
            grid_cache.emplace(key,
                CacheEntry{std::move(store), ++generation_counter, cache_keys.size() - 1,
                           entry_bytes});
            cache_bytes += entry_bytes;
        }

        // Caller must hold the exclusive lock. Recompute every entry's charge
        // from the store itself (its decoded-polyline cache grows after
        // insertion). O(entries), so it is called rarely.
        void resync_bytes_locked() const
        {
            inserts_since_resync = 0;
            size_t total = 0;
            for (auto& [k, e] : grid_cache) {
                (void)k;
                e.bytes = e.grid_store ? e.grid_store->residentBytes() : 0;
                total += e.bytes;
            }
            cache_bytes = total;
        }

        // Caller must hold the exclusive lock. Erases in O(1) by swapping the
        // victim's slot with the last key rather than searching cache_keys.
        void erase_locked(std::unordered_map<cv::Vec2i, CacheEntry>::iterator it) const
        {
            cache_bytes -= std::min(cache_bytes, it->second.bytes);
            const size_t slot = it->second.key_slot;
            const cv::Vec2i moved = cache_keys.back();
            cache_keys[slot] = moved;
            cache_keys.pop_back();
            if (moved != it->first) {
                grid_cache.at(moved).key_slot = slot;
            }
            grid_cache.erase(it);
        }

        // Caller must hold the exclusive lock. Same policy as before -- draw
        // eviction_sample_size keys uniformly at random and drop the one with
        // the oldest generation stamp -- but the sample is now drawn straight
        // from cache_keys, so it costs O(sample) instead of O(cache size).
        //
        // Evicting is always safe: the cache is transparent, so a dropped
        // store is reloaded from the same immutable .grid file on the next
        // miss. Eviction policy affects timing, never results.
        bool evict_one_locked() const
        {
            if (cache_keys.empty()) {
                return false;
            }
            std::uniform_int_distribution<size_t> dist(0, cache_keys.size() - 1);
            auto victim = grid_cache.end();
            uint64_t min_generation = std::numeric_limits<uint64_t>::max();
            for (size_t i = 0; i < eviction_sample_size; ++i) {
                auto it = grid_cache.find(cache_keys[dist(eviction_rng)]);
                if (it != grid_cache.end() && it->second.generation < min_generation) {
                    min_generation = it->second.generation;
                    victim = it;
                }
            }
            if (victim == grid_cache.end()) {
                return false;
            }
            erase_locked(victim);
            return true;
        }

        // Caller must hold the exclusive lock. Evict until both the byte
        // budget and the entry backstop hold. Always keeps at least one entry.
        void evict_to_budget_locked() const
        {
            while (((byte_budget && cache_bytes > max_cache_bytes)
                    || grid_cache.size() > max_cache_entries)
                   && grid_cache.size() > 1) {
                if (!evict_one_locked()) {
                    break;
                }
            }
        }

        NormalGridVolume::CacheStats cacheStats() const {
            NormalGridVolume::CacheStats stats;
            stats.gridHits = cache_hits.load(std::memory_order_relaxed);
            stats.gridMisses = cache_misses.load(std::memory_order_relaxed);

            std::shared_lock<std::shared_mutex> lock(mutex);
            stats.liveGridEntries = grid_cache.size();
            for (const auto& [key, entry] : grid_cache) {
                (void)key;
                if (!entry.grid_store) {
                    continue;
                }
                const auto grid_stats = entry.grid_store->cacheStats();
                stats.decodedPathHits += grid_stats.decodedPathHits;
                stats.decodedPathMisses += grid_stats.decodedPathMisses;
                stats.decodedPathEvictions += grid_stats.decodedPathEvictions;
                stats.decodedPathEntries += grid_stats.decodedPathEntries;
                stats.decodedPathBytes += grid_stats.decodedPathBytes;
            }
            return stats;
        }

        void resetCacheStats() const {
            cache_hits.store(0, std::memory_order_relaxed);
            cache_misses.store(0, std::memory_order_relaxed);
            last_stat_time = std::chrono::steady_clock::now();

            std::shared_lock<std::shared_mutex> lock(mutex);
            for (const auto& [key, entry] : grid_cache) {
                (void)key;
                if (entry.grid_store) {
                    entry.grid_store->resetCacheStats();
                }
            }
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
                    if (hit_rate < 99.0)
                        std::cout << "[GridStore Cache] Hitrate Warning Triggered: Hits: " << hits << ", Misses: " << misses << ", Total: " << total << ", Hit Rate: " << std::fixed << std::setprecision(2) << hit_rate << "%" << std::endl;
                    last_stat_time = now;
                }
            }
        }
    };

    NormalGridVolume::NormalGridVolume(const std::string& path)
        : NormalGridVolume(path, 0) {}

    NormalGridVolume::NormalGridVolume(const std::string& path, int level)
        : pimpl_(std::make_unique<pimpl>(path, level)) {}
 
    std::optional<NormalGridVolume::GridQueryResult> NormalGridVolume::query(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query(point, plane_idx);
    }

    std::shared_ptr<const GridStore> NormalGridVolume::query_nearest(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query_nearest(point, plane_idx);
    }

    std::shared_ptr<const GridStore> NormalGridVolume::get_grid(int plane_idx, int slice_idx) const {
        return pimpl_->get_grid(plane_idx, slice_idx);
    }

    NormalGridVolume::CacheStats NormalGridVolume::cacheStats() const {
        return pimpl_->cacheStats();
    }

    void NormalGridVolume::resetCacheStats() const {
        pimpl_->resetCacheStats();
    }

    double NormalGridVolume::coordinateScale() const {
        return pimpl_->coordinate_scale;
    }

    double NormalGridVolume::outputSpiralStep() const {
        return pimpl_->output_spiral_step;
    }

    int NormalGridVolume::level() const {
        return pimpl_->selected_level;
    }

    NormalGridVolume::~NormalGridVolume() = default;
    NormalGridVolume::NormalGridVolume(NormalGridVolume&&) noexcept = default;
    NormalGridVolume& NormalGridVolume::operator=(NormalGridVolume&&) noexcept = default;
    const utils::Json& NormalGridVolume::metadata() const {
        return pimpl_->metadata;
    }
} // namespace vc::core::util
