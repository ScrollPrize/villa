#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/HashFunctions.hpp"

#include <nlohmann/json.hpp>
#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/core/util/RemoteGridFile.hpp"

#include <future>

#include <vc_reader/source_reader.hpp>
#include <vc_reader/zarr.hpp>

 #include <filesystem>
 #include <fstream>
#include <algorithm>
#include <array>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <deque>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <system_error>
#include <thread>
#include <unordered_set>

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

    // A byte count from the environment, or nullopt when the variable is unset
    // or not a number. Unlike env_size, 0 is a value here: it is how the byte
    // budget is switched off.
    static std::optional<size_t> env_bytes(const char* name)
    {
        const char* e = std::getenv(name);
        if (!e || !*e) {
            return std::nullopt;
        }
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        if (end == e || v < 0) {
            return std::nullopt;
        }
        return static_cast<size_t>(v);
    }

     struct NormalGridVolume::pimpl {
         std::string base_path;
         int sparse_volume;
         nlohmann::json metadata;
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
         // Budget in BYTES instead of the entry cap, on by default.
         //
         // With VC_GRID_CACHE_BYTES unset the cache is bounded by
         // kDefaultCacheBytes of accounted bytes (but see budget_floor_entries),
         // and the entry cap relaxes to a 65,536 backstop unless
         // VC_GRID_CACHE_ENTRIES says otherwise. VC_GRID_CACHE_BYTES=<bytes>
         // sets another budget. VC_GRID_CACHE_BYTES=0 switches the budget off:
         // the cache then holds at most 512 entries, as it did before, and none
         // of the byte accounting below runs.
         //
         // An entry count is the wrong unit here. A cached GridStore costs the
         // .grid file it maps -- whose size varies by an order of magnitude
         // between slices -- plus up to 2 MiB of decoded seglists and the
         // bucket indices it has decoded, so any entry cap is simultaneously
         // too loose (a worst case of entries x 2 MiB and more) and too tight
         // (it evicts small stores that cost almost nothing to keep). A byte
         // budget bounds the cache directly and does not need retuning when
         // the grids change.
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
         // The accounting charges each store its whole mapped file plus what
         // it has put on the heap: the polylines and the bucket indices it has
         // decoded, each counted as it is allocated (GridStore::residentBytes).
         // The file is charged in full although only touched pages are
         // resident, and on the stores measured it is nearly all of the charge
         // (PHerc1447 at eight threads: 95.6 % at the end of a 1 cm^2 trace,
         // against 3.5 % for the polylines and 0.9 % for the bucket indices,
         // and 96-100 % through a 47 cm^2 trace held at the default budget), so
         // the RSS cost stays well under the budget there (1.8 GB at a 4 GiB
         // budget in the sweep above). That is a property of those stores,
         // not of the accounting: a store keeps about 110 bytes of heap for
         // every bucket it was asked for, a pass over all buckets of a sparse
         // slice can hold many times its file, and the charge then follows
         // the heap. Mappings are file-backed, so concurrent tracers working
         // the same grids share those pages through the page cache.
         //
         // The default of 2 GiB keeps the resident cost near the ~1 GiB the
         // 512-entry cap was sized for (on that store the charge is about 2x
         // what is resident, see above) and holds the working set of a seeded
         // trace of up to about 1 cm^2 on an 8.64 um store; larger traces
         // evict, as before (a 200-generation patch, VC3D's default, stops
         // gaining at about 8 GiB at eight threads).
         // It is kept small and fixed because batch runs start one tracer per
         // core. On a store of large slices the charge is much further above
         // what is resident (about 9x on PHercParis4 at 2.4 um, where a trace
         // reads a small window of slices of up to 5 MiB), so such a store
         // wants a larger VC_GRID_CACHE_BYTES.
         static constexpr size_t kDefaultCacheBytes = size_t{2} << 30;
         // The entry cap the cache had before the byte budget.
         static constexpr size_t kLegacyCacheEntries = 512;
         std::optional<size_t> env_cache_bytes = env_bytes("VC_GRID_CACHE_BYTES");
         // 0 = byte budget off.
         size_t max_cache_bytes = env_cache_bytes.value_or(kDefaultCacheBytes);
         bool byte_budget = max_cache_bytes > 0;
         // The built-in budget never evicts below the old cap. Slices of a
         // large volume can be big enough (PHercParis4's 2.4 um grids run to
         // about 5 MiB each) that fewer than 512 of them fit in the default,
         // and the default must not hold less than the cache used to. An
         // explicit VC_GRID_CACHE_BYTES is taken at its word.
         size_t budget_floor_entries = env_cache_bytes ? 0 : kLegacyCacheEntries;
         // Entry cap. 512 without a byte budget. With one it becomes a backstop so
         // a volume of unusually tiny slices cannot grow the map without bound.
         size_t max_cache_entries =
             env_size("VC_GRID_CACHE_ENTRIES", byte_budget ? 65536 : kLegacyCacheEntries);
         mutable size_t cache_bytes = 0;
         // A store's caches (decoded polylines, bucket indices) keep growing
         // AFTER it is inserted (on hits), so the running total drifts low. It
         // is therefore re-summed from the stores themselves, both every
         // kResyncEveryInserts insertions and every kResyncEveryHits hits, so a
         // working set that is fully resident (no more insertions) is still
         // held to the budget.
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

         // Remote streaming (kNormalGridsRemoteMarker in base_path): base_path is
         // remote_url's place under a cache root, <root>/<scheme>/<host>/<path>, and
         // files missing there are downloaded by vc_reader's file cache under the
         // root's disk budget and free-space floor (VC3D's cache settings). A grid
         // the store does not have is remembered in grid_cache as a null entry.
         bool remote = false;
         std::string remote_url;
         // Shared with the range fetchers of grids, which may outlive this object.
         mutable std::shared_ptr<vcr::sources::EncodedFiles> files;
         mutable std::atomic<uint64_t> fetch_failures{0};
         mutable std::mutex fetch_error_mutex;
         mutable std::string last_fetch_error;
         // Read-ahead: when a slice has to be fetched, its neighbours on the same plane (the
         // tracer's frontier reaches them next) are opened too, each as one plain blocking
         // open on its own background thread. No queue: a neighbour already cached, already
         // in flight, or over the in-flight limit is simply not started; the file cache's
         // per-file lock makes a tracer thread that needs one of them wait for that download.
         int readahead_radius = 4;
         int readahead_limit = 16;
         mutable std::mutex readahead_mutex;
         mutable std::condition_variable readahead_cv;
         mutable std::unordered_set<uint64_t> readahead_inflight;
         mutable int readahead_running = 0;

         // Range reads (VC_NORMAL_GRID_RANGE_READS=1, streaming stores only): a grid
         // file that is not cached whole is read a part at a time instead of being
         // downloaded (EncodedFiles::readRanges, with the store's access), into a
         // sparse mirror beside where the whole file would go (RemoteGridFile), or
         // into memory when the disk is under the cache's free-space floor. Queries
         // fetch the tiles of buckets they need; read-ahead fetches the same tile on
         // the neighbouring slices. A file the range reader cannot open (a store
         // without range support, a file that is not GridStore v3) is downloaded
         // whole, as without the option.
         bool range_reads = false;
         uint64_t range_min_free_bytes = 0;
         // Under readahead_mutex: (plane, slice, tile) keys read-ahead was started
         // from, and those it has prefetched.
         mutable std::unordered_set<uint64_t> range_ahead_triggered, range_ahead_fetched;
         static constexpr float range_ahead_margin = 144.0f;  // query radius 80 + one bucket
         // Each plane's file layout, learnt from its first file (under readahead_mutex).
         mutable std::array<std::optional<RemoteGridFile::Layout>, 3> plane_layouts;
         // Range-read opens in flight (range_file). Under readahead_mutex.
         mutable std::unordered_map<uint64_t, std::shared_future<std::optional<std::shared_ptr<RemoteGridFile>>>> range_opening;
         // Mirrors opened so far, by slice (null: the store does not have it), so
         // read-ahead warms a mirror without putting grids in grid_cache. Only the
         // tracer's own lookups add to grid_cache, as with whole files, so what it
         // holds and evicts does not depend on read-ahead timing (the tracer's
         // path caches follow grid objects, and eviction order changes results).
         // Holds no descriptors. Capped by count, and with grid_cache by the
         // bytes files hold in memory (VC_NORMAL_GRID_RANGE_MEMORY_BYTES,
         // default 1 GiB; see enforce_range_memory_cap): a dropped file is
         // replayed from its pack when opened again, with nothing fetched.
         // Under readahead_mutex.
         mutable std::unordered_map<uint64_t, std::shared_ptr<RemoteGridFile>> range_files;
         mutable std::deque<uint64_t> range_files_order;
         static constexpr size_t kRangeFilesCap = 8192;
         uint64_t range_memory_cap = uint64_t(1) << 30;
         // Every file still alive (held by a grid in grid_cache, say), so a slice
         // dropped from range_files is never opened a second time while its first
         // file lives: tiles read ahead into one must be seen by the other.
         // Under readahead_mutex.
         mutable std::unordered_map<uint64_t, std::weak_ptr<RemoteGridFile>> range_live;
         mutable std::string last_range_fallback;  // under fetch_error_mutex
         mutable std::atomic<uint64_t> range_over_cap_checks{0}, range_trimmed_at{0};
         mutable std::mutex range_trim_mutex;

         pimpl(const std::string& path, int requested_level) : base_path(path) {
            load_remote_marker();
            if (!ensure_local_file("metadata.json")) {
                throw std::runtime_error(missing_file_error("metadata.json"));
            }
            metadata = nlohmann::json::parse(std::ifstream(fs::path(base_path) / "metadata.json"));
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

                nlohmann::json level_metadata;
                const std::string level_metadata_name =
                    "metadata.level" + std::to_string(selected_level) + ".json";
                const fs::path level_metadata_path = fs::path(base_path) / level_metadata_name;
                const bool level_present = ensure_local_file(level_metadata_name);
                if (level_present && fs::exists(level_metadata_path)) {
                    level_metadata = nlohmann::json::parse(std::ifstream(level_metadata_path));
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

        }

        ~pimpl() {
            std::unique_lock<std::mutex> lock(readahead_mutex);
            readahead_cv.wait(lock, [&] { return readahead_running == 0; });
            if (range_reads && std::getenv("VC_NORMAL_GRID_RANGE_STATS")) {
                const auto st = RemoteGridFile::globalStats();
                std::cerr << "range-read stats: opens " << st.opens << ", requests " << st.requests << ", bytes " << st.bytes
                          << ", tiles loaded " << st.tilesLoaded << " (foreground " << st.fgLoads << ", read-ahead " << st.bgLoads
                          << "), foreground wait " << st.fgWaitNs / 1e9 << " s, failures " << st.failures
                          << ", in memory " << st.inMemory << ", tiles used by queries " << st.tilesUsed << std::endl;
            }
        }

        static uint64_t slice_key(int plane_idx, int slice_idx) {
            return (static_cast<uint64_t>(static_cast<uint32_t>(plane_idx)) << 32) | static_cast<uint32_t>(slice_idx);
        }

        // Runs `work` on a detached thread. The caller holds readahead_mutex and
        // has counted the worker in readahead_running; when no thread can be
        // started (std::system_error, e.g. EAGAIN), the count is given back so
        // the destructor does not wait for a worker that never ran, and false
        // is returned: read-ahead is skipped and queries fetch what they need.
        bool start_worker(std::function<void()> work) const {
            try {
                std::thread(std::move(work)).detach();
                return true;
            } catch (const std::system_error& e) {
                --readahead_running;
                readahead_cv.notify_all();
                static std::once_flag reported;
                std::call_once(reported, [&] {
                    std::cerr << "normal-grid read-ahead: cannot start a thread (" << e.what()
                              << "); reading without read-ahead" << std::endl;
                });
                return false;
            }
        }

        // Start background opens of the slices around (plane_idx, slice_idx) that are not cached yet.
        void read_ahead(int plane_idx, int slice_idx) const {
            std::error_code ec;
            std::lock_guard<std::mutex> lock(readahead_mutex);
            for (int distance = 1; distance <= readahead_radius; ++distance) {
                for (const int sign : {1, -1}) {
                    const int slice = slice_idx + sign * distance * sparse_volume;
                    if (slice < 0 || readahead_running >= readahead_limit) continue;
                    const uint64_t key = slice_key(plane_idx, slice);
                    if (readahead_inflight.count(key)) continue;
                    const std::string rel = relative_grid_path(plane_idx, slice);
                    if (fs::exists(fs::path(base_path) / rel, ec)) continue;
                    {
                        std::shared_lock<std::shared_mutex> cache(mutex);
                        if (grid_cache.count(cv::Vec2i(plane_idx, slice))) continue;  // known, present or absent
                    }
                    readahead_inflight.insert(key);
                    ++readahead_running;
                    if (!start_worker([this, key, rel, plane_idx, slice] {
                        try {
                            (void)files->open(remote_url + "/" + rel);
                        } catch (const vcr::zarr::NotFound&) {
                            // The store does not have it: remember that, so nobody asks again.
                            std::unique_lock<std::shared_mutex> cache(mutex);
                            if (!grid_cache.count(cv::Vec2i(plane_idx, slice))) insert_locked(cv::Vec2i(plane_idx, slice), nullptr);
                        } catch (const std::exception&) {
                            // The tracer's own open reports failures.
                        }
                        std::lock_guard<std::mutex> done(readahead_mutex);
                        readahead_inflight.erase(key);
                        --readahead_running;
                        readahead_cv.notify_all();
                    })) {
                        readahead_inflight.erase(key);  // the tracer's own read fetches it
                        return;
                    }
                }
            }
        }

        // The cache root under which `dir` is `url`'s place, <root>/<scheme>/<host>/<path>
        // (vcr::zarr::decodedCachePath), so that files vc_reader caches for url/<rel> land at dir/<rel>.
        static fs::path streamingCacheRoot(const fs::path& dir, const std::string& url) {
            const fs::path place = vcr::zarr::decodedCachePath(fs::path(), url);
            auto root = fs::absolute(dir).lexically_normal();
            if (root.filename().empty()) root = root.parent_path();
            const std::vector<fs::path> parts(place.begin(), place.end());
            for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
                if (root.filename() != *it) {
                    throw std::runtime_error("streaming normal grids for " + url + " must be cached at <root>/" +
                                             place.generic_string() + ", not at " + dir.string());
                }
                root = root.parent_path();
            }
            return root;
        }

        void load_remote_marker() {
            remote_url = normalGridsRemoteUrl(base_path).value_or(std::string());
            remote = !remote_url.empty();
            if (!remote) {
                return;
            }
            const fs::path root = streamingCacheRoot(base_path, remote_url);
            range_min_free_bytes = static_cast<uint64_t>(
                std::max<std::int64_t>(0, vc::settings::remoteOpenOptions(root).cacheMinFreeBytes.value_or(0)));
            const char* rr = std::getenv("VC_NORMAL_GRID_RANGE_READS");
            range_reads = rr && std::string(rr) == "1";
#ifdef _WIN32
            if (range_reads) {
                std::cerr << "VC_NORMAL_GRID_RANGE_READS is not available on Windows; normal grids are read whole"
                          << std::endl;
                range_reads = false;
            }
#endif
            if (const auto cap = env_bytes("VC_NORMAL_GRID_RANGE_MEMORY_BYTES")) range_memory_cap = *cap;
            // Range reads are many small requests, so latency, not bandwidth, bounds
            // them: they read further ahead with more in flight than whole files
            // (measured on a 150 ms, 6 MB/s link: 64 in flight took 1.6-1.9x as
            // long as 128). vc_reader shares connections, DNS answers and TLS
            // sessions across its handles, so this many do not each reconnect.
            // VC3D's download parallelism, when set, still decides the connections.
            int connections = 0;
            if (range_reads) {
                readahead_radius = static_cast<int>(env_size("VC_NORMAL_GRID_RANGE_AHEAD", 12));
                readahead_limit = static_cast<int>(env_size("VC_NORMAL_GRID_RANGE_THREADS", 128));
#ifndef _WIN32
                // Each request in flight holds a few descriptors (its connection and
                // libcurl's wakeup pair); past the open-file limit, connects and even
                // DNS lookups fail. Stay within an eighth of it (32 at macOS's
                // default 256; VC3D raises its own limit at startup).
                rlimit files{};
                if (::getrlimit(RLIMIT_NOFILE, &files) == 0 && files.rlim_cur != RLIM_INFINITY) {
                    const int fit = static_cast<int>(std::min<rlim_t>(files.rlim_cur / 8, 1 << 20));
                    if (fit < readahead_limit) {
                        readahead_limit = std::max(8, fit);
                        std::cerr << "normal-grid range reads: " << readahead_limit << " in flight (open-file limit "
                                  << files.rlim_cur << "; raise it, e.g. ulimit -n 10240, for faster reads)" << std::endl;
                    }
                }
#endif
                connections = vc::settings::remoteDownloadParallelism().value_or(std::max(readahead_limit, 16));
            }
            files = std::make_shared<vcr::sources::EncodedFiles>([root, connections](const std::string& location) {
                auto options = vc::settings::remoteOpenOptions(root);
                options.access = vcr::zarr::resolveAccess(location);
                if (connections) options.connections = connections;
                return options;
            });
        }

        // Build an intentional, actionable error message for a normal-grid file
        // that could not be made available locally. Distinguishes the remote-
        // streaming failure (fetch from remote_url failed) from a plain missing
        // local file so the caller can surface it instead of the generic
        // "Cannot open" thrown by Json::parse_file on a still-absent file.
        std::string missing_file_error(const std::string& rel) const {
            if (remote) {
                std::lock_guard<std::mutex> lock(fetch_error_mutex);
                return "Failed to fetch remote normal-grid file '" + rel +
                       "' from " + remote_url +
                       " (cached under " + base_path + ")" +
                       (last_fetch_error.empty() ? std::string(": the store does not have it")
                                                 : ": " + last_fetch_error);
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

        // Make base_path/rel available: present for a plain store; for a streaming store,
        // cached already or downloaded now by vc_reader (one download per file, concurrent
        // misses wait for it; budget and free-space floor apply). `pinned` keeps the
        // streamed file's descriptor, so eviction cannot take it before it is mapped.
        // False when the store does not have the file, or when fetching it failed: that
        // is counted, kept in last_fetch_error and reported on stderr once per message.
        bool ensure_local_file(const std::string& rel,
                               std::optional<vcr::sources::EncodedFiles::Opened>* pinned = nullptr) const {
            if (!remote) {
                std::error_code ec;
                return fs::exists(fs::path(base_path) / rel, ec);
            }
            try {
                auto opened = files->open(remote_url + "/" + rel);
                if (pinned) pinned->emplace(std::move(opened));
                return true;
            } catch (const vcr::zarr::NotFound&) {
                return false;
            } catch (const std::exception& e) {
                fetch_failures.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(fetch_error_mutex);
                if (last_fetch_error != e.what()) {
                    last_fetch_error = e.what();
                    std::cerr << "normal-grid fetch failed for " << rel << ": " << e.what() << std::endl;
                }
                return false;
            }
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

            if (range_reads) {
                range_read_ahead(point, plane_idx, slice_idx);
                const cv::Point2f p = in_plane(point, plane_idx);
                const int m = static_cast<int>(range_ahead_margin);
                return get_grid(plane_idx, slice_idx, cv::Rect(int(p.x) - m, int(p.y) - m, 2 * m, 2 * m));
            }
            return get_grid(plane_idx, slice_idx);
        }

        // In-plane coordinates of `point` on plane_idx's slices, in grid pixels.
        cv::Point2f in_plane(const cv::Point3f& point, int plane_idx) const {
            const float s = static_cast<float>(coordinate_scale);
            switch (plane_idx) {
                case 0: return {point.x * s, point.y * s};
                case 1: return {point.x * s, point.z * s};
                default: return {point.y * s, point.z * s};
            }
        }

        // Range reads: fetch, on background threads, the tiles of the region the
        // tracer queries around `point`, on this slice and the readahead_radius
        // slices on each side (the region moves little from slice to slice).
        // Started once per 128-px cell, so a tile is fetched while the tracer
        // nears it, and each tile of a slice is fetched once. (Measured: a wider
        // region costs more bytes than it saves waiting; a narrower one, the
        // query radius alone, saves few bytes and leaves the tracer waiting.)
        void range_read_ahead(const cv::Point3f& point, int plane_idx, int slice_idx) const {
            const cv::Point2f p = in_plane(point, plane_idx);
            if (p.x < 0 || p.y < 0) return;
            constexpr int cell_px = 128;
            auto key_of = [plane_idx](int slice, int a, int b) {
                return (uint64_t(plane_idx) << 62) ^ (uint64_t(uint32_t(slice)) << 32) ^ (uint64_t(uint32_t(a) & 0xffff) << 16) ^
                       uint64_t(uint32_t(b) & 0xffff);
            };
            const uint64_t here = key_of(slice_idx, int(p.x) / cell_px, int(p.y) / cell_px);
            thread_local uint64_t last_key = ~0ull;
            thread_local const void* last_owner = nullptr;
            if (last_key == here && last_owner == this) return;
            last_key = here;
            last_owner = this;
            std::lock_guard<std::mutex> lock(readahead_mutex);
            if (range_ahead_triggered.count(here)) return;
            // Tiles in the plane's bucket grid (origin bx, by), so a tile here is a tile of the file.
            int ox = 0, oy = 0, tile_px = RemoteGridFile::kTileBuckets * 64;
            if (const auto& layout = plane_layouts[plane_idx]) {
                ox = int(layout->bx), oy = int(layout->by), tile_px = RemoteGridFile::kTileBuckets * int(layout->cell);
            }
            const float reach = range_ahead_margin;
            const int tx0 = std::max(0, int(std::floor((p.x - reach - ox) / tile_px)));
            const int tx1 = std::max(0, int(std::floor((p.x + reach - ox) / tile_px)));
            const int ty0 = std::max(0, int(std::floor((p.y - reach - oy) / tile_px)));
            const int ty1 = std::max(0, int(std::floor((p.y + reach - oy) / tile_px)));
            const int ptx = int(std::floor((p.x - ox) / tile_px)), pty = int(std::floor((p.y - oy) / tile_px));
            std::vector<std::pair<int, int>> tiles{{ptx, pty}};  // the point's own tile first
            for (int ty = ty0; ty <= ty1; ++ty)
                for (int tx = tx0; tx <= tx1; ++tx)
                    if (tx != ptx || ty != pty) tiles.push_back({tx, ty});
            bool complete = true;  // every tile started or fetched already
            for (int distance = 0; distance <= readahead_radius; ++distance) {
                for (const int sign : {1, -1}) {
                    if (distance == 0 && sign < 0) continue;
                    const int slice = slice_idx + sign * distance * sparse_volume;
                    if (slice < 0) continue;
                    std::error_code exists_ec;  // read whole already: nothing to fetch
                    if (fs::exists(fs::path(base_path) / relative_grid_path(plane_idx, slice), exists_ec)) continue;
                    for (const auto& [tx, ty] : tiles) {
                        if (tx < 0 || ty < 0) continue;
                        const uint64_t key = key_of(slice, tx, ty);
                        if (range_ahead_fetched.count(key)) continue;
                        if (readahead_running >= readahead_limit) {
                            complete = false;  // over the limit: a later query here tries again
                            continue;
                        }
                        range_ahead_fetched.insert(key);
                        ++readahead_running;
                        const cv::Rect rect(ox + tx * tile_px, oy + ty * tile_px, tile_px, tile_px);
                        if (!start_worker([this, plane_idx, slice, rect] {
                            try {
                                std::error_code ec;
                                if (!fs::exists(fs::path(base_path) / relative_grid_path(plane_idx, slice), ec)) {
                                    const auto file = range_file(plane_idx, slice, rect);
                                    if (file && *file) {
                                        const auto b = bucket_rect((*file)->layout(), rect);
                                        (*file)->ensureBuckets(b.x0, b.y0, b.x1, b.y1, /*foreground=*/false);
                                    }
                                    enforce_range_memory_cap();
                                }
                            } catch (const std::exception&) {
                                // The tracer's own query fetches again and reports failures.
                            }
                            std::lock_guard<std::mutex> done(readahead_mutex);
                            --readahead_running;
                            readahead_cv.notify_all();
                        })) {
                            range_ahead_fetched.erase(key);  // a later query here tries again
                            last_key = ~0ull;
                            return;
                        }
                    }
                }
            }
            if (complete) range_ahead_triggered.insert(here);
            else last_key = ~0ull;
        }

        // Range reads: the mirror of the grid at (plane, slice), opened once and
        // kept in range_files. nullopt: read it whole instead; a null mirror: the
        // store does not have it. Concurrent opens of one slice (the tracer and
        // read-ahead) share one opener. `first_px`: the region the first query reads.
        using RangeOpen = std::optional<std::shared_ptr<RemoteGridFile>>;
        RangeOpen range_file(int plane_idx, int slice_idx, std::optional<cv::Rect> first_px) const {
            const uint64_t open_key = slice_key(plane_idx, slice_idx);
            std::promise<RangeOpen> promise;
            std::shared_future<RangeOpen> waiting;
            {
                std::lock_guard<std::mutex> lock(readahead_mutex);
                if (auto known = range_files.find(open_key); known != range_files.end()) return known->second;
                if (auto live = range_live.find(open_key); live != range_live.end()) {
                    if (auto file = live->second.lock()) {
                        remember_range_file_locked(open_key, file);
                        return RangeOpen(file);
                    }
                    range_live.erase(live);
                }
                auto it = range_opening.find(open_key);
                if (it != range_opening.end()) waiting = it->second;
                else range_opening.emplace(open_key, promise.get_future().share());
            }
            if (waiting.valid()) return waiting.get();
            RangeOpen result;
            try {
                result = open_range_now(plane_idx, relative_grid_path(plane_idx, slice_idx), first_px);
            } catch (...) {
                // nullopt: read whole
            }
            promise.set_value(result);
            std::lock_guard<std::mutex> lock(readahead_mutex);
            range_opening.erase(open_key);
            if (result) {
                if (*result) range_live[open_key] = *result;
                remember_range_file_locked(open_key, *result);
            }
            return result;
        }

        // Caller holds readahead_mutex.
        void remember_range_file_locked(uint64_t open_key, const std::shared_ptr<RemoteGridFile>& file) const {
            if (range_files.emplace(open_key, file).second) {
                range_files_order.push_back(open_key);
                uint64_t held = 0;
                for (const auto& [key, open] : range_files)
                    if (open) held += open->residentBytes();
                while (range_files_order.size() > 1 &&
                       (range_files_order.size() > kRangeFilesCap || held > range_memory_cap)) {
                    const auto oldest = range_files.find(range_files_order.front());
                    if (oldest->second) held -= std::min(held, oldest->second->residentBytes());
                    range_files.erase(oldest);
                    range_files_order.pop_front();
                }
            }
            if (range_live.size() > 2 * range_files.size() + 1024)
                std::erase_if(range_live, [](const auto& entry) { return entry.second.expired(); });
        }

        RangeOpen open_range_now(int plane_idx, const std::string& rel, std::optional<cv::Rect> first_px) const {
            const std::string grid_path = (fs::path(base_path) / rel).string();
            std::optional<RemoteGridFile::Layout> layout;
            {
                std::lock_guard<std::mutex> lock(readahead_mutex);
                layout = plane_layouts[plane_idx];
            }
            const std::string location = remote_url + "/" + rel;
            RemoteGridFile::Options options;
            options.min_free_bytes = range_min_free_bytes;
            options.charge = [files = files, location](const std::string& path, uint64_t bytes) {
                return files->chargeFile(location, path, static_cast<std::int64_t>(bytes));
            };
            if (layout) {
                options.layout = &*layout;
                if (first_px && layout->cell) options.first = bucket_rect(*layout, *first_px);
            }
            auto fetcher = [files = files, location](const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                                     const std::string& version) {
                try {
                    auto read = files->readRanges(location, ranges, version);
                    return RemoteGridFile::Fetched{std::move(read.parts), std::move(read.version)};
                } catch (const vcr::zarr::NotFound& e) {
                    throw RemoteGridFile::NotFound(e.what());
                } catch (const vcr::sources::EncodedFiles::Changed& e) {
                    throw RemoteGridFile::Changed(e.what());
                }
            };
            try {
                auto remote_file = std::make_shared<RemoteGridFile>(location, grid_path, std::move(fetcher), options);
                if (!layout) {
                    std::lock_guard<std::mutex> lock(readahead_mutex);
                    if (!plane_layouts[plane_idx]) plane_layouts[plane_idx] = remote_file->layout();
                }
                return remote_file;
            } catch (const RemoteGridFile::NotFound&) {
                return std::shared_ptr<RemoteGridFile>();
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(fetch_error_mutex);
                const std::string message = "normal-grid range read of " + rel + " unavailable, downloading it whole: " + e.what();
                if (message != last_range_fallback) {
                    last_range_fallback = message;
                    std::cerr << message << std::endl;
                }
                return std::nullopt;
            }
        }

        // Grid pixels to the buckets of a file with `layout`, inclusive.
        static RemoteGridFile::BucketRect bucket_rect(const RemoteGridFile::Layout& layout, const cv::Rect& px) {
            const int c = static_cast<int>(layout.cell);
            auto floor_div = [c](int v) { return v >= 0 ? v / c : -((-v + c - 1) / c); };
            return {floor_div(px.x - int(layout.bx)), floor_div(px.y - int(layout.by)),
                    floor_div(px.br().x - 1 - int(layout.bx)), floor_div(px.br().y - 1 - int(layout.by))};
        }

        // Range reads keep fetched bytes in anonymous memory, and files grow
        // after they are opened (queries fetch more tiles; a whole-slice read
        // fetches all of it). So the cap is checked as they grow: one load of
        // a counter per query, and when over the cap, now and then a trim.
        // Trims when the total has grown by a sixteenth of the cap since the
        // last trim (so it overshoots by that and one file's growth at most),
        // and every 64th query while it stays over (queries let go of grids).
        void enforce_range_memory_cap() const {
            if (!range_reads) return;
            const uint64_t total = RemoteGridFile::residentTotal();
            if (total <= range_memory_cap) return;
            const bool grown = total >= range_trimmed_at.load(std::memory_order_relaxed) +
                                            std::max<uint64_t>(range_memory_cap / 16, 1);
            if (!grown && range_over_cap_checks.fetch_add(1, std::memory_order_relaxed) % 64 != 0) return;
            std::unique_lock<std::mutex> trimming(range_trim_mutex, std::try_to_lock);
            if (!trimming) return;  // another thread is trimming
            trim_range_memory();
            range_trimmed_at.store(RemoteGridFile::residentTotal(), std::memory_order_relaxed);
        }

        // Drops, oldest first, the read-ahead files nothing else holds, then
        // the range-read grids grid_cache keeps, until the files in memory fit
        // the cap. A query still holding a grid keeps it (and its file) until
        // it lets go; a dropped slice is replayed from its pack when needed.
        void trim_range_memory() const {
            auto over = [&] { return RemoteGridFile::residentTotal() > range_memory_cap; };
            auto drop_unheld_files = [&] {
                std::lock_guard<std::mutex> lock(readahead_mutex);
                for (auto it = range_files_order.begin(); it != range_files_order.end() && over();) {
                    const auto file = range_files.find(*it);
                    if (file == range_files.end()) {
                        it = range_files_order.erase(it);
                    } else if (file->second && file->second.use_count() == 1) {
                        range_files.erase(file);
                        it = range_files_order.erase(it);
                    } else {
                        ++it;
                    }
                }
            };
            drop_unheld_files();
            if (!over()) return;
            {
                std::unique_lock<std::shared_mutex> lock(mutex);
                std::vector<std::pair<uint64_t, cv::Vec2i>> held;  // (generation, key)
                for (const auto& [key, entry] : grid_cache)
                    if (entry.grid_store && entry.grid_store->remoteResidentBytes() > 0)
                        held.push_back({entry.generation, key});
                std::sort(held.begin(), held.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                // Dropping a grid frees its file only once range_files lets go
                // of it too, so drop until what may be freed would fit.
                uint64_t freeable = 0;
                const uint64_t total = RemoteGridFile::residentTotal();
                const uint64_t excess = total > range_memory_cap ? total - range_memory_cap : 0;
                for (const auto& [generation, key] : held) {
                    if (freeable >= excess) break;
                    const auto it = grid_cache.find(key);
                    if (it == grid_cache.end()) continue;
                    freeable += it->second.grid_store->remoteResidentBytes();
                    erase_locked(it);
                }
            }
            drop_unheld_files();
        }

        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx,
                                                  std::optional<cv::Rect> first_px = std::nullopt) const {
            enforce_range_memory_cap();
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

            std::shared_ptr<GridStore> grid_store;
            std::optional<vcr::sources::EncodedFiles::Opened> pinned;
            std::error_code exists_ec;
            if (range_reads && !fs::exists(fs::path(base_path) / rel, exists_ec)) {
                if (auto opened = range_file(plane_idx, slice_idx, first_px)) {
                    if (!*opened) {  // the store does not have it
                        std::unique_lock<std::shared_mutex> lock(mutex);
                        insert_locked(key, nullptr);
                        return nullptr;
                    }
                    // Should a region fail to arrive, the store reads this
                    // slice whole the usual way (by value: the store can
                    // outlive this volume).
                    auto whole_file = [files = files, url = remote_url + "/" + rel,
                                       grid_path]() -> std::shared_ptr<const GridStore> {
                        try {
                            auto pinned = files->open(url);
                            return std::make_shared<GridStore>(grid_path);
                        } catch (const vcr::zarr::NotFound&) {
                            return nullptr;
                        }
                    };
                    grid_store = std::make_shared<GridStore>(std::move(*opened), std::move(whole_file));
                }
            }
            if (!grid_store) {
                if (remote && !range_reads) {
                    std::error_code ec;
                    if (!fs::exists(fs::path(base_path) / rel, ec)) read_ahead(plane_idx, slice_idx);
                }
                if (!ensure_local_file(rel, &pinned)) {
                    std::unique_lock<std::shared_mutex> lock(mutex);
                    insert_locked(key, nullptr);
                    return nullptr;
                }
                grid_store = std::make_shared<GridStore>(grid_path);
            }

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
            // store's cells, and the entry-cap mode does not need it.
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
        // from the store itself (its decoded-polyline and bucket-index caches
        // grow after insertion; each store keeps running counts, so this is
        // one sum over the entries). O(entries), so it is called rarely.
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
        // budget and the entry backstop hold. Always keeps at least one entry,
        // and the byte budget alone never goes below budget_floor_entries.
        void evict_to_budget_locked() const
        {
            while (((byte_budget && cache_bytes > max_cache_bytes
                     && grid_cache.size() > budget_floor_entries)
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
            stats.fetchFailures = fetch_failures.load(std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(fetch_error_mutex);
                stats.lastFetchError = last_fetch_error;
            }

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
                stats.bucketIndexEntries += grid_stats.bucketIndexEntries;
                stats.bucketIndexBytes += grid_stats.bucketIndexBytes;
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
    const nlohmann::json& NormalGridVolume::metadata() const {
        return pimpl_->metadata;
    }
std::optional<std::string> normalGridsRemoteUrl(const std::filesystem::path& dir)
{
    const fs::path marker = dir / kNormalGridsRemoteMarker;
    std::error_code ec;
    if (!fs::is_regular_file(marker, ec)) {
        return std::nullopt;
    }
    std::string url;
    try {
        url = nlohmann::json::parse(std::ifstream(marker)).value("url", std::string());
    } catch (const std::exception& e) {
        std::cerr << "Ignoring unreadable " << marker << ": " << e.what() << std::endl;
        return std::nullopt;
    }
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url;
}

bool writeNormalGridsRemoteMarker(const std::filesystem::path& dir, const std::string& url)
{
    if (normalGridsRemoteUrl(dir) == url) {
        return true;
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        return false;
    }
    auto marker = nlohmann::json::object();
    marker["url"] = url;
    std::ofstream out(dir / kNormalGridsRemoteMarker, std::ios::trunc);
    out << marker.dump(2);
    out.close();
    return static_cast<bool>(out);
}

bool hasNormalGridFiles(const std::filesystem::path& dir)
{
    std::error_code ec;
    return fs::is_directory(dir / "xy", ec) && fs::is_directory(dir / "xz", ec) &&
           fs::is_directory(dir / "yz", ec) && fs::is_regular_file(dir / "metadata.json", ec);
}

} // namespace vc::core::util
