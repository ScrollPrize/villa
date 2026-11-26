#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vc::core::util {

// A GridStore v3 file read from a remote store a part at a time.
//
// The file is laid out in anonymous memory of its full size (pages that are
// never written take no memory), so GridStore reads it exactly as it reads a
// whole file. Before a query reads a region, ensureBuckets() fetches what that
// region needs with range reads and writes it in place: the bucket index
// rows, the buckets' path offsets and the path records they name. Regions are
// loaded in tiles of kTileBuckets x kTileBuckets buckets.
//
// What was fetched is kept on disk in one compact pack file beside the local
// path (local_path + ".ranges"): a header naming the remote file's size and
// version, then records appended as tiles complete, each with a checksum:
// fetched bytes at an offset, the tiles they complete, "the whole file", or
// "stale". A later process replays the records up to the first one that does
// not check out, so a record cut off mid-write is ignored. The pack holds only
// fetched bytes, so it takes that much disk on every filesystem (a sparse
// mirror does not save space on APFS). When the disk is under its free-space
// floor, or the budget refuses the pack, nothing is written to disk.
//
// File layout (GridStore v3, big-endian u32): a 13-word header; the bucket
// index, num_buckets+1 cumulative counts; the flat path-offset list; the path
// records (start x, start y, n, then n int8 offsets); the JSON metadata.
class RemoteGridFile {
public:
    // The bytes of each (offset, length) range of the remote file, read
    // concurrently where possible. `version` is empty on the first read and
    // the version that read returned afterwards; a fetcher that can tell
    // versions apart throws Changed when the file is no longer that version.
    // A missing file throws NotFound.
    struct Fetched {
        std::vector<std::string> parts;
        std::string version;
    };
    using RangeFetcher = std::function<Fetched(const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                               const std::string& version)>;

    struct NotFound : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    struct Changed : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // What every file of one plane of a store shares: bounds, cell size and
    // bucket count. With it, opening a new file fetches the header and the
    // index rows of the first region together instead of one after the other.
    struct Layout {
        uint32_t bx = 0, by = 0, bw = 0, bh = 0, cell = 0, num_buckets = 0;
        bool operator==(const Layout&) const = default;
    };
    // Buckets [x0..x1] x [y0..y1], inclusive.
    struct BucketRect { int x0, y0, x1, y1; };

    struct Options {
        const Layout* layout = nullptr;         // the plane's layout, when known
        std::optional<BucketRect> first;        // the region the first query reads
        uint64_t min_free_bytes = 0;            // below this much free disk, write no pack
        bool in_memory = false;                 // never write a pack to disk
        // The cache's disk budget, holding it for each write to the pack (see
        // vcr::sources::EncodedFiles::writeFile): `budgeted(path, write)` runs
        // `write`, which writes only once fits(total) accepts what the pack
        // will occupy, and returns what the pack occupies afterwards, written
        // or not (nullopt: none there); the budget charges that. Used for every
        // pack written, appended to or reused. Without it the pack is not
        // limited, and writes to one pack in this process are still made one
        // at a time. When the budget refuses a write, the pack stops growing
        // (it is still read as it is).
        using Fits = std::function<bool(uint64_t total)>;
        using Write = std::function<std::optional<uint64_t>(const Fits& fits)>;
        std::function<void(const std::string& path, const Write& write)> budgeted;
    };

    // Replays the pack at local_path + ".ranges", or creates it when there is
    // none or it does not match the remote file.
    // Throws NotFound when the remote file does not exist, and
    // std::runtime_error when it is not a GridStore v3 file.
    RemoteGridFile(std::string url, const std::string& local_path, RangeFetcher fetcher, const Options& options);
    ~RemoteGridFile();
    RemoteGridFile(const RemoteGridFile&) = delete;
    RemoteGridFile& operator=(const RemoteGridFile&) = delete;

    static constexpr int kTileBuckets = 8;

    const char* data() const { return static_cast<const char*>(map_); }
    size_t size() const { return size_; }
    bool inMemory() const { return in_memory_; }  // no pack on disk
    // Bytes of the file held in memory (fetched or replayed, each byte once).
    uint64_t residentBytes() const { return resident_.load(); }
    // residentBytes() of every RemoteGridFile alive in this process.
    static uint64_t residentTotal();

    // Make the buckets [x0..x1] x [y0..y1] (inclusive) readable: their index
    // entries, their path offsets and every path record they name. Throws
    // when that fails; the buckets must then not be read (what failed is
    // fetched again by the next call). `foreground`: called by a query that
    // waits for it (counted separately).
    void ensureBuckets(int x0, int y0, int x1, int y1, bool foreground = true);
    // Make the whole file readable (one request).
    void ensureAll();
    Layout layout() const { return layout_; }
    const std::string& url() const { return url_; }

    struct Stats { uint64_t requests = 0, bytes = 0, tilesLoaded = 0, tilesReused = 0, opens = 0,
                   fgLoads = 0, fgWaitNs = 0, bgLoads = 0, failures = 0, inMemory = 0,
                   tilesUsed = 0; };  // tilesUsed: distinct tiles a foreground query asked for
    static Stats globalStats();
    // Counts a failed ensureBuckets() and reports it on stderr, once per message.
    static void reportFailure(const std::string& what);

private:
    // Fetched bytes not yet in the pack, by offset.
    using Pieces = std::vector<std::pair<uint64_t, std::string>>;

    void allocate();
    bool reuse();
    void create(const Options& options);
    void loadTiles(const std::vector<std::pair<int, int>>& tiles, Pieces& log);
    void writeAt(uint64_t offset, std::string bytes, Pieces* log);
    Fetched fetch(const std::vector<std::pair<uint64_t, uint64_t>>& ranges);
    uint32_t u32(uint64_t offset) const;  // bounds-checked
    // Appends the pieces not in the pack yet and then `tail` (records) in one write.
    void commit(const Pieces& pieces, const std::string& tail);
    bool writePack(const std::string& records);  // a new pack, renamed into place
    // Runs `write` with the pack's budget held, its fits() also keeping the
    // free-space floor; a budget that throws writes nothing.
    void withPack(const Options::Write& write);
    void chargePack();

    std::string url_;
    std::string pack_path_;
    RangeFetcher fetch_;
    std::string version_;
    bool in_memory_ = false;
    std::function<void(const std::string&, const Options::Write&)> budgeted_;
    void* map_ = nullptr;
    size_t size_ = 0;

    // Under pack_mutex_: the pack appended to (0: none, or appending stopped)
    // and the byte ranges [start, end) it holds.
    std::mutex pack_mutex_;
    uint64_t pack_dev_ = 0, pack_ino_ = 0, pack_identity_ = 0;
    uint64_t min_free_bytes_ = 0;
    std::map<uint64_t, uint64_t> packed_;
    // The byte ranges [start, end) written to memory, and their total.
    void place(uint64_t offset, const char* bytes, uint64_t n);
    std::mutex held_mutex_;
    std::map<uint64_t, uint64_t> held_;
    std::atomic<uint64_t> resident_{0};

    Layout layout_;
    // Bucket-index rows (and the total count) already in the mirror; under mutex_.
    std::set<int> rows_loaded_;
    bool total_loaded_ = false;
    // Header fields.
    uint32_t cell_ = 0, num_buckets_ = 0, buckets_offset_ = 0, paths_offset_ = 0, meta_offset_ = 0;
    uint64_t flat_offset_ = 0;  // the path-offset list
    int grid_w_ = 0, grid_h_ = 0;

    std::mutex mutex_;
    std::condition_variable loaded_cv_;
    std::set<std::pair<int, int>> tiles_done_, tiles_loading_, tiles_used_;
    bool all_done_ = false;
};

} // namespace vc::core::util
