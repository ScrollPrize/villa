#include "vc/core/util/RemoteGridFile.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <mutex>
#include <map>
#include <random>

namespace fs = std::filesystem;

#ifndef _WIN32

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::core::util {

namespace {
constexpr uint32_t kMagic = 0x56434753;  // "VCGS"
constexpr uint32_t kHeaderBytes = 13 * 4;
// Path records of one tile are gathered into ranges: offsets closer than
// kPathGap bytes share a request, and each request reads kPathPad bytes past
// its last offset so the last record normally arrives whole.
constexpr uint64_t kPathGap = 4096;
constexpr uint64_t kPathPad = 2048;
// Path-offset spans of adjacent rows closer than this many entries share a request.
constexpr uint64_t kFlatGapEntries = 2048;

// The pack file (little-endian): "VCGR", format, remote size (u64), a random
// identity (u64; every new pack has its own, so an append never goes to a pack
// that replaced ours, even one given our inode number), version length and
// bytes, crc32 of all that; then records: type (u8), payload length (u32),
// payload, crc32 of type, length and payload.
constexpr uint32_t kPackMagic = 0x52474356;  // "VCGR" read as little-endian
constexpr uint32_t kPackFormat = 1;
enum : uint8_t { kData = 1, kTiles = 2, kAll = 3, kStale = 4 };
// A data record's bytes are at most this long (larger writes are split).
constexpr uint64_t kMaxDataBytes = 16u << 20;
constexpr size_t kPackFixed = 28;  // the header up to the version bytes
// Larger than any GridStore file: a pack naming a bigger one is not trusted.
constexpr uint64_t kMaxFileBytes = uint64_t(1) << 40;
// Tile coordinates a pack may name (a grid of 2^31 buckets has fewer tiles).
constexpr int64_t kMaxTile = int64_t(1) << 24;

uint64_t newIdentity()
{
    std::random_device random;
    const uint64_t r = (uint64_t(random()) << 32) ^ random();
    return r ^ uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) ^ (uint64_t(::getpid()) << 17);
}

uint32_t be32(const char* p)
{
    uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}

uint32_t crc32(const char* p, size_t n, uint32_t crc = 0)
{
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ uint8_t(p[i])) & 0xff] ^ (crc >> 8);
    return ~crc;
}

void put(std::string& out, uint64_t v, int bytes)
{
    for (int i = 0; i < bytes; ++i) out.push_back(char((v >> (8 * i)) & 0xff));
}

uint64_t get(const char* p, int bytes)
{
    uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) v |= uint64_t(uint8_t(p[i])) << (8 * i);
    return v;
}

void putRecord(std::string& out, uint8_t type, const char* payload, size_t n, uint64_t offset = 0, bool with_offset = false)
{
    const size_t start = out.size();
    out.push_back(char(type));
    put(out, n + (with_offset ? 8 : 0), 4);
    if (with_offset) put(out, offset, 8);
    out.append(payload, n);
    put(out, crc32(out.data() + start, out.size() - start), 4);
}

void putData(std::string& out, uint64_t offset, const char* bytes, size_t n)
{
    for (size_t at = 0; at < n; at += kMaxDataBytes) {
        const size_t len = std::min<size_t>(kMaxDataBytes, n - at);
        putRecord(out, kData, bytes + at, len, offset + at, true);
    }
}

std::string tilesRecord(const std::vector<std::pair<int, int>>& tiles)
{
    std::string payload, out;
    for (const auto& [tx, ty] : tiles) {
        put(payload, uint32_t(tx), 4);
        put(payload, uint32_t(ty), 4);
    }
    if (!tiles.empty()) putRecord(out, kTiles, payload.data(), payload.size());
    return out;
}

std::string emptyRecord(uint8_t type)
{
    std::string out;
    putRecord(out, type, "", 0);
    return out;
}

// Adds [a, b) to the merged ranges and returns the parts it did not hold.
std::vector<std::pair<uint64_t, uint64_t>> addRange(std::map<uint64_t, uint64_t>& ranges, uint64_t a, uint64_t b)
{
    std::vector<std::pair<uint64_t, uint64_t>> added;
    if (a >= b) return added;
    auto it = ranges.upper_bound(a);
    if (it != ranges.begin() && std::prev(it)->second >= a) --it;
    uint64_t pos = a, lo = a, hi = b;
    while (it != ranges.end() && it->first <= b) {
        if (it->first > pos) added.push_back({pos, it->first});
        pos = std::max(pos, it->second);
        lo = std::min(lo, it->first);
        hi = std::max(hi, it->second);
        it = ranges.erase(it);
    }
    if (pos < b) added.push_back({pos, b});
    ranges[lo] = hi;
    return added;
}

bool preadAll(int fd, char* out, size_t n, uint64_t at)
{
    while (n > 0) {
        const ssize_t r = ::pread(fd, out, n, static_cast<off_t>(at));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        out += r;
        n -= size_t(r);
        at += uint64_t(r);
    }
    return true;
}

bool writeAll(int fd, const std::string& bytes)
{
    size_t at = 0;
    while (at < bytes.size()) {
        const ssize_t w = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        at += size_t(w);
    }
    return true;
}

std::atomic<uint64_t> g_requests{0}, g_bytes{0}, g_tiles_loaded{0}, g_tiles_reused{0}, g_opens{0}, g_fg_loads{0},
    g_fg_wait_ns{0}, g_bg_loads{0}, g_failures{0}, g_in_memory{0}, g_tiles_used{0};

// Merge [a, b) ranges whose gap is at most `gap`.
std::vector<std::pair<uint64_t, uint64_t>> mergeSpans(std::vector<std::pair<uint64_t, uint64_t>> spans, uint64_t gap)
{
    std::sort(spans.begin(), spans.end());
    std::vector<std::pair<uint64_t, uint64_t>> out;
    for (const auto& s : spans) {
        if (s.second <= s.first) continue;
        if (!out.empty() && s.first <= out.back().second + gap)
            out.back().second = std::max(out.back().second, s.second);
        else
            out.push_back(s);
    }
    return out;
}

bool belowFloor(const fs::path& dir, uint64_t min_free_bytes, uint64_t needed)
{
    if (min_free_bytes == 0) return false;
    std::error_code ec;
    const auto space = fs::space(dir, ec);
    return ec || space.available < min_free_bytes + needed;
}
}  // namespace

namespace {
std::atomic<uint64_t> g_resident{0};
}

uint64_t RemoteGridFile::residentTotal()
{
    return g_resident.load(std::memory_order_relaxed);
}

RemoteGridFile::Stats RemoteGridFile::globalStats()
{
    return {g_requests.load(), g_bytes.load(), g_tiles_loaded.load(), g_tiles_reused.load(), g_opens.load(),
            g_fg_loads.load(), g_fg_wait_ns.load(), g_bg_loads.load(), g_failures.load(), g_in_memory.load(),
            g_tiles_used.load()};
}

void RemoteGridFile::reportFailure(const std::string& what)
{
    ++g_failures;
    static std::mutex mutex;
    static std::string last;
    std::lock_guard lock(mutex);
    if (what == last) return;
    last = what;
    std::cerr << "normal-grid range read failed (those normals are not used): " << what << std::endl;
}

RemoteGridFile::RemoteGridFile(std::string url, const std::string& local_path, RangeFetcher fetcher, const Options& options)
    : url_(std::move(url)), pack_path_(local_path + ".ranges"), fetch_(std::move(fetcher)), budgeted_(options.budgeted),
      min_free_bytes_(options.min_free_bytes)
{
    ++g_opens;
    const fs::path dir = fs::path(local_path).parent_path();
    in_memory_ = options.in_memory;
    bool below_floor = false;
    if (!in_memory_) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        in_memory_ = bool(ec);
        below_floor = !in_memory_ && belowFloor(dir, options.min_free_bytes, 0);
    }
    // A pack already on disk is read even under the free-space floor (it is
    // then not written to: see chargePack); a new one is kept in memory.
    if (in_memory_ || !reuse()) {
        in_memory_ = in_memory_ || below_floor;
        create(options);
    }

    const char* h = data();
    cell_ = be32(h + 24);
    num_buckets_ = be32(h + 28);
    buckets_offset_ = be32(h + 36);
    paths_offset_ = be32(h + 40);
    meta_offset_ = be32(h + 44);
    const uint32_t bw = be32(h + 16), bh = be32(h + 20);
    layout_ = {be32(h + 8), be32(h + 12), bw, bh, cell_, num_buckets_};
    if (cell_ == 0) throw std::runtime_error("remote grid " + url_ + ": cell size 0");
    grid_w_ = static_cast<int>((uint64_t(bw) + cell_ - 1) / cell_);
    grid_h_ = static_cast<int>((uint64_t(bh) + cell_ - 1) / cell_);
    flat_offset_ = uint64_t(buckets_offset_) + (uint64_t(num_buckets_) + 1) * 4;
    // What the range reads rely on; GridStore checks the rest as for a whole file.
    if (grid_w_ <= 0 || grid_h_ <= 0 || uint64_t(grid_w_) * uint64_t(grid_h_) != num_buckets_ ||
        buckets_offset_ < kHeaderBytes || flat_offset_ > paths_offset_ || paths_offset_ > meta_offset_ ||
        meta_offset_ > size_)
        throw std::runtime_error("remote grid " + url_ + ": header does not describe a readable GridStore v3 file");
}

RemoteGridFile::~RemoteGridFile()
{
    if (map_) ::munmap(map_, size_);
    g_resident -= resident_.load();
}

namespace {
std::optional<uint64_t> fileSize(const std::string& path)
{
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return std::nullopt;
    return uint64_t(st.st_size);
}
}  // namespace

void RemoteGridFile::withPack(const Options::Write& write)
{
    const fs::path dir = fs::path(pack_path_).parent_path();
    const Options::Write floored = [&](const Options::Fits& budget) -> std::optional<uint64_t> {
        const auto before = fileSize(pack_path_).value_or(0);
        const Options::Fits fits = [&](uint64_t total) {
            if (belowFloor(dir, min_free_bytes_, total > before ? total - before : 0)) return false;
            return budget(total);
        };
        return write(fits);
    };
    if (!budgeted_) {
        static std::mutex mutex;
        static std::map<std::string, std::shared_ptr<std::mutex>> per_pack;
        std::shared_ptr<std::mutex> one;
        {
            std::lock_guard lock(mutex);
            auto& slot = per_pack[pack_path_];
            if (!slot) slot = std::make_shared<std::mutex>();
            one = slot;
        }
        std::lock_guard lock(*one);
        (void)floored([](uint64_t) { return true; });
        return;
    }
    try {
        budgeted_(pack_path_, floored);
    } catch (const std::exception&) {
        // The budget could not be asked: write() did not run, or what it wrote
        // was not recorded; the caller's flags say which, and the next start
        // reconciles the budget with the disk.
    }
}

// Charges what the pack occupies now, as the newest in the budget (other
// processes may have appended to it). When the budget would not take it, or
// the disk is under its free-space floor, the pack stops growing (it is still
// read as it is).
void RemoteGridFile::chargePack()
{
    if (in_memory_) return;
    bool fits_now = false;
    withPack([&](const Options::Fits& fits) {
        const auto size = fileSize(pack_path_);
        if (size) fits_now = fits(*size);
        return size;
    });
    if (!fits_now) {
        std::lock_guard lock(pack_mutex_);
        pack_ino_ = 0;
    }
}

// Memory for the whole file; pages nothing is written to stay unallocated.
void RemoteGridFile::allocate()
{
    map_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw std::runtime_error("cannot map memory for " + url_);
    }
}

// Replays the pack an earlier process left: the records up to the first one
// that does not check out (one cut off mid-write, say). False when there is
// no usable pack: none, another remote size, or marked stale.
bool RemoteGridFile::reuse()
{
    const int fd = ::open(pack_path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct Closer { int fd; ~Closer() { ::close(fd); } } closer{fd};
    struct stat st{};
    if (::fstat(fd, &st) != 0) return false;
    const uint64_t length = uint64_t(st.st_size);

    char fixed[kPackFixed];
    if (length < sizeof(fixed) + 4 || !preadAll(fd, fixed, sizeof(fixed), 0)) return false;
    const uint64_t vlen = get(fixed + 24, 4);
    if (get(fixed, 4) != kPackMagic || get(fixed + 4, 4) != kPackFormat || vlen > 4096 ||
        length < sizeof(fixed) + vlen + 4)
        return false;
    std::string header(sizeof(fixed) + vlen + 4, '\0');
    if (!preadAll(fd, header.data(), header.size(), 0)) return false;
    if (crc32(header.data(), header.size() - 4) != get(header.data() + header.size() - 4, 4)) return false;
    const uint64_t recorded_size = get(fixed + 8, 8);
    if (recorded_size < kHeaderBytes || recorded_size > kMaxFileBytes) return false;

    size_ = recorded_size;
    try {
        allocate();
    } catch (const std::exception&) {
        return false;  // a new pack is made instead
    }
    std::map<uint64_t, uint64_t> packed;
    std::set<std::pair<int, int>> done;
    bool all = false, stale = false;
    uint64_t pos = header.size();
    std::string record;
    while (pos + 9 <= length) {
        char head[5];
        if (!preadAll(fd, head, 5, pos)) break;
        const uint8_t type = uint8_t(head[0]);
        const uint64_t n = get(head + 1, 4);
        if (n > kMaxDataBytes + 8 || pos + 5 + n + 4 > length) break;
        record.resize(5 + n + 4);
        if (!preadAll(fd, record.data(), record.size(), pos)) break;
        if (crc32(record.data(), 5 + n) != get(record.data() + 5 + n, 4)) break;
        const char* payload = record.data() + 5;
        if (type == kData) {
            if (n < 8) break;
            const uint64_t offset = get(payload, 8), bytes = n - 8;
            if (offset > size_ || bytes > size_ - offset) break;
            place(offset, payload + 8, bytes);
            addRange(packed, offset, offset + bytes);
        } else if (type == kTiles) {
            if (n % 8) break;
            bool sane = true;
            for (uint64_t i = 0; i < n; i += 8) {
                const int64_t tx = int32_t(get(payload + i, 4)), ty = int32_t(get(payload + i + 4, 4));
                if (tx < 0 || ty < 0 || tx >= kMaxTile || ty >= kMaxTile) sane = false;
                else done.insert({int(tx), int(ty)});
            }
            if (!sane) break;
        } else if (type == kAll) {
            all = true;
        } else if (type == kStale) {
            stale = true;
            break;
        } else {
            break;
        }
        pos += record.size();
    }
    // The header must be in it; a tile is complete only with its bytes, which
    // are always appended with it or before it.
    const auto first = packed.find(0);
    if (stale || first == packed.end() || first->second < kHeaderBytes ||
        (all && !(first->second >= size_))) {
        ::munmap(map_, size_);
        map_ = nullptr;
        std::lock_guard lock(held_mutex_);
        held_.clear();
        g_resident -= resident_.exchange(0);  // what place() charged while replaying
        return false;
    }
    version_ = header.substr(sizeof(fixed), vlen);
    pack_identity_ = get(fixed + 16, 8);
    packed_ = std::move(packed);
    tiles_done_ = std::move(done);
    all_done_ = all;
    pack_dev_ = uint64_t(st.st_dev);
    pack_ino_ = uint64_t(st.st_ino);
    if (pos < length) {
        // A record that does not check out: what follows it would never be
        // read, so write the records that did into a new pack.
        std::string records;
        for (const auto& [a, b] : packed_) putData(records, a, data() + a, b - a);
        records += tilesRecord({tiles_done_.begin(), tiles_done_.end()});
        if (all_done_) records += emptyRecord(kAll);
        if (!writePack(records)) pack_ino_ = 0;
    }
    chargePack();  // used again: the newest in the budget
    // The rows of complete tiles are in place (and so is the total, which
    // loading any tile reads).
    for (const auto& [tx, ty] : tiles_done_)
        for (int y = ty * kTileBuckets; y < (ty + 1) * kTileBuckets; ++y) rows_loaded_.insert(y);
    total_loaded_ = all_done_ || !tiles_done_.empty();
    return true;
}

// Writes a new pack (header, then `records`) beside the pack path and renames
// it into place; a process appending to an older pack then stops appending.
bool RemoteGridFile::writePack(const std::string& records)
{
    std::string out;
    put(out, kPackMagic, 4);
    put(out, kPackFormat, 4);
    const uint64_t identity = newIdentity();
    put(out, size_, 8);
    put(out, identity, 8);
    put(out, version_.size(), 4);
    out += version_;
    put(out, crc32(out.data(), out.size()), 4);
    out += records;
    // Written, renamed into place and charged with the budget held, so no
    // eviction comes between charging the pack and publishing it.
    bool published = false;
    withPack([&](const Options::Fits& fits) -> std::optional<uint64_t> {
        if (!fits(out.size())) return fileSize(pack_path_);
        const std::string temporary = pack_path_ + ".tmp." + std::to_string(::getpid());
        const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) return fileSize(pack_path_);
        struct stat st{};
        const bool ok = writeAll(fd, out) && ::fstat(fd, &st) == 0;
        ::close(fd);
        if (!ok || ::rename(temporary.c_str(), pack_path_.c_str()) != 0) {
            ::unlink(temporary.c_str());
            return fileSize(pack_path_);
        }
        pack_dev_ = uint64_t(st.st_dev);
        pack_ino_ = uint64_t(st.st_ino);
        pack_identity_ = identity;
        published = true;
        return uint64_t(st.st_size);
    });
    if (!published) pack_ino_ = 0;
    return published;
}

// A new file: the header, and with the plane's layout the first region's
// index rows, in one round.
void RemoteGridFile::create(const Options& options)
{
    std::vector<std::pair<uint64_t, uint64_t>> first_ranges{{0, kHeaderBytes}};
    std::vector<int> early_rows;
    const Layout* layout = options.layout;
    if (layout && options.first && layout->cell) {
        const uint64_t gw = (uint64_t(layout->bw) + layout->cell - 1) / layout->cell;
        const int gh = static_cast<int>((uint64_t(layout->bh) + layout->cell - 1) / layout->cell);
        if (gw > 0 && gh > 0 && gw * uint64_t(gh) == layout->num_buckets) {
            const auto& first = *options.first;
            const int ra = std::clamp(first.y0, 0, gh - 1) / kTileBuckets * kTileBuckets;
            const int rb = std::min(gh, (std::clamp(first.y1, 0, gh - 1) / kTileBuckets + 1) * kTileBuckets) - 1;
            const uint64_t a = uint64_t(ra) * gw, b = (uint64_t(rb) + 1) * gw + 1;
            first_ranges.push_back({kHeaderBytes + a * 4, (b - a) * 4});
            first_ranges.push_back({kHeaderBytes + uint64_t(layout->num_buckets) * 4, 4});
            for (int y = ra; y <= rb; ++y) early_rows.push_back(y);
        }
    }
    auto first = fetch(first_ranges);
    const std::string& header = first.parts.at(0);
    if (be32(header.data()) != kMagic || be32(header.data() + 4) != 3)
        throw std::runtime_error("remote grid " + url_ + " is not a GridStore v3 file");
    size_ = uint64_t(be32(header.data() + 44)) + be32(header.data() + 48);
    if (size_ < kHeaderBytes) throw std::runtime_error("remote grid " + url_ + ": bad metadata offset");
    // Early rows only when the file has the layout they assumed.
    const Layout actual{be32(header.data() + 8), be32(header.data() + 12), be32(header.data() + 16),
                        be32(header.data() + 20), be32(header.data() + 24), be32(header.data() + 28)};
    const bool early = !early_rows.empty() && actual == *layout && be32(header.data() + 36) == kHeaderBytes &&
                       first_ranges[2].first + 4 <= size_;

    allocate();
    Pieces pieces;
    writeAt(0, first.parts[0], &pieces);
    if (early) {
        writeAt(first_ranges[1].first, std::move(first.parts[1]), &pieces);
        writeAt(first_ranges[2].first, std::move(first.parts[2]), &pieces);
        rows_loaded_.insert(early_rows.begin(), early_rows.end());
        total_loaded_ = true;
    }

    // The new pack first holds the header and the first rows (writePack
    // charges it, and checks the floor, before writing).
    if (!in_memory_) {
        std::string records;
        for (const auto& [o, b] : pieces) putData(records, o, b.data(), b.size());
        if (writePack(records)) {
            for (const auto& [o, b] : pieces) addRange(packed_, o, o + b.size());
        } else {
            in_memory_ = true;  // the disk would not take it: nothing to reuse later
        }
    }
    if (in_memory_) ++g_in_memory;
}

uint32_t RemoteGridFile::u32(uint64_t offset) const
{
    if (offset > size_ || size_ - offset < 4)
        throw std::runtime_error("remote grid " + url_ + ": offset out of bounds (corrupt file)");
    return be32(data() + offset);
}

void RemoteGridFile::writeAt(uint64_t offset, std::string bytes, Pieces* log)
{
    if (offset > size_ || bytes.size() > size_ - offset)
        throw std::runtime_error("remote grid " + url_ + ": write past the end of the file");
    place(offset, bytes.data(), bytes.size());
    if (log && !in_memory_) log->push_back({offset, std::move(bytes)});
}

// Copies in the bytes memory does not hold yet. Bytes already held are never
// written again, so queries reading them race with nothing; and the copy is
// made under the lock, so once any writer of a range returns, it is in place.
void RemoteGridFile::place(uint64_t offset, const char* bytes, uint64_t n)
{
    std::lock_guard lock(held_mutex_);
    for (const auto& [a, b] : addRange(held_, offset, offset + n)) {
        std::memcpy(static_cast<char*>(map_) + a, bytes + (a - offset), b - a);
        resident_ += b - a;
        g_resident += b - a;
    }
}

RemoteGridFile::Fetched RemoteGridFile::fetch(const std::vector<std::pair<uint64_t, uint64_t>>& ranges)
{
    std::string version;
    {
        std::lock_guard lock(mutex_);
        version = version_;
    }
    auto fetched = fetch_(ranges, version);
    if (fetched.parts.size() != ranges.size()) throw std::runtime_error("range fetch returned the wrong number of parts");
    g_requests += ranges.size();
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (fetched.parts[i].size() != ranges[i].second) throw std::runtime_error("short range read from " + url_);
        g_bytes += fetched.parts[i].size();
    }
    // Bytes of no known version (a whole copy read in place, say) or of another
    // one are never mixed with ours.
    if (!version.empty() && fetched.version != version)
        throw Changed("remote grid " + url_ + " is no longer version " + version);
    std::lock_guard lock(mutex_);
    if (version_.empty()) version_ = fetched.version;
    return fetched;
}

// One append holding the pieces the pack lacks, then `tail`. A tile record is
// only ever in the pack after the bytes it needs: they are in this append or
// were in one that succeeded before it. After a failed or short append (the
// pack may end in a cut-off record) nothing more is appended.
void RemoteGridFile::commit(const Pieces& pieces, const std::string& tail)
{
    if (in_memory_) return;
    std::lock_guard lock(pack_mutex_);
    if (pack_ino_ == 0) return;
    // packed_ may run ahead of the file only when this append fails, and then
    // nothing is appended again.
    std::string out;
    for (const auto& [offset, bytes] : pieces)
        for (const auto& [a, b] : addRange(packed_, offset, offset + bytes.size()))
            putData(out, a, bytes.data() + (a - offset), b - a);
    out += tail;
    if (out.empty()) return;
    // With the budget held, so appends of other readers of this pack, here or
    // in another process, are made and charged one at a time, each fitted to
    // the size the pack has when it is made.
    bool ok = false;
    withPack([&](const Options::Fits& fits) -> std::optional<uint64_t> {
        const int fd = ::open(pack_path_.c_str(), O_RDWR | O_APPEND | O_CLOEXEC);
        if (fd < 0) return fileSize(pack_path_);  // evicted, or the disk is gone
        struct stat st{};
        char identity[8];
        if (::fstat(fd, &st) == 0 && uint64_t(st.st_dev) == pack_dev_ && uint64_t(st.st_ino) == pack_ino_ &&
            preadAll(fd, identity, 8, 16) && get(identity, 8) == pack_identity_ &&
            fits(uint64_t(st.st_size) + out.size())) {
            const ssize_t n = ::write(fd, out.data(), out.size());
            ok = n == ssize_t(out.size());
            // A short write is cut off again: the pack holds only what was charged.
            if (!ok && n > 0 && ::ftruncate(fd, st.st_size) != 0) {}
        }
        struct stat after{};
        const bool sized = ::fstat(fd, &after) == 0;
        ::close(fd);
        if (!sized) return fileSize(pack_path_);
        return uint64_t(after.st_size);
    });
    if (!ok) pack_ino_ = 0;  // nothing more is appended; the pack is read as it is
}

void RemoteGridFile::ensureAll()
{
    {
        std::lock_guard lock(mutex_);
        if (all_done_) return;
    }
    {
        auto whole = fetch({{0, size_}});
        writeAt(0, std::move(whole.parts.at(0)), nullptr);
    }
    // Into the pack a part at a time: one append of a whole file would be
    // more than a single write() takes.
    constexpr uint64_t kChunk = 64u << 20;
    for (uint64_t at = 0; at < size_ && !in_memory_; at += kChunk) {
        const uint64_t n = std::min<uint64_t>(kChunk, size_ - at);
        commit({{at, std::string(data() + at, n)}}, at + n >= size_ ? emptyRecord(kAll) : std::string());
    }
    std::lock_guard lock(mutex_);
    all_done_ = true;
    loaded_cv_.notify_all();
}

void RemoteGridFile::ensureBuckets(int x0, int y0, int x1, int y1, bool foreground)
{
    const auto t0 = std::chrono::steady_clock::now();
    struct Timer {
        bool fg; std::chrono::steady_clock::time_point t0; bool waited = false;
        ~Timer() { if (fg && waited) g_fg_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); }
    } timer{foreground, t0};
    x0 = std::clamp(x0, 0, grid_w_ - 1);
    x1 = std::clamp(x1, 0, grid_w_ - 1);
    y0 = std::clamp(y0, 0, grid_h_ - 1);
    y1 = std::clamp(y1, 0, grid_h_ - 1);
    if (x0 > x1 || y0 > y1) return;

    std::vector<std::pair<int, int>> wanted, mine;
    for (int ty = y0 / kTileBuckets; ty <= y1 / kTileBuckets; ++ty)
        for (int tx = x0 / kTileBuckets; tx <= x1 / kTileBuckets; ++tx) wanted.push_back({tx, ty});
    {
        std::unique_lock lock(mutex_);
        if (foreground)
            for (const auto& t : wanted)
                if (tiles_used_.insert(t).second) ++g_tiles_used;
        if (all_done_) return;
        for (const auto& t : wanted)
            if (!tiles_done_.count(t) && tiles_loading_.insert(t).second) mine.push_back(t);
    }
    if (!mine.empty()) {
        timer.waited = true;
        (foreground ? g_fg_loads : g_bg_loads) += mine.size();
        Pieces log;
        try {
            loadTiles(mine, log);
        } catch (const Changed&) {
            // Bytes of another version may be in the pack: never reuse it.
            commit({}, emptyRecord(kStale));
            std::lock_guard lock(mutex_);
            for (const auto& t : mine) tiles_loading_.erase(t);
            loaded_cv_.notify_all();
            throw;
        } catch (...) {
            std::lock_guard lock(mutex_);
            for (const auto& t : mine) tiles_loading_.erase(t);
            loaded_cv_.notify_all();
            throw;
        }
        {
            std::lock_guard lock(mutex_);
            for (const auto& t : mine) {
                tiles_loading_.erase(t);
                tiles_done_.insert(t);
            }
            g_tiles_loaded += mine.size();
            loaded_cv_.notify_all();
        }
        // The bytes are in memory already: queries waiting for these tiles go
        // on while they are appended to the pack.
        commit(log, tilesRecord(mine));
    } else {
        g_tiles_reused += 1;
    }
    // Wait for tiles another thread is loading; one that failed fails this call too.
    std::unique_lock lock(mutex_);
    bool failed = false;
    for (const auto& t : wanted) if (!tiles_done_.count(t)) timer.waited = true;
    loaded_cv_.wait(lock, [&] {
        if (all_done_) return true;
        for (const auto& t : wanted)
            if (!tiles_done_.count(t)) {
                if (!tiles_loading_.count(t)) {
                    failed = true;
                    return true;
                }
                return false;
            }
        return true;
    });
    if (failed && !all_done_) throw std::runtime_error("loading a grid tile of " + url_ + " failed");
}

void RemoteGridFile::loadTiles(const std::vector<std::pair<int, int>>& tiles, Pieces& log)
{
    const uint64_t gw = static_cast<uint64_t>(grid_w_);
    auto tileRows = [&](int ty) { return std::pair<int, int>{ty * kTileBuckets, std::min(grid_h_, (ty + 1) * kTileBuckets) - 1}; };
    auto tileCols = [&](int tx) { return std::pair<int, int>{tx * kTileBuckets, std::min(grid_w_, (tx + 1) * kTileBuckets) - 1}; };

    // A: the bucket index entries of whole rows (plus the one after the last
    // row, which ends its last bucket) and the total count at the end.
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    std::vector<int> new_rows;
    bool need_total;
    {
        std::lock_guard lock(mutex_);
        need_total = !total_loaded_;
        for (const auto& [tx, ty] : tiles) {
            (void)tx;
            const auto [ra, rb] = tileRows(ty);
            for (int y = ra; y <= rb; ++y)
                if (!rows_loaded_.count(y)) {
                    spans.push_back({uint64_t(y) * gw, (uint64_t(y) + 1) * gw + 1});
                    new_rows.push_back(y);
                }
        }
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto& [a, b] : mergeSpans(spans, 0))
        ranges.push_back({buckets_offset_ + a * 4, (b - a) * 4});
    if (need_total) ranges.push_back({buckets_offset_ + uint64_t(num_buckets_) * 4, 4});
    if (!ranges.empty()) {
        auto fetched = fetch(ranges);
        Pieces rows;
        for (size_t i = 0; i < ranges.size(); ++i) writeAt(ranges[i].first, std::move(fetched.parts[i]), &rows);
        // Other tiles of these rows skip this step once they count as loaded,
        // and their tile records rely on the rows being in the pack: so the
        // rows go into it first (if that fails, nothing more is appended).
        commit(rows, std::string());
        std::lock_guard lock(mutex_);
        rows_loaded_.insert(new_rows.begin(), new_rows.end());
        total_loaded_ = true;
    }
    const uint64_t total_entries = u32(buckets_offset_ + uint64_t(num_buckets_) * 4);
    if (flat_offset_ + total_entries * 4 > paths_offset_)
        throw std::runtime_error("remote grid " + url_ + ": path-offset list out of bounds (corrupt file)");

    // B: the path offsets of the tiles' buckets, one span per tile row.
    std::vector<std::pair<uint64_t, uint64_t>> flat;
    for (const auto& [tx, ty] : tiles) {
        const auto [ra, rb] = tileRows(ty);
        const auto [ca, cb] = tileCols(tx);
        for (int y = ra; y <= rb; ++y) {
            const uint64_t s = u32(buckets_offset_ + (uint64_t(y) * gw + ca) * 4);
            const uint64_t e = u32(buckets_offset_ + (uint64_t(y) * gw + cb + 1) * 4);
            if (e < s || e > total_entries)
                throw std::runtime_error("remote grid " + url_ + ": bucket index out of bounds (corrupt file)");
            if (e > s) flat.push_back({s, e});
        }
    }
    if (flat.empty()) return;
    ranges.clear();
    for (const auto& [a, b] : mergeSpans(flat, kFlatGapEntries))
        ranges.push_back({flat_offset_ + a * 4, (b - a) * 4});
    {
        auto fetched = fetch(ranges);
        for (size_t i = 0; i < ranges.size(); ++i) writeAt(ranges[i].first, std::move(fetched.parts[i]), &log);
    }

    // C: the path records those offsets name.
    std::vector<uint64_t> offsets;
    for (const auto& [s, e] : flat)
        for (uint64_t i = s; i < e; ++i) offsets.push_back(u32(flat_offset_ + i * 4));
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    const uint64_t paths_end = meta_offset_;
    if (!offsets.empty() && paths_offset_ + offsets.back() + 12 > paths_end)
        throw std::runtime_error("remote grid " + url_ + ": path offset out of bounds (corrupt file)");
    std::vector<std::pair<uint64_t, uint64_t>> clusters;  // [first, last] offset
    for (const uint64_t o : offsets) {
        if (!clusters.empty() && o <= clusters.back().second + kPathGap)
            clusters.back().second = o;
        else
            clusters.push_back({o, o});
    }
    ranges.clear();
    std::vector<std::pair<uint64_t, uint64_t>> covered;  // absolute [a, b) read so far in this step
    for (const auto& [a, b] : clusters) {
        const uint64_t start = paths_offset_ + a;
        const uint64_t end = std::min<uint64_t>(paths_end, paths_offset_ + b + kPathPad);
        ranges.push_back({start, end - start});
        covered.push_back({start, end});
    }
    auto readAndWrite = [&](const std::vector<std::pair<uint64_t, uint64_t>>& rs) {
        auto fetched = fetch(rs);
        for (size_t i = 0; i < rs.size(); ++i) writeAt(rs[i].first, std::move(fetched.parts[i]), &log);
    };
    readAndWrite(ranges);
    // A record longer than the padding: read what is missing, until every record is whole.
    auto missing = [&](uint64_t a, uint64_t b) {
        // The part of [a, b) not in `covered`, as one range from the first gap.
        covered = mergeSpans(covered, 0);
        uint64_t pos = a;
        for (const auto& [ca, cb] : covered) {
            if (cb <= pos) continue;
            if (ca > pos) break;
            pos = cb;
            if (pos >= b) return std::pair<uint64_t, uint64_t>{0, 0};
        }
        return std::pair<uint64_t, uint64_t>{pos, b};
    };
    for (int round = 0; round < 4; ++round) {
        std::vector<std::pair<uint64_t, uint64_t>> need;
        for (const uint64_t o : offsets) {
            const uint64_t rec = paths_offset_ + o;
            auto gap = missing(rec, rec + 12);
            if (gap.second == 0) {
                const uint64_t end = rec + 12 + u32(rec + 8);
                if (end > paths_end) throw std::runtime_error("path record of " + url_ + " runs past the paths");
                gap = missing(rec, end);
            }
            if (gap.second != 0) need.push_back(gap);
        }
        if (need.empty()) return;
        need = mergeSpans(need, 0);
        ranges.clear();
        for (const auto& [a, b] : need) {
            ranges.push_back({a, b - a});
            covered.push_back({a, b});
        }
        readAndWrite(ranges);
    }
    throw std::runtime_error("path records of " + url_ + " still incomplete after four reads");
}

}  // namespace vc::core::util

#else  // _WIN32

// Range reads use POSIX files and anonymous memory and are not built on
// Windows: NormalGridVolume reads every grid whole there (it never enables
// them), and a RemoteGridFile cannot be opened.
namespace vc::core::util {

namespace {
std::atomic<uint64_t> g_failures{0};
}

uint64_t RemoteGridFile::residentTotal()
{
    return 0;
}

RemoteGridFile::Stats RemoteGridFile::globalStats()
{
    Stats stats;
    stats.failures = g_failures.load();
    return stats;
}

void RemoteGridFile::reportFailure(const std::string& what)
{
    ++g_failures;
    std::cerr << "normal-grid range read failed (those normals are not used): " << what << std::endl;
}

RemoteGridFile::RemoteGridFile(std::string url, const std::string&, RangeFetcher, const Options&) : url_(std::move(url))
{
    throw std::runtime_error("normal-grid range reads are not available on Windows: " + url_);
}

RemoteGridFile::~RemoteGridFile() = default;

void RemoteGridFile::ensureBuckets(int, int, int, int, bool)
{
    throw std::logic_error("normal-grid range reads are not available on Windows");
}

void RemoteGridFile::ensureAll()
{
    throw std::logic_error("normal-grid range reads are not available on Windows");
}

}  // namespace vc::core::util

#endif  // _WIN32
