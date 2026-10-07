#include "vc/core/types/FiberCollection.hpp"

#include <sqlite3.h>
#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace vc::fibers
{
namespace
{
constexpr int kApplicationId = 0x56434643;  // VCFC
constexpr int kMaxBlockPoints = 257;
void validateMinLength(double length)
{
    if (!std::isfinite(length) || length < 0)
        throw std::runtime_error("Invalid minimum fiber length");
}
const char* familyPredicate(FamilyFilter family)
{
    switch (family) {
    case FamilyFilter::All: return "";
    case FamilyFilter::Horizontal: return " AND f.family='H' ";
    case FamilyFilter::Vertical: return " AND f.family='V' ";
    case FamilyFilter::None: return " AND 0 ";
    }
    throw std::runtime_error("Invalid fiber family filter");
}
void check(int rc, sqlite3* db)
{
    if (rc != SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(db));
}
class Statement
{
public:
    sqlite3_stmt* value{};
    explicit Statement(sqlite3* db, const char* sql) : db_(db) { check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr), db); }
    ~Statement() { sqlite3_finalize(value); }
    void bind(int index, int64_t n) { check(sqlite3_bind_int64(value, index, n), db_); }
    void bind(int index, double n) { check(sqlite3_bind_double(value, index, n), db_); }
    void reset() { check(sqlite3_reset(value), db_); }
    bool row()
    {
        const int rc = sqlite3_step(value);
        if (rc == SQLITE_ROW)
            return true;
        if (rc == SQLITE_DONE)
            return false;
        throw std::runtime_error(sqlite3_errmsg(db_));
    }

private:
    sqlite3* db_;
};
std::string text(sqlite3_stmt* s, int column)
{
    const auto* p = sqlite3_column_text(s, column);
    return p ? std::string(reinterpret_cast<const char*>(p), sqlite3_column_bytes(s, column)) : "";
}
Summary readSummary(sqlite3_stmt* s)
{
    Summary out;
    out.id = sqlite3_column_int64(s, 0);
    out.name = text(s, 1);
    out.family = text(s, 2);
    out.pointCount = sqlite3_column_int64(s, 3);
    out.length = sqlite3_column_double(s, 4);
    for (int i = 0; i < 3; ++i) {
        out.bounds.lower[i] = sqlite3_column_double(s, 5 + 2 * i);
        out.bounds.upper[i] = sqlite3_column_double(s, 6 + 2 * i);
    }
    return out;
}
Block readBlock(sqlite3_stmt* s)
{
    Block out;
    out.id = sqlite3_column_int64(s, 0);
    out.fiberId = sqlite3_column_int64(s, 1);
    out.firstSegment = sqlite3_column_int64(s, 2);
    out.family = text(s, 3);
    const int n = sqlite3_column_bytes(s, 4);
    if (n < 48 || n % 24 || n > kMaxBlockPoints * 24)
        throw std::runtime_error("Invalid fiber block size");
    const auto* src = static_cast<const unsigned char*>(sqlite3_column_blob(s, 4));
    out.points.resize(n / 24);
    static_assert(sizeof(Point) == 24);
    if constexpr (std::endian::native == std::endian::little) {
        std::memcpy(out.points.data(), src, size_t(n));
    } else {
        for (auto& p : out.points)
            for (double& x : p) {
                uint64_t bits = 0;
                for (int b = 0; b < 8; ++b)
                    bits |= uint64_t(*src++) << (8 * b);
                x = std::bit_cast<double>(bits);
            }
    }
    for (const auto& p : out.points)
        for (double x : p) {
            if (!std::isfinite(x))
                throw std::runtime_error("Non-finite fiber coordinate");
        }
    return out;
}
constexpr const char* summaryColumns = "id,name,family,point_count,length,min_x,max_x,min_y,max_y,min_z,max_z";
uint64_t collectionSalt(const std::string& uuid)
{
    uint64_t salt = 14695981039346656037ULL;
    for (unsigned char c : uuid)
        salt = (salt ^ c) * 1099511628211ULL;
    return salt;
}
// Explicit integer hashing gives the same priorities on every platform,
// independent of SQL row order and of std::hash.
uint64_t fiberPriority(int64_t id, uint64_t salt)
{
    uint64_t x = uint64_t(id) ^ salt;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
}  // namespace

Point ViewRegion::project(const Point& point) const
{
    Point out{};
    for (size_t i = 0; i < 3; ++i) {
        const double d = point[i] - origin[i];
        out[0] += d * u[i];
        out[1] += d * v[i];
        out[2] += d * n[i];
    }
    return out;
}

bool ViewRegion::contains(const ViewRegion& other) const
{
    if (u != other.u || v != other.v || n != other.n)
        return false;
    const auto offset = project(other.origin);
    for (int i = 0; i < 3; ++i)
        if (!(other.local.lower[i] + offset[i] >= local.lower[i] && other.local.upper[i] + offset[i] <= local.upper[i]))
            return false;
    return true;
}

struct FiberCollection::Impl {
    sqlite3* db{};
    struct Candidate {
        int64_t fiber{}, block{};
        Bounds bounds;
    };
    struct CandidateCache {
        Bounds coverage;
        int minPoints{};
        double minLength{};
        FamilyFilter family{};
        std::vector<Candidate> rows;
    };
    struct CachedBlock {
        Block block;
        size_t bytes{};
        std::list<int64_t>::iterator recency;
        uint64_t generation{};
    };
    size_t cacheLimit{}, blockBytes{};
    uint64_t generation{};
    std::optional<CandidateCache> candidates;
    std::list<int64_t> recentBlocks;
    std::unordered_map<int64_t, CachedBlock> blocks;
    ViewCacheStats stats;
    ~Impl()
    {
        if (db)
            sqlite3_close(db);
    }
};
FiberCollection::FiberCollection(const std::filesystem::path& path, size_t cacheBytes) : impl_(std::make_unique<Impl>())
{
    impl_->cacheLimit = cacheBytes;
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    if (extension != ".afv")
        throw std::runtime_error("Expected an Automated Fiber Volume (.afv)");
    const auto name = path.u8string();
    const int rc = sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()), &impl_->db, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
    check(rc, impl_->db);
    sqlite3_busy_timeout(impl_->db, 1000);
    check(
        sqlite3_exec(impl_->db, "PRAGMA query_only=ON; PRAGMA cache_size=-2048; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;", nullptr, nullptr, nullptr),
        impl_->db);
    Statement app(impl_->db, "PRAGMA application_id");
    Statement version(impl_->db, "PRAGMA user_version");
    if (!app.row() || sqlite3_column_int(app.value, 0) != kApplicationId || !version.row() || sqlite3_column_int(version.value, 0) != 1)
        throw std::runtime_error("Not a supported Automated Fiber Volume");
    if (metadata("complete") != "true")
        throw std::runtime_error("Incomplete Automated Fiber Volume");
    // Preparing this query also detects builds without the RTree extension.
    Statement spatialIndex(impl_->db, "SELECT id FROM block_bounds LIMIT 0");
}
FiberCollection::~FiberCollection() = default;
std::string FiberCollection::metadata(const std::string& key) const
{
    Statement s(impl_->db, "SELECT value FROM metadata WHERE key=?");
    check(sqlite3_bind_text(s.value, 1, key.c_str(), -1, SQLITE_TRANSIENT), impl_->db);
    return s.row() ? text(s.value, 0) : "";
}
Summary FiberCollection::summary(int64_t id) const
{
    const std::string sql = std::string("SELECT ") + summaryColumns + " FROM fibers WHERE id=?";
    Statement s(impl_->db, sql.c_str());
    s.bind(1, id);
    if (!s.row())
        throw std::runtime_error("Unknown fiber ID");
    return readSummary(s.value);
}
std::vector<Summary> FiberCollection::catalogByLength(double beforeLength, int64_t afterId, int limit, double minLength, FamilyFilter family) const
{
    validateMinLength(minLength);
    if (std::isnan(beforeLength) || beforeLength < 0)
        throw std::runtime_error("Invalid fiber catalog cursor");
    // Written without OR so that SQLite walks the fibers_length index instead
    // of sorting the whole table for every page.
    const std::string sql = std::string("SELECT ") + summaryColumns +
                            " FROM fibers f WHERE length<=?1 AND NOT (length=?1 AND id<=?2) AND length>=?3 " + familyPredicate(family) +
                            " ORDER BY length DESC,id ASC LIMIT ?4";
    Statement s(impl_->db, sql.c_str());
    s.bind(1, beforeLength);
    s.bind(2, afterId);
    s.bind(3, minLength);
    s.bind(4, int64_t(std::clamp(limit, 1, 1000)));
    std::vector<Summary> out;
    while (s.row())
        out.push_back(readSummary(s.value));
    return out;
}
ViewCacheStats FiberCollection::viewCacheStats() const
{
    auto stats = impl_->stats;
    stats.retainedBytes = impl_->blockBytes + (impl_->candidates ? impl_->candidates->rows.capacity() * sizeof(Impl::Candidate) : 0);
    return stats;
}
RegionPage FiberCollection::viewRegion(
    const ViewRegion& view,
    int minPoints,
    size_t maximum,
    int64_t selected,
    size_t budget,
    const std::atomic_bool* cancel,
    double minLength,
    FamilyFilter family) const
{
    validateMinLength(minLength);
    for (size_t i = 0; i < 3; ++i)
        if (!std::isfinite(view.box.lower[i]) || !std::isfinite(view.box.upper[i]) || view.box.lower[i] > view.box.upper[i] ||
            !std::isfinite(view.local.lower[i]) || !std::isfinite(view.local.upper[i]) || view.local.lower[i] > view.local.upper[i] ||
            !std::isfinite(view.origin[i]) || !std::isfinite(view.u[i]) || !std::isfinite(view.v[i]) || !std::isfinite(view.n[i]))
            throw std::runtime_error("Invalid fiber view bounds");
    if (budget < 48)
        throw std::runtime_error("Fiber view budget is too small");
    auto checkCancelled = [&]() {
        if (cancel && cancel->load())
            throw std::runtime_error("Fiber view cancelled");
    };
    checkCancelled();
    ++impl_->generation;
    RegionPage out;
    if (family == FamilyFilter::None)
        return out;
    struct ProgressGuard {
        sqlite3* db;
        ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
    } guard{impl_->db};
    if (cancel)
        sqlite3_progress_handler(
            impl_->db, 1000, [](void* p) { return static_cast<const std::atomic_bool*>(p)->load() ? 1 : 0; },
            const_cast<std::atomic_bool*>(cancel));

    // Candidates are keyed by fiber, so a fiber's number of blocks does not
    // change its chance of being drawn, and offscreen fibers never enter.
    std::map<int64_t, std::vector<int64_t>> candidates;
    auto contains = [](const Bounds& outer, const Bounds& inner) {
        for (int i = 0; i < 3; ++i)
            if (inner.lower[i] < outer.lower[i] || inner.upper[i] > outer.upper[i])
                return false;
        return true;
    };
    auto intersects = [](const Bounds& a, const Bounds& b) {
        for (int i = 0; i < 3; ++i)
            if (a.upper[i] < b.lower[i] || a.lower[i] > b.upper[i])
                return false;
        return true;
    };
    auto append = [&](const Impl::Candidate& row) {
        if (intersects(row.bounds, view.box))
            candidates[row.fiber].push_back(row.block);
    };
    const auto& cached = impl_->candidates;
    if (cached && cached->minPoints == minPoints && cached->minLength == minLength && cached->family == family &&
        contains(cached->coverage, view.box)) {
        ++impl_->stats.candidateHits;
        for (const auto& row : cached->rows) {
            checkCancelled();
            append(row);
        }
    } else {
        impl_->candidates.reset();
        Impl::CandidateCache next{view.box, minPoints, minLength, family, {}};
        const size_t maxRows = impl_->cacheLimit / 4 / sizeof(Impl::Candidate);
        bool retain = maxRows > 0;
        if (retain) {
            // Cache metadata around the viewport and neighboring layers. Actual
            // membership and clipping are still tested against the exact view.
            for (int i = 0; i < 3; ++i) {
                const double margin = std::clamp((view.box.upper[i] - view.box.lower[i]) * .15, 16.0, 256.0);
                next.coverage.lower[i] -= margin;
                next.coverage.upper[i] += margin;
            }
            next.rows.reserve(std::min(size_t(4096), maxRows));
        }
        const std::string sql = std::string(
                                    // Materialize sorted lightweight RTree rows before touching the
                                    // geometry table. RTree traversal order otherwise causes scattered
                                    // table-page reads in large oblique views. LIMIT prevents flattening
                                    // this ordered subquery into the joins; temp data may spill to disk.
                                    "SELECT k.fiber_id,k.id,r.min_x,r.max_x,r.min_y,r.max_y,r.min_z,r.max_z FROM "
                                    "(SELECT * FROM block_bounds WHERE max_x>=? AND min_x<=? "
                                    "AND max_y>=? AND min_y<=? AND max_z>=? AND min_z<=? ORDER BY id LIMIT -1) r "
                                    "JOIN blocks k ON k.id=r.id JOIN fibers f ON f.id=k.fiber_id "
                                    "WHERE f.point_count>=? AND f.length>=? ") +
                                familyPredicate(family);
        Statement s(impl_->db, sql.c_str());
        ++impl_->stats.candidateQueries;
        for (int i = 0; i < 3; ++i) {
            s.bind(1 + 2 * i, next.coverage.lower[i]);
            s.bind(2 + 2 * i, next.coverage.upper[i]);
        }
        s.bind(7, int64_t(std::max(2, minPoints)));
        s.bind(8, minLength);
        while (s.row()) {
            checkCancelled();
            Impl::Candidate row{sqlite3_column_int64(s.value, 0), sqlite3_column_int64(s.value, 1), {}};
            for (int i = 0; i < 3; ++i) {
                row.bounds.lower[i] = sqlite3_column_double(s.value, 2 + 2 * i);
                row.bounds.upper[i] = sqlite3_column_double(s.value, 3 + 2 * i);
            }
            append(row);
            if (retain) {
                if (next.rows.size() == maxRows) {
                    retain = false;
                    std::vector<Impl::Candidate>().swap(next.rows);
                } else {
                    if (next.rows.size() == next.rows.capacity())
                        next.rows.reserve(std::min(maxRows, next.rows.capacity() * 2));
                    next.rows.push_back(row);
                }
            }
        }
        // Never publish partially read or cancelled spatial coverage.
        if (retain)
            impl_->candidates = std::move(next);
    }
    std::vector<std::pair<uint64_t, int64_t>> order;
    const auto salt = maximum ? collectionSalt(metadata("uuid")) : 0;
    for (const auto& [id, blocks] : candidates)
        if (!maximum || id != selected)
            order.emplace_back(maximum ? fiberPriority(id, salt) : 0, id);
    if (maximum) {
        std::sort(order.begin(), order.end());
        if (candidates.contains(selected))
            order.insert(order.begin(), {0, selected});
    }
    Statement blockQuery(
        impl_->db,
        "SELECT b.id,b.fiber_id,b.first_segment,f.family,b.points FROM blocks b "
        "JOIN fibers f ON f.id=b.fiber_id WHERE b.id=?");
    size_t visible = 0;
    for (const auto& [priority, id] : order) {
        checkCancelled();
        bool hit = false;
        for (auto blockId : candidates.at(id)) {
            checkCancelled();
            Block decoded;
            const Block* blockPtr;
            auto found = impl_->blocks.find(blockId);
            if (found != impl_->blocks.end()) {
                ++impl_->stats.blockHits;
                found->second.generation = impl_->generation;
                impl_->recentBlocks.splice(impl_->recentBlocks.begin(), impl_->recentBlocks, found->second.recency);
                blockPtr = &found->second.block;
            } else {
                ++impl_->stats.blockReads;
                blockQuery.reset();
                blockQuery.bind(1, blockId);
                if (!blockQuery.row())
                    throw std::runtime_error("Missing fiber block");
                decoded = readBlock(blockQuery.value);
                blockPtr = &decoded;
                // Include container/string overhead as well as coordinate bytes.
                const size_t bytes = decoded.points.capacity() * sizeof(Point) + decoded.family.capacity() + sizeof(Impl::CachedBlock) + 128;
                const size_t limit = impl_->cacheLimit - impl_->cacheLimit / 4;
                if (bytes <= limit) {
                    while (impl_->blockBytes + bytes > limit) {
                        const auto old = impl_->recentBlocks.back();
                        // A dense slice can exceed the cache. Do not evict the
                        // previous/current pass's working set for another scan:
                        // that would cause zero hits on every subsequent frame.
                        if (impl_->blocks.at(old).generation + 1 >= impl_->generation)
                            break;
                        impl_->blockBytes -= impl_->blocks.at(old).bytes;
                        impl_->blocks.erase(old);
                        impl_->recentBlocks.pop_back();
                    }
                    if (impl_->blockBytes + bytes <= limit) {
                        impl_->recentBlocks.push_front(blockId);
                        auto inserted = impl_->blocks.emplace(
                            blockId, Impl::CachedBlock{std::move(decoded), bytes, impl_->recentBlocks.begin(), impl_->generation});
                        impl_->blockBytes += bytes;
                        blockPtr = &inserted.first->second.block;
                    }
                }
            }
            const auto& block = *blockPtr;
            // Keep original coordinates and segment indices in contiguous runs.
            // The GUI clips their endpoints precisely at the viewport boundary.
            Block run{block.id, block.fiberId, block.firstSegment, block.family, {}};
            auto flush = [&]() {
                if (!run.points.empty()) {
                    out.blocks.push_back(std::move(run));
                    run = {block.id, block.fiberId, 0, block.family, {}};
                }
            };
            auto previous = view.project(block.points.front());
            for (size_t i = 1; i < block.points.size(); ++i) {
                const auto a = previous, b = view.project(block.points[i]);
                previous = b;
                double lo, hi;
                if (!clipSegment(a, b, view.local, lo, hi)) {
                    flush();
                    continue;
                }
                const size_t bytes = run.points.empty() ? 48 : 24;
                if (out.decodedBytes + bytes > budget) {
                    flush();
                    out.complete = false;
                    return out;
                }
                if (run.points.empty()) {
                    run.firstSegment = block.firstSegment + int64_t(i) - 1;
                    run.points.push_back(block.points[i - 1]);
                }
                run.points.push_back(block.points[i]);
                out.decodedBytes += bytes;
                hit = true;
            }
            flush();
        }
        if (hit && ++visible == maximum)
            break;
    }
    return out;
}
bool clipSegment(const Point& a, const Point& b, const Bounds& box, double& lo, double& hi)
{
    lo = 0;
    hi = 1;
    for (int i = 0; i < 3; ++i) {
        const double d = b[i] - a[i];
        if (d == 0) {
            if (a[i] < box.lower[i] || a[i] > box.upper[i])
                return false;
        } else {
            double x = (box.lower[i] - a[i]) / d, y = (box.upper[i] - a[i]) / d;
            if (x > y)
                std::swap(x, y);
            lo = std::max(lo, x);
            hi = std::min(hi, y);
            if (lo > hi)
                return false;
        }
    }
    return true;
}
std::vector<Block> FiberCollection::fiberBlocks(int64_t id, int64_t first, int limit) const
{
    Statement s(
        impl_->db,
        "SELECT b.id,b.fiber_id,b.first_segment,f.family,b.points FROM blocks b "
        "JOIN fibers f ON f.id=b.fiber_id WHERE b.fiber_id=? AND b.first_segment>=? "
        "ORDER BY b.first_segment LIMIT ?");
    s.bind(1, id);
    s.bind(2, first);
    s.bind(3, int64_t(std::clamp(limit, 1, 1024)));
    std::vector<Block> out;
    while (s.row())
        out.push_back(readBlock(s.value));
    return out;
}
Point FiberCollection::point(int64_t id, int64_t index) const
{
    const auto info = summary(id);
    if (index < 0 || index >= info.pointCount)
        throw std::runtime_error("Fiber point index out of bounds");
    Statement s(
        impl_->db,
        "SELECT b.id,b.fiber_id,b.first_segment,f.family,b.points FROM blocks b "
        "JOIN fibers f ON f.id=b.fiber_id WHERE b.fiber_id=? AND b.first_segment<=? "
        "ORDER BY b.first_segment DESC LIMIT 1");
    s.bind(1, id);
    s.bind(2, index);
    if (!s.row())
        throw std::runtime_error("Missing fiber block");
    const auto b = readBlock(s.value);
    return b.points.at(index - b.firstSegment);
}
std::string FiberCollection::annotation(int64_t id) const
{
    Statement s(impl_->db, "SELECT annotation FROM fibers WHERE id=?");
    s.bind(1, id);
    if (!s.row())
        throw std::runtime_error("Unknown fiber ID");
    return text(s.value, 0);
}
}  // namespace vc::fibers
