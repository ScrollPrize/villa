// RemoteGridFile: a GridStore v3 file read a part at a time answers every query
// exactly as the whole file does, fetches a small part of it, is reused by a
// later opener, and fails safely: a failed or corrupt read is reported and
// leaves nothing wrong behind, a changed remote file is never mixed into an old
// mirror, and a disk under its floor keeps the mirror in memory.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AnonymousPages.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/RemoteGridFile.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using vc::core::util::GridStore;
using vc::core::util::RemoteGridFile;

namespace {

using Paths = std::vector<std::shared_ptr<std::vector<cv::Point>>>;

// A grid like a normal-grid slice: many short polylines over a large plane.
// About the size of a real slice (~0.8 MB, 125 x 94 buckets).
std::string makeGrid(const fs::path& path, unsigned seed, int paths = 12000)
{
    GridStore gs(cv::Rect(0, 0, 8000, 6000), 64);
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> x(0, 7990), y(0, 5990), step(-6, 6), length(2, 40);
    // Lines added in raster order of their starts, as a slice's lines are traced.
    std::vector<cv::Point> starts;
    for (int p = 0; p < paths; ++p) starts.emplace_back(x(random), y(random));
    std::sort(starts.begin(), starts.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y / 64 != b.y / 64 ? a.y < b.y : a.x < b.x;
    });
    for (int p = 0; p < paths; ++p) {
        std::vector<cv::Point> line{starts[p]};
        const int n = p % 97 == 0 ? 400 : length(random);  // a few records longer than the padding
        for (int i = 1; i < n; ++i)
            line.push_back({std::clamp(line.back().x + step(random), 0, 7999), std::clamp(line.back().y + step(random), 0, 5999)});
        gs.add(line);
    }
    gs.save(path.string());
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

// Serves bytes like a remote store: versions, request counts, injected failures.
struct Remote {
    std::mutex mutex;
    std::string bytes, version = "v1";
    std::atomic<uint64_t> requests{0}, served{0};
    std::atomic<int> failNext{0};
    bool missing = false;

    RemoteGridFile::RangeFetcher fetcher()
    {
        return [this](const std::vector<std::pair<uint64_t, uint64_t>>& ranges, const std::string& wanted) {
            std::lock_guard lock(mutex);
            if (missing) throw RemoteGridFile::NotFound("no such grid");
            if (failNext > 0) {
                --failNext;
                throw std::runtime_error("connection reset");
            }
            if (!wanted.empty() && wanted != version) throw RemoteGridFile::Changed("grid changed");
            RemoteGridFile::Fetched out;
            out.version = version;
            for (const auto& [offset, length] : ranges) {
                ++requests;
                served += length;
                out.parts.push_back(bytes.substr(offset, length));
            }
            return out;
        };
    }
};

std::vector<std::vector<cv::Point>> points(const Paths& paths)
{
    std::vector<std::vector<cv::Point>> out;
    for (const auto& p : paths) out.push_back(*p);
    return out;
}

std::vector<cv::Rect> randomRects(unsigned seed, int n)
{
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> x(-100, 8000), y(-100, 6000), size(1, 400);
    std::vector<cv::Rect> rects;
    for (int i = 0; i < n; ++i) rects.emplace_back(x(random), y(random), size(random), size(random));
    return rects;
}

struct Scratch {
    fs::path dir = fs::temp_directory_path() / ("vc_remote_grid_" + std::to_string(vc::pages::pid()));
    Scratch()
    {
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    ~Scratch()
    {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

std::shared_ptr<RemoteGridFile> open(Remote& remote, const fs::path& local, RemoteGridFile::Options options = {})
{
    return std::make_shared<RemoteGridFile>("test://grid", local.string(), remote.fetcher(), options);
}

}  // namespace

TEST_CASE("Remote grid: every query matches the whole file, from a small part of it")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 1);
    GridStore whole((s.dir / "whole.grid").string());
    GridStore partial(open(remote, s.dir / "mirror" / "000000.grid"));
    CHECK(partial.numSegments() == whole.numSegments());  // reads the whole file
    remote.requests = remote.served = 0;

    GridStore fresh(open(remote, s.dir / "fresh" / "000000.grid"));
    const auto small = fresh.get(cv::Point2f(1000, 1000), 80);
    CHECK(points(small) == points(whole.get(cv::Point2f(1000, 1000), 80)));
    CHECK_FALSE(small.empty());
    MESSAGE("served " << remote.served.load() << " of " << remote.bytes.size() << " in " << remote.requests.load() << " ranges");
    CHECK(remote.served < remote.bytes.size() / 10);

    for (const auto& rect : randomRects(2, 300)) CHECK(points(fresh.get(rect)) == points(whole.get(rect)));
    CHECK(points(fresh.get_all()) == points(whole.get_all()));
    CHECK(fresh.numSegments() == whole.numSegments());
}

TEST_CASE("Remote grid: the plane's layout fetches the first region with the header")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 3);
    GridStore whole((s.dir / "whole.grid").string());
    const auto layout = open(remote, s.dir / "learn" / "g.grid")->layout();

    RemoteGridFile::Options options;
    options.layout = &layout;
    options.first = RemoteGridFile::BucketRect{14, 14, 18, 18};
    remote.requests = 0;
    GridStore hinted(open(remote, s.dir / "hinted" / "g.grid", options));
    const auto rounds = remote.requests.load();
    CHECK(rounds == 3);  // header, the region's index rows, the total
    CHECK(points(hinted.get(cv::Point2f(1000, 1000), 80)) == points(whole.get(cv::Point2f(1000, 1000), 80)));

    // A layout the file does not have is not trusted: the rows are read again.
    RemoteGridFile::Layout wrong = layout;
    wrong.bw += 64;
    wrong.num_buckets = ((wrong.bw + 63) / 64) * ((wrong.bh + 63) / 64);
    options.layout = &wrong;
    GridStore misled(open(remote, s.dir / "misled" / "g.grid", options));
    for (const auto& rect : randomRects(4, 100)) CHECK(points(misled.get(rect)) == points(whole.get(rect)));
}

namespace {
uint32_t testCrc32(const std::string& bytes)
{
    uint32_t crc = ~0u;
    for (const unsigned char c : bytes) {
        crc ^= c;
        for (int k = 0; k < 8; ++k) crc = (crc & 1) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
    }
    return ~crc;
}

// A pack record as RemoteGridFile writes one: type, little-endian length,
// payload, crc32.
std::string packRecord(uint8_t type, const std::string& payload)
{
    std::string r(1, char(type));
    for (int i = 0; i < 4; ++i) r.push_back(char((payload.size() >> (8 * i)) & 0xff));
    r += payload;
    const uint32_t crc = testCrc32(r);
    for (int i = 0; i < 4; ++i) r.push_back(char((crc >> (8 * i)) & 0xff));
    return r;
}

std::string tilePayload(int tx, int ty)
{
    std::string p;
    for (const uint32_t v : {uint32_t(tx), uint32_t(ty)})
        for (int i = 0; i < 4; ++i) p.push_back(char((v >> (8 * i)) & 0xff));
    return p;
}

void append(const fs::path& path, const std::string& bytes)
{
    std::ofstream(path, std::ios::binary | std::ios::app) << bytes;
}
}  // namespace

TEST_CASE("Remote grid: a later opener replays the pack; a damaged, cut-off or stale pack is not trusted")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 5);
    GridStore whole((s.dir / "whole.grid").string());
    const auto local = s.dir / "m" / "000000.grid";
    const fs::path pack = local.string() + ".ranges";
    const auto rects = randomRects(6, 60);
    const auto before = RemoteGridFile::globalStats();
    {
        GridStore first(open(remote, local));
        for (const auto& rect : rects) (void)first.get(rect);
    }
    const auto after = RemoteGridFile::globalStats();
    const uint64_t fetched = after.bytes - before.bytes, requests = after.requests - before.requests;
    REQUIRE(fs::exists(pack));
    CHECK_FALSE(fs::exists(local));  // never where the whole file goes
    // Only what was fetched is on disk (each piece once, with a small record
    // header), not a file of the remote file's size.
    CHECK(fs::file_size(pack) <= fetched + 21 * requests + 8 * 4096);
    CHECK(fs::file_size(pack) < remote.bytes.size());
    for (const auto& entry : fs::directory_iterator(s.dir / "m")) CHECK(entry.path() == pack);
    remote.requests = 0;
    {
        GridStore again(open(remote, local));
        for (const auto& rect : rects) CHECK(points(again.get(rect)) == points(whole.get(rect)));
    }
    CHECK(remote.requests == 0);

    // Not a pack: a new one.
    {
        std::ofstream(pack, std::ios::binary | std::ios::trunc) << "VCGR garbage";
    }
    remote.requests = 0;
    {
        GridStore again(open(remote, local));
        for (const auto& rect : rects) CHECK(points(again.get(rect)) == points(whole.get(rect)));
    }
    CHECK(remote.requests > 0);

    // A record another opener is still appending, or one that does not check
    // out, is not read: either would name tile (15, 11), which nobody loaded.
    const cv::Point2f far(15 * 512 + 256, 11 * 512 + 256);
    REQUIRE_FALSE(whole.get(far, 80).empty());
    for (const bool cut : {true, false}) {
        const auto fresh = s.dir / (cut ? "cut" : "bad") / "g.grid";
        const fs::path freshPack = fresh.string() + ".ranges";
        {
            GridStore first(open(remote, fresh));
            (void)first.get(cv::Point2f(100, 100), 80);
        }
        const auto before = fs::file_size(freshPack);
        std::string record = packRecord(2, tilePayload(15, 11));
        if (cut)
            record.resize(record.size() - 3);
        else
            record[6] ^= 1;
        append(freshPack, record);
        {
            GridStore again(open(remote, fresh));
            CHECK(points(again.get(far, 80)) == points(whole.get(far, 80)));
            CHECK(points(again.get(cv::Point2f(100, 100), 80)) == points(whole.get(cv::Point2f(100, 100), 80)));
        }
        // The damaged tail was dropped, so what the second opener added is
        // read by a third without fetching.
        CHECK(fs::file_size(freshPack) > before);
        remote.requests = 0;
        {
            GridStore third(open(remote, fresh));
            CHECK(points(third.get(far, 80)) == points(whole.get(far, 80)));
        }
        CHECK(remote.requests == 0);
    }

    // A byte flipped inside a data record: everything from that record on is
    // ignored, and what it held is fetched again.
    {
        std::string bytes;
        {
            std::ifstream in(pack, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        bytes[bytes.size() / 2] ^= 0x5a;
        std::ofstream(pack, std::ios::binary | std::ios::trunc) << bytes;
    }
    {
        GridStore again(open(remote, local));
        for (const auto& rect : rects) CHECK(points(again.get(rect)) == points(whole.get(rect)));
    }

    // A pack marked stale is never reused.
    append(pack, packRecord(4, ""));
    remote.requests = 0;
    {
        GridStore again(open(remote, local));
        CHECK(points(again.get(rects[0])) == points(whole.get(rects[0])));
    }
    CHECK(remote.requests > 0);
}

TEST_CASE("Remote grid: resident bytes count what is held in memory, each byte once")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 21);
    const auto local = s.dir / "m" / "g.grid";
    uint64_t held = 0;
    {
        auto file = open(remote, local);
        const uint64_t start = file->residentBytes();
        CHECK(start >= 52);  // the header at least
        GridStore store(file);
        (void)store.get(cv::Point2f(300, 300), 80);
        const uint64_t one = file->residentBytes();
        CHECK(one > start);
        (void)store.get(cv::Point2f(300, 300), 80);  // nothing new
        CHECK(file->residentBytes() == one);
        for (const auto& rect : randomRects(22, 30)) (void)store.get(rect);
        held = file->residentBytes();
        CHECK(held > one);
        CHECK(held < remote.bytes.size());
    }
    // Replayed from the pack: the same bytes are held again.
    auto replayed = open(remote, local);
    CHECK(replayed->residentBytes() == held);
    replayed->ensureAll();
    CHECK(replayed->residentBytes() == remote.bytes.size());
}

TEST_CASE("Remote grid: a failed read is reported, not cached, and fetched again")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 7);
    GridStore whole((s.dir / "whole.grid").string());
    GridStore partial(open(remote, s.dir / "m" / "g.grid"));
    const auto failures = RemoteGridFile::globalStats().failures;

    // Each step of a tile's load can fail; nothing it half-loaded is ever read.
    for (int failAt = 0; failAt < 3; ++failAt) {
        const cv::Point2f at(500.f + 700.f * failAt, 1500.f);
        remote.failNext = 0;
        // Let `failAt` steps succeed, then fail one.
        std::atomic<int> calls{-1000};  // the open itself never fails
        auto fetch = remote.fetcher();
        auto flaky = [&](const auto& ranges, const std::string& version) {
            if (calls++ == failAt) throw std::runtime_error("connection reset");
            return fetch(ranges, version);
        };
        GridStore once(std::make_shared<RemoteGridFile>("test://grid", (s.dir / ("f" + std::to_string(failAt)) / "g.grid").string(),
                                                        flaky, RemoteGridFile::Options{}));
        calls = 0;
        // The failed step is fetched again within the same query.
        CHECK(points(once.get(at, 80)) == points(whole.get(at, 80)));
        CHECK(calls > failAt);
        CHECK_FALSE(whole.get(at, 80).empty());
    }
    CHECK(RemoteGridFile::globalStats().failures == failures);  // recovered, so nothing reported
    (void)partial;

    // What the retries completed is whole in the packs: a later opener reads
    // it back with no network at all.
    auto offline = [](const auto&, const std::string&) -> RemoteGridFile::Fetched {
        throw std::runtime_error("offline");
    };
    for (int failAt = 0; failAt < 3; ++failAt) {
        const cv::Point2f at(500.f + 700.f * failAt, 1500.f);
        GridStore later(std::make_shared<RemoteGridFile>("test://grid", (s.dir / ("f" + std::to_string(failAt)) / "g.grid").string(),
                                                         offline, RemoteGridFile::Options{}));
        CHECK(points(later.get(at, 80)) == points(whole.get(at, 80)));
    }
}

TEST_CASE("Remote grid: an opener never appends to a pack that replaced its own")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 23);
    const auto local = s.dir / "m" / "g.grid";
    const fs::path pack = local.string() + ".ranges";
    auto first = open(remote, local);
    GridStore firstStore(first);
    (void)firstStore.get(cv::Point2f(300, 300), 80);
    {
        // Another pack takes its place in the same file (so the same inode, as a
        // file system that reuses inode numbers would give a new one).
        GridStore other(open(remote, s.dir / "other" / "g.grid"));
        (void)other.get(cv::Point2f(4000, 3000), 80);
    }
    std::string replacement;
    {
        std::ifstream in(s.dir / "other" / "g.grid.ranges", std::ios::binary);
        replacement.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::ofstream(pack, std::ios::binary | std::ios::trunc) << replacement;
    GridStore whole((s.dir / "whole.grid").string());
    const cv::Point2f far(7000, 5000);
    CHECK(points(firstStore.get(far, 80)) == points(whole.get(far, 80)));  // still answers from memory
    CHECK(fs::file_size(pack) == replacement.size());                     // but appends nothing there
}

TEST_CASE("Remote grid: a pack the disk budget refuses stops growing")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 24);
    GridStore whole((s.dir / "whole.grid").string());
    std::atomic<int> charges{0};
    RemoteGridFile::Options options;
    options.charge = [&](const std::string&, uint64_t) { return charges++ == 0; };  // only the first fits
    const fs::path pack = (s.dir / "m" / "g.grid").string() + ".ranges";
    auto file = open(remote, s.dir / "m" / "g.grid", options);
    GridStore store(file);
    const auto rects = randomRects(25, 200);
    size_t i = 0;
    for (; i < rects.size() && charges < 2; ++i) CHECK(points(store.get(rects[i])) == points(whole.get(rects[i])));
    REQUIRE(charges >= 2);  // refused once
    const auto size = fs::file_size(pack);
    for (; i < rects.size(); ++i) CHECK(points(store.get(rects[i])) == points(whole.get(rects[i])));
    CHECK(fs::file_size(pack) == size);
}

TEST_CASE("Remote grid: a changed remote file is not mixed into the mirror")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "v1.grid", 9);
    const auto local = s.dir / "m" / "g.grid";
    const auto v1_whole = std::make_unique<GridStore>((s.dir / "v1.grid").string());
    {
        GridStore v1(open(remote, local));
        CHECK(points(v1.get(cv::Point2f(300, 300), 80)) == points(v1_whole->get(cv::Point2f(300, 300), 80)));
    }
    {
        std::lock_guard lock(remote.mutex);
        remote.bytes = makeGrid(s.dir / "v2.grid", 10);
        remote.version = "v2";
    }
    GridStore v2_whole((s.dir / "v2.grid").string());
    {
        // The old mirror is reused, and its old tiles still read as before, but a
        // new tile would come from the new file: refused.
        GridStore reused(open(remote, local));
        CHECK(points(reused.get(cv::Point2f(300, 300), 80)) == points(v1_whole->get(cv::Point2f(300, 300), 80)));
        CHECK(reused.get(cv::Point2f(7000, 5500), 80).empty());
    }
    // Marked stale: the next opener builds a mirror of the new file.
    GridStore fresh(open(remote, local));
    for (const auto& rect : randomRects(11, 50)) CHECK(points(fresh.get(rect)) == points(v2_whole.get(rect)));
}

TEST_CASE("Remote grid: corrupt files raise or read nothing, never past the mirror")
{
    Scratch s;
    Remote remote;
    const auto good = makeGrid(s.dir / "whole.grid", 12);
    SUBCASE("not a grid")
    {
        remote.bytes = std::string(4096, 'x');
        CHECK_THROWS(open(remote, s.dir / "a" / "g.grid"));
    }
    SUBCASE("header past the end")
    {
        remote.bytes = good;
        // paths offset beyond the metadata
        const uint32_t huge = __builtin_bswap32(0x7fffffffu);
        std::memcpy(remote.bytes.data() + 40, &huge, 4);
        CHECK_THROWS(open(remote, s.dir / "b" / "g.grid"));
    }
    SUBCASE("bucket index out of bounds")
    {
        remote.bytes = good;
        const uint32_t huge = __builtin_bswap32(0x7ffffff0u);
        for (int i = 0; i < 2000; ++i) std::memcpy(remote.bytes.data() + 52 + 4 * i, &huge, 4);
        GridStore partial(open(remote, s.dir / "c" / "g.grid"));
        for (const auto& rect : randomRects(13, 40)) (void)partial.get(rect);  // no crash
        CHECK(partial.get(cv::Point2f(100, 100), 80).empty());
    }
    SUBCASE("path offsets out of bounds")
    {
        remote.bytes = good;
        uint32_t buckets = 0;
        std::memcpy(&buckets, good.data() + 28, 4);
        buckets = __builtin_bswap32(buckets);
        const uint32_t bucketsOffset = 52;
        const uint64_t flat = bucketsOffset + (uint64_t(buckets) + 1) * 4;
        const uint32_t huge = __builtin_bswap32(0x7ffffff0u);
        for (int i = 0; i < 3000; ++i) std::memcpy(remote.bytes.data() + flat + 4 * i, &huge, 4);
        GridStore partial(open(remote, s.dir / "d" / "g.grid"));
        for (const auto& rect : randomRects(14, 40)) (void)partial.get(rect);  // no crash
    }
}

TEST_CASE("Remote grid: missing files, memory mirrors and the free-space floor")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 15);
    GridStore whole((s.dir / "whole.grid").string());

    remote.missing = true;
    CHECK_THROWS_AS(open(remote, s.dir / "x" / "g.grid"), RemoteGridFile::NotFound);
    remote.missing = false;

    RemoteGridFile::Options memory;
    memory.in_memory = true;
    auto inMemory = open(remote, s.dir / "mem" / "g.grid", memory);
    CHECK(inMemory->inMemory());
    GridStore a(inMemory);
    for (const auto& rect : randomRects(16, 50)) CHECK(points(a.get(rect)) == points(whole.get(rect)));
    CHECK_FALSE(fs::exists(s.dir / "mem" / "g.grid.ranges"));

    RemoteGridFile::Options starved;
    starved.min_free_bytes = uint64_t(1) << 62;  // more than any disk has free
    auto floor = open(remote, s.dir / "floor" / "g.grid", starved);
    CHECK(floor->inMemory());
    GridStore b(floor);
    for (const auto& rect : randomRects(17, 50)) CHECK(points(b.get(rect)) == points(whole.get(rect)));
    CHECK_FALSE(fs::exists(s.dir / "floor" / "g.grid.ranges"));

    // The disk budget: a mirror it refuses is kept in memory; one it takes is
    // charged what it occupies as it grows, far below its apparent size.
    RemoteGridFile::Options refused;
    refused.charge = [](const std::string&, uint64_t) { return false; };
    auto noRoom = open(remote, s.dir / "refused" / "g.grid", refused);
    CHECK(noRoom->inMemory());
    CHECK_FALSE(fs::exists(s.dir / "refused" / "g.grid.ranges"));

    std::mutex chargeMutex;
    std::vector<std::pair<std::string, uint64_t>> charges;
    RemoteGridFile::Options budgeted;
    budgeted.charge = [&](const std::string& path, uint64_t bytes) {
        std::lock_guard lock(chargeMutex);
        charges.push_back({path, bytes});
        return true;
    };
    {
        GridStore c(open(remote, s.dir / "charged" / "g.grid", budgeted));
        for (const auto& rect : randomRects(20, 12)) CHECK(points(c.get(rect)) == points(whole.get(rect)));
    }  // closing charges the last growth
    REQUIRE(charges.size() >= 2);
    for (const auto& [path, bytes] : charges) CHECK(path == (s.dir / "charged" / "g.grid.ranges").string());
    CHECK(charges.back().second > charges.front().second);
    CHECK(charges.back().second < remote.bytes.size());
}

TEST_CASE("Remote grid: concurrent queries and concurrent openers of one mirror")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 18);
    GridStore whole((s.dir / "whole.grid").string());
    const auto local = s.dir / "m" / "g.grid";
    const auto rects = randomRects(19, 400);

    // Openers in parallel, as two processes sharing a cache would; each queries.
    std::vector<std::thread> threads;
    std::atomic<int> mismatches{0};
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            GridStore mine(open(remote, local));
            for (std::size_t i = t; i < rects.size(); i += 8)
                if (points(mine.get(rects[i])) != points(whole.get(rects[i]))) ++mismatches;
        });
    for (auto& thread : threads) thread.join();
    CHECK(mismatches == 0);

    // Many threads on one store.
    GridStore shared(open(remote, s.dir / "n" / "g.grid"));
    threads.clear();
    for (int t = 0; t < 16; ++t)
        threads.emplace_back([&, t] {
            for (std::size_t i = t; i < rects.size(); i += 16)
                if (points(shared.get(rects[i])) != points(whole.get(rects[i]))) ++mismatches;
        });
    for (auto& thread : threads) thread.join();
    CHECK(mismatches == 0);
}

TEST_CASE("Remote grid: a region that fails to arrive is recovered, never read as empty")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 11);
    GridStore whole((s.dir / "whole.grid").string());
    const auto rects = randomRects(12, 40);

    SUBCASE("a one-off failure is fetched again")
    {
        GridStore store(open(remote, s.dir / "mirror" / "000000.grid"));
        for (const auto& rect : rects) {
            remote.failNext = 1;
            CHECK(points(store.get(rect)) == points(whole.get(rect)));
        }
    }

    SUBCASE("persistent range failures fall back to reading the file whole")
    {
        int wholeReads = 0;
        GridStore store(open(remote, s.dir / "mirror" / "000000.grid"), [&]() -> std::shared_ptr<const GridStore> {
            ++wholeReads;
            return std::make_shared<GridStore>((s.dir / "whole.grid").string());
        });
        remote.failNext = 1 << 30;
        for (const auto& rect : rects) CHECK(points(store.get(rect)) == points(whole.get(rect)));
        CHECK(store.numSegments() == whole.numSegments());
        CHECK(points(store.get_all()).size() == points(whole.get_all()).size());
        CHECK(wholeReads == 1);  // switched once, then every query reads the whole copy
    }

    SUBCASE("a remote file that changed version is read whole at its new version, not mixed")
    {
        auto mirror = open(remote, s.dir / "mirror" / "000000.grid");
        const std::string v2 = makeGrid(s.dir / "v2.grid", 13);
        GridStore newer((s.dir / "v2.grid").string());
        GridStore store(mirror, [&]() -> std::shared_ptr<const GridStore> {
            return std::make_shared<GridStore>((s.dir / "v2.grid").string());
        });
        {
            std::lock_guard lock(remote.mutex);
            remote.bytes = v2;
            remote.version = "v2";
        }
        for (const auto& rect : rects) CHECK(points(store.get(rect)) == points(newer.get(rect)));
    }

    SUBCASE("without a fallback the failure is reported, and the region is fetched again later")
    {
        GridStore store(open(remote, s.dir / "mirror" / "000000.grid"));
        const auto failuresBefore = RemoteGridFile::globalStats().failures;
        remote.failNext = 1 << 30;
        CHECK(store.get(rects[0]).empty());
        CHECK(RemoteGridFile::globalStats().failures > failuresBefore);
        remote.failNext = 0;
        CHECK(points(store.get(rects[0])) == points(whole.get(rects[0])));
    }
}

namespace {
// A disk budget like vc_reader's: each file is charged its whole size, and a
// size the limit cannot take is refused.
struct Budget {
    std::mutex mutex;
    std::map<std::string, uint64_t> charged;
    uint64_t limit;
    explicit Budget(uint64_t limit) : limit(limit) {}
    std::function<bool(const std::string&, uint64_t)> charge()
    {
        return [this](const std::string& path, uint64_t bytes) {
            std::lock_guard lock(mutex);
            uint64_t others = 0;
            for (const auto& [p, b] : charged) if (p != path) others += b;
            if (others + bytes > limit) return false;
            charged[path] = bytes;
            return true;
        };
    }
    // Every pack on disk is within what was charged for it, and the charges within the limit.
    void check(const std::vector<fs::path>& packs)
    {
        std::lock_guard lock(mutex);
        uint64_t total = 0;
        for (const auto& [p, b] : charged) total += b;
        CHECK(total <= limit);
        for (const auto& pack : packs) {
            std::error_code ec;
            const auto size = fs::file_size(pack, ec);
            if (ec) continue;  // never written
            INFO(pack.string());
            CHECK(size <= charged[pack.string()]);
        }
    }
};
}  // namespace

TEST_CASE("Remote grid: a pack never holds more than its budget accepted")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 21);
    GridStore whole((s.dir / "whole.grid").string());

    SUBCASE("one writer under a 16 KiB budget")
    {
        Budget budget(16 << 10);
        RemoteGridFile::Options options;
        options.charge = budget.charge();
        const auto local = s.dir / "m" / "000000.grid";
        GridStore store(open(remote, local, options));
        for (const auto& rect : randomRects(22, 60)) {
            CHECK(points(store.get(rect)) == points(whole.get(rect)));  // answers come from memory
            budget.check({fs::path(local.string() + ".ranges")});
        }
    }

    SUBCASE("concurrent writers sharing a budget")
    {
        Budget budget(96 << 10);
        RemoteGridFile::Options options;
        options.charge = budget.charge();
        std::vector<fs::path> packs;
        std::vector<std::unique_ptr<GridStore>> stores;
        for (int i = 0; i < 3; ++i) {
            const auto local = s.dir / ("w" + std::to_string(i)) / "000000.grid";
            packs.push_back(local.string() + ".ranges");
            stores.push_back(std::make_unique<GridStore>(open(remote, local, options)));
        }
        std::vector<std::thread> threads;
        std::atomic<int> wrong{0};
        for (int t = 0; t < 6; ++t)
            threads.emplace_back([&, t] {
                for (const auto& rect : randomRects(30 + t, 40))
                    if (points(stores[t % 3]->get(rect)) != points(whole.get(rect))) ++wrong;
            });
        for (auto& thread : threads) thread.join();
        CHECK(wrong == 0);
        budget.check(packs);
    }

    SUBCASE("a budget that throws stops the pack, not the queries")
    {
        RemoteGridFile::Options options;
        std::atomic<int> calls{0};
        options.charge = [&](const std::string&, uint64_t) -> bool {
            if (++calls > 1) throw std::runtime_error("database is locked");
            return true;
        };
        GridStore store(open(remote, s.dir / "t" / "000000.grid", options));
        for (const auto& rect : randomRects(23, 20)) CHECK(points(store.get(rect)) == points(whole.get(rect)));
    }
}

TEST_CASE("Remote grid: under the free-space floor a pack on disk is still read, and not grown")
{
    Scratch s;
    Remote remote;
    remote.bytes = makeGrid(s.dir / "whole.grid", 24);
    GridStore whole((s.dir / "whole.grid").string());
    const auto local = s.dir / "m" / "000000.grid";
    const auto rects = randomRects(25, 20);
    {
        GridStore first(open(remote, local));
        for (const auto& rect : rects) (void)first.get(rect);
    }
    const auto pack = fs::path(local.string() + ".ranges");
    const auto packed = fs::file_size(pack);
    remote.requests = 0;
    RemoteGridFile::Options starved;
    starved.min_free_bytes = std::numeric_limits<uint64_t>::max() / 2;  // always below the floor
    GridStore again(open(remote, local, starved));
    for (const auto& rect : rects) CHECK(points(again.get(rect)) == points(whole.get(rect)));
    CHECK(remote.requests == 0);  // all from the pack
    for (const auto& rect : randomRects(26, 20)) CHECK(points(again.get(rect)) == points(whole.get(rect)));
    CHECK(fs::file_size(pack) == packed);  // new regions stay in memory
}
