// Coverage for core/src/SparseChunkSpool.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/SparseChunkSpool.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace vc::core::util;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_spool_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

SparseChunkRecordU8x7 makeRec(uint8_t z, uint8_t y, uint8_t x, uint8_t v0 = 0)
{
    SparseChunkRecordU8x7 r;
    r.z = z; r.y = y; r.x = x;
    for (int i = 0; i < 7; ++i) r.values[i] = static_cast<uint8_t>(v0 + i);
    return r;
}

} // namespace

TEST_CASE("SparseChunkIndex operator== and hash")
{
    SparseChunkIndex a{1, 2, 3};
    SparseChunkIndex b{1, 2, 3};
    SparseChunkIndex c{1, 2, 4};
    CHECK(a == b);
    CHECK_FALSE(a == c);
    SparseChunkIndexHash h;
    CHECK(h(a) == h(b));
    // Different keys should usually differ (but not guaranteed) — just smoke.
    CHECK(h(a) != 0);
}

TEST_CASE("sparseChunkIndexLess is a strict weak ordering on (z,y,x)")
{
    CHECK(sparseChunkIndexLess({1, 0, 0}, {2, 0, 0}));
    CHECK_FALSE(sparseChunkIndexLess({2, 0, 0}, {1, 0, 0}));
    CHECK(sparseChunkIndexLess({1, 0, 0}, {1, 1, 0}));
    CHECK(sparseChunkIndexLess({1, 1, 0}, {1, 1, 5}));
    CHECK_FALSE(sparseChunkIndexLess({1, 1, 5}, {1, 1, 5}));
}

TEST_CASE("constructor: zero chunk dim throws")
{
    auto d = tmpDir("ctor_zero");
    CHECK_THROWS_AS(SparseChunkSpool(d, {0, 64, 64}, {128, 128, 128}, 1024),
                    std::runtime_error);
    CHECK_THROWS_AS(SparseChunkSpool(d, {64, 0, 64}, {128, 128, 128}, 1024),
                    std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("constructor: chunk dim > 255 throws")
{
    auto d = tmpDir("ctor_big");
    CHECK_THROWS_AS(SparseChunkSpool(d, {256, 64, 64}, {1024, 1024, 1024}, 1024),
                    std::runtime_error);
    fs::remove_all(d);
}

TEST_CASE("appendChunkRecords / readChunkRecords in-memory mode round-trip")
{
    auto d = tmpDir("mem_rt");
    SparseChunkSpool spool(d, {64, 64, 64}, {128, 128, 128}, /*inMemoryMaxBytes=*/1024 * 1024);
    SparseChunkIndex k{0, 0, 0};
    std::vector<SparseChunkRecordU8x7> recs = {
        makeRec(0, 0, 0, 1),
        makeRec(1, 1, 1, 2),
    };
    spool.appendChunkRecords(k, recs);

    std::vector<SparseChunkRecordU8x7> out;
    bool got = spool.readChunkRecords(k, out);
    CHECK(got);
    REQUIRE(out.size() == 2);
    CHECK(out[0].z == 0);
    CHECK(out[1].x == 1);
    CHECK(int(out[0].values[0]) == 1);
    fs::remove_all(d);
}

TEST_CASE("appendChunkRecords: empty record list is a no-op")
{
    auto d = tmpDir("empty");
    SparseChunkSpool spool(d, {64, 64, 64}, {128, 128, 128}, 1024);
    spool.appendChunkRecords({0, 0, 0}, {});
    CHECK(spool.touchedChunks().empty());
    CHECK(spool.stats().appendedRecords == 0);
    fs::remove_all(d);
}

TEST_CASE("readChunkRecords: missing chunk returns false")
{
    auto d = tmpDir("missing");
    SparseChunkSpool spool(d, {64, 64, 64}, {128, 128, 128}, 1024);
    std::vector<SparseChunkRecordU8x7> out;
    CHECK_FALSE(spool.readChunkRecords({9, 9, 9}, out));
    CHECK(out.empty());
    fs::remove_all(d);
}

TEST_CASE("inMemoryMaxBytes=0 always spills to disk")
{
    auto d = tmpDir("nomem");
    SparseChunkSpool spool(d, {64, 64, 64}, {128, 128, 128}, 0);
    SparseChunkIndex k{0, 0, 0};
    spool.appendChunkRecords(k, {makeRec(0, 0, 0)});
    auto stats = spool.stats();
    CHECK(stats.spillFiles >= 1);
    CHECK(stats.appendedRecords == 1);
    CHECK(stats.inMemoryBytes == 0);
    // touched
    auto touched = spool.touchedChunks();
    REQUIRE(touched.size() == 1);
    CHECK(touched[0] == k);
    // read back through readChunkRecords
    std::vector<SparseChunkRecordU8x7> out;
    CHECK(spool.readChunkRecords(k, out));
    CHECK(out.size() == 1);
    fs::remove_all(d);
}

TEST_CASE("appending across many chunks evicts oldest via memOrder")
{
    auto d = tmpDir("evict");
    // Budget is very small (just 1 record = 10 bytes).
    SparseChunkSpool spool(d, {64, 64, 64}, {255, 255, 255}, /*budget=*/10);
    // Append into 3 distinct chunks. Older ones must be spilled.
    for (uint32_t i = 0; i < 3; ++i) {
        SparseChunkIndex k{i, 0, 0};
        spool.appendChunkRecords(k, {makeRec(0, 0, 0, static_cast<uint8_t>(i))});
    }
    // All 3 chunks should be readable (some from memory, others from spill).
    for (uint32_t i = 0; i < 3; ++i) {
        std::vector<SparseChunkRecordU8x7> out;
        CHECK(spool.readChunkRecords({i, 0, 0}, out));
        CHECK_FALSE(out.empty());
    }
    auto s = spool.stats();
    CHECK(s.appendedRecords == 3);
    CHECK(s.spillFiles >= 1);
    fs::remove_all(d);
}

TEST_CASE("touchedChunks lists every chunk we've appended to")
{
    auto d = tmpDir("touched");
    SparseChunkSpool spool(d, {64, 64, 64}, {255, 255, 255}, 1024);
    spool.appendChunkRecords({0, 0, 0}, {makeRec(1, 1, 1)});
    spool.appendChunkRecords({1, 0, 0}, {makeRec(2, 2, 2)});
    auto touched = spool.touchedChunks();
    CHECK(touched.size() == 2);
    fs::remove_all(d);
}

TEST_CASE("accessors: chunkShape / volumeShape / spoolDir")
{
    auto d = tmpDir("accessors");
    Shape3 cs{16, 32, 48};
    Shape3 vs{128, 256, 384};
    SparseChunkSpool spool(d, cs, vs, 1024);
    CHECK(spool.chunkShape() == cs);
    CHECK(spool.volumeShape() == vs);
    CHECK(spool.spoolDir() == d);
    fs::remove_all(d);
}

// --- SparseChunkSpoolBuffer ---

TEST_CASE("SparseChunkSpoolBuffer::emit batches per-chunk; flushAll writes to spool")
{
    auto d = tmpDir("buf");
    SparseChunkSpool spool(d, {64, 64, 64}, {128, 128, 128}, 1024);
    SparseChunkSpoolBuffer buf(spool);
    std::array<uint8_t, 7> v{1, 2, 3, 4, 5, 6, 7};
    // Two emits in the same chunk (z=0, y=0, x=0)
    buf.emit(1, 2, 3, v);
    buf.emit(4, 5, 6, v);
    // One emit in a different chunk
    buf.emit(80, 0, 0, v);
    buf.flushAll();
    auto s = spool.stats();
    CHECK(s.appendedRecords == 3);
    // Two distinct chunks touched
    CHECK(spool.touchedChunks().size() == 2);
    fs::remove_all(d);
}
