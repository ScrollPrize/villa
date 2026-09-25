// Shard writers against both sharding index locations. VC's own arrays put
// the index at the start of the shard; zarr-python (and the published volcomp
// exports) put it at the end, optionally followed by a crc32c. Every shard-
// mutating ZarrArray call must keep the index where the metadata says it is:
// write_shard, write_inner_chunk_to_shard (new and existing inner chunks),
// write_chunk on a sharded array, mark_inner_chunk_empty and write_empty_shard.
// Index codecs the writers cannot produce are refused before any file changes.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utils/zarr.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kInner = 16;                       // inner chunk edge
constexpr std::size_t kInnerBytes = kInner * kInner * kInner;
constexpr std::size_t kPerShard = 8;                     // 2 x 2 x 2 inner chunks

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_zarr_shard_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

// Uncompressed uint8 v3 array, 64 x 32 x 32 = two 32^3 shards of 16^3 inner chunks.
utils::ZarrMetadata makeMeta(const std::string& location, bool crc,
                             const std::string& extraIndexCodec = {})
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {64, 32, 32};
    meta.chunks = {32, 32, 32};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0.0;
    utils::ShardConfig sc;
    sc.sub_chunks = {kInner, kInner, kInner};
    sc.index_location = location;
    utils::ZarrCodecConfig bytes_c;
    bytes_c.name = "bytes";
    bytes_c.configuration = std::make_shared<utils::JsonValue>(
        utils::Json({{"endian", extraIndexCodec == "big" ? "big" : "little"}}));
    sc.index_codecs.push_back(bytes_c);
    if (!extraIndexCodec.empty() && extraIndexCodec != "big") {
        utils::ZarrCodecConfig c;
        c.name = extraIndexCodec;
        sc.index_codecs.push_back(c);
    }
    if (crc) {
        utils::ZarrCodecConfig c;
        c.name = "crc32c";
        sc.index_codecs.push_back(c);
    }
    meta.shard_config = sc;
    return meta;
}

std::vector<std::byte> pattern(unsigned tag)
{
    std::vector<std::byte> v(kInnerBytes);
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = std::byte(static_cast<std::uint8_t>((i * 7 + tag * 31 + 1) & 0xFF));
    return v;
}

std::array<std::size_t, 3> innerCoord(std::size_t shardZ, std::size_t linear)
{
    return {shardZ * 2 + (linear >> 2), (linear >> 1) & 1, linear & 1};
}

std::vector<std::byte> readFile(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::byte> out(c.size());
    std::memcpy(out.data(), c.data(), c.size());
    return out;
}

// Parse the shard's index from where the metadata says it lives, verify its
// crc32c when present, and check every entry is in bounds and clear of the
// index bytes themselves.
std::vector<std::pair<std::uint64_t, std::uint64_t>>
checkLayout(const fs::path& shard, bool atEnd, bool crc)
{
    auto bytes = readFile(shard);
    const std::size_t idx = kPerShard * 16;
    const std::size_t total = idx + (crc ? 4 : 0);
    REQUIRE(bytes.size() >= total);
    const std::size_t at = atEnd ? bytes.size() - total : 0;
    if (crc) {
        const auto want = utils::detail::crc32c(
            std::span<const std::byte>(bytes.data() + at, idx));
        std::uint32_t got = 0;
        for (int i = 0; i < 4; ++i)
            got |= std::uint32_t(std::to_integer<std::uint8_t>(bytes[at + idx + i])) << (8 * i);
        CHECK(got == want);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> entries;
    for (std::size_t i = 0; i < kPerShard; ++i) {
        const auto off = utils::detail::read_le64(bytes.data() + at + i * 16);
        const auto n = utils::detail::read_le64(bytes.data() + at + i * 16 + 8);
        entries.emplace_back(off, n);
        if (n == 0 || n == ~std::uint64_t(0)) continue;
        CHECK(off + n <= bytes.size());
        // the payload must not overlap the index
        CHECK((off + n <= at || off >= at + total));
    }
    return entries;
}

void checkChunk(const utils::ZarrArray& a, std::array<std::size_t, 3> c,
                const std::optional<std::vector<std::byte>>& want)
{
    CAPTURE(c[0]); CAPTURE(c[1]); CAPTURE(c[2]);
    auto got = a.read_chunk(c);   // partial read: one index entry, then the payload
    if (!want) {
        CHECK_FALSE(got.has_value());
    } else {
        REQUIRE(got.has_value());
        CHECK(*got == *want);
    }
    // whole-shard read + extract agrees
    auto whole = a.read_whole_shard(c);
    if (!whole) {
        CHECK_FALSE(want.has_value());
        return;
    }
    std::array<std::size_t, 3> inner{c[0] % 2, c[1] % 2, c[2] % 2};
    auto ex = a.extract_inner_chunk(whole->span(), inner);
    CHECK(ex.has_value() == want.has_value());
    if (ex && want) CHECK(*ex == *want);
}

void roundTrip(const std::string& location, bool crc)
{
    CAPTURE(location); CAPTURE(crc);
    const bool atEnd = location == "end";
    auto d = tmpDir(location + (crc ? "_crc" : ""));
    auto arr = utils::ZarrArray::create(d / "arr", makeMeta(location, crc));
    REQUIRE(arr.is_sharded());

    // expected contents of every inner chunk of shard 0 and shard 1
    std::vector<std::optional<std::vector<std::byte>>> want(2 * kPerShard);

    // 1. a whole shard, inner chunk 3 absent
    std::vector<std::optional<std::vector<std::byte>>> inner(kPerShard);
    for (std::size_t i = 0; i < kPerShard; ++i)
        if (i != 3) inner[i] = pattern(unsigned(i));
    arr.write_shard(std::array<std::size_t, 3>{0, 0, 0}, inner);
    for (std::size_t i = 0; i < kPerShard; ++i) want[i] = inner[i];
    const fs::path shard0 = d / "arr" / "c" / "0" / "0" / "0";
    const fs::path shard1 = d / "arr" / "c" / "1" / "0" / "0";
    checkLayout(shard0, atEnd, crc);

    // 2. update an existing inner chunk and fill the absent one
    arr.write_inner_chunk_to_shard(innerCoord(0, 1), pattern(101));
    want[1] = pattern(101);
    arr.write_inner_chunk_to_shard(innerCoord(0, 3), pattern(103));
    want[3] = pattern(103);
    // 3. the encoding write path routes to the shard as well
    arr.write_chunk(innerCoord(0, 5), pattern(105));
    want[5] = pattern(105);
    // 4. known-empty sentinel
    arr.mark_inner_chunk_empty(innerCoord(0, 6));
    want[6].reset();
    // 5. a fresh shard created by an inner write, then updated again
    arr.write_inner_chunk_to_shard(innerCoord(1, 2), pattern(202));
    arr.write_inner_chunk_to_shard(innerCoord(1, 2), pattern(212));
    want[kPerShard + 2] = pattern(212);

    auto e0 = checkLayout(shard0, atEnd, crc);
    CHECK(e0[6].first == ~std::uint64_t(0) - 1);
    CHECK(e0[6].second == 0);
    auto e1 = checkLayout(shard1, atEnd, crc);
    for (std::size_t i = 0; i < kPerShard; ++i)
        if (i != 2) CHECK(e1[i].first == ~std::uint64_t(0));

    CHECK(arr.inner_chunk_is_empty(innerCoord(0, 6)));
    CHECK_FALSE(arr.inner_chunk_exists(innerCoord(0, 6)));
    CHECK(arr.inner_chunk_exists(innerCoord(0, 3)));
    CHECK(arr.inner_chunk_exists(innerCoord(1, 2)));
    CHECK_FALSE(arr.inner_chunk_exists(innerCoord(1, 0)));

    // read back through a freshly opened array
    auto re = utils::ZarrArray::open(d / "arr");
    CHECK(re.metadata().shard_config->index_location == location);
    for (std::size_t s = 0; s < 2; ++s)
        for (std::size_t i = 0; i < kPerShard; ++i)
            checkChunk(re, innerCoord(s, i), want[s * kPerShard + i]);

    // 6. an all-empty shard
    arr.write_empty_shard(std::array<std::size_t, 3>{1, 0, 0});
    checkLayout(shard1, atEnd, crc);
    CHECK(arr.inner_chunk_is_empty(innerCoord(1, 2)));
    CHECK_FALSE(re.read_chunk(innerCoord(1, 2)).has_value());

    fs::remove_all(d);
}

} // namespace

TEST_CASE("crc32c matches the Castagnoli check value")
{
    const std::string s = "123456789";
    CHECK(utils::detail::crc32c(std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(s.data()), s.size())) == 0xE3069283u);
}

TEST_CASE("sharded writes: index_location=start (VC default) is unchanged")
{
    roundTrip("start", false);
}

TEST_CASE("sharded writes: index_location=start with a crc32c index")
{
    roundTrip("start", true);
}

TEST_CASE("sharded writes: index_location=end, write + update + read back")
{
    roundTrip("end", false);
}

TEST_CASE("sharded writes: index_location=end with a crc32c index (zarr-python layout)")
{
    roundTrip("end", true);
}

TEST_CASE("sharded writes: start layout keeps the fixed front index")
{
    // The start-index fast path patches 16 bytes in place; the payload of an
    // updated chunk lands 4k-aligned after the existing data.
    auto d = tmpDir("start_fixed");
    auto arr = utils::ZarrArray::create(d / "arr", makeMeta("start", false));
    arr.write_inner_chunk_to_shard(innerCoord(0, 0), pattern(1));
    const fs::path shard0 = d / "arr" / "c" / "0" / "0" / "0";
    auto e = checkLayout(shard0, false, false);
    CHECK(e[0].first == 4096);
    CHECK(e[0].second == kInnerBytes);
    arr.write_inner_chunk_to_shard(innerCoord(0, 0), pattern(2));
    e = checkLayout(shard0, false, false);
    CHECK(e[0].first == 8192);
    CHECK(*arr.read_chunk(innerCoord(0, 0)) == pattern(2));
    fs::remove_all(d);
}

TEST_CASE("sharded writes: unwritable index codecs are refused before any file changes")
{
    for (const std::string bad : {"big", "zstd"}) {
        CAPTURE(bad);
        auto d = tmpDir("reject");
        auto arr = utils::ZarrArray::create(d / "arr", makeMeta("end", false, bad));
        const fs::path shard0 = d / "arr" / "c" / "0" / "0" / "0";
        std::vector<std::optional<std::vector<std::byte>>> inner(kPerShard, pattern(0));
        CHECK_THROWS(arr.write_shard(std::array<std::size_t, 3>{0, 0, 0}, inner));
        CHECK_THROWS(arr.write_inner_chunk_to_shard(innerCoord(0, 0), pattern(0)));
        CHECK_THROWS(arr.write_chunk(innerCoord(0, 0), pattern(0)));
        CHECK_THROWS(arr.mark_inner_chunk_empty(innerCoord(0, 0)));
        CHECK_THROWS(arr.write_empty_shard(std::array<std::size_t, 3>{0, 0, 0}));
        CHECK_FALSE(fs::exists(d / "arr" / "c"));
        fs::remove_all(d);
    }
}
