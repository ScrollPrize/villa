// volcomp codec through VcDataset: v2 round trip at q, v3 sharded arrays with
// the index at the end of the shard (the zarr default and the layout of the
// volcomp exports), and the shim's guards. The codec is portable (AVX2 at
// runtime on x86-64, plain C elsewhere), so nothing here is skipped.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VcDataset.hpp"
#include "utils/volcomp_codec.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t N = utils::kVolcompChunkBytes;

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_ds_volcomp_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

// Smooth field + texture + noise, CT-like; the same shape the volcomp tests use.
std::vector<uint8_t> synth(uint32_t seed)
{
    std::vector<uint8_t> v(N);
    uint32_t r = seed | 1u;
    auto rng = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
    for (uint32_t z = 0; z < 128; ++z)
        for (uint32_t y = 0; y < 128; ++y)
            for (uint32_t x = 0; x < 128; ++x) {
                double s = 120 + 50 * std::sin(z * 0.05 + seed) * std::cos(y * 0.07) +
                           30 * std::sin((x + y) * 0.11);
                s += 12 * std::sin(x * 0.9 + z * 0.3) + double(rng() % 9) - 4.0;
                int iv = int(s + 0.5);
                v[(std::size_t(z) * 128 + y) * 128 + x] =
                    uint8_t(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
            }
    return v;
}

double psnr(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    double se = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        se += d * d;
    }
    if (se == 0) return 999.0;
    return 10.0 * std::log10(255.0 * 255.0 * double(a.size()) / se);
}

void put_le64(std::vector<std::byte>& out, uint64_t v)
{
    for (int i = 0; i < 8; ++i) out.push_back(std::byte((v >> (8 * i)) & 0xFF));
}

} // namespace

TEST_CASE("volcomp: v2 dataset round trip at q, q recorded in .zarray")
{
    REQUIRE(utils::volcomp_available());
    MESSAGE("volcomp kernels: " << utils::volcomp_kernels());
    auto d = tmpDir("v2");
    auto ds = vc::createZarrDataset(d, "arr", {256, 128, 128}, {128, 128, 128},
                                    vc::VcDtype::uint8, "volcomp", "/", 0,
                                    /*compressionLevel = q*/ 4);
    REQUIRE(ds);
    auto src = synth(3);
    CHECK(ds->writeChunk(1, 0, 0, src.data(), src.size()));

    // metadata carries the codec and q
    std::ifstream f(d / "arr" / ".zarray");
    std::string meta((std::istreambuf_iterator<char>(f)), {});
    CHECK(meta.find("\"id\": \"volcomp\"") != std::string::npos);
    CHECK(meta.find("\"q\": 4") != std::string::npos);
    // the chunk on disk is a bare volcomp stream at that q
    {
        std::ifstream c(d / "arr" / "1" / "0" / "0", std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(c)), {});
        REQUIRE(raw.size() > 8);
        auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.data()), raw.size());
        CHECK(utils::is_volcomp_compressed(sp));
        CHECK(utils::volcomp_chunk_q(sp) == doctest::Approx(4.0f));
        CHECK(raw.size() < N / 5);
    }

    // reopen and read back: lossy but close, and the fill chunk is zeros
    vc::VcDataset re(d / "arr");
    std::vector<uint8_t> out(N, 0);
    CHECK(re.readChunk(1, 0, 0, out.data()));
    CHECK(psnr(src, out) > 38.0);
    std::vector<uint8_t> fill(N, 7);
    CHECK_FALSE(re.readChunkOrFill(0, 0, 0, fill.data()));
    CHECK(std::all_of(fill.begin(), fill.end(), [](uint8_t v) { return v == 0; }));
    fs::remove_all(d);
}

TEST_CASE("volcomp: createZarrDataset rejects non-128^3 chunks")
{
    if (!utils::volcomp_available()) return;
    auto d = tmpDir("shape");
    CHECK_THROWS(vc::createZarrDataset(d, "arr", {64, 64, 64}, {64, 64, 64},
                                       vc::VcDtype::uint8, "volcomp"));
    CHECK_THROWS(vc::createZarrDataset(d, "arr16", {128, 128, 128}, {128, 128, 128},
                                       vc::VcDtype::uint16, "volcomp"));
    fs::remove_all(d);
}

TEST_CASE("volcomp: v3 sharded array with index_location=end and crc32c index")
{
    if (!utils::volcomp_available()) return;
    auto d = tmpDir("v3end");
    // 2 x 1 x 1 inner chunks per shard (256 x 128 x 128 shard), one shard.
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {256, 128, 128};
    meta.chunks = {256, 128, 128};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0.0;
    utils::ShardConfig sc;
    sc.sub_chunks = {128, 128, 128};
    sc.index_location = "end";
    {
        utils::ZarrCodecConfig bytes_c;
        bytes_c.name = "bytes";
        bytes_c.configuration = std::make_shared<utils::JsonValue>(
            utils::Json({{"endian", "little"}}));
        sc.index_codecs.push_back(bytes_c);
        utils::ZarrCodecConfig crc;
        crc.name = "crc32c";
        sc.index_codecs.push_back(crc);
        utils::ZarrCodecConfig vc;
        vc.name = "volcomp";
        vc.configuration = std::make_shared<utils::JsonValue>(
            utils::JsonValue{{"q", utils::Json(8.0)}});
        sc.sub_codecs.push_back(vc);
    }
    meta.shard_config = sc;
    utils::ZarrArray::create(d / "arr", meta);
    {
        std::ifstream f(d / "arr" / "zarr.json");
        std::string j((std::istreambuf_iterator<char>(f)), {});
        CHECK(j.find("\"index_location\":\"end\"") != std::string::npos ||
              j.find("\"index_location\": \"end\"") != std::string::npos);
    }

    // Hand-build the shard: chunk 1 present (volcomp), chunk 0 missing,
    // then the 2-entry index and a (not verified) crc32c.
    auto src = synth(11);
    utils::VolcompCodecParams p;
    p.q = 8.0f;
    auto enc = utils::volcomp_encode(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(src.data()), src.size()), p);
    std::vector<std::byte> shard(enc.begin(), enc.end());
    const uint64_t missing = ~uint64_t(0);
    put_le64(shard, missing); put_le64(shard, missing);   // chunk 0: absent
    put_le64(shard, 0);       put_le64(shard, enc.size()); // chunk 1: [0, n)
    for (int i = 0; i < 4; ++i) shard.push_back(std::byte(0xAB));  // crc32c placeholder
    fs::create_directories(d / "arr" / "c" / "0" / "0");
    {
        std::ofstream f(d / "arr" / "c" / "0" / "0" / "0", std::ios::binary);
        f.write(reinterpret_cast<const char*>(shard.data()), std::streamsize(shard.size()));
    }

    // VcDataset reads through the partial-read path (index entry then payload)
    vc::VcDataset ds(d / "arr");
    REQUIRE(ds.defaultChunkShape().size() == 3);
    CHECK(ds.defaultChunkShape()[0] == 128);
    std::vector<uint8_t> out(N, 1);
    CHECK(ds.readChunk(1, 0, 0, out.data()));
    CHECK(psnr(src, out) > 38.0);
    std::vector<uint8_t> fill(N, 7);
    CHECK_FALSE(ds.readChunkOrFill(0, 0, 0, fill.data()));
    CHECK(std::all_of(fill.begin(), fill.end(), [](uint8_t v) { return v == 0; }));

    // ZarrArray whole-shard extraction agrees
    auto arr = utils::ZarrArray::open(d / "arr", vc::buildZarrCodecRegistry(1));
    CHECK(utils::is_canonical_volcomp(arr.metadata()));
    std::vector<std::size_t> idx{1, 0, 0};
    auto whole = arr.read_chunk(idx);
    REQUIRE(whole.has_value());
    REQUIRE(whole->size() == N);
    CHECK(std::memcmp(whole->data(), out.data(), N) == 0);

    // Writes keep the trailing index: fill the absent chunk 0 and replace
    // chunk 1 through VcDataset (write_chunk -> encode -> shard), then read
    // both back through a fresh dataset. The index stays the file's last
    // bytes, followed by a valid crc32c.
    auto src0 = synth(21), src1 = synth(22);
    {
        vc::VcDataset w(d / "arr");
        CHECK(w.writeChunk(0, 0, 0, src0.data(), src0.size()));
        CHECK(w.writeChunk(1, 0, 0, src1.data(), src1.size()));
    }
    {
        vc::VcDataset r(d / "arr");
        std::vector<uint8_t> a(N), b(N);
        CHECK(r.readChunk(0, 0, 0, a.data()));
        CHECK(r.readChunk(1, 0, 0, b.data()));
        CHECK(psnr(src0, a) > 38.0);
        CHECK(psnr(src1, b) > 38.0);
    }
    {
        std::ifstream f(d / "arr" / "c" / "0" / "0" / "0", std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
        REQUIRE(raw.size() > 36);
        const auto* idx = reinterpret_cast<const std::byte*>(raw.data()) + raw.size() - 36;
        uint32_t crc = 0;
        for (int i = 0; i < 4; ++i) crc |= uint32_t(uint8_t(raw[raw.size() - 4 + i])) << (8 * i);
        CHECK(crc == utils::detail::crc32c(std::span<const std::byte>(idx, 32)));
        for (int c = 0; c < 2; ++c) {
            const auto off = utils::detail::read_le64(idx + c * 16);
            const auto n = utils::detail::read_le64(idx + c * 16 + 8);
            CHECK(off + n <= raw.size() - 36);
            auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.data()) + off, n);
            CHECK(utils::is_volcomp_compressed(sp));
        }
    }
    fs::remove_all(d);
}

namespace {
// A smooth surface-teacher-like probability field: a wavy sheet, u8 = round(255 p).
std::vector<uint8_t> synthProb(uint32_t seed)
{
    std::vector<uint8_t> v(N);
    for (uint32_t z = 0; z < 128; ++z)
        for (uint32_t y = 0; y < 128; ++y)
            for (uint32_t x = 0; x < 128; ++x) {
                const double sheet = 64.0 + 18.0 * std::sin(y * 0.06 + seed) * std::cos(x * 0.045);
                const double dist = (double(z) - sheet) / 2.5;
                const double p = std::exp(-dist * dist) +
                                 0.35 * std::exp(-std::pow((double(z) - sheet - 30.0) / 4.0, 2));
                const double pc = p > 1.0 ? 1.0 : p;
                v[(std::size_t(z) * 128 + y) * 128 + x] = uint8_t(std::lround(255.0 * pc));
            }
    return v;
}
} // namespace

TEST_CASE("volcomp: surface chunks (mode 6) decode through the vc3d codec path, threshold exact")
{
    CHECK(utils::volcomp_format_revision() >= 4u);
    MESSAGE("volcomp " << utils::volcomp_version() << " format revision " << utils::volcomp_format_revision());
    auto src = synthProb(5);
    std::size_t above = 0;
    for (auto v : src) above += v >= 128;
    REQUIRE(above > N / 100);   // the sheet is really there
    REQUIRE(above < N / 2);

    auto raw = std::span<const std::byte>(reinterpret_cast<const std::byte*>(src.data()), src.size());
    auto enc = utils::volcomp_surface_encode(raw, 48.0f, 128);
    REQUIRE(enc.size() > 16);
    CHECK(std::to_integer<int>(enc[5]) == 6);   // header mode byte
    CHECK(utils::is_volcomp_compressed(enc));
    CHECK(utils::volcomp_chunk_q(enc) == doctest::Approx(48.0f));
    auto info = utils::volcomp_surface_info(enc);
    REQUIRE(info.has_value());
    CHECK(info->thr == 128u);
    MESSAGE("surface chunk: " << enc.size() << " bytes, refinement margin " << info->margin);
    // a mode-0 chunk is not a surface chunk
    utils::VolcompCodecParams p;
    CHECK_FALSE(utils::volcomp_surface_info(utils::volcomp_encode(raw, p)).has_value());

    auto checkExact = [&](const std::vector<std::byte>& dec, const char* what) {
        REQUIRE(dec.size() == N);
        std::size_t wrong = 0, maxErr = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const auto d = std::to_integer<uint8_t>(dec[i]);
            wrong += (src[i] >= 128) != (d >= 128);
            const std::size_t e = std::size_t(std::abs(int(d) - int(src[i])));
            maxErr = std::max(maxErr, e);
        }
        MESSAGE(what << ": threshold mismatches " << wrong << ", max abs error " << maxErr);
        CHECK(wrong == 0);
    };

    // plain decode (the default read path)
    auto dec = utils::volcomp_decode(enc, N);
    checkExact(dec, "volcomp_decode");
    std::vector<uint8_t> decU8(N);
    std::memcpy(decU8.data(), dec.data(), N);
    CHECK(psnr(src, decU8) > 30.0);

    // block decode agrees with the whole-chunk decode
    std::vector<std::byte> blk(4096);
    for (unsigned b : {0u, 3u, 4u, 7u}) {
        utils::volcomp_decode_block_into(enc, b, b, 7 - b, blk);
        for (unsigned z = 0; z < 16; ++z)
            for (unsigned y = 0; y < 16; ++y)
                CHECK(std::memcmp(blk.data() + (z * 16 + y) * 16,
                                  dec.data() + ((std::size_t(b) * 16 + z) * 128 + (b * 16 + y)) * 128 +
                                      (7 - b) * 16,
                                  16) == 0);
    }

    // opt-in smoothing keeps the exact threshold too
    std::vector<std::byte> sm(N);
    utils::volcomp_decode_smooth_into(enc, sm, 0.6f);
    checkExact(sm, "volcomp_decode_smooth (opt-in)");

    // the same stream stored as an inner chunk of a v3 shard (index at the
    // end, as the published exports) reads back through VcDataset unchanged
    auto d = tmpDir("surface");
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {128, 128, 256};
    meta.chunks = {128, 128, 256};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.fill_value = 0.0;
    utils::ShardConfig sc;
    sc.sub_chunks = {128, 128, 128};
    sc.index_location = "end";
    utils::ZarrCodecConfig vcodec;
    vcodec.name = "volcomp";
    vcodec.configuration = std::make_shared<utils::JsonValue>(utils::JsonValue{{"q", utils::Json(48.0)}});
    sc.sub_codecs.push_back(vcodec);
    meta.shard_config = sc;
    {
        auto arr = utils::ZarrArray::create(d / "arr", meta, vc::buildZarrCodecRegistry(1));
        arr.write_inner_chunk_to_shard(std::array<std::size_t, 3>{0, 0, 1}, enc);
    }
    vc::VcDataset ds(d / "arr");
    std::vector<uint8_t> out(N, 1);
    CHECK(ds.readChunk(0, 0, 1, out.data()));
    CHECK(std::memcmp(out.data(), dec.data(), N) == 0);
    fs::remove_all(d);
}

TEST_CASE("volcomp: shim guards")
{
    if (!utils::volcomp_available()) return;
    std::vector<std::byte> small(64);
    utils::VolcompCodecParams p;
    CHECK_THROWS(utils::volcomp_encode(small, p));
    std::vector<std::byte> raw(N, std::byte(0));
    p.q = 0.0f;
    CHECK_THROWS(utils::volcomp_encode(raw, p));
    CHECK_THROWS(utils::volcomp_decode(small, N));
    CHECK_FALSE(utils::is_volcomp_compressed(small));
    p.q = 8.0f;
    auto enc = utils::volcomp_encode(raw, p);
    CHECK(enc.size() < 2048);  // an all-zero chunk is a few hundred bytes
    CHECK_THROWS(utils::volcomp_decode(enc, N - 1));
    std::vector<std::byte> blk(4096, std::byte(9));
    utils::volcomp_decode_block_into(enc, 7, 7, 7, blk);
    CHECK(std::all_of(blk.begin(), blk.end(), [](std::byte b) { return b == std::byte(0); }));
}

// Live check against a published volcomp export (zarr v3, 1024^3 shards with
// the index at the end, crc32c index, 128^3 volcomp inner chunks). Opt in
// with VC_VOLCOMP_LIVE_URL=<volume .zarr url>; VC_VOLCOMP_LIVE_LEVEL selects
// the pyramid level (default 3, whose shards are a few tens of MB).
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include <cstdlib>

TEST_CASE("volcomp: live HTTP pyramid open + chunk fetch (opt-in)")
{
    const char* url = std::getenv("VC_VOLCOMP_LIVE_URL");
    if (!url || !utils::volcomp_available()) {
        MESSAGE("Skipping: set VC_VOLCOMP_LIVE_URL to run");
        return;
    }
    int level = 3;
    if (const char* l = std::getenv("VC_VOLCOMP_LIVE_LEVEL")) level = std::atoi(l);
    auto opened = vc::render::openHttpZarrPyramid(url);
    REQUIRE(!opened.fetchers.empty());
    MESSAGE("levels: " << opened.fetchers.size() << " dtype u8: " << (opened.dtype == vc::render::ChunkDtype::UInt8));
    std::size_t li = 0;
    for (std::size_t i = 0; i < opened.levelNumbers.size(); ++i)
        if (opened.levelNumbers[i] == level) li = i;
    CHECK(opened.chunkShapes[li] == std::array<int, 3>{128, 128, 128});
    CHECK(opened.storageChunkShapes[li] == std::array<int, 3>{1024, 1024, 1024});
    const auto& shape = opened.shapes[li];
    // a chunk in the middle of the volume
    vc::render::ChunkKey key{};
    key.level = level;
    key.iz = shape[0] / 128 / 2;
    key.iy = shape[1] / 128 / 2;
    key.ix = shape[2] / 128 / 2;
    auto res = opened.fetchers[li]->fetch(key);
    MESSAGE("fetch status " << int(res.status) << " msg '" << res.message << "' bytes " << res.bytes.size());
    REQUIRE(res.status == vc::render::ChunkFetchStatus::Found);
    REQUIRE(res.bytes.size() == N);
    std::size_t nz = 0;
    for (auto b : res.bytes) nz += b != std::byte(0);
    MESSAGE("nonzero voxels: " << nz);
    CHECK(nz > N / 10);
    CHECK(res.hasPersistentBytes);
    CHECK(utils::is_volcomp_compressed(res.persistentBytes));
    // a chunk outside the scroll (corner) is Missing, not an error
    vc::render::ChunkKey corner{};
    corner.level = level;
    auto res2 = opened.fetchers[li]->fetch(corner);
    CHECK((res2.status == vc::render::ChunkFetchStatus::Missing ||
           res2.status == vc::render::ChunkFetchStatus::Found));
}

