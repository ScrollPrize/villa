// volcodec_ng codec through VcDataset: exact v2 round trips, v3 sharded arrays
// with the index at the end of the shard (as the volcomp exports use), the
// shim's guards, and the encoded bytes of five synthetic chunks pinned, so a
// change to the vendored format fails here. The codec is portable (AVX2 at
// runtime on x86-64, plain C elsewhere, same bytes), so nothing is skipped.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/VcDataset.hpp"
#include "utils/volcodec_ng_codec.hpp"
#include "utils/volcomp_codec.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t N = utils::kVolcodecNgChunkBytes;

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_ds_volcodec_ng_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

std::span<const std::byte> bytes(const std::vector<uint8_t>& v)
{
    return {reinterpret_cast<const std::byte*>(v.data()), v.size()};
}

// Smooth field + texture + noise, CT-like; the same field the volcomp test uses.
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

// Integer-only chunks for the format pin (no libm, so every platform builds
// the same input): constant, noise, ramp, blocks with noise, a thin layer in
// zero air. The codec source's own test builds the same five.
std::vector<uint8_t> pinChunk(int kind)
{
    std::vector<uint8_t> v(N);
    uint32_t r = 0x9E3779B9u + uint32_t(kind);
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
    for (uint32_t z = 0; z < 128; ++z)
        for (uint32_t y = 0; y < 128; ++y)
            for (uint32_t x = 0; x < 128; ++x) {
                unsigned val = 0;
                switch (kind) {
                case 0: val = 77; break;
                case 1: val = rnd() & 255u; break;
                case 2: val = (x + 2 * y + 3 * z) & 255u; break;
                case 3: val = 100 + ((x / 16 + y / 16 + z / 16) & 1u) * 60 + (rnd() % 9u); break;
                case 4: {
                    const int d = int(z) - 64 - int((x * 3 + y * 5) % 23u);
                    val = d > -6 && d < 6 ? 140 + unsigned(6 - std::abs(d)) * 12 + (rnd() % 5u) : 0;
                    break;
                }
                }
                v[(std::size_t(z) * 128 + y) * 128 + x] = uint8_t(val);
            }
    return v;
}

uint64_t fnv1a(uint64_t h, std::span<const std::byte> p)
{
    for (auto b : p) h = (h ^ uint64_t(uint8_t(b))) * 1099511628211ull;
    return h;
}

std::vector<char> readFile(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

void put_le64(std::vector<std::byte>& out, uint64_t v)
{
    for (int i = 0; i < 8; ++i) out.push_back(std::byte((v >> (8 * i)) & 0xFF));
}

} // namespace

TEST_CASE("volcodec_ng: encoded bytes of the pinned chunks")
{
    MESSAGE("volcodec_ng kernels: " << utils::volcodec_ng_kernels());
    CHECK(utils::volcodec_ng_format_revision() == 2u);
    const int modes[5] = {3, 2, 6, 6, 6};  // constant, noise stored raw, then lossless
    uint64_t h = 1469598103934665603ull;
    for (int k = 0; k < 5; ++k) {
        const auto src = pinChunk(k);
        const auto enc = utils::volcodec_ng_encode(bytes(src));
        CHECK(utils::volcodec_ng_chunk_mode(enc) == modes[k]);
        h = fnv1a(h, enc);
        std::vector<std::byte> out(N);
        utils::volcodec_ng_decode_into(enc, out);
        CHECK(std::memcmp(out.data(), src.data(), N) == 0);
    }
    // FNV-1a over the five streams, as written by the codec source at the
    // vendored version on every build it was checked with.
    CHECK(h == 0xcda62457224282e0ull);
}

TEST_CASE("volcodec_ng: v2 dataset round trip is exact")
{
    auto d = tmpDir("v2");
    auto ds = vc::createZarrDataset(d, "arr", {256, 128, 128}, {128, 128, 128},
                                    vc::VcDtype::uint8, "volcodec_ng", "/");
    REQUIRE(ds);
    auto src = synth(3);
    CHECK(ds->writeChunk(1, 0, 0, src.data(), src.size()));

    // metadata carries the codec, with no parameters
    {
        std::ifstream f(d / "arr" / ".zarray");
        std::string meta((std::istreambuf_iterator<char>(f)), {});
        CHECK(meta.find("\"compressor\": {\"id\": \"volcodec_ng\"}") != std::string::npos);
        auto parsed = utils::detail::parse_zarray(meta);
        CHECK(parsed.compressor_id == "volcodec_ng");
    }
    // the chunk on disk is a bare volcodec_ng stream in the lossless mode,
    // and smaller than raw
    {
        auto raw = readFile(d / "arr" / "1" / "0" / "0");
        auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.data()), raw.size());
        CHECK(utils::is_volcodec_ng_compressed(sp));
        CHECK(utils::volcodec_ng_chunk_mode(sp) == 6);
        CHECK(raw.size() < N / 2);
    }

    // reopen and read back: every voxel exact, and the fill chunk is zeros
    vc::VcDataset re(d / "arr");
    std::vector<uint8_t> out(N, 0);
    CHECK(re.readChunk(1, 0, 0, out.data()));
    CHECK(out == src);
    std::vector<uint8_t> fill(N, 7);
    CHECK_FALSE(re.readChunkOrFill(0, 0, 0, fill.data()));
    CHECK(std::all_of(fill.begin(), fill.end(), [](uint8_t v) { return v == 0; }));

    // rewriting through the reopened dataset stays lossless
    auto src2 = synth(4);
    CHECK(re.writeChunk(0, 0, 0, src2.data(), src2.size()));
    vc::VcDataset again(d / "arr");
    CHECK(again.readChunk(0, 0, 0, out.data()));
    CHECK(out == src2);
    fs::remove_all(d);
}

TEST_CASE("volcodec_ng: createZarrDataset rejects non-128^3 chunks and uint16")
{
    auto d = tmpDir("shape");
    CHECK_THROWS(vc::createZarrDataset(d, "arr", {64, 64, 64}, {64, 64, 64},
                                       vc::VcDtype::uint8, "volcodec_ng"));
    CHECK_THROWS(vc::createZarrDataset(d, "arr16", {128, 128, 128}, {128, 128, 128},
                                       vc::VcDtype::uint16, "volcodec_ng"));
    fs::remove_all(d);
}

TEST_CASE("volcodec_ng: v3 sharded array with index_location=end and crc32c index")
{
    auto d = tmpDir("v3end");
    // 4 x 1 x 1 inner chunks per shard (512 x 128 x 128 shard), one shard.
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v3;
    meta.shape = {512, 128, 128};
    meta.chunks = {512, 128, 128};
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
        vc.name = "volcodec_ng";
        sc.sub_codecs.push_back(vc);
    }
    meta.shard_config = sc;
    utils::ZarrArray::create(d / "arr", meta);

    // Hand-build the shard: chunk 1 present, chunk 0 missing, chunk 2 a
    // constant chunk, chunk 3 incompressible noise (stored raw), then the
    // 4-entry index and a (not verified) crc32c.
    const auto src1 = synth(11), src2 = pinChunk(0), src3 = pinChunk(1);
    std::vector<std::byte> shard;
    std::vector<std::pair<uint64_t, uint64_t>> entries{{~uint64_t(0), ~uint64_t(0)}};
    for (const auto* s : {&src1, &src2, &src3}) {
        auto enc = utils::volcodec_ng_encode(bytes(*s));
        entries.emplace_back(shard.size(), enc.size());
        shard.insert(shard.end(), enc.begin(), enc.end());
    }
    for (auto [off, n] : entries) { put_le64(shard, off); put_le64(shard, n); }
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
    CHECK(out == src1);
    CHECK(ds.readChunk(2, 0, 0, out.data()));
    CHECK(out == src2);
    CHECK(ds.readChunk(3, 0, 0, out.data()));
    CHECK(out == src3);
    std::vector<uint8_t> fill(N, 7);
    CHECK_FALSE(ds.readChunkOrFill(0, 0, 0, fill.data()));
    CHECK(std::all_of(fill.begin(), fill.end(), [](uint8_t v) { return v == 0; }));

    // ZarrArray whole-shard extraction agrees
    auto arr = utils::ZarrArray::open(d / "arr", vc::buildZarrCodecRegistry(1));
    CHECK(arr.stores_chunks_with_codec("volcodec_ng"));
    std::vector<std::size_t> idx{1, 0, 0};
    auto whole = arr.read_chunk(idx);
    REQUIRE(whole.has_value());
    REQUIRE(whole->size() == N);
    CHECK(std::memcmp(whole->data(), src1.data(), N) == 0);

    // Writes keep the trailing index: fill the absent chunk 0 and replace
    // chunk 1, then read everything back through a fresh dataset; the index
    // stays the file's last bytes, followed by a valid crc32c.
    auto w0 = synth(21), w1 = synth(22);
    {
        vc::VcDataset w(d / "arr");
        CHECK(w.writeChunk(0, 0, 0, w0.data(), w0.size()));
        CHECK(w.writeChunk(1, 0, 0, w1.data(), w1.size()));
    }
    {
        vc::VcDataset r(d / "arr");
        std::vector<uint8_t> a(N);
        CHECK(r.readChunk(0, 0, 0, a.data()));
        CHECK(a == w0);
        CHECK(r.readChunk(1, 0, 0, a.data()));
        CHECK(a == w1);
        CHECK(r.readChunk(3, 0, 0, a.data()));
        CHECK(a == src3);
    }
    {
        auto raw = readFile(d / "arr" / "c" / "0" / "0" / "0");
        constexpr std::size_t kIndex = 4 * 16 + 4;
        REQUIRE(raw.size() > kIndex);
        const auto* ix = reinterpret_cast<const std::byte*>(raw.data()) + raw.size() - kIndex;
        uint32_t crc = 0;
        for (int i = 0; i < 4; ++i) crc |= uint32_t(uint8_t(raw[raw.size() - 4 + i])) << (8 * i);
        CHECK(crc == utils::detail::crc32c(std::span<const std::byte>(ix, kIndex - 4)));
        for (int c = 0; c < 4; ++c) {
            const auto off = utils::detail::read_le64(ix + c * 16);
            const auto n = utils::detail::read_le64(ix + c * 16 + 8);
            CHECK(off + n <= raw.size() - kIndex);
            auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw.data()) + off, n);
            CHECK(utils::is_volcodec_ng_compressed(sp));
        }
    }
    fs::remove_all(d);
}

TEST_CASE("volcodec_ng: shim guards")
{
    std::vector<std::byte> small(100);
    CHECK_THROWS_AS((void)utils::volcodec_ng_encode(small), std::runtime_error);

    const auto src = synth(5);
    auto enc = utils::volcodec_ng_encode(bytes(src));
    REQUIRE(utils::volcodec_ng_chunk_mode(enc) == 6);
    std::vector<std::byte> out(N), shortOut(N - 1);
    CHECK_THROWS_AS(utils::volcodec_ng_decode_into(enc, shortOut), std::runtime_error);

    // a volcomp chunk is not a volcodec_ng chunk
    utils::VolcompCodecParams p;
    p.q = 0.0f;
    auto vcomp = utils::volcomp_encode(bytes(src), p);
    CHECK_FALSE(utils::is_volcodec_ng_compressed(vcomp));
    CHECK(utils::volcodec_ng_chunk_mode(vcomp) == -1);
    CHECK_THROWS_AS(utils::volcodec_ng_decode_into(vcomp, out), std::runtime_error);

    // truncated and damaged streams throw instead of returning wrong voxels
    auto cut = enc;
    cut.resize(enc.size() / 2);
    CHECK_THROWS_AS(utils::volcodec_ng_decode_into(cut, out), std::runtime_error);
    auto extra = enc;
    extra.push_back(std::byte{0});
    CHECK_THROWS_AS(utils::volcodec_ng_decode_into(extra, out), std::runtime_error);

    // a chunk mode this build does not read is refused by name
    auto other = enc;
    other[5] = std::byte{4};
    try {
        utils::volcodec_ng_decode_into(other, out);
        CHECK(false);
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("chunk mode not supported") != std::string::npos);
    }

    // so is a lossless stream of another version (its first payload byte)
    auto newer = enc;
    newer[12] = std::byte{4};
    try {
        utils::volcodec_ng_decode_into(newer, out);
        CHECK(false);
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("unsupported stream version") != std::string::npos);
    }

    // and the untouched stream still decodes exactly
    utils::volcodec_ng_decode_into(enc, out);
    CHECK(std::memcmp(out.data(), src.data(), N) == 0);
}
