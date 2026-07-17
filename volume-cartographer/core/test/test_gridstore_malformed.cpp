// Hardening tests for the GridStore on-disk loader (load_mmap).
//
// The v3 loader reads bucket/path offsets and a cell_size straight from
// untrusted file bytes. Two defects could turn a corrupt file into a crash or
// out-of-bounds read rather than a clean error:
//   - cell_size == 0 -> integer divide-by-zero (SIGFPE) when computing grid_size_
//   - an inverted/oversized bucket-index range -> uint32 underflow -> OOB read
// These tests build a valid file, corrupt one header word, and assert the
// loader now throws std::runtime_error instead of crashing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using vc::core::util::GridStore;

namespace {

std::string tmpPath(const std::string& name)
{
    auto p = std::filesystem::temp_directory_path() / ("vc_gridstore_malformed_" + name);
    return p.string();
}

std::vector<cv::Point> diagonal(int n, cv::Point origin = {0, 0})
{
    std::vector<cv::Point> pts;
    for (int i = 0; i < n; ++i) pts.emplace_back(origin.x + i, origin.y + i);
    return pts;
}

// Build a valid v3 GridStore file and return its raw bytes.
std::vector<uint8_t> makeValidFileBytes(const std::string& path)
{
    std::filesystem::remove(path);
    {
        GridStore gs(cv::Rect(0, 0, 100, 100), 10);
        gs.add(diagonal(5, {0, 0}));
        gs.add(diagonal(5, {50, 50}));
        gs.save(path);
    }
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    std::filesystem::remove(path);
    return bytes;
}

// Overwrite header word `wordIndex` (0-based u32, big-endian) in-place.
void patchHeaderWordBE(std::vector<uint8_t>& bytes, size_t wordIndex, uint32_t value)
{
    const size_t off = wordIndex * 4;
    REQUIRE(off + 4 <= bytes.size());
    bytes[off + 0] = static_cast<uint8_t>((value >> 24) & 0xFF);
    bytes[off + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
    bytes[off + 2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    bytes[off + 3] = static_cast<uint8_t>(value & 0xFF);
}

void writeBytes(const std::string& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

// Build a raw 44-byte (11 big-endian u32 words) header from the given words.
std::vector<uint8_t> makeHeaderBytes(const std::vector<uint32_t>& words)
{
    std::vector<uint8_t> bytes(words.size() * 4, 0);
    for (size_t i = 0; i < words.size(); ++i) patchHeaderWordBE(bytes, i, words[i]);
    return bytes;
}

const uint32_t GRIDSTORE_MAGIC = 0x56434753; // "VCGS"

} // namespace

// Header layout (big-endian u32 words):
//   0=magic 1=version 2=bounds.x 3=bounds.y 4=bounds.width 5=bounds.height
//   6=cell_size 7=num_buckets 8=num_paths 9=buckets_offset 10=paths_offset

TEST_CASE("GridStore: a valid file still loads (hardening does not reject good files)")
{
    const auto src = tmpPath("valid_src.bin");
    auto bytes = makeValidFileBytes(src);
    const auto path = tmpPath("valid.bin");
    writeBytes(path, bytes);

    GridStore loaded(path);
    CHECK(loaded.size() == cv::Size(100, 100));
    CHECK(loaded.get_all().size() == 2);

    std::filesystem::remove(path);
}

TEST_CASE("GridStore: cell_size == 0 throws instead of SIGFPE")
{
    const auto src = tmpPath("cell0_src.bin");
    auto bytes = makeValidFileBytes(src);
    patchHeaderWordBE(bytes, 6, 0); // cell_size = 0
    const auto path = tmpPath("cell0.bin");
    writeBytes(path, bytes);

    CHECK_THROWS_AS(GridStore{path}, std::runtime_error);

    std::filesystem::remove(path);
}

TEST_CASE("GridStore: buckets_offset past EOF throws instead of OOB read")
{
    const auto src = tmpPath("badoff_src.bin");
    auto bytes = makeValidFileBytes(src);
    patchHeaderWordBE(bytes, 9, 0xFFFFFFF0u); // buckets_offset way past EOF
    const auto path = tmpPath("badoff.bin");
    writeBytes(path, bytes);

    CHECK_THROWS_AS(GridStore{path}, std::runtime_error);

    std::filesystem::remove(path);
}

TEST_CASE("GridStore: paths_offset past EOF throws")
{
    const auto src = tmpPath("badpaths_src.bin");
    auto bytes = makeValidFileBytes(src);
    patchHeaderWordBE(bytes, 10, 0xFFFFFFF0u); // paths_offset way past EOF
    const auto path = tmpPath("badpaths.bin");
    writeBytes(path, bytes);

    CHECK_THROWS_AS(GridStore{path}, std::runtime_error);

    std::filesystem::remove(path);
}

TEST_CASE("GridStore: v1 header with huge num_buckets throws instead of OOM")
{
    // A 44-byte v1 file claiming ~4.3e9 buckets. Without the descriptor-region
    // bound check this drives grid_bucket_descriptors_.resize(0xFFFFFFFF)
    // (~64 GiB) before any per-bucket end check, i.e. a DoS from a tiny file.
    const auto bytes = makeHeaderBytes({
        GRIDSTORE_MAGIC, 1u,     // magic, version=1 (legacy descriptor path)
        0u, 0u, 1u, 1u,          // bounds x,y,w,h
        1u,                      // cell_size = 1 (passes > 0)
        0xFFFFFFFFu,             // num_buckets (hostile)
        0u,                      // num_paths
        44u, 44u                 // buckets_offset, paths_offset
    });
    const auto path = tmpPath("evil_v1.bin");
    writeBytes(path, bytes);

    CHECK_THROWS_AS(GridStore{path}, std::runtime_error);

    std::filesystem::remove(path);
}

TEST_CASE("GridStore: width near INT_MAX loads without signed-overflow UB")
{
    // bounds.width = INT_MAX previously overflowed the signed grid_size_
    // ceil-divide (bounds.width + cell_size - 1 > INT_MAX) — undefined behavior
    // that UBSan traps. With the 64-bit ceil-divide it computes cleanly. Use a
    // minimal empty v1 file (num_buckets=0) so construction completes; the point
    // is that it must not invoke UB or crash.
    const auto bytes = makeHeaderBytes({
        GRIDSTORE_MAGIC, 1u,     // magic, version=1
        0u, 0u, 0x7FFFFFFFu, 1u, // bounds x,y, width=INT_MAX, height=1
        1u,                      // cell_size = 1
        0u,                      // num_buckets = 0 (empty)
        0u,                      // num_paths
        44u, 44u                 // buckets_offset, paths_offset
    });
    const auto path = tmpPath("ovf_v1.bin");
    writeBytes(path, bytes);

    CHECK_NOTHROW(GridStore{path}); // must not overflow/crash

    std::filesystem::remove(path);
}
