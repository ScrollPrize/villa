#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/ChunkedTensor.hpp"

#include <filesystem>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path makeDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() /
             ("vc_chunked_meta_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

struct ConstantChunkComputer {
    enum { BORDER = 0 };
    enum { CHUNK_SIZE = 4 };
    enum { FILL_V = 0 };
    const std::string UNIQUE_ID_STRING = "destructor_cleanup";

    template <typename Chunk, typename Element>
    void compute(const Chunk&, Chunk& small, const cv::Vec3i&)
    {
        small.fill(static_cast<Element>(7));
    }
};

} // namespace

TEST_CASE("Chunked3d destruction releases mapped chunks before removing transient cache")
{
    const auto cacheRoot = makeDir("mapped_cleanup");
    const auto cacheFamily = cacheRoot / "destructor_cleanup";
    ConstantChunkComputer compute;

    {
        Chunked3d<std::uint8_t, ConstantChunkComputer> chunks(
            compute, std::array<int, 3>{4, 4, 4}, nullptr, 0, cacheRoot);
        const auto* chunk = chunks.chunk_safe(cv::Vec3i{0, 0, 0});
        REQUIRE(chunk != nullptr);
        CHECK(chunk[0] == 7);
        REQUIRE(fs::exists(cacheFamily));
        CHECK_FALSE(fs::is_empty(cacheFamily));
    }

    // On Windows this removal only succeeds if the mapped view was released
    // before Chunked3d attempted to delete its per-instance cache directory.
    CHECK(fs::is_empty(cacheFamily));
    fs::remove_all(cacheRoot);
}
