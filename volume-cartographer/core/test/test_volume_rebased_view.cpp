// Volume::NewRebasedView: a level-shifted view over a pyramid whose fine
// levels are absent, the shape fiber-prediction exports take (/3 upward).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "utils/zarr.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace
{

fs::path temporaryDirectory()
{
    std::mt19937_64 rng(std::random_device{}());
    auto path = fs::temp_directory_path() /
                ("vc_volume_rebased_view_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

void createZyx(const fs::path& path,
               std::array<std::size_t, 3> shape,
               unsigned char value)
{
    utils::ZarrMetadata metadata;
    metadata.version = utils::ZarrVersion::v2;
    metadata.shape = {shape[0], shape[1], shape[2]};
    metadata.chunks = {shape[0], shape[1], shape[2]};
    metadata.dtype = utils::ZarrDtype::uint8;
    metadata.compressor_id.clear();
    metadata.fill_value = 0.0;
    auto array = utils::ZarrArray::create(path, metadata);
    std::vector<std::byte> bytes(shape[0] * shape[1] * shape[2],
                                 static_cast<std::byte>(value));
    const std::array<std::size_t, 3> key{0, 0, 0};
    array.write_chunk(key, bytes);
}

}  // namespace

TEST_CASE("NewRebasedView shifts a sparse pyramid by whole levels")
{
    const auto dir = temporaryDirectory();
    // Published from /3 upward: the level-0 frame is 8x the stored array.
    createZyx(dir / "presence.zarr" / "3", {4, 4, 4}, 200);
    createZyx(dir / "presence.zarr" / "4", {2, 2, 2}, 100);
    const auto source = Volume::New(dir / "presence.zarr");
    REQUIRE(source->shape() == std::array<int, 3>{32, 32, 32});
    REQUIRE(source->firstPresentScaleLevel() == 3);

    SUBCASE("level 0 is the source itself")
    {
        CHECK(Volume::NewRebasedView(source, 0) == source);
    }

    SUBCASE("a one-level rebase halves the frame and keeps the gap")
    {
        const auto view = Volume::NewRebasedView(source, 1);
        REQUIRE(view);
        CHECK(view->shape() == std::array<int, 3>{16, 16, 16});
        CHECK(view->numScales() == 4);
        CHECK_FALSE(view->hasScaleLevel(0));
        CHECK_FALSE(view->hasScaleLevel(1));
        CHECK(view->hasScaleLevel(2));
        CHECK(view->hasScaleLevel(3));
        CHECK(view->firstPresentScaleLevel() == 2);
        CHECK(view->shape(2) == std::array<int, 3>{4, 4, 4});
        CHECK(view->id() == source->id() + "-vc-base-1");
        CHECK(view->id() != source->id());
        // The chunk source opens through the factory with the shifted levels,
        // and the fetchers moved with them: the view's level 2 is the source's
        // /3 (value 200) and its level 3 the source's /4 (value 100).
        auto cache = view->sharedChunkCache();
        REQUIRE(cache);
        CHECK(cache->numLevels() == 4);
        const auto firstVoxel = [&](int level) {
            const auto chunk = cache->getChunkBlocking(level, 0, 0, 0);
            REQUIRE(chunk.status == vc::render::ChunkStatus::Data);
            REQUIRE(chunk.bytes);
            REQUIRE_FALSE(chunk.bytes->empty());
            return static_cast<int>(std::to_integer<unsigned char>((*chunk.bytes)[0]));
        };
        CHECK(firstVoxel(2) == 200);
        CHECK(firstVoxel(3) == 100);
    }

    SUBCASE("rebasing onto the coarsest present level leaves one level")
    {
        const auto view = Volume::NewRebasedView(source, 4);
        REQUIRE(view);
        CHECK(view->shape() == std::array<int, 3>{2, 2, 2});
        CHECK(view->numScales() == 1);
        CHECK(view->hasScaleLevel(0));
        CHECK(view->firstPresentScaleLevel() == 0);
    }

    SUBCASE("the logical frame follows the source's, not the padded stored level")
    {
        // A source whose recorded frame is smaller than what the stored level
        // implies (30 vs 4 x 8 = 32): the view halves the recorded frame with
        // the pyramid's ceiling rule instead of re-synthesizing it from /3.
        utils::Json metadata = utils::Json::object();
        metadata["uuid"] = "presence-30";
        metadata["slices"] = 30;
        metadata["height"] = 30;
        metadata["width"] = 30;
        const fs::path root = dir / "presence.zarr";
        const auto exact = Volume::NewFromPreparedChunkedSource(
            [root]() { return vc::render::openLocalZarrPyramid(root); }, metadata);
        REQUIRE(exact->shape() == std::array<int, 3>{30, 30, 30});
        const auto view = Volume::NewRebasedView(exact, 1);
        CHECK(view->shape() == std::array<int, 3>{15, 15, 15});
        CHECK(view->shape(2) == std::array<int, 3>{4, 4, 4});
        const auto twice = Volume::NewRebasedView(exact, 2);
        CHECK(twice->shape() == std::array<int, 3>{8, 8, 8});
        CHECK(twice->firstPresentScaleLevel() == 1);
    }

    SUBCASE("rebasing past the pyramid is refused")
    {
        CHECK_THROWS_AS(Volume::NewRebasedView(source, 5), std::runtime_error);
        CHECK_THROWS_AS(Volume::NewRebasedView(nullptr, 1), std::invalid_argument);
        CHECK_THROWS_AS(Volume::NewRebasedView(source, -1), std::invalid_argument);
    }

    fs::remove_all(dir);
}
