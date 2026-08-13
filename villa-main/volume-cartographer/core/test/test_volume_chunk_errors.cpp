#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::array<int, 3> kShape{4, 4, 4};
constexpr std::array<int, 3> kChunkShape{2, 2, 2};
constexpr std::size_t kChunkBytes = 8;

std::vector<std::byte> chunkBytes(std::byte value)
{
    return std::vector<std::byte>(kChunkBytes, value);
}

std::vector<vc::render::ChunkCache::LevelInfo> singleLevelInfo()
{
    vc::render::ChunkCache::LevelInfo level;
    level.shape = kShape;
    level.chunkShape = kChunkShape;
    return {level};
}

std::unique_ptr<vc::render::ChunkCache> makeCache(
    std::shared_ptr<vc::render::IChunkFetcher> fetcher,
    double fillValue = 0.0)
{
    vc::render::ChunkCache::Options options;
    options.maxConcurrentReads = 1;

    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers;
    fetchers.push_back(std::move(fetcher));

    return std::make_unique<vc::render::ChunkCache>(
        singleLevelInfo(),
        std::move(fetchers),
        fillValue,
        vc::render::ChunkDtype::UInt8,
        std::move(options));
}

class ThrowingChunkFetcher final : public vc::render::IChunkFetcher {
public:
    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        if (key.level == 0 && key.iz == 0 && key.iy == 0 && key.ix == 0) {
            throw std::runtime_error("synthetic downloader exception");
        }

        vc::render::ChunkFetchResult result;
        result.status = vc::render::ChunkFetchStatus::Found;
        result.bytes = chunkBytes(std::byte{7});
        return result;
    }
};

class MissingChunkFetcher final : public vc::render::IChunkFetcher {
public:
    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey&) override
    {
        vc::render::ChunkFetchResult result;
        result.status = vc::render::ChunkFetchStatus::Missing;
        return result;
    }
};

class HttpErrorChunkFetcher final : public vc::render::IChunkFetcher {
public:
    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        if (key.level == 0 && key.iz == 0 && key.iy == 0 && key.ix == 0) {
            vc::render::ChunkFetchResult result;
            result.status = vc::render::ChunkFetchStatus::HttpError;
            result.httpStatus = 400;
            result.message = "HTTP 400 fetching 0/0/0/0";
            return result;
        }

        vc::render::ChunkFetchResult result;
        result.status = vc::render::ChunkFetchStatus::Found;
        result.bytes = chunkBytes(std::byte{7});
        return result;
    }
};

} // namespace

TEST_CASE("ChunkCache converts fetch exceptions to chunk errors")
{
    auto cache = makeCache(std::make_shared<ThrowingChunkFetcher>());

    vc::render::ChunkResult result;
    CHECK_NOTHROW(result = cache->getChunkBlocking(0, 0, 0, 0));
    CHECK(result.status == vc::render::ChunkStatus::Error);
    CHECK(result.error.find("synthetic downloader exception") != std::string::npos);
}

TEST_CASE("Volume::readZYX reports downloader chunk errors outside OpenMP workers")
{
    auto cache = makeCache(std::make_shared<ThrowingChunkFetcher>());
    Array3D<uint8_t> out({4, 4, 4});

    bool threw = false;
    try {
        Volume::readZYX(out, {0, 0, 0}, *cache, 0);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        CHECK(message.find("Volume::read failed fetching chunk 0/0/0/0") != std::string::npos);
        CHECK(message.find("synthetic downloader exception") != std::string::npos);
    }

    CHECK(threw);
}

TEST_CASE("Volume::readZYX reports HTTP chunk errors outside OpenMP workers")
{
    auto cache = makeCache(std::make_shared<HttpErrorChunkFetcher>());
    Array3D<uint8_t> out({4, 4, 4});

    bool threw = false;
    try {
        Volume::readZYX(out, {0, 0, 0}, *cache, 0);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        CHECK(message.find("Volume::read failed fetching chunk 0/0/0/0") != std::string::npos);
        CHECK(message.find("HTTP 400 fetching 0/0/0/0") != std::string::npos);
    }

    CHECK(threw);
}

TEST_CASE("Volume::readZYX treats missing sparse chunks as fill value")
{
    auto cache = makeCache(std::make_shared<MissingChunkFetcher>(), 13.0);
    Array3D<uint8_t> out({4, 4, 4}, uint8_t{99});

    CHECK_NOTHROW(Volume::readZYX(out, {0, 0, 0}, *cache, 0));
    CHECK(std::all_of(out.data_.begin(), out.data_.end(), [](uint8_t value) {
        return value == 13;
    }));
}
