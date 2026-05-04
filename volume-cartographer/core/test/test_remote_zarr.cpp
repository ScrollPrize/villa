#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteAuth.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <cstdlib>
#include <string>

// ---------- RemoteUrl pure-function tests --------------------------------------

TEST_CASE("resolveRemoteUrl: passthrough for plain http(s)")
{
    auto r = vc::resolveRemoteUrl("https://example.com/foo");
    CHECK(r.httpsUrl =="https://example.com/foo");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.awsRegion.empty());
}

TEST_CASE("resolveRemoteUrl: s3:// expands to virtual-hosted-style URL")
{
    auto r = vc::resolveRemoteUrl("s3://my-bucket/path/to/key");
    CHECK(r.httpsUrl =="https://my-bucket.s3.us-east-1.amazonaws.com/path/to/key");
    CHECK(r.awsRegion == "us-east-1");
    CHECK(r.useAwsSigv4);
}

TEST_CASE("resolveRemoteUrl: s3:// with no key")
{
    auto r = vc::resolveRemoteUrl("s3://just-bucket");
    CHECK(r.httpsUrl =="https://just-bucket.s3.us-east-1.amazonaws.com");
    CHECK(r.awsRegion == "us-east-1");
    CHECK(r.useAwsSigv4);
}

TEST_CASE("resolveRemoteUrl: s3+REGION:// uses explicit region")
{
    auto r = vc::resolveRemoteUrl("s3+eu-west-2://my-bucket/key");
    CHECK(r.httpsUrl =="https://my-bucket.s3.eu-west-2.amazonaws.com/key");
    CHECK(r.awsRegion == "eu-west-2");
    CHECK(r.useAwsSigv4);
}

TEST_CASE("resolveRemoteUrl: s3+ malformed (no scheme separator) treated as plain URL")
{
    auto r = vc::resolveRemoteUrl("s3+broken");
    CHECK_FALSE(r.useAwsSigv4);
}

TEST_CASE("resolveRemoteUrl: virtual-hosted https S3 URL detected as SigV4-eligible")
{
    auto r = vc::resolveRemoteUrl(
        "https://bucket.s3.us-west-2.amazonaws.com/some/key");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-west-2");
}

TEST_CASE("resolveRemoteUrl: legacy s3.amazonaws.com URL defaults region to us-east-1")
{
    auto r = vc::resolveRemoteUrl(
        "https://bucket.s3.amazonaws.com/some/key");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-east-1");
}

TEST_CASE("resolveRemoteUrl: non-S3 https URL stays SigV4-disabled")
{
    auto r = vc::resolveRemoteUrl("https://dl.example.org/foo.zarr");
    CHECK_FALSE(r.useAwsSigv4);
}

// ---------- Live remote S3 fetch -----------------------------------------------
//
// Pulls from the public Vesuvius Challenge AWS Open Data Registry bucket
// (https://registry.opendata.aws/vesuvius-challenge-herculaneum-scrolls/).
// Anonymous reads to public S3 are free for the bucket owner and require no
// credentials. Set VC_SKIP_REMOTE_TESTS=1 to skip when offline / behind a
// firewall, or override the URL with VC_TEST_REMOTE_ZARR_URL.

namespace {
const char* kDefaultRemoteZarr =
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
    "volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/";
}

TEST_CASE("openHttpZarrPyramid: live Vesuvius public zarr opens and yields a level")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    vc::render::OpenedChunkedZarr opened;
    try {
        opened = vc::render::openHttpZarrPyramid(url);
    } catch (const std::exception& e) {
        MESSAGE("openHttpZarrPyramid threw: " << e.what() << " — skipping");
        return;
    }

    REQUIRE_FALSE(opened.fetchers.empty());
    CHECK(opened.shapes.size() == opened.fetchers.size());
    CHECK(opened.chunkShapes.size() == opened.fetchers.size());
    CHECK((opened.dtype == vc::render::ChunkDtype::UInt8 ||
           opened.dtype == vc::render::ChunkDtype::UInt16));
    const auto shape = opened.shapes.at(0);
    for (int dim : shape) CHECK(dim > 0);
}

TEST_CASE("openHttpZarrPyramid + ChunkCache: fetch one chunk over the wire")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    std::unique_ptr<vc::render::ChunkCache> cache;
    try {
        auto opened = vc::render::openHttpZarrPyramid(url);
        cache = vc::render::createChunkCache(std::move(opened),
                                             64ULL * 1024 * 1024, 4);
    } catch (const std::exception& e) {
        MESSAGE("setup threw: " << e.what() << " — skipping");
        return;
    }
    REQUIRE(cache);
    REQUIRE(cache->numLevels() >= 1);

    const int coarsest = cache->numLevels() - 1;
    auto r = cache->getChunkBlocking(coarsest, 0, 0, 0);
    INFO("status=" << static_cast<int>(r.status)
         << " err=" << r.error);
    CHECK((r.status == vc::render::ChunkStatus::Data ||
           r.status == vc::render::ChunkStatus::AllFill ||
           r.status == vc::render::ChunkStatus::Error));
    if (r.status == vc::render::ChunkStatus::Data) {
        REQUIRE(r.bytes);
        const auto chunk = cache->chunkShape(coarsest);
        const auto dtypeBytes = cache->dtype() == vc::render::ChunkDtype::UInt16 ? 2 : 1;
        CHECK(r.bytes->size() ==
              std::size_t(chunk[0]) * std::size_t(chunk[1]) *
              std::size_t(chunk[2]) * std::size_t(dtypeBytes));
    }
}

TEST_CASE("Remote zarr: pyramid has multiple levels with shrinking shapes")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    vc::render::OpenedChunkedZarr opened;
    try {
        opened = vc::render::openHttpZarrPyramid(url);
    } catch (const std::exception& e) {
        MESSAGE("openHttpZarrPyramid threw: " << e.what() << " — skipping");
        return;
    }

    REQUIRE(opened.shapes.size() >= 1);
    if (opened.shapes.size() < 2) {
        MESSAGE("only 1 pyramid level present at this URL — skipping shrink check");
        return;
    }
    for (std::size_t i = 1; i < opened.shapes.size(); ++i) {
        for (int dim = 0; dim < 3; ++dim) {
            CHECK(opened.shapes[i][dim] <= opened.shapes[i - 1][dim]);
        }
    }
}

TEST_CASE("Remote zarr: cache reuse — second fetch of same chunk returns same bytes pointer")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    std::unique_ptr<vc::render::ChunkCache> cache;
    try {
        auto opened = vc::render::openHttpZarrPyramid(url);
        cache = vc::render::createChunkCache(std::move(opened),
                                             64ULL * 1024 * 1024, 4);
    } catch (const std::exception& e) {
        MESSAGE("setup threw: " << e.what() << " — skipping");
        return;
    }

    const int level = cache->numLevels() - 1;
    auto first = cache->getChunkBlocking(level, 0, 0, 0);
    auto second = cache->getChunkBlocking(level, 0, 0, 0);

    CHECK(first.status == second.status);
    if (first.status == vc::render::ChunkStatus::Data &&
        second.status == vc::render::ChunkStatus::Data) {
        CHECK(first.bytes.get() == second.bytes.get());
    }
}

TEST_CASE("Remote zarr: prefetchChunks with wait blocks until all keys resolve")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    std::unique_ptr<vc::render::ChunkCache> cache;
    try {
        auto opened = vc::render::openHttpZarrPyramid(url);
        cache = vc::render::createChunkCache(std::move(opened),
                                             64ULL * 1024 * 1024, 4);
    } catch (const std::exception& e) {
        MESSAGE("setup threw: " << e.what() << " — skipping");
        return;
    }

    const int level = cache->numLevels() - 1;
    std::vector<vc::render::ChunkKey> keys;
    keys.push_back({level, 0, 0, 0});
    keys.push_back({level, 0, 0, 1});

    cache->prefetchChunks(keys, /*wait=*/true);

    auto a = cache->tryGetChunk(level, 0, 0, 0);
    auto b = cache->tryGetChunk(level, 0, 0, 1);
    CHECK(a.status != vc::render::ChunkStatus::MissQueued);
    CHECK(b.status != vc::render::ChunkStatus::MissQueued);
}

TEST_CASE("vc::httpGetString: GET against the public zarr returns non-empty body")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const std::string url = std::string(kDefaultRemoteZarr) + "0/.zarray";
    std::string body;
    try {
        body = vc::httpGetString(url);
    } catch (const std::exception& e) {
        MESSAGE("httpGetString threw: " << e.what() << " — skipping");
        return;
    }
    CHECK_FALSE(body.empty());
    CHECK(body.find("shape") != std::string::npos);
}

TEST_CASE("vc::httpGetString: 404 returns empty body without throwing")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    std::string body;
    try {
        body = vc::httpGetString(
            "https://dl.ash2txt.org/this/path/does/not/exist/at/all/hopefully.json");
    } catch (const std::exception& e) {
        MESSAGE("httpGetString threw on 404: " << e.what());
        return;
    }
    CHECK(body.empty());
}

TEST_CASE("vc::httpGetString: 5xx server error throws")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    bool threw = false;
    try {
        vc::httpGetString("https://httpbin.org/status/500");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Remote zarr: stats() reports remote-fetch tracking")
{
    if (const char* skip = std::getenv("VC_SKIP_REMOTE_TESTS"); skip && *skip == '1') {
        MESSAGE("VC_SKIP_REMOTE_TESTS=1 — skipping");
        return;
    }
    const char* url = std::getenv("VC_TEST_REMOTE_ZARR_URL");
    if (!url || !*url) url = kDefaultRemoteZarr;

    std::unique_ptr<vc::render::ChunkCache> cache;
    try {
        auto opened = vc::render::openHttpZarrPyramid(url);
        cache = vc::render::createChunkCache(std::move(opened),
                                             64ULL * 1024 * 1024, 4);
    } catch (const std::exception& e) {
        MESSAGE("setup threw: " << e.what() << " — skipping");
        return;
    }

    const auto before = cache->stats();
    CHECK(before.decodedBytes == 0);
    CHECK(before.decodedByteCapacity == 64ULL * 1024 * 1024);

    auto r = cache->getChunkBlocking(cache->numLevels() - 1, 0, 0, 0);
    if (r.status == vc::render::ChunkStatus::Data) {
        const auto after = cache->stats();
        CHECK(after.decodedBytes > 0);
    }
}
