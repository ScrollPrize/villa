// The loopback HTTP fixture, driven by vc_reader's own HTTP client: files, ranges,
// S3 listings with paging, signed buckets, handler answers, chunked bodies, and
// concurrent connections.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"
#include "vc/core/util/MemMap.hpp"

#include <vc_reader/native/http.hpp>
#include <vc_reader/s3_listing.hpp>
#include <vc_reader/zarr.hpp>

#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vc::test::LoopbackHttpServer;

namespace {

fs::path fixtureRoot()
{
    const auto root = fs::temp_directory_path() / ("vc_loopback_http_" + std::to_string(vc::memmap::pid()));
    fs::remove_all(root);
    for (const auto& [name, size] : {std::pair{"a.txt", 5}, {"dir/b.bin", 1000}, {"pfx/x/1", 1}, {"pfx/x/2", 2}, {"pfx/y", 3}}) {
        fs::create_directories((root / name).parent_path());
        std::ofstream(root / name, std::ios::binary) << (std::string(name) == "a.txt" ? std::string("hello") : std::string(std::size_t(size), 'b'));
    }
    return root;
}

std::string text(const std::vector<std::uint8_t>& body) { return std::string(body.begin(), body.end()); }

}  // namespace

TEST_CASE("Loopback server: files, ranges, handler, chunked, concurrency")
{
    const auto root = fixtureRoot();
    LoopbackHttpServer server;
    server.serveDirectory(root);
    server.setHandler([](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
        if (request.path == "/custom") return LoopbackHttpServer::Response{200, "custom:" + request.header("x-probe"), {}, false};
        if (request.path == "/chunked") return LoopbackHttpServer::Response{200, std::string(70000, 'c'), {}, true};
        return std::nullopt;
    });
    vcr::HttpClient http{vcr::HttpOptions{}};
    std::vector<std::uint8_t> body;

    REQUIRE(http.get(server.url() + "/a.txt", body));
    CHECK(text(body) == "hello");
    CHECK(http.head(server.url() + "/a.txt"));
    CHECK_FALSE(http.get(server.url() + "/missing.txt", body));
    CHECK_FALSE(http.head(server.url() + "/missing.txt"));
    CHECK(server.requestCount("/a.txt") == 2);
    CHECK(server.requestCount("/missing.txt") == 2);

    const auto ranged = http.get_range(server.url() + "/dir/b.bin", body, "10-19", 10);
    CHECK(ranged.status == 206);
    CHECK(body.size() == 10);
    CHECK(ranged.content_range == "bytes 10-19/1000");

    REQUIRE(http.get(server.url() + "/custom", body));
    CHECK(text(body) == "custom:");  // the client adds no X-Probe header
    REQUIRE(http.get(server.url() + "/chunked", body));
    CHECK(body.size() == 70000);
    CHECK(server.requests().back().header("host").starts_with("127.0.0.1"));

    // Eight connections at once, each served on its own thread.
    std::vector<std::future<bool>> reads;
    for (int i = 0; i < 8; ++i)
        reads.push_back(std::async(std::launch::async, [&, i] {
            vcr::HttpClient own{vcr::HttpOptions{}};
            std::vector<std::uint8_t> got;
            return own.get(server.url() + (i % 2 ? "/a.txt" : "/dir/b.bin"), got) && got.size() == (i % 2 ? 5u : 1000u);
        }));
    for (auto& read : reads) CHECK(read.get());
    fs::remove_all(root);
}

TEST_CASE("Loopback server: S3 objects, paged listings and signed buckets")
{
    const auto root = fixtureRoot();
    LoopbackHttpServer server;
    server.serveBucket("public", root);
    server.serveBucket("private", root, true);
    vcr::HttpClient http{vcr::HttpOptions{}};
    std::vector<std::uint8_t> body;

    REQUIRE(http.get(server.url() + "/public/pfx/y", body));
    CHECK(body.size() == 3);
    CHECK_FALSE(http.get(server.url() + "/public/pfx/none", body));

    // Everything under the bucket, three keys a page.
    const vcr::s3::Prefix all{server.url() + "/public", ""};
    REQUIRE(http.get(vcr::s3::listUrl(all, {}, {}, 3), body));
    auto page = vcr::s3::parseListPage(text(body));
    REQUIRE(page.objects.size() == 3);
    CHECK(page.objects[0].key == "a.txt");
    CHECK(page.objects[0].size == 5);
    CHECK(page.objects[1].key == "dir/b.bin");
    CHECK(page.objects[2].key == "pfx/x/1");
    REQUIRE_FALSE(page.nextContinuationToken.empty());
    REQUIRE(http.get(vcr::s3::listUrl(all, {}, page.nextContinuationToken, 3), body));
    page = vcr::s3::parseListPage(text(body));
    REQUIRE(page.objects.size() == 2);
    CHECK(page.objects[0].key == "pfx/x/2");
    CHECK(page.objects[1].key == "pfx/y");
    CHECK(page.nextContinuationToken.empty());

    // Delimited: one common prefix and one object under pfx/, one item a page.
    const vcr::s3::Prefix pfx{server.url() + "/public", "pfx/"};
    REQUIRE(http.get(vcr::s3::listUrl(pfx, "/", {}, 1), body));
    page = vcr::s3::parseListPage(text(body));
    CHECK(page.objects.empty());
    REQUIRE(page.commonPrefixes.size() == 1);
    CHECK(page.commonPrefixes[0] == "pfx/x/");
    REQUIRE_FALSE(page.nextContinuationToken.empty());
    REQUIRE(http.get(vcr::s3::listUrl(pfx, "/", page.nextContinuationToken, 1), body));
    page = vcr::s3::parseListPage(text(body));
    REQUIRE(page.objects.size() == 1);
    CHECK(page.objects[0].key == "pfx/y");
    CHECK(page.commonPrefixes.empty());
    CHECK(page.nextContinuationToken.empty());

    // The private bucket refuses anonymous requests and serves signed ones.
    CHECK_THROWS_AS(http.get(server.url() + "/private/a.txt", body), vcr::HttpDenied);
    const auto access = vcr::zarr::s3Access("s3://private/a.txt", vcr::Auth{"s3", "key", "secret", "", "us-east-1"}, server.url());
    vcr::HttpOptions signedOptions;
    signedOptions.auth = access.auth;
    vcr::HttpClient signedClient{signedOptions};
    REQUIRE(signedClient.get(access.url("s3://private/a.txt"), body));
    CHECK(text(body) == "hello");
    const auto requests = server.requests();
    REQUIRE(requests.size() >= 2);
    CHECK_FALSE(requests[requests.size() - 2].signedV4());
    CHECK(requests.back().signedV4());
    CHECK(requests.back().path == "/private/a.txt");
    fs::remove_all(root);
}
