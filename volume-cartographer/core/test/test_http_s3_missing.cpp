// Actual libcurl + HttpStore/ZarrArray regression tests; no remote bucket.
#include <doctest/doctest.h>
#include "LoopbackHttpServer.hpp"
#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>
#include <QCoreApplication>
#include <array>

namespace {
using Server = LoopbackHttpServer;
const std::string missing = "<Error><Code>NoSuchKey</Code></Error>";
const std::string bucket = "<Error><Code>NoSuchBucket</Code></Error>";
const std::string metadata = R"({"zarr_format":2,"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null})";
utils::HttpClient::Config config(std::size_t retries = 1) {
    utils::HttpClient::Config c;
    c.max_retries = retries;
    c.connect_timeout = c.transfer_timeout = std::chrono::seconds{2};
    return c;
}
}

TEST_CASE("generic HTTP 404 remains a missing object") {
    Server server([](auto&) { return Server::Reply{404, "<html>Not found</html>", false}; });
    auto r = utils::HttpClient(config()).get(server.url());
    CHECK(r.not_found());
    CHECK(server.requests().size() == 1);
    CHECK(utils::HttpClient(config()).head(server.url()).not_found());
    CHECK(server.requests().size() == 2);
}

TEST_CASE("complete S3 NoSuchKey is missing without retries") {
    for (const auto& body : {missing, std::string("<?xml version='1.0'?><Error xmlns='http://s3.amazonaws.com/doc/2006-03-01/'><Code> NoSuchKey </Code><Message>Missing</Message></Error>")}) {
        Server server([&](auto&) { return Server::Reply{404, body}; });
        auto r = utils::HttpClient(config()).get(server.url());
        CHECK(r.not_found());
        CHECK(server.requests().size() == 1);
    }
}

TEST_CASE("unconfirmed S3 404 retries on a fresh connection and returns original bytes") {
    for (const auto& body : {bucket, std::string{}, std::string("<html>Not found</html>"), std::string("<Error><Code>NoSuchKey</Code>")}) {
        Server server([&](const auto& req) { return req.connection == 0
            ? Server::Reply{404, body} : Server::Reply{200, std::string(64, 'x')}; });
        auto r = utils::HttpClient(config()).get(server.url());
        CHECK(r.ok());
        CHECK(r.body_string() == std::string(64, 'x'));
        const auto requests = server.requests();
        REQUIRE(requests.size() == 2);
        CHECK(requests[0].connection != requests[1].connection);
    }
}

TEST_CASE("exhausted or malformed S3 404 is an error, never missing") {
    for (const auto& body : {bucket, std::string{}, std::string("<html><Code>NoSuchKey</Code></html>"),
             std::string("<Error><Code>NoSuchKey</Code>"), std::string("<Error><Code>NoSuchKey</Code><Code>NoSuchBucket</Code></Error>"),
             std::string("<!DOCTYPE Error><Error><Code>NoSuchKey</Code></Error>"),
             std::string("<Error><Code><nested/>NoSuchKey</Code></Error>"), std::string("<Error><Code>NoSuchKey</Code></Error>trailing")}) {
        Server server([&](auto&) { return Server::Reply{404, body}; });
        auto r = utils::HttpClient(config()).get(server.url());
        CHECK(r.status_code == 404);
        CHECK_FALSE(r.not_found());
        CHECK_FALSE(r.error_message.empty());
        CHECK(server.requests().size() == 2);
    }
}

TEST_CASE("S3 HEAD 404 requires ranged GET confirmation") {
    Server server([](auto&) { return Server::Reply{404, missing}; });
    auto r = utils::HttpClient(config(0)).head(server.url());
    CHECK(r.not_found());
    CHECK(r.body.empty());
    const auto requests = server.requests();
    REQUIRE(requests.size() == 2);
    CHECK(requests[0].method == "HEAD");
    CHECK(requests[1].method == "GET");
    CHECK(requests[1].headers.find("Range: bytes=0-0") != std::string::npos);
}

TEST_CASE("S3 HEAD recovery retains full object size instead of the probe byte length") {
    int calls = 0;
    Server server([&](const auto& req) {
        if (++calls == 1) return Server::Reply{404, bucket};
        return req.method == "GET" ? Server::Reply{206, "x"} : Server::Reply{200, std::string(64, 'x')};
    });
    auto r = utils::HttpClient(config()).head(server.url());
    CHECK(r.ok());
    CHECK(r.content_length == 64);
    CHECK(r.body.empty());
    CHECK(server.requests().size() == 3);
}

TEST_CASE("S3 HEAD unconfirmed GET cannot become a missing object") {
    Server server([](auto&) { return Server::Reply{404, ""}; });
    auto r = utils::HttpClient(config(0)).head(server.url());
    CHECK_FALSE(r.not_found());
    CHECK_FALSE(r.error_message.empty());
    CHECK(r.body.empty());
    CHECK(server.requests().size() == 2);
}

TEST_CASE("ranged GET retains its byte range on a fresh retry") {
    Server server([](const auto& req) { return req.connection == 0
        ? Server::Reply{404, bucket} : Server::Reply{206, "34567"}; });
    auto r = utils::HttpClient(config()).get_range(server.url(), 3, 5);
    CHECK(r.status_code == 206);
    CHECK(r.body_string() == "34567");
    REQUIRE(server.requests().size() == 2);
    for (const auto& req : server.requests())
        CHECK(req.headers.find("Range: bytes=3-7") != std::string::npos);
}

TEST_CASE("HttpStore propagates fetch failures in all optional read APIs") {
    for (int status : {403, 404, 500}) {
        Server server([&](auto&) { return Server::Reply{status, bucket}; });
        utils::HttpStore store(server.url(""), config(0));
        CHECK_THROWS_AS(store.get_if_exists("key"), std::runtime_error);
        CHECK_THROWS_AS(store.get_partial("key", 0, 1), std::runtime_error);
        CHECK_THROWS_AS(store.exists("key"), std::runtime_error);
        CHECK_THROWS_AS(store.size_of("key"), std::runtime_error);
    }
}

TEST_CASE("HttpStore preserves confirmed sparse misses") {
    Server server([](auto&) { return Server::Reply{404, missing}; });
    utils::HttpStore store(server.url(""), config(0));
    CHECK_FALSE(store.get_if_exists("key").has_value());
    CHECK_FALSE(store.get_partial("key", 0, 1).has_value());
    CHECK_FALSE(store.exists("key"));
    CHECK_FALSE(store.size_of("key").has_value());
}

TEST_CASE("ZarrArray never substitutes fill values for an unconfirmed 404") {
    Server server([](const auto& req) {
        if (req.path == "/.zarray") return Server::Reply{200, metadata};
        if (req.path == "/0.0.0") return Server::Reply{404, bucket};
        return Server::Reply{404, missing};
    });
    auto store = std::make_shared<utils::HttpStore>(server.url(""), config(0));
    auto array = utils::ZarrArray::open(store, "");
    const std::array<std::size_t, 3> index{0, 0, 0};
    CHECK_THROWS_AS(array.read_chunk_encoded(index), std::runtime_error);
    CHECK_THROWS_AS(array.read_chunk(index), std::runtime_error);
}

TEST_CASE("automatic S3 policy uses the URL authority, not path or query text") {
    Server server([](auto&) { return Server::Reply{404, "", false}; });
    struct ProxyScope {
        QByteArray previous = qgetenv("http_proxy");
        bool present = qEnvironmentVariableIsSet("http_proxy");
        explicit ProxyScope(const std::string& url) { qputenv("http_proxy", QByteArray::fromStdString(url)); }
        ~ProxyScope() { if (present) qputenv("http_proxy", previous); else qunsetenv("http_proxy"); }
    } proxy(server.url(""));
    // This local HTTP proxy never forwards requests or performs origin DNS.
    for (const char* url : {"http://bucket.s3.us-east-1.amazonaws.com/key",
            "http://s3.us-west-2.amazonaws.com/bucket/key",
            "http://bucket.s3-us-west-2.amazonaws.com/key",
            "http://bucket.s3.cn-north-1.amazonaws.com.cn/key"}) {
        const auto r = utils::HttpClient(config(0)).get(url);
        CHECK_FALSE(r.not_found());
        CHECK_FALSE(r.error_message.empty());
    }
    for (const char* url : {"http://ordinary.example/s3.amazonaws.com/key",
            "http://s3.amazonaws.com.example/key", "http://ordinary.example/?url=s3.amazonaws.com"}) {
        CHECK(utils::HttpClient(config(0)).get(url).not_found());
    }
    CHECK(server.requests().size() == 7);
}

TEST_CASE("an unconfirmed S3 HEAD stays an error when retries are disabled") {
    Server server([](const auto& req) { return req.method == "HEAD"
        ? Server::Reply{404, bucket} : Server::Reply{206, "x"}; });
    const auto r = utils::HttpClient(config(0)).head(server.url());
    CHECK_FALSE(r.ok());
    CHECK_FALSE(r.not_found());
    CHECK_FALSE(r.error_message.empty());
    CHECK(server.requests().size() == 2);
}

TEST_CASE("zero-length HttpStore range keeps its no-request behavior") {
    Server server([](auto&) { return Server::Reply{200, "x"}; });
    utils::HttpStore store(server.url(""), config(0));
    CHECK_FALSE(store.get_partial("key", 0, 0).has_value());
    CHECK(server.requests().empty());
}

TEST_CASE("S3 identity remains strict when a retry loses S3 response headers") {
    int calls = 0;
    Server server([&](auto&) { return Server::Reply{404, "", ++calls == 1}; });
    const auto r = utils::HttpClient(config()).get(server.url());
    CHECK_FALSE(r.not_found());
    CHECK_FALSE(r.error_message.empty());
    CHECK(server.requests().size() == 2);
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qputenv("NO_PROXY", "127.0.0.1");
    qputenv("no_proxy", "127.0.0.1");
    return doctest::detail::runAll();
}
