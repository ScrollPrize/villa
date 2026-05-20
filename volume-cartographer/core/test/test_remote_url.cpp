#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/RemoteUrl.hpp"

#include <string>

using vc::resolveRemoteUrl;

TEST_CASE("s3:// with key uses default us-east-1")
{
    auto r = resolveRemoteUrl("s3://mybucket/path/to/key");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-east-1");
    CHECK(r.httpsUrl == "https://mybucket.s3.us-east-1.amazonaws.com/path/to/key");
}

TEST_CASE("s3:// without key drops trailing slash")
{
    auto r = resolveRemoteUrl("s3://mybucket");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-east-1");
    CHECK(r.httpsUrl == "https://mybucket.s3.us-east-1.amazonaws.com");
}

TEST_CASE("s3+REGION:// extracts region")
{
    auto r = resolveRemoteUrl("s3+eu-west-2://b/k");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "eu-west-2");
    CHECK(r.httpsUrl == "https://b.s3.eu-west-2.amazonaws.com/k");
}

TEST_CASE("s3+REGION:// with no key omits trailing path")
{
    auto r = resolveRemoteUrl("s3+ap-south-1://b");
    CHECK(r.httpsUrl == "https://b.s3.ap-south-1.amazonaws.com");
}

TEST_CASE("malformed s3+ without :// passes through as plain URL")
{
    auto r = resolveRemoteUrl("s3+region-no-slashes");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.awsRegion.empty());
    CHECK(r.httpsUrl == "s3+region-no-slashes");
}

TEST_CASE("https://bucket.s3.region.amazonaws.com is detected as S3")
{
    auto r = resolveRemoteUrl("https://b.s3.us-west-2.amazonaws.com/key");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-west-2");
    CHECK(r.httpsUrl == "https://b.s3.us-west-2.amazonaws.com/key");
}

TEST_CASE("https://bucket.s3.amazonaws.com defaults to us-east-1")
{
    auto r = resolveRemoteUrl("https://b.s3.amazonaws.com/key");
    CHECK(r.useAwsSigv4);
    CHECK(r.awsRegion == "us-east-1");
}

TEST_CASE("http:// with no S3 hostname passes through unchanged")
{
    auto r = resolveRemoteUrl("http://example.com/x");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.awsRegion.empty());
    CHECK(r.httpsUrl == "http://example.com/x");
}

TEST_CASE("https://example.com (non-S3) passes through")
{
    auto r = resolveRemoteUrl("https://example.com");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.awsRegion.empty());
    CHECK(r.httpsUrl == "https://example.com");
}

TEST_CASE("non-URL strings pass through")
{
    auto r = resolveRemoteUrl("/local/path");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.httpsUrl == "/local/path");
}

TEST_CASE("empty string passes through")
{
    auto r = resolveRemoteUrl("");
    CHECK_FALSE(r.useAwsSigv4);
    CHECK(r.httpsUrl.empty());
}
