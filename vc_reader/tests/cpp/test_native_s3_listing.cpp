// S3 ListObjectsV2 requests and responses: prefix URLs, page URLs, and the
// ListBucketResult parse (pagination, common prefixes, URL-encoded keys, errors).
#include "vc_reader/s3_listing.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace vcr::s3;

namespace {
void require(bool v, const std::string& message)
{
    if (!v) throw std::runtime_error(message);
}

bool throws(const std::string& xml)
{
    try {
        (void)parseListPage(xml);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}
} // namespace

int main()
{
    try {
        // Prefix URLs.
        const auto split = splitPrefixUrl("https://bucket.test/a/b.zarr//");
        require(split && split->origin == "https://bucket.test" && split->key == "a/b.zarr/", "split");
        require(splitPrefixUrl("https://bucket.test/a")->key == "a/", "split adds slash");
        const auto root = splitPrefixUrl("https://bucket.test/");
        require(root && root->origin == "https://bucket.test" && root->key.empty() &&
                    splitPrefixUrl("https://bucket.test")->key.empty(),
                "bucket root");
        require(!splitPrefixUrl("bucket.test/a") && !splitPrefixUrl("https:///a"), "no host");

        // Page URLs: sorted parameters, encoded values.
        require(listUrl({"https://b.test", "a b/c/"}) ==
                    "https://b.test/?encoding-type=url&list-type=2&max-keys=1000&prefix=a%20b%2Fc%2F",
                "list url");
        require(listUrl({"https://b.test", ""}, "/", "tok+/=", 100) ==
                    "https://b.test/?continuation-token=tok%2B%2F%3D&delimiter=%2F&encoding-type=url&list-type=2&max-keys=100",
                "list url with delimiter and token");
        bool rejected = false;
        try { (void)listUrl({"https://b.test", ""}, {}, {}, 0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "max-keys range");

        // Object URLs keep '/'.
        require(objectUrl("https://b.test", "a b/c+d.json") == "https://b.test/a%20b/c%2Bd.json", "object url");

        // A truncated page with URL-encoded keys and common prefixes.
        const auto page = parseListPage(R"(<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>bucket</Name><Prefix>root%2F</Prefix><KeyCount>3</KeyCount><MaxKeys>2</MaxKeys>
  <Delimiter>%2F</Delimiter><EncodingType>url</EncodingType><IsTruncated>true</IsTruncated>
  <NextContinuationToken>1ueGcxLPRx1Tr/XYExHnhbYLgveDs2J/wm36Hy4vbOwM=</NextContinuationToken>
  <Contents><Key>root/a+b.json</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>
    <ETag>&quot;abc&quot;</ETag><Size>12</Size><StorageClass>STANDARD</StorageClass></Contents>
  <Contents><Key>root/%C3%A9%26x</Key><Size>0</Size></Contents>
  <CommonPrefixes><Prefix>root/sub%20dir/</Prefix></CommonPrefixes>
</ListBucketResult>)");
        require(page.objects.size() == 2 && page.objects[0].key == "root/a b.json" && page.objects[0].size == 12 &&
                    page.objects[1].key == "root/\xC3\xA9&x" && page.objects[1].size == 0,
                "objects");
        require(page.commonPrefixes.size() == 1 && page.commonPrefixes[0] == "root/sub dir/",
                "common prefixes; the request Prefix is not one");
        require(page.nextContinuationToken == "1ueGcxLPRx1Tr/XYExHnhbYLgveDs2J/wm36Hy4vbOwM=", "token");

        // A last page without encoding: entities only, no token.
        const auto last = parseListPage(
            "<ListBucketResult><IsTruncated>false</IsTruncated>"
            "<Contents><Key>a&amp;b&#x20;c&#65;+</Key><Size>5</Size></Contents>"
            "<NextContinuationToken/></ListBucketResult>");
        require(last.objects.size() == 1 && last.objects[0].key == "a&b cA+" && last.nextContinuationToken.empty(),
                "last page");

        // Failures.
        require(throws("<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"), "error document");
        require(throws("") && throws("<?xml version=\"1.0\"?>\n"), "empty response");
        require(throws("<html><body>nope</body></html>"), "not a listing");
        require(throws("<ListBucketResult><Contents><Key>a</Key></ListBucketResult>"), "mismatched tags");
        require(throws("<ListBucketResult><Contents><Key>a</Key>"), "unterminated");
        require(throws("<ListBucketResult><Contents><Size>x</Size></Contents></ListBucketResult>"), "bad size");
        require(throws("<ListBucketResult><Contents><Key>&bogus;</Key></Contents></ListBucketResult>"), "entity");
        require(throws("<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>"), "truncated without token");
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
