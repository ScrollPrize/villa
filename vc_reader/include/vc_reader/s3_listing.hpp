#pragma once
// S3 ListObjectsV2: the request URL for one page of a prefix listing and the
// parse of its ListBucketResult response. Fetching is the caller's (anonymous
// or signed, through whatever HTTP stack it uses); this owns the protocol.
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vcr::s3 {

// A listable prefix: the endpoint a listing request goes to and the key prefix
// it lists. "https://bucket.s3.region.amazonaws.com/some/prefix" is origin
// "https://bucket.s3.region.amazonaws.com" and key "some/prefix/".
struct Prefix {
    std::string origin;
    std::string key;  // empty lists the bucket root; else ends with '/'
};

// Trailing slashes on the URL are ignored. nullopt when the URL has no scheme
// or no key path below its host.
[[nodiscard]] std::optional<Prefix> splitPrefixUrl(std::string_view url);

// The GET URL of one ListObjectsV2 page under `prefix`. With a delimiter
// ("/"), keys below the next delimiter are rolled up into common prefixes.
// Keys are requested URL-encoded, so any key survives the XML.
[[nodiscard]] std::string listUrl(const Prefix& prefix, std::string_view delimiter = {},
                                  std::string_view continuationToken = {}, int maxKeys = 1000);

struct Object {
    std::string key;  // the full key, decoded
    std::uint64_t size = 0;
};

struct ListPage {
    std::vector<Object> objects;
    std::vector<std::string> commonPrefixes;  // decoded, each ending with the delimiter
    std::string nextContinuationToken;        // empty on the last page
};

// Parse a ListBucketResult document. Throws std::runtime_error on an S3 Error
// document or malformed XML.
[[nodiscard]] ListPage parseListPage(std::string_view xml);

// The GET URL of the object `key` at `origin`, percent-encoded except '/'.
[[nodiscard]] std::string objectUrl(std::string_view origin, std::string_view key);

} // namespace vcr::s3
