#pragma once
// Objects at http(s):// and s3:// locations, read only, with an optional disk cache.
#include <cstdint>
#include <string>
#include <vector>

namespace helirec::remote {

// http://, https:// or s3://
bool is_remote(const std::string& location);
// The object's bytes; false: it does not exist (HTTP 404). s3://bucket/key is read from the bucket's public
// endpoint, https://bucket.s3.amazonaws.com/key (no credentials). Thread-safe.
// cache_dir (optional): the object, or the fact that it does not exist, is kept there and served from there the next
// time; with `refresh` it is fetched again all the same and the cached copy only serves when the server cannot be
// reached (metadata). Nothing is written to a cache directory with little free space left; nothing in it is ever
// deleted.
bool fetch(const std::string& location, std::vector<std::uint8_t>& out, const std::string& cache_dir = {},
           bool refresh = false);

}  // namespace helirec::remote
