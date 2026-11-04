#pragma once

#include "vc/core/util/RemoteAuth.hpp"

#include <vc_reader/s3_listing.hpp>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc {

// Fetch URL body as a string. Returns an empty string for 4xx misses, and
// throws for authentication and server errors.
std::string httpGetString(const std::string& url, const HttpAuth& auth = {});

// Fetch URL body as bytes. Returns an empty vector for 4xx misses, and throws
// for authentication and server errors.
std::vector<std::byte> httpGetBytes(const std::string& url, const HttpAuth& auth = {});

// Every object under `prefix` (S3 ListObjectsV2, 1000 keys a page), with the
// full keys as listed. `cancelled`, when given, is polled before each page;
// nullopt when it stops the listing. Throws when a page cannot be fetched or
// parsed.
std::optional<std::vector<vcr::s3::Object>> listS3Prefix(
    const vcr::s3::Prefix& prefix, const HttpAuth& auth = {},
    const std::function<bool()>& cancelled = {});

} // namespace vc
