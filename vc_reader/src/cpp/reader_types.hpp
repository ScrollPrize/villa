// The few types the reader's interfaces share, without the reader itself.
#pragma once

#include "vc_reader/exception_type.hpp"
#include "element_type.hpp"

#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>

namespace vcr {

using Index = std::vector<std::int64_t>;

inline std::string chunkKey(const Index& index, const std::string& separator)
{
    std::string key;
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
        if (axis) key += separator;
        key += std::to_string(index[axis]);
    }
    return key;
}

// How requests authenticate.
// A key below a remote location: slashes between them collapse, and a query (a signed
// URL's token) stays at the end, its bytes untouched. Anything without a scheme is a
// path, where ? and # are plain characters.
inline std::string joinLocation(const std::string& location, const std::string& key)
{
    const auto query = location.find("://") == std::string::npos ? std::string::npos : location.find_first_of("?#");
    auto end = query == std::string::npos ? location.size() : query;
    while (end > 0 && location[end - 1] == '/')
        --end;
    const auto start = std::min(key.find_first_not_of('/'), key.size());
    return location.substr(0, end) + "/" + key.substr(start) + (query == std::string::npos ? "" : location.substr(query));
}

// The location above `location`: its last path segment removed, trailing slashes ignored, a
// query or fragment kept as joinLocation keeps it. A location with nothing above it (a bare
// host, a path without a slash) comes back unchanged.
inline std::string parentLocation(const std::string& location)
{
    const bool url = location.find("://") != std::string::npos;
    const auto query = url ? location.find_first_of("?#") : std::string::npos;
    std::string path = location.substr(0, query);
    const std::string suffix = query == std::string::npos ? std::string() : location.substr(query);
    while (path.size() > 1 && path.back() == '/')
        path.pop_back();
    const auto authority = url ? path.find("://") + 3 : 0;
    const auto slash = path.rfind('/');
    if (slash == std::string::npos || slash < authority)
        return path + suffix;
    return path.substr(0, slash) + suffix;
}

struct Auth {
    std::string kind;  // "" (anonymous), "basic" or "s3"
    std::string user;  // basic: user name; s3: access key id
    std::string secret;  // basic: password; s3: secret access key
    std::string token;   // s3: session token (may be empty)
    std::string region;  // s3: the bucket's region
};

// 401/403: the request was not allowed (missing, wrong or expired credentials).
struct VCR_EXCEPTION_TYPE HttpDenied : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A chunk an offline read needs is in none of the disk layers.
struct VCR_EXCEPTION_TYPE CacheMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A read stopped because its cancel flag was set: chunks not started by then were not read.
struct VCR_EXCEPTION_TYPE Cancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace vcr
