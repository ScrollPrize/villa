// The few types the reader's interfaces share, without the reader itself.
#pragma once

#include "element_type.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vcr {

using Index = std::vector<std::int64_t>;

// How requests authenticate.
struct Auth {
    std::string kind;  // "" (anonymous), "basic" or "s3"
    std::string user;  // basic: user name; s3: access key id
    std::string secret;  // basic: password; s3: secret access key
    std::string token;   // s3: session token (may be empty)
    std::string region;  // s3: the bucket's region
};

// 401/403: the request was not allowed (missing, wrong or expired credentials).
struct HttpDenied : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A chunk an offline read needs is in none of the disk layers.
struct CacheMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace vcr
