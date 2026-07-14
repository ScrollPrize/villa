#pragma once

#include "vc/core/util/RemoteAuth.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vc {

// Fetch URL body as a string. Returns an empty string for 4xx misses, and
// throws for authentication and server errors.
std::string httpGetString(const std::string& url, const HttpAuth& auth = {});

// Fetch URL body as bytes. Returns an empty vector for 4xx misses, and throws
// for authentication and server errors.
std::vector<std::byte> httpGetBytes(const std::string& url, const HttpAuth& auth = {});

} // namespace vc
