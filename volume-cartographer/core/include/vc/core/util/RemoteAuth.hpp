#pragma once

#include <string>

#include <vc_reader/zarr.hpp>

namespace vc {

// AWS credentials used by remote HTTP/S3-backed volume and scroll workflows.
struct HttpAuth {
    std::string access_key;
    std::string secret_key;
    std::string session_token;  // optional (STS temporary credentials)
    std::string region;         // e.g. "us-east-1"

    [[nodiscard]] bool empty() const noexcept { return access_key.empty() && secret_key.empty(); }
};

// From the environment and the shared AWS files; empty when there are none.
inline HttpAuth loadAwsCredentials()
{
    const auto keys = vcr::zarr::awsCredentials();
    return keys ? HttpAuth{keys->user, keys->secret, keys->token, keys->region} : HttpAuth{};
}

inline vcr::Auth readerAuth(const HttpAuth& auth)
{
    return auth.empty() ? vcr::Auth{} : vcr::Auth{"s3", auth.access_key, auth.secret_key, auth.session_token, auth.region};
}

} // namespace vc
