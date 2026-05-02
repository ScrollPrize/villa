#pragma once

#include <string>

#include <utils/http_fetch.hpp>

namespace vc {

// AWS credentials used by remote HTTP/S3-backed volume and scroll workflows.
using HttpAuth = utils::AwsAuth;

inline HttpAuth loadAwsCredentials(const std::string& profile = "default")
{
    return utils::AwsAuth::load(profile);
}

} // namespace vc
