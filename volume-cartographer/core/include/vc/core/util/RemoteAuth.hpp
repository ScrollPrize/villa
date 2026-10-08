#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <utils/http_fetch.hpp>

namespace vc {

// AWS credentials used by remote HTTP/S3-backed volume and scroll workflows.
using HttpAuth = utils::AwsAuth;

inline HttpAuth loadAwsCredentials(const std::string& profile = "default")
{
    return utils::AwsAuth::load(profile);
}

// Keep credential-process execution off the public-S3 fast path.  Constructing
// this object never consults ambient AWS configuration; resolve() does so at
// most once, and callers invoke it only after an anonymous request is denied.
class DeferredAwsAuth final
{
public:
    using Loader = std::function<HttpAuth()>;

    DeferredAwsAuth(HttpAuth explicitAuth,
                    bool discoverAwsCredentials,
                    std::string region,
                    Loader loader = {})
        : explicitAuth_(std::move(explicitAuth))
        , discoverAwsCredentials_(discoverAwsCredentials)
        , region_(std::move(region))
        , loader_(std::move(loader))
    {
    }

    [[nodiscard]] bool available() const noexcept
    {
        return !explicitAuth_.empty() || discoverAwsCredentials_;
    }

    [[nodiscard]] HttpAuth resolve() const
    {
        std::lock_guard lock(mutex_);
        if (!resolved_) {
            auto auth = explicitAuth_;
            if (auth.empty() && discoverAwsCredentials_) {
                auth = loader_ ? loader_() : loadAwsCredentials();
            }
            if (!auth.empty() && auth.region.empty())
                auth.region = region_;
            if (auth.access_key.empty() || auth.secret_key.empty())
                auth = {};
            resolved_ = std::move(auth);
        }
        return *resolved_;
    }

private:
    HttpAuth explicitAuth_;
    bool discoverAwsCredentials_ = true;
    std::string region_;
    Loader loader_;
    mutable std::mutex mutex_;
    mutable std::optional<HttpAuth> resolved_;
};

} // namespace vc
