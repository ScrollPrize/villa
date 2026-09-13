#pragma once

#include <utils/http_fetch.hpp>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string_view>

namespace vc
{

[[nodiscard]] bool hasExplicitAwsCredentialError(std::string_view detail);

[[nodiscard]] bool isAwsAuthenticationFailure(long status, std::string_view detail);

// Selects anonymous S3 access on the first successful request and retains
// authenticated access only after anonymous access is denied.
class S3AuthFallback final
{
public:
    using Attempt = std::function<utils::HttpResponse(bool anonymous)>;

    struct Result {
        utils::HttpResponse response;
        std::optional<utils::HttpResponse> anonymousFailure;
        bool usedAnonymous = false;
    };

    S3AuthFallback(bool isS3, bool credentialsLoaded);

    [[nodiscard]] Result request(const Attempt& attempt) const;
    [[nodiscard]] bool usesAnonymous() const;

private:
    enum class Mode {
        Disabled,
        Undecided,
        Probing,
        Anonymous,
        Authenticated,
    };

    mutable std::mutex mutex_;
    mutable std::condition_variable probeFinished_;
    mutable Mode mode_;
};

}  // namespace vc
