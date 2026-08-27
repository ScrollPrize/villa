#pragma once

#include <utils/http_fetch.hpp>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string_view>

namespace vc
{

[[nodiscard]] bool hasExplicitAwsCredentialError(std::string_view detail);

[[nodiscard]] bool isAwsAuthenticationFailure(long status, std::string_view detail);

class S3AuthFallback final
{
public:
    using Attempt = std::function<utils::HttpResponse(bool anonymous)>;

    struct Result {
        utils::HttpResponse response;
        std::optional<utils::HttpResponse> authenticatedFailure;
        bool usedAnonymous = false;
    };

    S3AuthFallback(bool isS3, bool credentialsLoaded);

    [[nodiscard]] Result request(const Attempt& attempt) const;
    [[nodiscard]] bool usesAnonymous() const;

private:
    enum class Mode {
        Disabled,
        Authenticated,
        ProbingAnonymous,
        Anonymous,
        AuthenticatedOnly,
    };

    mutable std::mutex mutex_;
    mutable std::condition_variable probeFinished_;
    mutable Mode mode_;
    mutable std::uint64_t probeGeneration_ = 0;
};

}  // namespace vc
