#include "vc/core/util/S3AuthFallback.hpp"

#include <algorithm>
#include <array>

namespace vc
{

bool hasExplicitAwsCredentialError(std::string_view detail)
{
    constexpr std::array<std::string_view, 10> markers{
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidClientTokenId",
        "InvalidSignatureException",
        "InvalidToken",
        "RequestExpired",
        "RequestTimeTooSkewed",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
        "UnrecognizedClientException",
    };
    return std::any_of(markers.begin(), markers.end(), [&](const auto marker) { return detail.find(marker) != std::string_view::npos; });
}

bool isAwsAuthenticationFailure(long status, std::string_view detail)
{
    if (status >= 200 && status < 300)
        return false;
    return status == 401 || status == 403 ||
           detail.find("AccessDenied") != std::string_view::npos ||
           hasExplicitAwsCredentialError(detail);
}

S3AuthFallback::S3AuthFallback(bool isS3, bool credentialsLoaded)
    : mode_(isS3 && credentialsLoaded ? Mode::Undecided : Mode::Disabled)
{
}

S3AuthFallback::Result S3AuthFallback::request(const Attempt& attempt) const
{
    Mode mode;
    {
        std::unique_lock lock(mutex_);
        probeFinished_.wait(lock, [&] { return mode_ != Mode::Probing; });
        mode = mode_;
        if (mode == Mode::Undecided)
            mode_ = Mode::Probing;
    }

    if (mode == Mode::Disabled || mode == Mode::Authenticated)
        return {attempt(false), std::nullopt, false};

    if (mode == Mode::Anonymous) {
        auto anonymous = attempt(true);
        if (anonymous.status_code != 401 && anonymous.status_code != 403)
            return {std::move(anonymous), std::nullopt, true};

        auto authenticated = attempt(false);
        if (authenticated.ok() || authenticated.not_found()) {
            std::lock_guard lock(mutex_);
            if (mode_ == Mode::Anonymous)
                mode_ = Mode::Authenticated;
        }
        return {std::move(authenticated), std::move(anonymous), false};
    }

    try {
        auto anonymous = attempt(true);
        if (anonymous.status_code == 401 || anonymous.status_code == 403) {
            auto authenticated = attempt(false);
            {
                std::lock_guard lock(mutex_);
                mode_ = authenticated.ok() || authenticated.not_found()
                    ? Mode::Authenticated
                    : Mode::Undecided;
            }
            probeFinished_.notify_all();
            return {std::move(authenticated), std::move(anonymous), false};
        }

        {
            std::lock_guard lock(mutex_);
            mode_ = anonymous.ok() ? Mode::Anonymous : Mode::Undecided;
        }
        probeFinished_.notify_all();
        return {std::move(anonymous), std::nullopt, true};
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            mode_ = Mode::Undecided;
        }
        probeFinished_.notify_all();
        throw;
    }
}

bool S3AuthFallback::usesAnonymous() const
{
    std::lock_guard lock(mutex_);
    return mode_ == Mode::Anonymous;
}

}  // namespace vc
