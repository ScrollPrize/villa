#include "vc/core/util/S3AuthFallback.hpp"

#include <algorithm>
#include <array>

namespace vc
{

bool hasExplicitAwsCredentialError(std::string_view detail)
{
    constexpr std::array<std::string_view, 11> markers{
        "AccessDenied",
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
    return status == 401 || status == 403 || hasExplicitAwsCredentialError(detail);
}

S3AuthFallback::S3AuthFallback(bool isS3, bool credentialsLoaded) : mode_(isS3 && credentialsLoaded ? Mode::Authenticated : Mode::Disabled)
{
}

S3AuthFallback::Result S3AuthFallback::request(const Attempt& attempt) const
{
    Mode mode;
    std::uint64_t probeGeneration;
    {
        std::unique_lock lock(mutex_);
        probeFinished_.wait(lock, [&] { return mode_ != Mode::ProbingAnonymous; });
        mode = mode_;
        probeGeneration = probeGeneration_;
    }

    if (mode == Mode::Anonymous)
        return {attempt(true), std::nullopt, true};

    auto authenticated = attempt(false);
    if (mode != Mode::Authenticated || !isAwsAuthenticationFailure(authenticated.status_code, authenticated.body_string())) {
        return {std::move(authenticated), std::nullopt, false};
    }

    {
        std::unique_lock lock(mutex_);
        if (mode_ == Mode::ProbingAnonymous) {
            probeFinished_.wait(lock, [&] { return mode_ != Mode::ProbingAnonymous; });
        }
        if (mode_ == Mode::Anonymous) {
            lock.unlock();
            return {attempt(true), std::move(authenticated), true};
        }
        if (mode_ != Mode::Authenticated || probeGeneration_ != probeGeneration) {
            return {std::move(authenticated), std::nullopt, false};
        }
        mode_ = Mode::ProbingAnonymous;
    }

    try {
        auto anonymous = attempt(true);
        {
            std::lock_guard lock(mutex_);
            if (anonymous.ok() || anonymous.not_found()) {
                mode_ = Mode::Anonymous;
            } else if (isAwsAuthenticationFailure(anonymous.status_code, anonymous.body_string())) {
                mode_ = Mode::AuthenticatedOnly;
            } else {
                mode_ = Mode::Authenticated;
            }
            ++probeGeneration_;
        }
        probeFinished_.notify_all();
        return {std::move(anonymous), std::move(authenticated), true};
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            mode_ = Mode::Authenticated;
            ++probeGeneration_;
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
