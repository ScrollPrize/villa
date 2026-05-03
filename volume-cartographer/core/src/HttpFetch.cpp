#include "vc/core/util/HttpFetch.hpp"

#include <utils/http_fetch.hpp>

#include <chrono>
#include <stdexcept>

namespace vc {

namespace {

utils::HttpClient makeTextClient(const HttpAuth& auth)
{
    utils::HttpClient::Config cfg;
    cfg.aws_auth = auth;
    cfg.transfer_timeout = std::chrono::seconds{30};
    cfg.connect_timeout = std::chrono::seconds{5};
    cfg.max_retries = 2;
    return utils::HttpClient(std::move(cfg));
}

bool isAuthError(long status, const std::string& body)
{
    if (status == 401 || status == 403)
        return true;
    return body.find("ExpiredToken") != std::string::npos ||
           body.find("AccessDenied") != std::string::npos ||
           body.find("InvalidAccessKeyId") != std::string::npos ||
           body.find("SignatureDoesNotMatch") != std::string::npos ||
           body.find("TokenRefreshRequired") != std::string::npos ||
           body.find("InvalidToken") != std::string::npos;
}

std::string authErrorMessage(long status, const std::string& body)
{
    std::string message = "Access denied (HTTP " + std::to_string(status) + ")";
    const auto msgStart = body.find("<Message>");
    const auto msgEnd = body.find("</Message>");
    if (msgStart != std::string::npos && msgEnd != std::string::npos && msgEnd > msgStart + 9) {
        message = body.substr(msgStart + 9, msgEnd - (msgStart + 9)) +
                  " (HTTP " + std::to_string(status) + ")";
    }
    return message + ". Check your AWS credentials.";
}

} // namespace

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
    auto client = makeTextClient(auth);
    auto resp = client.get(url);
    if (resp.ok())
        return std::string(resp.body_string());

    if (resp.status_code >= 400) {
        const auto body = std::string(resp.body_string());
        if (isAuthError(resp.status_code, body))
            throw std::runtime_error(authErrorMessage(resp.status_code, body));
        if (resp.status_code >= 500) {
            throw std::runtime_error(
                "HTTP server error " + std::to_string(resp.status_code) + " fetching " + url);
        }
    }
    return {};
}

} // namespace vc
