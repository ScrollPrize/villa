#include "vc/core/util/HttpFetch.hpp"

#include <utils/http_fetch.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

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

bool hasGzipMagic(std::string_view body)
{
    return body.size() >= 2 &&
           static_cast<unsigned char>(body[0]) == 0x1f &&
           static_cast<unsigned char>(body[1]) == 0x8b;
}

std::string gzipInflate(std::string_view compressed)
{
    if (compressed.size() > std::numeric_limits<uInt>::max()) {
        throw std::runtime_error("Gzip HTTP response is too large to decompress");
    }

    z_stream stream{};
    if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
        throw std::runtime_error("Failed to initialize gzip decompressor");
    }

    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
    stream.avail_in = static_cast<uInt>(compressed.size());

    std::string output;
    std::array<char, 64 * 1024> buffer{};
    int rc = Z_OK;
    do {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = static_cast<uInt>(buffer.size());
        rc = inflate(&stream, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) {
            inflateEnd(&stream);
            throw std::runtime_error("Failed to decompress gzip HTTP response");
        }
        output.append(buffer.data(), buffer.size() - stream.avail_out);
    } while (rc != Z_STREAM_END);

    inflateEnd(&stream);
    return output;
}

} // namespace

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
    auto client = makeTextClient(auth);
    auto resp = client.get(url);
    if (resp.ok()) {
        auto body = std::string(resp.body_string());
        if (hasGzipMagic(body)) {
            return gzipInflate(body);
        }
        return body;
    }

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

std::vector<std::byte> httpGetBytes(const std::string& url, const HttpAuth& auth)
{
    auto client = makeTextClient(auth);
    auto resp = client.get(url);
    if (resp.ok()) {
        return std::move(resp.body);
    }

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
