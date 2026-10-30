#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/RemoteAuth.hpp"
#include "vc_reader/s3_listing.hpp"
#include "vc_reader/sftp.hpp"
#include "vc_reader/zarr.hpp"

#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace vc {

namespace {

std::string trimTrailingPathSlashes(std::string url)
{
    const auto query = url.find('?');
    const auto pathEnd = query == std::string::npos ? url.size() : query;
    auto trimmedEnd = pathEnd;
    while (trimmedEnd > 0 && url[trimmedEnd - 1] == '/') {
        --trimmedEnd;
    }
    if (trimmedEnd != pathEnd) {
        url.erase(trimmedEnd, pathEnd - trimmedEnd);
    }
    return url;
}

int parseBaseScaleSelector(std::string_view fragment)
{
    constexpr std::string_view key = "vc-base-scale";
    bool found = false;
    int level = 0;

    while (!fragment.empty()) {
        const auto separator = fragment.find('&');
        const auto item = fragment.substr(0, separator);
        fragment = separator == std::string_view::npos
            ? std::string_view{}
            : fragment.substr(separator + 1);

        const auto equals = item.find('=');
        if (equals == std::string_view::npos || item.substr(0, equals) != key) {
            throw std::invalid_argument(
                "unsupported remote volume selector '" + std::string(item) + "'");
        }
        if (found) {
            throw std::invalid_argument("duplicate remote volume selector 'vc-base-scale'");
        }
        found = true;

        const auto value = item.substr(equals + 1);
        if (value.empty()) {
            throw std::invalid_argument("remote volume selector vc-base-scale requires an integer");
        }
        int parsed = 0;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
            throw std::invalid_argument(
                "remote volume selector vc-base-scale must be an integer from 0 through " +
                std::to_string(kMaxRemoteVolumeBaseScale));
        }
        if (parsed < 0 || parsed > kMaxRemoteVolumeBaseScale) {
            throw std::invalid_argument(
                "remote volume selector vc-base-scale must be from 0 through " +
                std::to_string(kMaxRemoteVolumeBaseScale));
        }
        level = parsed;
    }

    if (!found) {
        throw std::invalid_argument("remote volume selector fragment is empty");
    }
    return level;
}

} // namespace

std::string normalizedRemoteUrl(const std::string& location)
{
    return trimTrailingPathSlashes(vcr::sftp::isSftpUrl(location) ? vcr::sftp::parseUrl(location).str()
                                                                  : vcr::zarr::httpUrl(location));
}

std::optional<std::string> s3Location(const std::string& url)
{
    if (url.starts_with("s3://"))
        return url;
    const auto https = vcr::zarr::httpUrl(url);
    const auto prefix = vcr::s3::splitPrefixUrl(https);
    if (!prefix || !https.starts_with("http"))
        return std::nullopt;
    const auto host = prefix->origin.substr(prefix->origin.find("://") + 3);
    const auto s3 = host.find(".s3.");
    if (s3 == std::string::npos || s3 == 0 || !host.ends_with(".amazonaws.com"))
        return std::nullopt;
    auto key = prefix->key;
    while (!key.empty() && key.back() == '/')
        key.pop_back();
    return "s3://" + host.substr(0, s3) + (key.empty() ? std::string() : "/" + key);
}

HttpAuth loadAwsCredentials()
{
    const auto keys = vcr::zarr::awsCredentials();
    return keys ? HttpAuth{keys->user, keys->secret, keys->token, keys->region} : HttpAuth{};
}

RemoteVolumeSpec parseRemoteVolumeSpec(const std::string& input)
{
    const auto fragmentPos = input.find('#');
    const std::string sourceInput = fragmentPos == std::string::npos
        ? input
        : input.substr(0, fragmentPos);
    const int baseScaleLevel = fragmentPos == std::string::npos
        ? 0
        : parseBaseScaleSelector(std::string_view(input).substr(fragmentPos + 1));

    std::string sourceUrl = normalizedRemoteUrl(sourceInput);
    while (!sourceUrl.empty() && sourceUrl.back() == '/') {
        sourceUrl.pop_back();
    }

    RemoteVolumeSpec spec;
    spec.sourceUrl = std::move(sourceUrl);
    spec.baseScaleLevel = baseScaleLevel;
    spec.hasBaseScaleSelector = fragmentPos != std::string::npos;
    spec.portableLocator = spec.sourceUrl;
    if (baseScaleLevel > 0) {
        spec.portableLocator += "#vc-base-scale=" + std::to_string(baseScaleLevel);
    }
    spec.useAwsSigv4 = s3Location(spec.sourceUrl).has_value();
    return spec;
}

std::string joinRemoteUrlPath(const std::string& baseUrl, const std::string& childPath)
{
    const auto queryPos = baseUrl.find('?');
    std::string path = queryPos == std::string::npos
        ? baseUrl
        : baseUrl.substr(0, queryPos);
    const std::string query = queryPos == std::string::npos
        ? std::string{}
        : baseUrl.substr(queryPos);

    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    std::size_t childStart = 0;
    while (childStart < childPath.size() && childPath[childStart] == '/') {
        ++childStart;
    }
    if (childStart < childPath.size()) {
        path.push_back('/');
        path.append(childPath, childStart, std::string::npos);
    }
    return path + query;
}

}  // namespace vc
