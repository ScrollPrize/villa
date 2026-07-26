#include "vc/core/util/RemoteUrl.hpp"

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

ResolvedUrl resolveRemoteUrl(const std::string& input)
{
    // Check for s3:// or s3+REGION:// prefix
    if (input.rfind("s3://", 0) == 0) {
        // s3://bucket/key — default region us-east-1
        std::string rest = input.substr(5);  // after "s3://"
        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string region = "us-east-1";
        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        return ResolvedUrl{httpsUrl, region, true};
    }

    if (input.rfind("s3+", 0) == 0) {
        // s3+REGION://bucket/key
        auto schemeEnd = input.find("://");
        if (schemeEnd == std::string::npos) {
            // Malformed — treat as plain URL
            return ResolvedUrl{input, {}, false};
        }

        std::string region = input.substr(3, schemeEnd - 3);  // between "s3+" and "://"
        std::string rest = input.substr(schemeEnd + 3);

        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        return ResolvedUrl{httpsUrl, region, true};
    }

    // Check if this is an S3 HTTPS URL by hostname pattern
    // e.g. https://bucket.s3.region.amazonaws.com/key
    //   or https://bucket.s3.amazonaws.com/key (us-east-1 default)
    if (input.rfind("https://", 0) == 0 || input.rfind("http://", 0) == 0) {
        auto schemeEnd = input.find("://");
        std::string rest = input.substr(schemeEnd + 3);
        auto slash = rest.find('/');
        std::string host = (slash != std::string::npos) ? rest.substr(0, slash) : rest;

        // Match *.s3.REGION.amazonaws.com or *.s3.amazonaws.com
        auto s3Pos = host.find(".s3.");
        if (s3Pos != std::string::npos && host.find(".amazonaws.com") != std::string::npos) {
            // Extract region from between .s3. and .amazonaws.com
            std::string afterS3 = host.substr(s3Pos + 4);  // after ".s3."
            auto amzPos = afterS3.find(".amazonaws.com");
            std::string region;
            if (amzPos != std::string::npos && amzPos > 0) {
                region = afterS3.substr(0, amzPos);
            } else {
                region = "us-east-1";
            }
            return ResolvedUrl{input, region, true};
        }
    }

    // HTTP/HTTPS — pass through
    return ResolvedUrl{input, {}, false};
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

    auto resolved = resolveRemoteUrl(sourceInput);
    // Preserve the legacy selector-free normalization exactly. Selector-aware
    // URLs additionally trim a path slash which occurs before a query.
    std::string sourceUrl = fragmentPos == std::string::npos
        ? resolved.httpsUrl
        : trimTrailingPathSlashes(std::move(resolved.httpsUrl));
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
    spec.awsRegion = std::move(resolved.awsRegion);
    spec.useAwsSigv4 = resolved.useAwsSigv4;
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
