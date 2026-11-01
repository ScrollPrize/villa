#include "vc/core/util/RemoteUrl.hpp"
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

RemoteVolumeSpec parseRemoteVolumeSpec(const std::string& input)
{
    const auto fragmentPos = input.find('#');
    const std::string sourceInput = fragmentPos == std::string::npos
        ? input
        : input.substr(0, fragmentPos);
    const int baseScaleLevel = fragmentPos == std::string::npos
        ? 0
        : parseBaseScaleSelector(std::string_view(input).substr(fragmentPos + 1));

    // The URL requests go to: ssh:// as sftp://, public s3:// as https://, no slash ending the path.
    RemoteVolumeSpec spec;
    spec.sourceUrl = trimTrailingPathSlashes(vcr::sftp::isSftpUrl(sourceInput) ? vcr::sftp::parseUrl(sourceInput).str()
                                                                               : vcr::zarr::httpUrl(sourceInput));
    spec.baseScaleLevel = baseScaleLevel;
    spec.hasBaseScaleSelector = fragmentPos != std::string::npos;
    spec.portableLocator = spec.sourceUrl;
    if (baseScaleLevel > 0) {
        spec.portableLocator += "#vc-base-scale=" + std::to_string(baseScaleLevel);
    }
    spec.useAwsSigv4 = vcr::zarr::s3Location(spec.sourceUrl).has_value();
    return spec;
}

}  // namespace vc
