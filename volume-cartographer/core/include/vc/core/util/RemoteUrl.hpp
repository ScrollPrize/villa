#pragma once

#include <optional>
#include <string>

namespace vc {

inline constexpr int kMaxRemoteVolumeBaseScale = 5;

// A remote volume has two deliberately separate identities:
//
// * sourceUrl is the fragment-free URL used for network requests.
// * portableLocator includes the canonical client-side selector, when any,
//   and is used when persisting or handing the view to another VC process.
//
struct RemoteVolumeSpec {
    std::string sourceUrl;
    std::string portableLocator;
    int baseScaleLevel = 0;
    bool hasBaseScaleSelector = false;
    bool useAwsSigv4 = false;
};

// The URL requests for a location go to, with no slash ending its path; an existing
// query is kept. ssh:// becomes sftp://.
std::string normalizedRemoteUrl(const std::string& location);

// The s3:// location (what vcr::zarr::s3Access signs) of an S3 URL; nullopt for any other.
std::optional<std::string> s3Location(const std::string& url);

// Parse #vc-base-scale=N before URL/S3 resolution. Unknown, duplicate, or
// malformed selectors throw std::invalid_argument. N is restricted to 0..5;
// zero canonicalizes to a selector-free portable locator.
RemoteVolumeSpec parseRemoteVolumeSpec(const std::string& input);

// Append a child object path before an existing query string. The query bytes
// are preserved exactly (no decoding, reordering, or re-encoding).
std::string joinRemoteUrlPath(const std::string& baseUrl, const std::string& childPath);

}  // namespace vc
