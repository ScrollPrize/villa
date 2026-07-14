#pragma once

#include <optional>
#include <string>

namespace vc {

struct ResolvedUrl {
    std::string httpsUrl;      // resolved HTTPS URL
    std::string awsRegion;     // empty if not S3
    bool useAwsSigv4 = false;  // true if s3:// and credentials detected
};

// Resolve s3://bucket/key -> https://... with optional SigV4 flag.
// Supports s3://bucket/key (defaults to us-east-1) and
// s3+REGION://bucket/key (explicit region).
// Passes through http:// and https:// URLs unchanged.
ResolvedUrl resolveRemoteUrl(const std::string& input);

inline constexpr int kMaxRemoteVolumeBaseScale = 5;

// A remote volume has two deliberately separate identities:
//
// * sourceUrl is the fragment-free URL used for network requests.
// * portableLocator includes the canonical client-side selector, when any,
//   and is used when persisting or handing the view to another VC process.
//
// Selector-free inputs retain the pre-existing resolve-then-trim behavior.
struct RemoteVolumeSpec {
    std::string sourceUrl;
    std::string portableLocator;
    int baseScaleLevel = 0;
    bool hasBaseScaleSelector = false;
    std::string awsRegion;
    bool useAwsSigv4 = false;
};

// Parse #vc-base-scale=N before URL/S3 resolution. Unknown, duplicate, or
// malformed selectors throw std::invalid_argument. N is restricted to 0..5;
// zero canonicalizes to a selector-free portable locator.
RemoteVolumeSpec parseRemoteVolumeSpec(const std::string& input);

// Append a child object path before an existing query string. The query bytes
// are preserved exactly (no decoding, reordering, or re-encoding).
std::string joinRemoteUrlPath(const std::string& baseUrl, const std::string& childPath);

}  // namespace vc
