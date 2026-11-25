#pragma once

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

// Parse #vc-base-scale=N before URL/S3 resolution. Unknown, duplicate, or
// malformed selectors throw std::invalid_argument. N is restricted to 0..5;
// zero canonicalizes to a selector-free portable locator.
RemoteVolumeSpec parseRemoteVolumeSpec(const std::string& input);

}  // namespace vc
