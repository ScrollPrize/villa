#include "vc/core/util/RemoteVolumeMetadata.hpp"

#include <cmath>
#include <cstddef>
#include <optional>

namespace vc
{

namespace
{

constexpr std::size_t kMaxMetadataDepth = 64;

bool collectSamplePixelSize(const utils::Json& node, std::size_t depth, std::optional<double>& result)
{
    if (depth > kMaxMetadataDepth) {
        return false;
    }

    if (node.is_object() && node.contains("samplePixelSize")) {
        const auto& value = node["samplePixelSize"];
        if (value.is_number()) {
            const double candidate = value.get_double();
            if (std::isfinite(candidate) && candidate > 0.0) {
                if (result && *result != candidate) {
                    return false;
                }
                result = candidate;
            }
        }
    }

    if (!node.is_object() && !node.is_array()) {
        return true;
    }
    for (const auto& child : node) {
        if (!collectSamplePixelSize(child, depth + 1, result)) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<double> findUnambiguousSamplePixelSizeMillimeters(const utils::Json& metadata)
{
    std::optional<double> result;
    if (!collectSamplePixelSize(metadata, 0, result)) {
        return std::nullopt;
    }
    return result;
}

}  // namespace vc
