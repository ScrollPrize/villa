#pragma once

#include "vc/core/util/LoadJson.hpp"

#include <string>

namespace vc3d {

// Catalog surfaces use representation-specific UUIDs internally so that
// different coordinate-space variants cannot collide. Keep that identity for
// application logic, but show the original catalog name in the surface panel.
inline std::string surfacePanelDisplayName(const std::string& internalId,
                                           const utils::Json& metadata)
{
    if (!metadata.is_object()) {
        return internalId;
    }

    const auto catalogSegmentId = vc::json::string_or(
        metadata, "vc_open_data_segment_id", std::string{});
    if (catalogSegmentId.empty()) {
        return internalId;
    }

    const auto originalName = vc::json::string_or(
        metadata, "name", std::string{});
    return originalName.empty() ? catalogSegmentId : originalName;
}

} // namespace vc3d
