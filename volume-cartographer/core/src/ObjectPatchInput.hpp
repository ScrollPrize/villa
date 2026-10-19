#pragma once
#include "vc/core/util/Tiff.hpp"
#include <stdexcept>
#include <string_view>

namespace vc::objects::detail {
// QuadSurface's display loader tolerates damaged tiles and can recover saves.
// Authoritative ingestion/detection must not interpret substituted pixels as
// evidence of new geometry or missing relationships.
inline void validatePatchFiles(const std::filesystem::path& source)
{
    if (std::filesystem::exists(source / ".vc-save-rollback"))
        throw std::runtime_error("patch has an incomplete save; recover it before ingestion or detection");
    for (const char* name : {"x.tif", "y.tif", "z.tif", "mask.tif"}) {
        const auto file = source / name;
        if (std::string_view(name) == "mask.tif" && !std::filesystem::exists(file)) continue;
        if (!isReadableTiff(file)) throw std::runtime_error("incomplete or unreadable TIFF: " + file.string());
    }
}
} // namespace vc::objects::detail
