#pragma once

#include <QString>

#include <filesystem>
#include <vector>

#include "vc/core/types/VolumePkg.hpp"

// How project attachments spell and match locations, shared by
// ProjectAttachmentController and its callers.
namespace vc3d::project_locations {

// True for any "<scheme>://" locator other than file://, whatever the
// scheme's case; such a location is never a local file.
bool looksRemote(const QString& location);

// The spelling stored for a local location: absolute (a relative one
// resolves against `projectDir`, as VolumePkg resolves stored entries),
// lexically normal, and without a trailing separator.
QString storedLocalLocation(const QString& location, const std::filesystem::path& projectDir);

// The stored spelling of the entry in `entries` that names `location` under
// vc::project::sameLocation, or an empty string.
QString findEntry(const std::vector<vc::project::Entry>& entries,
                  const QString& location,
                  const std::filesystem::path& projectDir);

}  // namespace vc3d::project_locations
