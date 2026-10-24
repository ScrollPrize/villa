#pragma once

#include <string>

#include "vc/core/util/Uuid.hpp"

namespace vc::util {

/** Resolve the object UUID written to a new TIFXYZ meta.json.
 *
 * An explicit value (for example, a catalog identity supplied by a production
 * pipeline) is kept as given. Otherwise the new surface gets a fresh random
 * UUID: the identity is never derived from the output directory or input
 * file name, so artifact directories can be named after a processing role
 * (for example, "output_tifxyz") without sharing an identity.
 */
inline std::string resolveTifxyzUuid(const std::string& explicitUuid = {})
{
    return explicitUuid.empty() ? newUuid() : explicitUuid;
}

}  // namespace vc::util
