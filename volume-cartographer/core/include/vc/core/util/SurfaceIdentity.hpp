#pragma once

#include "utils/Json.hpp"
#include "vc/core/util/Uuid.hpp"

namespace vc::core::util {

// Derived local surfaces retain coordinate provenance, not catalog membership.
// The caller persists the returned UUID rather than deriving identity from a path.
inline std::string newDerivedSurfaceIdentity(utils::Json& metadata)
{
    if (metadata.is_object()) metadata.erase("vc_open_data_representation");
    return vc::util::newUuid();
}

} // namespace vc::core::util
