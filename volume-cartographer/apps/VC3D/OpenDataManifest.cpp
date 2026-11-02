#include "OpenDataManifest.hpp"

namespace vc3d::opendata {
bool OpenDataResourceSelection::allowsVolume(const std::string& volumeId) const
{
    if (!volumeIds)
        return true;
    return std::find(volumeIds->begin(), volumeIds->end(), volumeId) !=
           volumeIds->end();
}

bool OpenDataResourceSelection::allowsRepresentation(
    std::size_t volumeIndex, std::size_t artifactIndex,
    vcr::catalog::OpenDataRepresentationKind kind, const std::string& volumeId) const
{
    if (!allowsVolume(volumeId))
        return false;
    if (representations) {
        const bool listed = std::any_of(
            representations->begin(), representations->end(),
            [&](const vcr::catalog::OpenDataRepresentationRef& ref) {
                return ref.volumeIndex == volumeIndex &&
                       ref.artifactIndex == artifactIndex;
            });
        if (!listed)
            return false;
    }
    if (kinds) {
        if (std::find(kinds->begin(), kinds->end(), kind) == kinds->end())
            return false;
    }
    return true;
}


}
