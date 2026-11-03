#include "OpenDataVolumeOrientation.hpp"

#include "vc/core/types/VolumePkg.hpp"

namespace vc3d::opendata {

std::optional<int> windingChiralityOf(const vcr::catalog::OpenDataOrientation& orientation)
{
    if (!orientation.zTopToBottom || !orientation.leftHandedCoordinates) {
        return std::nullopt;
    }
    return *orientation.zTopToBottom != *orientation.leftHandedCoordinates ? -1 : 1;
}

std::optional<CatalogVolumeSense> catalogVolumeSense(const VolumePkg& pkg, const std::string& volumeId)
{
    const auto view = pkg.catalogView(volumeId);
    if (!view) {
        return std::nullopt;
    }
    const auto* volume = pkg.catalog()->findSample(view->sampleId)->findVolume(view->volumeId);
    return CatalogVolumeSense{view->sampleId + '/' + view->volumeId, volume->orientation};
}

std::string orientationText(const vcr::catalog::OpenDataOrientation& orientation)
{
    const auto text = [](const std::optional<bool>& value) {
        return value ? (*value ? "1" : "0") : "?";
    };
    return std::string("z-top-to-bottom ") + text(orientation.zTopToBottom) + ", left-handed " +
           text(orientation.leftHandedCoordinates);
}

} // namespace vc3d::opendata
