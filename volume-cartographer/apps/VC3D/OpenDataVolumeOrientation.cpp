#include "OpenDataVolumeOrientation.hpp"

#include "vc/core/types/VolumePkg.hpp"

namespace vc3d::opendata {

VolumeOrientation volumeOrientationOf(const vcr::catalog::OpenDataVolume& volume)
{
    VolumeOrientation orientation;
    orientation.zTopToBottom = volume.zDirectionIsTopToBottom;
    orientation.leftHandedCoordinates = volume.leftHandedCoordinates;
    return orientation;
}

std::optional<int> windingChiralityOf(const VolumeOrientation& orientation)
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
    return CatalogVolumeSense{view->sampleId + '/' + view->volumeId, volumeOrientationOf(*volume)};
}

std::string orientationText(const VolumeOrientation& orientation)
{
    const auto text = [](const std::optional<bool>& value) {
        return value ? (*value ? "1" : "0") : "?";
    };
    return std::string("z-top-to-bottom ") + text(orientation.zTopToBottom) + ", left-handed " +
           text(orientation.leftHandedCoordinates);
}

} // namespace vc3d::opendata
