#include "OpenDataVolumeOrientation.hpp"

#include <stdexcept>

#include "vc/core/types/VolumePkg.hpp"

namespace vc3d::opendata {

std::optional<int> windingChiralityOf(const vcr::catalog::OpenDataOrientation& orientation)
{
    if (!orientation.zTopToBottom || !orientation.leftHandedCoordinates) {
        return std::nullopt;
    }
    return *orientation.zTopToBottom != *orientation.leftHandedCoordinates ? -1 : 1;
}

WindingSense statedWindingSense(std::optional<int> umbilicusHandedness,
                                const std::optional<CatalogVolumeSense>& catalog)
{
    if (umbilicusHandedness) {
        if (*umbilicusHandedness != 1 && *umbilicusHandedness != -1)
            throw std::runtime_error("the umbilicus' winding_handedness must be +1 or -1");
        return {*umbilicusHandedness, false};
    }
    if (catalog)
        if (const auto sense = windingChiralityOf(catalog->orientation)) return {*sense, true};
    throw std::runtime_error(
        "the winding sense is unknown: the umbilicus states no winding_handedness and the catalog "
        "does not orient this volume. Set winding_handedness (+1 or -1) on the umbilicus.");
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
