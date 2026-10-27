#pragma once
#include <vc_reader/catalog.hpp>

#include <optional>
#include <string>

class VolumePkg;

namespace vc3d::opendata {

// The two catalog volume properties that between them fix which way the
// scroll spirals in the volume's own axes. The catalog (vesuvius-atlas)
// records no spiral direction of its own; its convention is that every
// scroll shows the same spiral seen from its top, so the direction in a
// volume's coordinates follows from which end of z is the top and whether
// the axes are mirrored. Either property unset means the catalog does not
// know for this volume.
struct VolumeOrientation {
    // properties.z_direction_is_top_to_bottom: slice z = 0 is the top.
    std::optional<bool> zTopToBottom;
    // properties.left_handed_coordinates: the xy plane is mirrored.
    std::optional<bool> leftHandedCoordinates;
};

// The two properties as the volume's catalog entry states them; a property
// that is absent or not a boolean (the strings "true" / "false" are read as
// booleans) is unset.
[[nodiscard]] VolumeOrientation volumeOrientationOf(const vcr::catalog::OpenDataVolume& volume);

// The fiber map's winding sense for a volume the catalog orients in full:
// +1 when the winding grows with theta = atan2(dy, dx) about the umbilicus,
// -1 against it (GlobalResult::chirality); nullopt while either property
// is unset.
//
// Derivation: the orient-segment tool (ScrollPrize/infra, jrudolph) applies
// the catalog's convention as "U, which runs outside-to-inside, grows with
// atan2(dy, dx) when z is top-to-bottom, falls when it is not, and either
// reading flips under left-handed axes". Inward with theta is outward
// against it, and the fiber map's winding grows outward, so the sense is
// -1 exactly when one of the two properties holds. Checked against
// PHerc0139 volume 20260102150214 (top-to-bottom, right-handed): -1 is the
// sense whose map is consistent, +1 mirrors it into hundreds of errors.
[[nodiscard]] std::optional<int> windingChiralityOf(const VolumeOrientation& orientation);

// The catalog volume a loaded volume shows ("<sample>/<volume>": one entry and
// one winding sense for all its pyramid levels) and its orientation, from the
// catalog the package was loaded with; nullopt for other volumes.
struct CatalogVolumeSense {
    std::string volume;
    VolumeOrientation orientation;
};
[[nodiscard]] std::optional<CatalogVolumeSense> catalogVolumeSense(const VolumePkg& pkg, const std::string& volumeId);

// The orientation as comparable text, for dependency checks.
[[nodiscard]] std::string orientationText(const VolumeOrientation& orientation);

} // namespace vc3d::opendata
