#pragma once

#include "OpenDataManifest.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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

// "PHerc0139/20260102150214@L0" (the vc-open-data-coordinate-space tag) ->
// {"PHerc0139", "20260102150214"}; nullopt when the text is not of that
// shape.
[[nodiscard]] std::optional<std::pair<std::string, std::string>>
sampleAndVolumeOfCoordinateSpace(std::string_view coordinateSpace);

// The catalog volume a coordinate space names, as one comparable string
// without the pyramid level ("PHerc0139/20260102150214" for either "@L0"
// or "@L2" - two levels of one volume are one catalog entry and one
// winding sense); empty when the text names none.
[[nodiscard]] std::string catalogVolumeOfCoordinateSpace(std::string_view coordinateSpace);

// The two properties as the volume's catalog entry states them; a property
// that is absent or not a boolean (the strings "true" / "false" are read as
// booleans) is unset.
[[nodiscard]] VolumeOrientation volumeOrientationOf(const OpenDataVolume& volume);

// nullopt when the manifest has no such sample or volume.
[[nodiscard]] std::optional<VolumeOrientation> findVolumeOrientation(
    const OpenDataManifest& manifest,
    std::string_view sampleId,
    std::string_view volumeId);

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

// Orientation lookups against the catalog manifest VC3D caches on disk
// (the copy the catalog window last fetched). One parse per manifest
// file version and coordinate space: the file is large, so the answer is
// memoized on the file's size and mtime and only re-read when either
// moves. Safe to call from any thread.
class CatalogVolumeOrientationLookup {
public:
    explicit CatalogVolumeOrientationLookup(
        std::filesystem::path manifestPath = cachedOpenDataManifestPath());

    // One observation of the catalog for a coordinate space: the answer and
    // the version of the manifest file it was read from, taken together so
    // a caller cannot pair an answer with another version's token.
    struct CatalogSense {
        // nullopt when there is no cached manifest, it does not parse, the
        // coordinate space does not name a sample and volume, or the
        // manifest has no such volume. A manifest that has the volume but
        // leaves the properties unset answers with both fields unset.
        std::optional<VolumeOrientation> orientation;
        // The manifest version the answer is bound to (manifestToken()):
        // empty when the space is empty, "absent" when there is no file,
        // else its size and mtime - or "unstable" when no version could
        // be read whole, which equals no version a stat reports, so a
        // build on that answer never passes for one on any version.
        std::string manifestToken;
    };
    // The answer is bound to one version of the file: the file is measured
    // before and after the parse and re-read while the two disagree, and a
    // file that keeps moving answers nullopt, unmemoized, as "unstable".
    [[nodiscard]] CatalogSense resolve(std::string_view coordinateSpace);

    // Test seam: run between each parse and the measurement after it, to
    // stand in for a catalog refresh landing mid-read.
    void setAfterParseHookForTesting(std::function<void()> hook);
    [[nodiscard]] std::optional<VolumeOrientation> lookup(std::string_view coordinateSpace)
    {
        return resolve(coordinateSpace).orientation;
    }

    // The manifest version a rebuild for this coordinate space would read
    // (CatalogSense::manifestToken), from a stat and no parse: empty for an
    // empty space, since the catalog is not consulted for one. Comparable
    // between builds: equal tokens for equal spaces mean the same answer.
    [[nodiscard]] std::string manifestToken(std::string_view coordinateSpace) const;

    [[nodiscard]] const std::filesystem::path& manifestPath() const noexcept
    {
        return _manifestPath;
    }

private:
    struct FileToken {
        std::uintmax_t size = 0;
        std::filesystem::file_time_type mtime{};
        bool operator==(const FileToken& other) const noexcept
        {
            return size == other.size && mtime == other.mtime;
        }
    };
    struct Memo {
        std::string coordinateSpace;
        // Unset when the file was absent at the time.
        std::optional<FileToken> token;
        std::optional<VolumeOrientation> value;
    };

    [[nodiscard]] std::optional<FileToken> fileToken() const;
    [[nodiscard]] static std::string tokenText(const std::optional<FileToken>& token);

    std::filesystem::path _manifestPath;
    std::mutex _mutex;
    std::optional<Memo> _memo;
    std::function<void()> _afterParseHook;
};

} // namespace vc3d::opendata
