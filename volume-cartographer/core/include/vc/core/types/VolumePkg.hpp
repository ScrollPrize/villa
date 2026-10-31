#pragma once
#include <vc_reader/catalog.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "utils/Json.hpp"
#include "vc/core/util/Rect3D.hpp"

#include "vc/core/types/ProjectDocument.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>

class Volume;

namespace vc::project {

inline constexpr std::string_view kFiberLasagnaTag = "vc-lasagna-fiber";
inline constexpr std::string_view kAnonymousRemoteAuthTag =
    "vc-remote-auth:anonymous";

struct Entry {
    std::string location;
    std::vector<std::string> tags;
    bool operator==(const Entry&) const = default;
};

[[nodiscard]] bool hasEntryTag(const Entry& entry, std::string_view tag);
[[nodiscard]] bool usesAnonymousRemoteAuth(const Entry& entry);
[[nodiscard]] bool isFiberLasagnaEntry(const Entry& entry);

enum class Category { Volumes, NormalGrids };

struct LoadOptions {
    bool failOnRemoteError = false;
    // Leave entries unresolved until resolveDeferredEntries().
    bool deferResolution = false;
    // The Open Data catalog: remote entries that show a catalog volume take
    // their id, voxel size and name from it (vcr::catalog::OpenDataManifest::volumeAt).
    std::shared_ptr<const vcr::catalog::OpenDataManifest> catalog;
};

std::filesystem::path resolveLocalPath(const std::string& location,
                                       const std::filesystem::path& base = {});

std::string validateLocation(Category category, const std::string& location);
std::string validateSingleVolumeLocation(const std::string& location);
// The catalog dataset a remote location shows (OpenDataManifest::volumeAt at
// its base level); nullopt for other locations.
std::optional<vcr::catalog::OpenDataVolumeView> catalogVolumeView(const vcr::catalog::OpenDataManifest*,
                                                                  const std::string& location);
// The volume metadata (uuid, voxelsize, name) a catalog gives a location: the
// catalog volume's ID at every level and the voxel size of the opened grid.
// Empty for other locations.
utils::Json catalogVolumeMetadata(const vcr::catalog::OpenDataManifest*, const std::string& location);

// The open volume's voxels in the store: those of the volume objects made on it
// are kept in (the catalog volume it shows, so every working level and derived
// dataset of it shares them; else the volume itself) at `scale` = 2^-level of
// its own. What views and queries of the open volume target, and the
// coordinates of objects made in it. `voxelSizeUm` is a voxel of `volume`.
struct AnnotationVolume : vc::objects::VolumeVoxels {
    std::optional<double> voxelSizeUm;
};

}

// Resources opened from a standalone .vcproj workspace document.
class VolumePkg : public std::enable_shared_from_this<VolumePkg>
{
public:
    enum class AttachVolumeResult {
        Attached,
        AlreadyAttached,
        VolumeIdConflict,
    };
    enum class AttachLasagnaResult {
        Attached,
        AlreadyAttached,
        VolumeIdConflict,
    };
    struct PreparedVolumeAttachment {
        std::string location;
        std::vector<std::string> tags;
        std::shared_ptr<Volume> volume;
    };

    static std::shared_ptr<VolumePkg> open(const std::filesystem::path& path,
                                           const vc::project::LoadOptions& opts = {});
    vc::project::ProjectDocument& document() { return document_; }
    const vc::project::ProjectDocument& document() const { return document_; }

    ~VolumePkg();

    [[nodiscard]] std::filesystem::path path() const { return path_; }
    [[nodiscard]] std::filesystem::path directory() const { return path_.parent_path(); }
    [[nodiscard]] bool writable() const;

    // The project document's name.
    [[nodiscard]] std::string name() const;
    void setName(const std::string& v);

    // Reload the project file, resolving added resources and dropping removed ones.
    bool reload();

    [[nodiscard]] const std::vector<vc::project::Entry>& volumeEntries() const;
    [[nodiscard]] const std::vector<vc::project::Entry>& normalGridEntries() const;
    // Role-filtered views over the canonical lasagna_datasets collection.
    [[nodiscard]] std::vector<vc::project::Entry> lasagnaDatasetEntries() const;
    [[nodiscard]] std::vector<vc::project::Entry> fiberInferenceDatasetEntries() const;
    [[nodiscard]] const std::vector<vc::project::Entry>& allLasagnaDatasetEntries() const;
    [[nodiscard]] std::optional<vc::project::Entry>
    matchingVolumeEntry(const std::string& location) const;
    // Resources belong to the project file; returns empty (no per-author ownership).
    [[nodiscard]] std::string entryOwner(const std::string& location) const;

    bool addVolumeEntry(const std::string& location, std::vector<std::string> tags = {});
    // Record an already-loaded volume and its tags as one project change.
    // `volume` must have been loaded from `location`.
    AttachVolumeResult attachPreparedVolume(
        const std::string& location,
        std::vector<std::string> tags,
        const std::shared_ptr<Volume>& volume);
    bool addNormalGridEntry(const std::string& location, std::vector<std::string> tags = {});
    bool addLasagnaDatasetEntry(const std::string& location,
                                std::vector<std::string> tags = {});
    AttachLasagnaResult attachPreparedLasagnaDataset(
        const std::string& manifestLocation,
        std::vector<std::string> manifestTags,
        bool fiberInference,
        const std::vector<PreparedVolumeAttachment>& preparedVolumes,
        bool updateSelection = true);
    bool removeEntry(const std::string& location);
    // Reopen a loaded remote volume at working level `level` (its entry's
    // #vc-base-scale), keeping its ID; false when it is already there.
    bool setVolumeLevel(const std::string& volumeId, int level);

    [[nodiscard]] std::string selectedLasagnaDataset() const;
    void setSelectedLasagnaDataset(std::string location);
    void clearSelectedLasagnaDataset();
    [[nodiscard]] std::string selectedFiberInferenceDataset() const;
    void setSelectedFiberInferenceDataset(std::string location);
    void clearSelectedFiberInferenceDataset();

    // Viewer bbox in volume coordinates; disabling retains the saved limits.
    [[nodiscard]] std::optional<Rect3D> focusBounds() const { return focusBounds_; }
    [[nodiscard]] bool focusBoundsEnabled() const { return focusBoundsEnabled_; }
    // Bounds are source annotation-frame XYZ, never current-volume voxel units.
    void setFocusBounds(std::optional<Rect3D> bounds, bool enabled);
    // The last viewed volume (empty when none).
    [[nodiscard]] std::string currentVolumeId() const { return currentVolumeId_; }
    void setCurrentVolumeId(const std::string& id);

    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    [[nodiscard]] bool entryResolutionDeferred() const noexcept {
        return opts_.deferResolution;
    }
    [[nodiscard]] bool hasLoadedVolumeEntry(const std::string& location) const;
    std::shared_ptr<Volume> volume() const;
    std::shared_ptr<Volume> volume(const std::string& id) const;
    bool addSingleVolume(const std::string& volumeDirName);
    bool removeSingleVolume(const std::string& volumeIdOrDirName);

    [[nodiscard]] std::vector<std::filesystem::path> normalGridPaths() const;

    [[nodiscard]] std::vector<std::string> volumeTags(const std::string& volumeId) const;

    // The catalog the package was loaded with (LoadOptions::catalog).
    [[nodiscard]] const std::shared_ptr<const vcr::catalog::OpenDataManifest>& catalog() const { return opts_.catalog; }
    // What a loaded remote volume shows in that catalog; nullopt for other volumes.
    [[nodiscard]] std::optional<vcr::catalog::OpenDataVolumeView> catalogView(const std::string& volumeId) const;
    // An overlay samples in the base volume's voxels; refused only when the
    // catalog places the two on different grids (catalog volume and level).
    [[nodiscard]] bool overlayCompatible(const std::string& baseId, const std::string& overlayId) const;

    [[nodiscard]] bool isRemote() const;

    // Completes a deferred load. Ordinary open() callers remain eager.
    void resolveDeferredEntries();

private:
    VolumePkg();

    vc::project::ProjectDocument document_;
    std::filesystem::path path_;
    std::string name_;
    vc::project::LoadOptions opts_;

    std::vector<vc::project::Entry> volumes_;
    std::vector<vc::project::Entry> normalGrids_;
    std::vector<vc::project::Entry> lasagnaDatasets_;
    std::optional<std::string> selectedLasagnaDataset_;
    std::optional<std::string> selectedFiberInferenceDataset_;
    std::optional<Rect3D> focusBounds_;
    bool focusBoundsEnabled_ = false;
    std::string currentVolumeId_;

    std::map<std::string, std::shared_ptr<Volume>> loadedVolumes_;
    std::map<std::string, std::vector<std::string>> volumeTagsByID_;
    std::vector<std::filesystem::path> resolvedNormalGridPaths_;

    void resolveAll();
    void resolveVolumeEntry(const vc::project::Entry& e);
    void resolveNormalGridEntry(const vc::project::Entry& e);

    // Read resources and view state from the document.
    void readDocument();
    // Write one resource; false when it is already there, unchanged.
    bool writeEntry(const char* kind, const vc::project::Entry& entry);
    void writeView();
    nlohmann::json viewJson() const;
    // Drop loaded volumes no entry backs any more and resolve the entries
    // that are not in `previous`.
    void followEntries(const std::vector<vc::project::Entry>& previous);
};

namespace vc::project {
// Where `volume`'s annotations are kept (AnnotationVolume), from `pkg`'s
// catalog when it has one.
AnnotationVolume annotationVolume(const VolumePkg* pkg, const Volume& volume);
}
