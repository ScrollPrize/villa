#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "utils/Json.hpp"

class Volume;
class Segmentation;
class QuadSurface;

namespace vc::project {

struct Entry {
    std::string location;
    std::vector<std::string> tags;
};

enum class Category { Volumes, Segments, NormalGrids };

struct LoadOptions {
    std::filesystem::path remoteCacheRoot;
    bool failOnRemoteError = false;
    bool deferResolution = false;
};

bool isLocationRemote(const std::string& location);
std::filesystem::path resolveLocalPath(const std::string& location,
                                       const std::filesystem::path& base = {});

std::string validateLocation(Category category, const std::string& location);
std::string validateSingleVolumeLocation(const std::string& location);

}

class VolumePkg : public std::enable_shared_from_this<VolumePkg>
{
public:
    static std::shared_ptr<VolumePkg> newEmpty();
    static std::shared_ptr<VolumePkg> newEmpty(
        const vc::project::LoadOptions& opts);
    // Build a package without updating the session autosave or implicitly
    // persisting later mutations. Call save() when the package is complete.
    static std::shared_ptr<VolumePkg> newDetached(
        const vc::project::LoadOptions& opts = {});
    static std::shared_ptr<VolumePkg> load(const std::filesystem::path& jsonFile,
                                           const vc::project::LoadOptions& opts = {});
    static std::shared_ptr<VolumePkg> loadAutosave(const vc::project::LoadOptions& opts = {});

    static std::shared_ptr<VolumePkg> New(const std::filesystem::path& jsonFile);

    static void setLoadFirstSegmentationDirectory(const std::string& dirName);
    static void setAutosaveRoot(const std::filesystem::path& dir);
    static std::filesystem::path autosaveRoot();
    static std::filesystem::path autosaveFile();

    ~VolumePkg();

    void save(const std::filesystem::path& target);
    void saveAutosave();
    [[nodiscard]] std::filesystem::path path() const;

    [[nodiscard]] std::string name() const;
    void setName(const std::string& v);
    [[nodiscard]] int version() const;

    [[nodiscard]] const std::vector<vc::project::Entry>& volumeEntries() const;
    [[nodiscard]] const std::vector<vc::project::Entry>& segmentEntries() const;
    [[nodiscard]] const std::vector<vc::project::Entry>& normalGridEntries() const;
    [[nodiscard]] const std::vector<vc::project::Entry>& lasagnaDatasetEntries() const;

    bool addVolumeEntry(const std::string& location, std::vector<std::string> tags = {});
    bool mergeVolumeEntryTags(const std::string& location, const std::vector<std::string>& tags);
    // Replace singleton keyed tags and merge ordinary tags in one operation,
    // refreshing a loaded remote volume at most once.
    bool reconcileVolumeEntryTags(
        const std::string& location,
        const std::vector<std::string>& tags,
        const std::vector<std::string>& singletonPrefixes);
    bool reconcileSegmentsEntryTags(
        const std::string& location,
        const std::vector<std::string>& tags,
        const std::vector<std::string>& singletonPrefixes);
    bool reconcileNormalGridEntryTags(
        const std::string& location,
        const std::vector<std::string>& tags,
        const std::vector<std::string>& singletonPrefixes);
    bool relocateSegmentsEntry(const std::string& oldLocation,
                               const std::string& newLocation);
    bool relocateNormalGridEntry(const std::string& oldLocation,
                                 const std::string& newLocation);
    bool addSegmentsEntry(const std::string& location, std::vector<std::string> tags = {});
    bool addNormalGridEntry(const std::string& location, std::vector<std::string> tags = {});
    bool addLasagnaDatasetEntry(const std::string& location,
                                std::vector<std::string> tags = {});
    bool reconcileLasagnaDatasetEntryTags(
        const std::string& location,
        const std::vector<std::string>& tags,
        const std::vector<std::string>& singletonPrefixes);
    bool removeEntry(const std::string& location);

    void setOutputSegments(const std::string& location);
    void clearOutputSegments();
    [[nodiscard]] bool hasOutputSegments() const;
    [[nodiscard]] std::filesystem::path outputSegmentsPath() const;

    [[nodiscard]] std::string selectedLasagnaDataset() const;
    void setSelectedLasagnaDataset(std::string location);
    void clearSelectedLasagnaDataset();
    [[nodiscard]] std::filesystem::path selectedLasagnaDatasetPath() const;

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
    bool addVolume(const std::shared_ptr<Volume>& volume);
    bool addSingleVolume(const std::string& volumeDirName);
    bool removeSingleVolume(const std::string& volumeIdOrDirName);
    bool reloadSingleVolume(const std::string& volumeId);

    [[nodiscard]] bool hasSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;
    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);

    [[nodiscard]] std::vector<std::filesystem::path> normalGridPaths() const;
    [[nodiscard]] std::vector<std::filesystem::path> normal3dZarrPaths() const;

    [[nodiscard]] std::vector<std::string> volumeTags(const std::string& volumeId) const;
    [[nodiscard]] std::vector<std::string> segmentationTags(const std::string& segmentId) const;

    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<QuadSurface> loadSurface(const std::string& id);
    std::shared_ptr<QuadSurface> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);

    [[nodiscard]] bool isRemote() const;

    void setSegmentsChangedCallback(std::function<void()> cb);

    [[nodiscard]] bool hasRemoteCacheRoot() const;
    [[nodiscard]] std::string remoteCacheRootOrEmpty() const;
    void setRemoteCacheRoot(const std::filesystem::path& dir);
    // Completes a deferred load. Ordinary load() callers remain eager.
    void resolveDeferredEntries();

    [[nodiscard]] std::string getVolpkgDirectory() const;
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;
    [[nodiscard]] std::vector<std::filesystem::path> availableSegmentPaths() const;
    [[nodiscard]] std::filesystem::path findSegmentPathByName(const std::string& dirName) const;
    void setSegmentationDirectory(const std::string& dirName);
    void refreshSegmentations();
    bool addSingleSegmentation(const std::string& id);
    bool removeSingleSegmentation(const std::string& id);
    bool reloadSingleSegmentation(const std::string& id);

private:
    VolumePkg();

    std::filesystem::path path_;
    std::string name_ = "Untitled";
    int version_ = 1;
    vc::project::LoadOptions opts_;
    std::filesystem::path remoteCacheRoot_;
    bool automaticPersistence_ = true;

    std::vector<vc::project::Entry> volumes_;
    std::vector<vc::project::Entry> segments_;
    std::vector<vc::project::Entry> normalGrids_;
    std::vector<vc::project::Entry> lasagnaDatasets_;
    std::optional<std::string> outputSegments_;
    std::optional<std::string> selectedLasagnaDataset_;

    std::map<std::string, std::shared_ptr<Volume>> loadedVolumes_;
    std::map<std::string, std::vector<std::string>> volumeTagsByID_;
    std::map<std::string, std::shared_ptr<Segmentation>> loadedSegmentations_;
    // Segmentations previously resolved for a segments entry, keyed by entry
    // location. Retained across setSegmentationDirectory() switches so
    // returning to a directory reuses the same Segmentation objects (and any
    // surfaces they have loaded) instead of redoing that work from disk.
    std::map<std::string, std::map<std::string, std::shared_ptr<Segmentation>>>
        segmentationsByLocation_;
    std::string activeSegmentsLocation_;
    std::map<std::string, std::vector<std::string>> segmentationTagsByID_;
    std::vector<std::filesystem::path> resolvedNormalGridPaths_;

    void resolveAll();
    void resolveVolumeEntry(const vc::project::Entry& e);
    void resolveSegmentsEntry(const vc::project::Entry& e);
    void resolveNormalGridEntry(const vc::project::Entry& e);
    void notifySegmentsChanged();

    void persistProjectState();
    void writeJsonTo(const std::filesystem::path& target) const;
    void readJsonFrom(const std::filesystem::path& source);
    [[nodiscard]] utils::Json toJson() const;
    void fromJson(const utils::Json& j);

    static std::filesystem::path autosaveRoot_;
    static std::optional<std::string> loadFirstSegmentationDir_;

    mutable std::mutex segmentsMutex_;
    std::function<void()> segmentsChangedCb_;
};
