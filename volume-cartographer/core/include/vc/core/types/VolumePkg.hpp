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
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"

namespace vc::project {

struct Entry {
    std::string location;
    std::vector<std::string> tags;
};

enum class Category { Volumes, Segments, NormalGrids };

struct LoadOptions {
    std::filesystem::path remoteCacheRoot;
    bool failOnRemoteError = false;
};

bool isLocationRemote(const std::string& location);
std::filesystem::path resolveLocalPath(const std::string& location,
                                       const std::filesystem::path& base = {});

std::string validateLocation(Category category, const std::string& location);

}

class VolumePkg : public std::enable_shared_from_this<VolumePkg>
{
public:
    static std::shared_ptr<VolumePkg> newEmpty();
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

    bool addVolumeEntry(const std::string& location, std::vector<std::string> tags = {});
    bool addSegmentsEntry(const std::string& location, std::vector<std::string> tags = {});
    bool addNormalGridEntry(const std::string& location, std::vector<std::string> tags = {});
    bool removeEntry(const std::string& location);

    void setOutputSegments(const std::string& location);
    void clearOutputSegments();
    [[nodiscard]] bool hasOutputSegments() const;
    [[nodiscard]] std::filesystem::path outputSegmentsPath() const;

    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    std::shared_ptr<Volume> volume();
    std::shared_ptr<Volume> volume(const std::string& id);
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

    std::vector<vc::project::Entry> volumes_;
    std::vector<vc::project::Entry> segments_;
    std::vector<vc::project::Entry> normalGrids_;
    std::optional<std::string> outputSegments_;

    std::map<std::string, std::shared_ptr<Volume>> loadedVolumes_;
    std::map<std::string, std::vector<std::string>> volumeTagsByID_;
    std::map<std::string, std::shared_ptr<Segmentation>> loadedSegmentations_;
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
