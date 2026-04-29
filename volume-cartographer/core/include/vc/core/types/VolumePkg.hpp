#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include "utils/Json.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/RemoteScroll.hpp"

namespace vc {

enum class DataSourceType {
    VolumesDir,
    ZarrVolume,
    SegmentsDir,
    Segment,
    NormalGrid,
    NormalDirVolume,
    SyncDir
};

enum class SyncDirection {
    Pull,
    Push,
    Both
};

std::string sync_direction_to_string(SyncDirection d);
SyncDirection sync_direction_from_string(const std::string& s);

enum class LocationKind {
    Local,
    Remote
};

std::string data_source_type_to_string(DataSourceType t);
DataSourceType data_source_type_from_string(const std::string& s);
std::string location_kind_to_string(LocationKind k);
LocationKind location_kind_from_string(const std::string& s);

struct DataSource {
    std::string id;
    DataSourceType type = DataSourceType::SegmentsDir;
    std::string location;
    LocationKind location_kind = LocationKind::Local;
    bool recursive = true;
    bool track_changes = false;
    bool enabled = true;
    std::vector<std::string> children;
    std::vector<std::string> tags;

    std::string sync_remote;
    SyncDirection sync_direction = SyncDirection::Pull;

    bool imported = false;

    [[nodiscard]] bool is_remote() const noexcept
    {
        return location_kind == LocationKind::Remote;
    }
};

struct Group {
    std::string id;
    std::string name;
    std::vector<std::string> source_ids;
};

struct LinkedProject {
    std::string path;
    bool read_only = true;
    std::string id_prefix;
};

LocationKind infer_location_kind(const std::string& location);

class Volpkg {
public:
    Volpkg() = default;

    std::string name;
    int version = 1;
    std::string description;

    std::vector<DataSource> data_sources;
    std::vector<Group> groups;
    std::vector<LinkedProject> linked_projects;

    struct Origin {
        std::string kind;
        std::filesystem::path root;
        utils::Json legacy_config;
    };
    std::optional<Origin> origin;

    std::string active_segments_source_id;
    std::string output_segments_source_id;

    [[nodiscard]] utils::Json to_json() const;
    static Volpkg from_json(const utils::Json& j);

    void save_to_file(const std::filesystem::path& path) const;
    static Volpkg load_from_file(const std::filesystem::path& path);

    static bool looks_like_volpkg(const std::filesystem::path& dir);
    static Volpkg from_volpkg(const std::filesystem::path& volpkg_root);
    [[nodiscard]] bool is_volpkg_compatible() const;

    [[nodiscard]] const DataSource* find_source(const std::string& id) const;
    DataSource* find_source(const std::string& id);

    [[nodiscard]] std::vector<const DataSource*>
        sources_of_type(DataSourceType t) const;
    [[nodiscard]] std::vector<const DataSource*>
        sources_with_tag(const std::string& tag) const;
    [[nodiscard]] std::vector<const DataSource*>
        sources_in_group(const std::string& group_id) const;

    void merge_linked_projects();

    [[nodiscard]] std::filesystem::path
        resolve_local(const DataSource& ds) const;
    [[nodiscard]] std::string remote_url(const DataSource& ds) const;
    [[nodiscard]] std::string resolve_location(const DataSource& ds) const;

    [[nodiscard]] std::filesystem::path
        resolve_segments_dir(const std::string& id_or_name) const;
    [[nodiscard]] std::filesystem::path
        resolve_active_segments_dir() const;
    [[nodiscard]] std::filesystem::path
        resolve_output_segments_dir() const;
    [[nodiscard]] std::filesystem::path
        resolve_volumes_dir() const;
    [[nodiscard]] std::vector<std::filesystem::path>
        all_segments_dirs() const;
    [[nodiscard]] std::filesystem::path
        support_file_path(const std::string& filename) const;

    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void set_path(const std::filesystem::path& p) { path_ = p; }

private:
    std::filesystem::path path_;
};

} // namespace vc

class VolumePkg
{
public:
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    explicit VolumePkg(std::shared_ptr<vc::Volpkg> project);
    ~VolumePkg();
    static std::shared_ptr<VolumePkg> New(const std::filesystem::path& fileLocation);
    static std::shared_ptr<VolumePkg> New(std::shared_ptr<vc::Volpkg> project);

    [[nodiscard]] std::shared_ptr<vc::Volpkg> project() const { return project_; }
    void setProject(std::shared_ptr<vc::Volpkg> project);

    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    std::shared_ptr<Volume> volume();
    std::shared_ptr<Volume> volume(const std::string& id);
    [[nodiscard]] bool hasSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;

    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;
    [[nodiscard]] std::string getVolpkgDirectory() const;

    [[nodiscard]] bool isValidVolumeDirectory(const std::filesystem::path& dirpath) const;
    bool addVolume(const std::shared_ptr<Volume>& volume);
    bool addSingleVolume(const std::string& volumeDirName);
    bool removeSingleVolume(const std::string& volumeIdOrDirName);
    bool reloadSingleVolume(const std::string& volumeId);

    bool addVolumeAt(const std::filesystem::path& dirpath);
    bool addSegmentationAt(const std::filesystem::path& dirpath,
                           const std::string& group);

    [[nodiscard]] std::vector<std::string>
        segmentationIDsInGroup(const std::string& group) const;

    void refreshSegmentations();
    static void setLoadFirstSegmentationDirectory(const std::string& dirName);

    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<QuadSurface> loadSurface(const std::string& id);
    std::shared_ptr<QuadSurface> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);
    bool addSingleSegmentation(const std::string& id);
    bool removeSingleSegmentation(const std::string& id);
    bool reloadSingleSegmentation(const std::string& id);

private:
    utils::Json config_;
    std::filesystem::path rootDir_;
    std::shared_ptr<vc::Volpkg> project_;
    std::map<std::string, std::shared_ptr<Volume>> volumes_;
    std::map<std::string, std::shared_ptr<Segmentation>> segmentations_;
    std::string currentSegmentationDir_ = "paths";
    std::map<std::string, std::string> segmentationDirectories_;
    std::set<std::string> loadedSegmentationDirs_;
    static std::optional<std::string> loadFirstSegmentationDir_;

    void loadSegmentationsFromDirectory(const std::string& dirName);
    void ensureSegmentScrollSource();
    void ingestLocalProjectSources();
};
