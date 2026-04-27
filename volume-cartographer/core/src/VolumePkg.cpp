#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>
#include <cstring>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"

constexpr auto CONFIG = "config.json";

std::optional<std::string> VolumePkg::loadFirstSegmentationDir_{};

void VolumePkg::setLoadFirstSegmentationDirectory(const std::string& dirName)
{
    if (dirName.empty()) {
        loadFirstSegmentationDir_.reset();
        return;
    }
    loadFirstSegmentationDir_ = dirName;
}

VolumePkg::VolumePkg(const std::filesystem::path& fileLocation) : rootDir_{fileLocation}
{
    if (fileLocation.empty()) {
        config_ = utils::Json::parse(R"({"name":"Volpkg","version":8})");
        currentSegmentationDir_.clear();
        return;
    }
    auto configPath = fileLocation / ::CONFIG;
    config_ = vc::json::load_json_file(configPath);
    vc::json::require_fields(config_, {"name", "version"}, configPath.string());

    for (const auto& d : {"volumes", "paths", "traces", "transforms", "renders", "backups"}) {
        if (!std::filesystem::exists(rootDir_ / d)) {
            std::filesystem::create_directory(rootDir_ / d);
        }
    }

    project_ = std::make_shared<vc::Volpkg>(vc::Volpkg::from_volpkg(fileLocation));

    if (loadFirstSegmentationDir_ && !loadFirstSegmentationDir_->empty()) {
        currentSegmentationDir_ = *loadFirstSegmentationDir_;
    }

    ingestLocalProjectSources();

    if (loadFirstSegmentationDir_ && !loadFirstSegmentationDir_->empty()) {
        const auto& first = *loadFirstSegmentationDir_;
        std::vector<std::string> toUnload;
        for (const auto& [segId, dir] : segmentationDirectories_) {
            if (dir != first) toUnload.push_back(segId);
        }
        for (const auto& id : toUnload) {
            segmentations_.erase(id);
            segmentationDirectories_.erase(id);
        }
        loadedSegmentationDirs_.clear();
        loadedSegmentationDirs_.insert(first);
    }

    ensureSegmentScrollSource();
}

std::shared_ptr<VolumePkg> VolumePkg::New(const std::filesystem::path& fileLocation)
{
    return std::make_shared<VolumePkg>(fileLocation);
}

VolumePkg::VolumePkg(std::shared_ptr<vc::Volpkg> project)
    : VolumePkg(project && project->is_volpkg_compatible()
                    ? project->origin->root
                    : std::filesystem::path{})
{
    if (project) {
        project_ = std::move(project);
        ingestLocalProjectSources();
    }
}

std::shared_ptr<VolumePkg> VolumePkg::New(std::shared_ptr<vc::Volpkg> project)
{
    return std::make_shared<VolumePkg>(std::move(project));
}

void VolumePkg::setProject(std::shared_ptr<vc::Volpkg> project)
{
    project_ = std::move(project);
}

void VolumePkg::ingestLocalProjectSources()
{
    if (!project_) return;
    namespace fs = std::filesystem;

    auto resolveLocal = [&](const vc::DataSource& ds) -> fs::path {
        if (ds.location_kind != vc::LocationKind::Local) return {};
        try { return project_->resolve_local(ds); }
        catch (const std::exception&) { return {}; }
    };

    for (const auto* ds : project_->sources_of_type(vc::DataSourceType::ZarrVolume)) {
        if (!ds || !ds->enabled) continue;
        const auto p = resolveLocal(*ds);
        if (p.empty() || !fs::exists(p)) continue;
        addVolumeAt(p);
    }
    for (const auto* ds : project_->sources_of_type(vc::DataSourceType::VolumesDir)) {
        if (!ds || !ds->enabled) continue;
        const auto p = resolveLocal(*ds);
        if (p.empty() || !fs::exists(p) || !fs::is_directory(p)) continue;
        if (!ds->recursive) {
            addVolumeAt(p);
            continue;
        }
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(p, ec)) {
            if (!entry.is_directory(ec)) continue;
            addVolumeAt(entry.path());
        }
    }

    for (const auto* ds : project_->sources_of_type(vc::DataSourceType::SegmentsDir)) {
        if (!ds || !ds->enabled) continue;
        const auto p = resolveLocal(*ds);
        if (p.empty() || !fs::exists(p) || !fs::is_directory(p)) continue;
        if (!ds->recursive) {
            addSegmentationAt(p, ds->id);
        } else {
            std::error_code ec;
            for (const auto& entry : fs::directory_iterator(p, ec)) {
                if (!entry.is_directory(ec)) continue;
                const auto name = entry.path().filename().string();
                if (name.empty() || name[0] == '.' || name == ".tmp") continue;
                addSegmentationAt(entry.path(), ds->id);
            }
        }
        if (currentSegmentationDir_.empty()) {
            currentSegmentationDir_ = ds->id;
        }
    }
    for (const auto* ds : project_->sources_of_type(vc::DataSourceType::Segment)) {
        if (!ds || !ds->enabled) continue;
        const auto p = resolveLocal(*ds);
        if (p.empty() || !fs::exists(p)) continue;
        addSegmentationAt(p, ds->id);
        if (currentSegmentationDir_.empty()) {
            currentSegmentationDir_ = ds->id;
        }
    }

    for (const auto* ds : project_->sources_of_type(vc::DataSourceType::SegmentsDir)) {
        if (!ds || !ds->enabled) continue;
        if (ds->location_kind == vc::LocationKind::Remote) {
            loadedSegmentationDirs_.insert(ds->id);
            if (currentSegmentationDir_.empty()) {
                currentSegmentationDir_ = ds->id;
            }
        }
    }

    ensureSegmentScrollSource();
}



std::string VolumePkg::name() const
{
    auto name = config_["name"].get_string();
    if (name != "NULL") {
        return name;
    }

    return "UnnamedVolume";
}

int VolumePkg::version() const { return config_["version"].get_int(); }

bool VolumePkg::hasVolumes() const { return !volumes_.empty(); }

bool VolumePkg::hasVolume(const std::string& id) const
{
    return volumes_.count(id) > 0;
}

std::size_t VolumePkg::numberOfVolumes() const
{
    return volumes_.size();
}

std::vector<std::string> VolumePkg::volumeIDs() const
{
    std::vector<std::string> ids;
    for (const auto& v : volumes_) {
        ids.emplace_back(v.first);
    }
    return ids;
}

std::shared_ptr<Volume> VolumePkg::volume()
{
    if (volumes_.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return volumes_.begin()->second;
}

std::shared_ptr<Volume> VolumePkg::volume(const std::string& id)
{
    return volumes_.at(id);
}

bool VolumePkg::isValidVolumeDirectory(const std::filesystem::path& dirpath) const
{
    if (!std::filesystem::is_directory(dirpath)) {
        return false;
    }

    if (!std::filesystem::exists(dirpath / "meta.json") &&
        !std::filesystem::exists(dirpath / "metadata.json")) {
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dirpath)) {
        if (!entry.is_directory()) {
            continue;
        }

        if (std::filesystem::exists(entry.path() / ".zarray")) {
            return true;
        }
    }

    return false;
}

bool VolumePkg::addVolume(const std::shared_ptr<Volume>& volume)
{
    if (!volume) {
        Logger()->warn("Cannot add null volume to package");
        return false;
    }

    const auto volumeId = volume->id();
    auto result = volumes_.emplace(volumeId, volume);
    if (!result.second) {
        Logger()->warn("Volume '{}' already exists in package", volumeId);
        return false;
    }

    Logger()->info("Added external volume '{}' from '{}'", volumeId, volume->path().string());
    return true;
}

bool VolumePkg::addSingleVolume(const std::string& volumeDirName)
{
    return addVolumeAt(rootDir_ / "volumes" / volumeDirName);
}

bool VolumePkg::removeSingleVolume(const std::string& volumeIdOrDirName)
{
    auto direct = volumes_.find(volumeIdOrDirName);
    if (direct != volumes_.end()) {
        volumes_.erase(direct);
        Logger()->info("Removed volume '{}'", volumeIdOrDirName);
        return true;
    }

    for (auto it = volumes_.begin(); it != volumes_.end(); ++it) {
        if (it->second->path().filename().string() == volumeIdOrDirName) {
            Logger()->info("Removed volume '{}' (directory '{}')",
                           it->first,
                           volumeIdOrDirName);
            volumes_.erase(it);
            return true;
        }
    }

    Logger()->warn("Cannot remove volume '{}': not found", volumeIdOrDirName);
    return false;
}

bool VolumePkg::addVolumeAt(const std::filesystem::path& dirpath)
{
    if (!std::filesystem::exists(dirpath) || !std::filesystem::is_directory(dirpath)) {
        Logger()->warn("addVolumeAt: not a directory: {}", dirpath.string());
        return false;
    }
    if (!isValidVolumeDirectory(dirpath)) {
        Logger()->warn("addVolumeAt: invalid volume dir: {}", dirpath.string());
        return false;
    }
    try {
        auto v = Volume::New(dirpath);
        if (hasVolume(v->id())) {
            Logger()->info("addVolumeAt: volume '{}' already loaded", v->id());
            return false;
        }
        return addVolume(v);
    } catch (const std::exception& e) {
        Logger()->warn("addVolumeAt failed for '{}': {}", dirpath.string(), e.what());
        return false;
    }
}

bool VolumePkg::addSegmentationAt(const std::filesystem::path& dirpath,
                                  const std::string& group)
{
    if (!std::filesystem::exists(dirpath) || !std::filesystem::is_directory(dirpath)) {
        Logger()->warn("addSegmentationAt: not a directory: {}", dirpath.string());
        return false;
    }
    const std::string groupName = group.empty() ? currentSegmentationDir_ : group;
    try {
        auto s = Segmentation::New(std::filesystem::canonical(dirpath));
        auto [it, inserted] = segmentations_.emplace(s->id(), s);
        if (!inserted) {
            Logger()->warn("addSegmentationAt: segment '{}' already loaded", s->id());
            return false;
        }
        segmentationDirectories_[s->id()] = groupName;
        loadedSegmentationDirs_.insert(groupName);
        if (!volumes_.empty()) {
            auto scrollName = config_["name"].get_string();
            auto volumeUuid = volumes_.begin()->second->id();
            s->ensureScrollSource(scrollName, volumeUuid);
        }
        return true;
    } catch (const std::exception& e) {
        Logger()->warn("addSegmentationAt failed for '{}': {}",
                       dirpath.string(), e.what());
        return false;
    }
}

std::vector<std::string>
VolumePkg::segmentationIDsInGroup(const std::string& group) const
{
    std::vector<std::string> ids;
    for (const auto& [segId, dir] : segmentationDirectories_) {
        if (dir == group) ids.push_back(segId);
    }
    return ids;
}

bool VolumePkg::reloadSingleVolume(const std::string& volumeId)
{
    auto it = volumes_.find(volumeId);
    if (it == volumes_.end()) {
        if (!addSingleVolume(volumeId)) {
            return false;
        }
        return true;
    }

    const auto volumePath = it->second->path();
    removeSingleVolume(volumeId);
    return addSingleVolume(volumePath.filename().string());
}

bool VolumePkg::hasSegmentations() const
{
    return !segmentations_.empty();
}


std::shared_ptr<Segmentation> VolumePkg::segmentation(const std::string& id)
{
    auto it = segmentations_.find(id);
    if (it == segmentations_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::string> VolumePkg::segmentationIDs() const
{
    std::vector<std::string> ids;
    // Only return IDs from the current directory
    for (const auto& s : segmentations_) {
        auto it = segmentationDirectories_.find(s.first);
        if (it != segmentationDirectories_.end() && it->second == currentSegmentationDir_) {
            ids.emplace_back(s.first);
        }
    }
    return ids;
}


void VolumePkg::loadSegmentationsFromDirectory(const std::string& dirName)
{
    // DO NOT clear existing segmentations - we keep all directories in memory
    // Only remove segmentations from this specific directory
    std::vector<std::string> toRemove;
    for (const auto& pair : segmentationDirectories_) {
        if (pair.second == dirName) {
            toRemove.push_back(pair.first);
        }
    }

    // Remove old segmentations from this directory
    for (const auto& id : toRemove) {
        segmentations_.erase(id);
        segmentationDirectories_.erase(id);
    }

    std::filesystem::path segDir;
    if (!rootDir_.empty()) {
        segDir = rootDir_ / dirName;
    } else if (project_) {
        try { segDir = project_->resolve_segments_dir(dirName); }
        catch (const std::exception&) { segDir.clear(); }
    }
    if (segDir.empty() || !std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", dirName);
        return;
    }

    // Load segmentations from the specified directory
    int loadedCount = 0;
    int skippedCount = 0;
    int failedCount = 0;
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            // Skip hidden directories and .tmp folders
            const auto dirName_ = dirpath.filename().string();
            if (dirName_.empty() || dirName_[0] == '.' || dirName_ == ".tmp") {
                skippedCount++;
                continue;
            }
            try {
                auto s = Segmentation::New(dirpath);
                auto result = segmentations_.emplace(s->id(), s);
                if (result.second) {
                    // Track which directory this segmentation came from
                    segmentationDirectories_[s->id()] = dirName;
                    loadedCount++;
                } else {
                    Logger()->warn("Duplicate segment ID '{}' - already loaded from different path, skipping: {}",
                                   s->id(), dirpath.string());
                    skippedCount++;
                }
            }
            catch (const std::exception &exc) {
                Logger()->warn("Failed to load segment dir: {} - {}", dirpath.string(), exc.what());
                failedCount++;
            }
        }
    }
    Logger()->info("Loaded {} segments from '{}' (skipped={}, failed={})",
                   loadedCount, dirName, skippedCount, failedCount);
}

void VolumePkg::ensureSegmentScrollSource()
{
    if (segmentations_.empty() || volumes_.empty()) {
        return;
    }

    auto scrollName = config_["name"].get_string();
    auto vol = volumes_.begin()->second;
    auto volumeUuid = vol->id();

    for (auto& [id, seg] : segmentations_) {
        seg->ensureScrollSource(scrollName, volumeUuid);
    }
}

void VolumePkg::setSegmentationDirectory(const std::string& dirName)
{
    if (currentSegmentationDir_ == dirName) {
        return;
    }
    if (loadedSegmentationDirs_.find(dirName) == loadedSegmentationDirs_.end()) {
        std::filesystem::path segDir;
        if (!rootDir_.empty()) {
            segDir = rootDir_ / dirName;
        } else if (project_) {
            try { segDir = project_->resolve_segments_dir(dirName); }
            catch (const std::exception&) { /* leave empty */ }
        }
        if (segDir.empty() || !std::filesystem::exists(segDir)) {
            Logger()->warn("Segmentation directory '{}' does not exist", dirName);
        } else {
            loadSegmentationsFromDirectory(dirName);
            loadedSegmentationDirs_.insert(dirName);
        }
    }
    currentSegmentationDir_ = dirName;
}

auto VolumePkg::getSegmentationDirectory() const -> std::string
{
    return currentSegmentationDir_;
}

auto VolumePkg::getVolpkgDirectory() const -> std::string
{
    if (rootDir_.empty() && project_) {
        if (project_->is_volpkg_compatible()) {
            return project_->origin->root.string();
        }
        if (!project_->path().empty()) {
            return project_->path().parent_path().string();
        }
    }
    return rootDir_;
}


auto VolumePkg::getAvailableSegmentationDirectories() const -> std::vector<std::string>
{
    std::vector<std::string> dirs;

    if (rootDir_.empty()) {
        dirs.insert(dirs.end(),
                    loadedSegmentationDirs_.begin(),
                    loadedSegmentationDirs_.end());
        return dirs;
    }

    const std::vector<std::string> commonDirs = {"paths", "traces", "export"};
    for (const auto& dir : commonDirs) {
        if (std::filesystem::exists(rootDir_ / dir) && std::filesystem::is_directory(rootDir_ / dir)) {
            dirs.push_back(dir);
        }
    }

    return dirs;
}

void VolumePkg::removeSegmentation(const std::string& id)
{
    // Check if segmentation exists
    auto it = segmentations_.find(id);
    if (it == segmentations_.end()) {
        throw std::runtime_error("Segmentation not found: " + id);
    }

    // Get the path before removing
    std::filesystem::path segPath = it->second->path();

    // Remove from internal map
    segmentations_.erase(it);

    // Delete the physical folder
    if (std::filesystem::exists(segPath)) {
        std::filesystem::remove_all(segPath);
    }
}

void VolumePkg::refreshSegmentations()
{
    const auto segDir = rootDir_ / currentSegmentationDir_;
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", currentSegmentationDir_);
        return;
    }

    // Build a set of current segmentation paths on disk for the current directory
    std::set<std::filesystem::path> diskPaths;
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            // Skip hidden directories and .tmp folders
            const auto dirName = dirpath.filename().string();
            if (dirName.empty() || dirName[0] == '.' || dirName == ".tmp") {
                continue;
            }
            diskPaths.insert(dirpath);
        }
    }

    // Find segmentations to remove (loaded from current directory but not on disk anymore)
    std::vector<std::string> toRemove;
    for (const auto& seg : segmentations_) {
        auto dirIt = segmentationDirectories_.find(seg.first);
        if (dirIt != segmentationDirectories_.end() && dirIt->second == currentSegmentationDir_) {
            // This segmentation belongs to the current directory
            // Check if it still exists on disk
            if (diskPaths.find(seg.second->path()) == diskPaths.end()) {
                // Not on disk anymore - mark for removal
                toRemove.push_back(seg.first);
            }
        }
    }

    // Remove segmentations that no longer exist
    for (const auto& id : toRemove) {
        Logger()->info("Removing segmentation '{}' - no longer exists on disk", id);

        // Get the path before removing the segmentation
        std::filesystem::path segPath;
        auto segIt = segmentations_.find(id);
        if (segIt != segmentations_.end()) {
            segPath = segIt->second->path();
        }

        // Remove from segmentations map
        segmentations_.erase(id);

        // Remove from directories map
        segmentationDirectories_.erase(id);
    }

    // Find and add new segmentations (on disk but not in memory)
    // Build a set of currently loaded paths for O(1) lookup
    std::set<std::filesystem::path> loadedPaths;
    for (const auto& seg : segmentations_) {
        loadedPaths.insert(seg.second->path());
    }

    for (const auto& diskPath : diskPaths) {
        if (loadedPaths.find(diskPath) == loadedPaths.end()) {
            try {
                auto s = Segmentation::New(diskPath);
                segmentations_.emplace(s->id(), s);
                segmentationDirectories_[s->id()] = currentSegmentationDir_;
                Logger()->info("Added new segmentation '{}'", s->id());
            }
            catch (const std::exception &exc) {
                Logger()->warn("Failed to load segment dir: {} - {}", diskPath.string(), exc.what());
            }
        }
    }
}

bool VolumePkg::isSurfaceLoaded(const std::string& id) const
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return false;
    }
    return segIt->second->isSurfaceLoaded();
}

std::shared_ptr<QuadSurface> VolumePkg::loadSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        Logger()->error("Cannot load surface - segmentation {} not found", id);
        return nullptr;
    }
    return segIt->second->loadSurface();
}

std::shared_ptr<QuadSurface> VolumePkg::getSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return nullptr;
    }
    return segIt->second->getSurface();
}


std::vector<std::string> VolumePkg::getLoadedSurfaceIDs() const
{
    std::vector<std::string> ids;
    for (const auto& [id, seg] : segmentations_) {
        if (seg->isSurfaceLoaded()) {
            ids.push_back(id);
        }
    }
    return ids;
}

void VolumePkg::unloadAllSurfaces()
{
    for (auto& [id, seg] : segmentations_) {
        seg->unloadSurface();
    }
}

bool VolumePkg::unloadSurface(const std::string& id)
{
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        return false;
    }
    segIt->second->unloadSurface();
    return true;
}


void VolumePkg::loadSurfacesBatch(const std::vector<std::string>& ids)
{
    std::vector<std::shared_ptr<Segmentation>> toLoad;
    for (const auto& id : ids) {
        auto segIt = segmentations_.find(id);
        if (segIt != segmentations_.end() && !segIt->second->isSurfaceLoaded() && segIt->second->canLoadSurface()) {
            toLoad.push_back(segIt->second);
        }
    }

#pragma omp parallel for schedule(dynamic,1)
    for (auto & seg : toLoad) {
        try {
            seg->loadSurface();
        } catch (const std::exception& e) {
            Logger()->error("Failed to load surface for {}: {}", seg->id(), e.what());
        }
    }
}

VolumePkg::~VolumePkg()
{
}

bool VolumePkg::addSingleSegmentation(const std::string& id)
{
    if (segmentations_.count(id)) return false;
    return addSegmentationAt(rootDir_ / currentSegmentationDir_ / id,
                             currentSegmentationDir_);
}

bool VolumePkg::removeSingleSegmentation(const std::string& id)
{
    auto it = segmentations_.find(id);
    if (it == segmentations_.end()) {
        Logger()->warn("Cannot remove segment {} - not found", id);
        return false; // Don't crash, just return false
    }

    // Check if this segment belongs to the current directory
    auto dirIt = segmentationDirectories_.find(id);
    if (dirIt != segmentationDirectories_.end()) {
        // Only log if it's from a different directory
        if (dirIt->second != currentSegmentationDir_) {
            Logger()->debug("Removing segment {} from {} directory (current is {})",
                          id, dirIt->second, currentSegmentationDir_);
        }
    }

    // Unload surface if loaded
    it->second->unloadSurface();

    // Remove from maps
    segmentations_.erase(it);
    segmentationDirectories_.erase(id);

    Logger()->info("Removed segmentation: {}", id);
    return true;
}

bool VolumePkg::reloadSingleSegmentation(const std::string& id)
{
    // First check if the segment exists on disk
    std::filesystem::path segPath = rootDir_ / currentSegmentationDir_ / id;

    if (!std::filesystem::exists(segPath) || !std::filesystem::is_directory(segPath)) {
        Logger()->warn("Cannot reload - segment directory does not exist: {}", segPath.string());
        return false;
    }

    removeSingleSegmentation(id);

    return addSingleSegmentation(id);
}

namespace vc {

namespace {

const std::vector<std::string>& kLegacySegmentsDirs()
{
    static const std::vector<std::string> v = {"paths", "traces", "export"};
    return v;
}

} // namespace

std::string data_source_type_to_string(DataSourceType t)
{
    switch (t) {
        case DataSourceType::VolumesDir:      return "volumes_dir";
        case DataSourceType::ZarrVolume:      return "zarr_volume";
        case DataSourceType::SegmentsDir:     return "segments_dir";
        case DataSourceType::Segment:         return "segment";
        case DataSourceType::NormalGrid:      return "normal_grid";
        case DataSourceType::NormalDirVolume: return "normal_dir_volume";
        case DataSourceType::SyncDir:         return "sync_dir";
    }
    throw std::runtime_error("unknown DataSourceType");
}

DataSourceType data_source_type_from_string(const std::string& s)
{
    if (s == "volumes_dir")       return DataSourceType::VolumesDir;
    if (s == "zarr_volume")       return DataSourceType::ZarrVolume;
    if (s == "segments_dir")      return DataSourceType::SegmentsDir;
    if (s == "segment")           return DataSourceType::Segment;
    if (s == "normal_grid")       return DataSourceType::NormalGrid;
    if (s == "normal_dir_volume") return DataSourceType::NormalDirVolume;
    if (s == "sync_dir")          return DataSourceType::SyncDir;
    throw std::runtime_error("unknown data source type: " + s);
}

std::string sync_direction_to_string(SyncDirection d)
{
    switch (d) {
        case SyncDirection::Pull: return "pull";
        case SyncDirection::Push: return "push";
        case SyncDirection::Both: return "both";
    }
    throw std::runtime_error("unknown SyncDirection");
}

SyncDirection sync_direction_from_string(const std::string& s)
{
    if (s == "pull") return SyncDirection::Pull;
    if (s == "push") return SyncDirection::Push;
    if (s == "both") return SyncDirection::Both;
    throw std::runtime_error("unknown sync direction: " + s);
}

LocationKind infer_location_kind(const std::string& location)
{
    static constexpr const char* kRemoteSchemes[] = {
        "http://", "https://", "s3://"
    };
    for (const char* scheme : kRemoteSchemes) {
        const std::string s(scheme);
        if (location.size() >= s.size()
            && location.compare(0, s.size(), s) == 0)
        {
            return LocationKind::Remote;
        }
    }
    return LocationKind::Local;
}

std::string location_kind_to_string(LocationKind k)
{
    switch (k) {
        case LocationKind::Local:  return "local";
        case LocationKind::Remote: return "remote";
    }
    throw std::runtime_error("unknown LocationKind");
}

LocationKind location_kind_from_string(const std::string& s)
{
    if (s == "local")  return LocationKind::Local;
    if (s == "remote") return LocationKind::Remote;
    throw std::runtime_error("unknown location kind: " + s);
}

utils::Json Volpkg::to_json() const
{
    auto root = utils::Json::object();
    root["name"] = name;
    root["version"] = version;
    if (!description.empty()) {
        root["description"] = description;
    }

    auto sources = utils::Json::array();
    for (const auto& ds : data_sources) {
        if (ds.imported) continue;
        auto s = utils::Json::object();
        s["id"] = ds.id;
        s["type"] = data_source_type_to_string(ds.type);
        s["location"] = ds.location;
        s["location_kind"] = location_kind_to_string(ds.location_kind);
        s["recursive"] = ds.recursive;
        s["track_changes"] = ds.track_changes;
        if (!ds.enabled) s["enabled"] = false;
        if (!ds.children.empty()) {
            auto arr = utils::Json::array();
            for (const auto& c : ds.children) arr.push_back(c);
            s["children"] = arr;
        }
        if (!ds.tags.empty()) {
            auto arr = utils::Json::array();
            for (const auto& t : ds.tags) arr.push_back(t);
            s["tags"] = arr;
        }
        if (ds.type == DataSourceType::SyncDir) {
            s["sync_remote"] = ds.sync_remote;
            s["sync_direction"] = sync_direction_to_string(ds.sync_direction);
        }
        sources.push_back(s);
    }
    root["data_sources"] = sources;

    if (!groups.empty()) {
        auto g = utils::Json::array();
        for (const auto& gr : groups) {
            auto o = utils::Json::object();
            o["id"] = gr.id;
            o["name"] = gr.name;
            auto arr = utils::Json::array();
            for (const auto& s : gr.source_ids) arr.push_back(s);
            o["source_ids"] = arr;
            g.push_back(o);
        }
        root["groups"] = g;
    }

    if (!linked_projects.empty()) {
        auto lp = utils::Json::array();
        for (const auto& link : linked_projects) {
            auto o = utils::Json::object();
            o["path"] = link.path;
            o["read_only"] = link.read_only;
            if (!link.id_prefix.empty()) o["id_prefix"] = link.id_prefix;
            lp.push_back(o);
        }
        root["linked_projects"] = lp;
    }

    if (origin) {
        auto o = utils::Json::object();
        o["kind"] = origin->kind;
        o["root"] = origin->root.string();
        if (!origin->legacy_config.is_null()) {
            o["legacy_config"] = origin->legacy_config;
        }
        root["origin"] = o;
    }

    if (!active_segments_source_id.empty()) {
        root["active_segments_source_id"] = active_segments_source_id;
    }
    if (!output_segments_source_id.empty()) {
        root["output_segments_source_id"] = output_segments_source_id;
    }

    return root;
}

Volpkg Volpkg::from_json(const utils::Json& j)
{
    Volpkg p;
    if (j.contains("name"))        p.name = j["name"].get_string();
    if (j.contains("version"))     p.version = j["version"].get_int();
    if (j.contains("description")) p.description = j["description"].get_string();

    if (j.contains("data_sources")) {
        const auto& sources = j["data_sources"];
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto& s = sources[i];
            DataSource ds;
            ds.id = s["id"].get_string();
            ds.type = data_source_type_from_string(s["type"].get_string());
            ds.location = s["location"].get_string();
            ds.location_kind = location_kind_from_string(
                s.value("location_kind", std::string("local")));
            ds.recursive = s.value("recursive", true);
            ds.track_changes = s.value("track_changes", false);
            ds.enabled = s.value("enabled", true);
            if (s.contains("children")) {
                ds.children = s["children"].get_string_array();
            }
            if (s.contains("tags")) {
                ds.tags = s["tags"].get_string_array();
            }
            if (s.contains("sync_remote")) {
                ds.sync_remote = s["sync_remote"].get_string();
            }
            if (s.contains("sync_direction")) {
                ds.sync_direction = sync_direction_from_string(
                    s["sync_direction"].get_string());
            }
            p.data_sources.push_back(std::move(ds));
        }
    }

    if (j.contains("groups")) {
        const auto& gs = j["groups"];
        for (std::size_t i = 0; i < gs.size(); ++i) {
            const auto& g = gs[i];
            Group gr;
            gr.id = g["id"].get_string();
            gr.name = g.value("name", gr.id);
            if (g.contains("source_ids")) {
                gr.source_ids = g["source_ids"].get_string_array();
            }
            p.groups.push_back(std::move(gr));
        }
    }

    if (j.contains("linked_projects")) {
        const auto& ls = j["linked_projects"];
        for (std::size_t i = 0; i < ls.size(); ++i) {
            const auto& l = ls[i];
            LinkedProject lp;
            lp.path = l["path"].get_string();
            lp.read_only = l.value("read_only", true);
            lp.id_prefix = l.value("id_prefix", std::string());
            p.linked_projects.push_back(std::move(lp));
        }
    }

    if (j.contains("origin")) {
        const auto& o = j["origin"];
        Origin origin;
        origin.kind = o.value("kind", std::string("project"));
        origin.root = o.value("root", std::string());
        if (o.contains("legacy_config")) {
            origin.legacy_config = o["legacy_config"];
        }
        p.origin = origin;
    }

    p.active_segments_source_id =
        j.value("active_segments_source_id", std::string());
    p.output_segments_source_id =
        j.value("output_segments_source_id", std::string());

    return p;
}

void Volpkg::save_to_file(const std::filesystem::path& path) const
{
    auto j = to_json();
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream o(tmp);
        if (!o) {
            throw std::runtime_error("failed to open for write: " + tmp.string());
        }
        o << j.dump(2) << '\n';
    }
    std::filesystem::rename(tmp, path);
}

Volpkg Volpkg::load_from_file(const std::filesystem::path& path)
{
    auto j = vc::json::load_json_file(path);
    auto p = Volpkg::from_json(j);
    p.set_path(path);
    p.merge_linked_projects();
    return p;
}

bool Volpkg::looks_like_volpkg(const std::filesystem::path& dir)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return false;
    }
    const auto cfg = dir / "config.json";
    if (!std::filesystem::exists(cfg, ec)) {
        return false;
    }
    try {
        auto j = vc::json::load_json_file(cfg);
        return j.contains("name") && j.contains("version");
    } catch (...) {
        return false;
    }
}

Volpkg Volpkg::from_volpkg(const std::filesystem::path& volpkg_root)
{
    Volpkg p;
    const auto cfg = volpkg_root / "config.json";
    auto j = vc::json::load_json_file(cfg);
    vc::json::require_fields(j, {"name", "version"}, cfg.string());

    p.name = j["name"].get_string();
    p.version = j["version"].get_int();

    Origin origin;
    origin.kind = "volpkg";
    origin.root = volpkg_root;
    auto legacy = j;
    legacy.erase("name");
    legacy.erase("version");
    if (!legacy.empty()) {
        origin.legacy_config = legacy;
    }
    p.origin = origin;

    {
        const auto vdir = volpkg_root / "volumes";
        if (std::filesystem::exists(vdir)) {
            DataSource ds;
            ds.id = "volumes";
            ds.type = DataSourceType::VolumesDir;
            ds.location = "volumes";
            ds.location_kind = LocationKind::Local;
            ds.recursive = true;
            ds.track_changes = true;
            p.data_sources.push_back(std::move(ds));
        }
    }

    for (const auto& name : kLegacySegmentsDirs()) {
        const auto d = volpkg_root / name;
        if (!std::filesystem::exists(d) || !std::filesystem::is_directory(d)) {
            continue;
        }
        DataSource ds;
        ds.id = name;
        ds.type = DataSourceType::SegmentsDir;
        ds.location = name;
        ds.location_kind = LocationKind::Local;
        ds.recursive = true;
        ds.track_changes = true;
        p.data_sources.push_back(std::move(ds));
    }

    for (const auto& ds : p.data_sources) {
        if (ds.type == DataSourceType::SegmentsDir) {
            p.active_segments_source_id = ds.id;
            p.output_segments_source_id = ds.id;
            break;
        }
    }

    return p;
}

bool Volpkg::is_volpkg_compatible() const
{
    return origin.has_value() && origin->kind == "volpkg"
        && !origin->root.empty();
}

const DataSource* Volpkg::find_source(const std::string& id) const
{
    for (const auto& ds : data_sources) {
        if (ds.id == id) return &ds;
    }
    return nullptr;
}

DataSource* Volpkg::find_source(const std::string& id)
{
    for (auto& ds : data_sources) {
        if (ds.id == id) return &ds;
    }
    return nullptr;
}

std::vector<const DataSource*>
Volpkg::sources_of_type(DataSourceType t) const
{
    std::vector<const DataSource*> out;
    for (const auto& ds : data_sources) {
        if (ds.type == t) out.push_back(&ds);
    }
    return out;
}

std::vector<const DataSource*>
Volpkg::sources_with_tag(const std::string& tag) const
{
    std::vector<const DataSource*> out;
    for (const auto& ds : data_sources) {
        for (const auto& t : ds.tags) {
            if (t == tag) { out.push_back(&ds); break; }
        }
    }
    return out;
}

void Volpkg::merge_linked_projects()
{
    if (linked_projects.empty()) return;

    auto base_dir = [&]() -> std::filesystem::path {
        if (is_volpkg_compatible()) return origin->root;
        if (!path_.empty()) return path_.parent_path();
        return std::filesystem::current_path();
    };

    std::set<std::string> seen;
    std::vector<std::pair<LinkedProject, std::filesystem::path>> todo;
    for (const auto& lp : linked_projects) {
        std::filesystem::path p = lp.path;
        if (!p.is_absolute()) p = base_dir() / p;
        todo.emplace_back(lp, p);
    }

    while (!todo.empty()) {
        auto [link, linkPath] = todo.back();
        todo.pop_back();

        std::error_code ec;
        auto canon = std::filesystem::weakly_canonical(linkPath, ec).string();
        if (ec || canon.empty()) canon = linkPath.string();
        if (seen.count(canon)) continue;
        seen.insert(canon);

        Volpkg other;
        try { other = Volpkg::load_from_file(linkPath); }
        catch (const std::exception&) { continue; }

        for (auto ds : other.data_sources) {
            if (ds.imported) continue;
            if (!link.id_prefix.empty()) ds.id = link.id_prefix + ds.id;
            ds.imported = true;
            data_sources.push_back(std::move(ds));
        }
        for (const auto& sub : other.linked_projects) {
            std::filesystem::path p = sub.path;
            if (!p.is_absolute()) p = linkPath.parent_path() / sub.path;
            todo.emplace_back(sub, p);
        }
    }
}

std::vector<const DataSource*>
Volpkg::sources_in_group(const std::string& group_id) const
{
    std::vector<const DataSource*> out;
    for (const auto& g : groups) {
        if (g.id != group_id) continue;
        for (const auto& sid : g.source_ids) {
            if (const auto* ds = find_source(sid)) out.push_back(ds);
        }
        break;
    }
    return out;
}

std::filesystem::path Volpkg::resolve_local(const DataSource& ds) const
{
    if (ds.location_kind != LocationKind::Local) {
        throw std::runtime_error(
            "resolve_local called on remote source: " + ds.id);
    }

    std::filesystem::path loc{ds.location};
    if (loc.is_absolute()) {
        return loc;
    }

    if (origin && !origin->root.empty()) {
        return origin->root / loc;
    }
    if (!path_.empty()) {
        return path_.parent_path() / loc;
    }
    return std::filesystem::absolute(loc);
}

std::string Volpkg::remote_url(const DataSource& ds) const
{
    if (ds.location_kind != LocationKind::Remote) {
        throw std::runtime_error(
            "remote_url called on local source: " + ds.id);
    }
    return ds.location;
}

std::string Volpkg::resolve_location(const DataSource& ds) const
{
    if (ds.location_kind == LocationKind::Remote) {
        return ds.location;
    }
    return resolve_local(ds).string();
}

std::filesystem::path
Volpkg::resolve_segments_dir(const std::string& id_or_name) const
{
    if (const auto* ds = find_source(id_or_name);
        ds && ds->type == DataSourceType::SegmentsDir
           && ds->location_kind == LocationKind::Local)
    {
        return resolve_local(*ds);
    }
    if (is_volpkg_compatible()) {
        return origin->root / id_or_name;
    }
    throw std::runtime_error(
        "cannot resolve segments dir '" + id_or_name + "'");
}

std::filesystem::path Volpkg::resolve_active_segments_dir() const
{
    std::string id = active_segments_source_id;
    if (id.empty()) {
        for (const auto& ds : data_sources) {
            if (ds.type == DataSourceType::SegmentsDir) {
                id = ds.id;
                break;
            }
        }
    }
    if (!id.empty()) {
        return resolve_segments_dir(id);
    }
    if (is_volpkg_compatible()) {
        return origin->root / "paths";
    }
    throw std::runtime_error("no active segments dir in project");
}

std::filesystem::path Volpkg::resolve_output_segments_dir() const
{
    if (!output_segments_source_id.empty()) {
        return resolve_segments_dir(output_segments_source_id);
    }
    return resolve_active_segments_dir();
}

std::filesystem::path Volpkg::resolve_volumes_dir() const
{
    for (const auto& ds : data_sources) {
        if (ds.type == DataSourceType::VolumesDir
            && ds.location_kind == LocationKind::Local)
        {
            return resolve_local(ds);
        }
    }
    if (is_volpkg_compatible()) {
        return origin->root / "volumes";
    }
    throw std::runtime_error("no volumes dir in project");
}

std::filesystem::path
Volpkg::support_file_path(const std::string& filename) const
{
    if (is_volpkg_compatible()) {
        return origin->root / filename;
    }
    if (!path_.empty()) {
        return path_.parent_path() / filename;
    }
    throw std::runtime_error(
        "cannot resolve support file '" + filename + "': project is unsaved");
}

std::vector<std::filesystem::path> Volpkg::all_segments_dirs() const
{
    std::vector<std::filesystem::path> out;
    for (const auto& ds : data_sources) {
        if (ds.type == DataSourceType::SegmentsDir
            && ds.location_kind == LocationKind::Local)
        {
            out.push_back(resolve_local(ds));
        }
    }
    if (out.empty() && is_volpkg_compatible()) {
        for (const char* name : {"paths", "traces", "export"}) {
            auto p = origin->root / name;
            if (std::filesystem::exists(p)
                && std::filesystem::is_directory(p))
            {
                out.push_back(std::move(p));
            }
        }
    }
    return out;
}

} // namespace vc
