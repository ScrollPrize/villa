#include "vc/core/types/VolumePkg.hpp"

#include <set>
#include <utility>
#include <cstring>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"

constexpr auto CONFIG = "config.json";

VolumePkg::VolumePkg(const std::filesystem::path& fileLocation) : rootDir_{fileLocation}
{
    auto configPath = fileLocation / ::CONFIG;
    config_ = vc::json::load_json_file(configPath);
    vc::json::require_fields(config_, {"name", "version"}, configPath.string());

    std::vector<std::string> dirs = {"volumes","paths","traces","transforms","renders","backups"};

    for (const auto& d : dirs) {
        if (not std::filesystem::exists(rootDir_ / d)) {
            std::filesystem::create_directory(rootDir_ / d);
        }

    }

    for (const auto& entry : std::filesystem::directory_iterator(rootDir_ / "volumes")) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            auto v = Volume::New(dirpath);
            volumes_.emplace(v->id(), v);
        }
    }

    auto availableDirs = getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        loadSegmentationsFromDirectory(dirName);
    }

    ensureSegmentScrollSource();
}

std::shared_ptr<VolumePkg> VolumePkg::New(const std::filesystem::path& fileLocation)
{
    return std::make_shared<VolumePkg>(fileLocation);
}


std::string VolumePkg::name() const
{
    auto name = config_["name"].get<std::string>();
    if (name != "NULL") {
        return name;
    }

    return "UnnamedVolume";
}

int VolumePkg::version() const { return config_["version"].get<int>(); }

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

    // Check if directory exists
    const auto segDir = rootDir_ / dirName;
    if (!std::filesystem::exists(segDir)) {
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

    auto scrollName = config_["name"].get<std::string>();
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
    currentSegmentationDir_ = dirName;
}

auto VolumePkg::getSegmentationDirectory() const -> std::string
{
    return currentSegmentationDir_;
}

auto VolumePkg::getVolpkgDirectory() const -> std::string
{
    return rootDir_;
}


auto VolumePkg::getAvailableSegmentationDirectories() const -> std::vector<std::string>
{
    std::vector<std::string> dirs;

    // Check for common segmentation directories
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
    // Check if already loaded
    if (segmentations_.find(id) != segmentations_.end()) {
        return false; // Already exists
    }

    // Build the path to the segment
    std::filesystem::path segPath = rootDir_ / currentSegmentationDir_ / id;

    if (!std::filesystem::exists(segPath) || !std::filesystem::is_directory(segPath)) {
        Logger()->warn("Segment directory does not exist: {}", segPath.string());
        return false;
    }

    try {
        auto s = Segmentation::New(segPath);
        if (!volumes_.empty()) {
            auto scrollName = config_["name"].get<std::string>();
            auto volumeUuid = volumes_.begin()->second->id();
            s->ensureScrollSource(scrollName, volumeUuid);
        }
        segmentations_.emplace(s->id(), s);
        segmentationDirectories_[s->id()] = currentSegmentationDir_;
        Logger()->info("Added segmentation: {}", id);
        return true;
    } catch (const std::exception& e) {
        Logger()->error("Failed to add segmentation {}: {}", id, e.what());
        return false;
    }
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

    // Remove if it exists (this also unloads the surface)
    removeSingleSegmentation(id);

    // Add it back
    return addSingleSegmentation(id);
}