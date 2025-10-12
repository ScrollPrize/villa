#include "vc/core/types/VolumePkg.hpp"

#include <set>
#include <utility>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Logging.hpp"

constexpr auto CONFIG = "config.json";

inline auto VolsDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "volumes";
}

inline auto SegsDir(const std::filesystem::path& baseDir, const std::string& dirName = "paths") -> std::filesystem::path
{
    return baseDir / dirName;
}

inline auto RendDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "renders";
}

inline auto TfmDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "transforms";
}

inline auto PreviewDirs(const std::filesystem::path& baseDir) -> std::vector<std::filesystem::path>
{
    return { baseDir / "volumes_preview_half", baseDir / "volumes_masked", baseDir / "volumes_previews"};
}

inline auto ReqDirs(const std::filesystem::path& baseDir) -> std::vector<std::filesystem::path>
{
    return {baseDir, VolsDir(baseDir), SegsDir(baseDir), RendDir(baseDir), TfmDir(baseDir)};
}

inline void keep(const std::filesystem::path& dir)
{
    if (not std::filesystem::exists(dir / ".vckeep")) {
        std::ofstream(dir / ".vckeep", std::ostream::ate);
    }
}

VolumePkg::VolumePkg(const std::filesystem::path& fileLocation) : rootDir_{fileLocation}
{
    config_ = Metadata(fileLocation / ::CONFIG);

    for (const auto& d : ::ReqDirs(rootDir_)) {
        if (not std::filesystem::exists(d)) {
            Logger()->warn(
                "Creating missing VolumePkg directory: {}",
                d.filename().string());
            std::filesystem::create_directory(d);
        }
        if (d != rootDir_) {
            ::keep(d);
        }
    }

    for (const auto& entry : std::filesystem::directory_iterator(::VolsDir(rootDir_))) {
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
}

auto VolumePkg::New(std::filesystem::path fileLocation) -> std::shared_ptr<VolumePkg>
{
    return std::make_shared<VolumePkg>(fileLocation);
}


auto VolumePkg::name() const -> std::string
{
    // Gets the Volume name from the configuration file
    auto name = config_.get<std::string>("name");
    if (name != "NULL") {
        return name;
    }

    return "UnnamedVolume";
}

auto VolumePkg::version() const -> int { return config_.get<int>("version"); }

auto VolumePkg::materialThickness() const -> double
{
    return config_.get<double>("materialthickness");
}

auto VolumePkg::metadata() const -> Metadata { return config_; }

void VolumePkg::saveMetadata() { config_.save(); }

void VolumePkg::saveMetadata(const std::filesystem::path& filePath)
{
    config_.save(filePath);
}

// VOLUME FUNCTIONS //
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

std::vector<std::string> VolumePkg::volumeNames() const
{
    std::vector<std::string> names;
    for (const auto& v : volumes_) {
        names.emplace_back(v.second->name());
    }
    return names;
}

std::shared_ptr<Volume> VolumePkg::newVolume(std::string name)
{
    // Generate a uuid
    auto uuid = DateTime();

    // Get dir name if not specified
    if (name.empty()) {
        name = uuid;
    }

    // Make the volume directory
    auto volDir = ::VolsDir(rootDir_) / uuid;
    if (!std::filesystem::exists(volDir)) {
        std::filesystem::create_directory(volDir);
    } else {
        throw std::runtime_error("Volume directory already exists");
    }

    // Make the volume
    auto r = volumes_.emplace(uuid, Volume::New(volDir, uuid, name));
    if (!r.second) {
        auto msg = "Volume already exists with ID " + uuid;
        throw std::runtime_error(msg);
    }

    // Return the Volume Pointer
    return r.first->second;
}

const std::shared_ptr<Volume> VolumePkg::volume() const
{
    if (volumes_.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return volumes_.begin()->second;
}

std::shared_ptr<Volume> VolumePkg::volume()
{
    if (volumes_.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return volumes_.begin()->second;
}

const std::shared_ptr<Volume> VolumePkg::volume(const std::string& id) const
{
    return volumes_.at(id);
}

std::shared_ptr<Volume> VolumePkg::volume(const std::string& id)
{
    return volumes_.at(id);
}

// SEGMENTATION FUNCTIONS //
bool VolumePkg::hasSegmentations() const
{
    return !segmentations_.empty();
}

std::size_t VolumePkg::numberOfSegmentations() const
{
    return segmentations_.size();
}

const std::shared_ptr<Segmentation> VolumePkg::segmentation(const std::string& id) const
{
    return segmentations_.at(id);
}

std::vector<std::filesystem::path> VolumePkg::segmentationFiles()
{
    return segmentation_files_;
}

auto VolumePkg::segmentation(const std::string& id)
    -> std::shared_ptr<Segmentation>
{
    return segmentations_.at(id);
}

auto VolumePkg::segmentationIDs() const -> std::vector<std::string>
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

auto VolumePkg::segmentationNames() const -> std::vector<std::string>
{
    std::vector<std::string> names;
    for (const auto& s : segmentations_) {
        names.emplace_back(s.second->name());
    }
    return names;
}



// SEGMENTATION DIRECTORY METHODS //
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
        
        // Remove from files vector
        auto it = std::remove_if(segmentation_files_.begin(), segmentation_files_.end(),
            [&id, this](const std::filesystem::path& path) {
                auto segIt = segmentations_.find(id);
                return segIt != segmentations_.end() && segIt->second->path() == path;
            });
        segmentation_files_.erase(it, segmentation_files_.end());
    }
    
    // Check if directory exists
    const auto segDir = ::SegsDir(rootDir_, dirName);
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", dirName);
        return;
    }
    
    // Load segmentations from the specified directory
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            try {
                auto s = Segmentation::New(dirpath);
                segmentations_.emplace(s->id(), s);
                segmentation_files_.push_back(dirpath);
                // Track which directory this segmentation came from
                segmentationDirectories_[s->id()] = dirName;
            }
            catch (const std::exception &exc) {
                std::cout << "WARNING: some exception occured, skipping segment dir: " << dirpath << std::endl;
                std::cerr << exc.what();
            }
        }
    }
}

void VolumePkg::setSegmentationDirectory(const std::string& dirName)
{
    // Just change the current directory - all segmentations are already loaded
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
    
    // Remove from files vector
    auto fileIt = std::find(segmentation_files_.begin(), 
                           segmentation_files_.end(), segPath);
    if (fileIt != segmentation_files_.end()) {
        segmentation_files_.erase(fileIt);
    }
    
    // Delete the physical folder
    if (std::filesystem::exists(segPath)) {
        std::filesystem::remove_all(segPath);
    }
}

void VolumePkg::refreshSegmentations()
{
    const auto segDir = ::SegsDir(rootDir_, currentSegmentationDir_);
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", currentSegmentationDir_);
        return;
    }
    
    // Build a set of current segmentation paths on disk for the current directory
    std::set<std::filesystem::path> diskPaths;
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
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
        
        // Remove from files vector if we have a path
        if (!segPath.empty()) {
            auto fileIt = std::find(segmentation_files_.begin(), 
                                  segmentation_files_.end(), 
                                  segPath);
            if (fileIt != segmentation_files_.end()) {
                segmentation_files_.erase(fileIt);
            }
        }
    }
    
    // Find and add new segmentations (on disk but not in memory)
    for (const auto& diskPath : diskPaths) {
        bool found = false;
        for (const auto& seg : segmentations_) {
            if (seg.second->path() == diskPath) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            try {
                auto s = Segmentation::New(diskPath);
                segmentations_.emplace(s->id(), s);
                segmentation_files_.push_back(diskPath);
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
    return loadedSurfaces_.count(id) > 0;
}

std::shared_ptr<SurfaceMeta> VolumePkg::loadSurface(const std::string& id)
{
    // Check if already loaded
    if (auto it = loadedSurfaces_.find(id); it != loadedSurfaces_.end()) {
        return it->second;
    }

    // Check if segmentation exists
    auto segIt = segmentations_.find(id);
    if (segIt == segmentations_.end()) {
        throw std::runtime_error("Segmentation not found: " + id);
    }

    auto seg = segIt->second;

    // Check if it's the right format
    if (!seg->metadata().hasKey("format") ||
        seg->metadata().get<std::string>("format") != "tifxyz") {
        return nullptr;
    }

    // Load the surface
    try {
        auto sm = std::make_shared<SurfaceMeta>(seg->path());
        sm->surface();
        sm->readOverlapping();
        loadedSurfaces_[id] = sm;
        return sm;
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface {}: {}", id, e.what());
        return nullptr;
    }
}

std::shared_ptr<SurfaceMeta> VolumePkg::getSurface(const std::string& id)
{
    if (auto it = loadedSurfaces_.find(id); it != loadedSurfaces_.end()) {
        return it->second;
    }
    return nullptr;
}


std::vector<std::string> VolumePkg::getLoadedSurfaceIDs() const
{
    std::vector<std::string> ids;
    ids.reserve(loadedSurfaces_.size());
    for (const auto& [id, _] : loadedSurfaces_) {
        ids.push_back(id);
    }
    return ids;
}

void VolumePkg::unloadAllSurfaces()
{
    loadedSurfaces_.clear();
}

bool VolumePkg::unloadSurface(const std::string& id)
{
    auto it = loadedSurfaces_.find(id);
    if (it != loadedSurfaces_.end()) {
        loadedSurfaces_.erase(it);
        Logger()->debug("Unloaded surface: {}", id);
        return true;
    }
    return false;
}


void VolumePkg::loadSurfacesBatch(const std::vector<std::string>& ids)
{
    std::vector<std::pair<std::string, std::shared_ptr<SurfaceMeta>>> toLoad;

    // Determine which surfaces need loading
    for (const auto& id : ids) {
        if (!isSurfaceLoaded(id)) {
            auto segIt = segmentations_.find(id);
            if (segIt != segmentations_.end() &&
                segIt->second->metadata().hasKey("format") &&
                segIt->second->metadata().get<std::string>("format") == "tifxyz") {
                toLoad.push_back({id, nullptr});
                }
        }
    }

    // Parallel loading
#pragma omp parallel for
    for (size_t i = 0; i < toLoad.size(); i++) {
        try {
            auto seg = segmentations_.at(toLoad[i].first);
            auto sm = std::make_shared<SurfaceMeta>(seg->path());
            sm->surface();
            sm->readOverlapping();
            toLoad[i].second = sm;
        } catch (const std::exception& e) {
            Logger()->error("Failed to load surface {}: {}", toLoad[i].first, e.what());
        }
    }

    // Store loaded surfaces
    {
        for (const auto& [id, surface] : toLoad) {
            if (surface) {
                loadedSurfaces_[id] = surface;
            }
        }
    }
}