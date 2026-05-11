#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pwd.h>
#include <unistd.h>

#include "vc/core/util/Logging.hpp"
#include "vc/core/util/RemoteUrl.hpp"

namespace fs = std::filesystem;

std::filesystem::path VolumePkg::autosaveRoot_;
std::optional<std::string> VolumePkg::loadFirstSegmentationDir_{};

namespace vc::project {

bool isLocationRemote(const std::string& location)
{
    if (location.rfind("s3://", 0) == 0) return true;
    if (location.rfind("s3+", 0) == 0) return true;
    if (location.rfind("http://", 0) == 0) return true;
    if (location.rfind("https://", 0) == 0) return true;
    return false;
}

fs::path resolveLocalPath(const std::string& location, const fs::path& base)
{
    constexpr const char* kFile = "file://";
    fs::path p = (location.rfind(kFile, 0) == 0)
        ? fs::path(location.substr(std::strlen(kFile)))
        : fs::path(location);
    if (p.is_absolute() || base.empty()) return p;
    return base / p;
}

}

namespace {

std::string asciiLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool hasZarrMarkerAtRoot(const fs::path& dir)
{
    return fs::exists(dir / ".zarray")
        || fs::exists(dir / ".zgroup")
        || fs::exists(dir / ".zattrs")
        || fs::exists(dir / "zarr.json");
}

bool isSingleZarrVolumeDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    bool meta = fs::exists(dir / "meta.json")
             || fs::exists(dir / "metadata.json")
             || hasZarrMarkerAtRoot(dir);
    if (!meta) return false;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_directory() && fs::exists(e.path() / ".zarray")) return true;
    }
    return false;
}

bool isSegmentDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    return Segmentation::checkDir(dir);
}

bool looksLikeTifxyzDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    return fs::exists(dir / "x.tif")
        && fs::exists(dir / "y.tif")
        && fs::exists(dir / "z.tif");
}

bool isNormalGridDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    return fs::is_directory(dir / "xy")
        && fs::is_directory(dir / "xz")
        && fs::is_directory(dir / "yz")
        && fs::exists(dir / "metadata.json");
}

bool isDirectRemoteZarrLocation(std::string location)
{
    location = asciiLower(std::move(location));
    const auto fragment = location.find('#');
    if (fragment != std::string::npos) location.erase(fragment);
    const auto query = location.find('?');
    if (query != std::string::npos) location.erase(query);
    while (!location.empty() && location.back() == '/') location.pop_back();
    constexpr std::string_view suffix = ".zarr";
    return location.size() >= suffix.size()
        && location.compare(location.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<fs::path> immediateSubdirs(const fs::path& dir)
{
    std::vector<fs::path> out;
    if (!fs::is_directory(dir)) return out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_directory()) continue;
        const auto name = e.path().filename().string();
        if (name.empty() || name[0] == '.' || name == ".tmp") continue;
        out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool anyImmediateSubdir(const fs::path& dir, bool (*test)(const fs::path&))
{
    for (const auto& child : immediateSubdirs(dir)) {
        if (test(child)) return true;
    }
    return false;
}

std::string trimTrailingSeparators(std::string value)
{
    while (value.size() > 1 && (value.back() == '/' || value.back() == '\\')) {
        value.pop_back();
    }
    return value;
}

fs::path normalizedLocalPath(const std::string& location, const fs::path& base)
{
    return vc::project::resolveLocalPath(trimTrailingSeparators(location), base).lexically_normal();
}

std::string normalizedPathName(std::string value)
{
    value = trimTrailingSeparators(std::move(value));
    fs::path path(value);
    std::string name = path.filename().string();
    if (name.empty() && path.has_parent_path()) {
        name = path.parent_path().filename().string();
    }
    return name;
}

bool sameLocalSegmentsLocation(const vc::project::Entry& entry,
                               const std::string& location,
                               const fs::path& base)
{
    if (entry.location == location) return true;
    if (vc::project::isLocationRemote(entry.location) || vc::project::isLocationRemote(location)) {
        return false;
    }
    return normalizedLocalPath(entry.location, base) == normalizedLocalPath(location, base);
}

bool matchesSegmentsDirectoryName(const vc::project::Entry& entry,
                                  const std::string& dirName,
                                  const fs::path& base)
{
    if (vc::project::isLocationRemote(entry.location)) return false;
    const auto requested = asciiLower(trimTrailingSeparators(dirName));
    const auto requestedName = asciiLower(normalizedPathName(dirName));
    const auto entryPath = normalizedLocalPath(entry.location, base);
    return asciiLower(entryPath.filename().string()) == requested
        || (!requestedName.empty() && asciiLower(entryPath.filename().string()) == requestedName)
        || asciiLower(entryPath.string()) == requested
        || asciiLower(trimTrailingSeparators(entry.location)) == requested;
}

const vc::project::Entry* findSegmentsEntryByLocation(const std::vector<vc::project::Entry>& entries,
                                                      const std::string& location,
                                                      const fs::path& base)
{
    if (location.empty()) return nullptr;
    for (const auto& entry : entries) {
        if (sameLocalSegmentsLocation(entry, location, base)) return &entry;
    }
    return nullptr;
}

const vc::project::Entry* findSegmentsEntryByDirectoryName(const std::vector<vc::project::Entry>& entries,
                                                           const std::string& dirName,
                                                           const fs::path& base)
{
    if (dirName.empty()) return nullptr;
    for (const auto& entry : entries) {
        if (matchesSegmentsDirectoryName(entry, dirName, base)) return &entry;
    }
    return nullptr;
}

const vc::project::Entry* firstLocalSegmentsEntry(const std::vector<vc::project::Entry>& entries)
{
    for (const auto& entry : entries) {
        if (!vc::project::isLocationRemote(entry.location)) return &entry;
    }
    return nullptr;
}

fs::path defaultAutosaveRoot()
{
    if (!VolumePkg::autosaveRoot().empty()) return VolumePkg::autosaveRoot();
    const struct passwd* pw = getpwuid(geteuid());
    if (pw == nullptr || pw->pw_dir == nullptr || pw->pw_dir[0] == '\0') return {};
    return fs::path(pw->pw_dir) / ".VC3D";
}

void atomicWriteString(const fs::path& target, const std::string& text)
{
    fs::create_directories(target.parent_path());
    auto tmp = target;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open " + tmp.string() + " for write");
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) throw std::runtime_error("write failed for " + tmp.string());
    }
    fs::rename(tmp, target);
}

utils::Json entriesToJson(const std::vector<vc::project::Entry>& entries)
{
    auto arr = utils::Json::array();
    for (const auto& e : entries) {
        if (e.tags.empty()) {
            arr.push_back(utils::Json(e.location));
        } else {
            auto obj = utils::Json::object();
            obj["location"] = e.location;
            auto t = utils::Json::array();
            for (const auto& s : e.tags) t.push_back(utils::Json(s));
            obj["tags"] = t;
            arr.push_back(obj);
        }
    }
    return arr;
}

std::vector<vc::project::Entry> entriesFromJson(const utils::Json& arr)
{
    std::vector<vc::project::Entry> out;
    if (!arr.is_array()) return out;
    for (const auto& v : arr) {
        vc::project::Entry e;
        if (v.is_string()) {
            e.location = v.get_string();
        } else if (v.is_object()) {
            e.location = v.at("location").get_string();
            if (v.contains("tags")) {
                e.tags = v.at("tags").get_string_array();
            }
        } else {
            continue;
        }
        if (!e.location.empty()) out.push_back(std::move(e));
    }
    return out;
}

}

namespace vc::project {

std::string validateLocation(Category category, const std::string& location)
{
    if (location.empty()) return "Location is empty.";

    if (isLocationRemote(location)) {
        if (category != Category::Volumes) {
            return "Remote locations are only supported for volumes.";
        }
        const auto schemeEnd = location.find("://");
        if (schemeEnd == std::string::npos) {
            return "Remote URL is missing scheme separator (expected '://').";
        }
        if (location.size() <= schemeEnd + 3) {
            return "Remote URL is missing host/bucket after scheme.";
        }
        return {};
    }

    const auto path = resolveLocalPath(location);
    std::error_code ec;
    if (!fs::exists(path, ec)) return "Path does not exist: " + path.string();
    if (!fs::is_directory(path, ec)) return "Path is not a directory: " + path.string();

    switch (category) {
        case Category::Volumes:
            if (isSingleZarrVolumeDir(path)) return {};
            if (anyImmediateSubdir(path, &isSingleZarrVolumeDir)) return {};
            return "Not a zarr volume and contains no zarr volumes (expected meta.json + chunk-level .zarray).";
        case Category::Segments:
            if (isSegmentDir(path)) return {};
            if (anyImmediateSubdir(path, &isSegmentDir)) return {};
            return "Not a segment directory and contains no segments (expected tifxyz layout with meta.json).";
        case Category::NormalGrids:
            if (isNormalGridDir(path)) return {};
            if (anyImmediateSubdir(path, &isNormalGridDir)) return {};
            return "Not a normal-grid directory (expected xy/, xz/, yz/ subdirs and metadata.json).";
    }
    return "Unknown category.";
}

}

VolumePkg::VolumePkg() = default;

VolumePkg::~VolumePkg()
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    segmentsChangedCb_ = nullptr;
}

void VolumePkg::setAutosaveRoot(const fs::path& dir) { autosaveRoot_ = dir; }
fs::path VolumePkg::autosaveRoot() { return autosaveRoot_; }

void VolumePkg::setLoadFirstSegmentationDirectory(const std::string& dirName)
{
    if (dirName.empty()) {
        loadFirstSegmentationDir_.reset();
        return;
    }
    loadFirstSegmentationDir_ = dirName;
}

fs::path VolumePkg::autosaveFile()
{
    const auto root = defaultAutosaveRoot();
    if (root.empty()) return {};
    return root / "current_project.json";
}

std::shared_ptr<VolumePkg> VolumePkg::newEmpty()
{
    return std::shared_ptr<VolumePkg>(new VolumePkg());
}

std::shared_ptr<VolumePkg> VolumePkg::load(const fs::path& jsonFile,
                                           const vc::project::LoadOptions& opts)
{
    auto p = std::shared_ptr<VolumePkg>(new VolumePkg());
    p->opts_ = opts;
    p->path_ = jsonFile;
    p->readJsonFrom(jsonFile);
    p->resolveAll();
    return p;
}

std::shared_ptr<VolumePkg> VolumePkg::loadAutosave(const vc::project::LoadOptions& opts)
{
    const auto file = autosaveFile();
    if (file.empty() || !fs::exists(file)) return nullptr;
    auto p = std::shared_ptr<VolumePkg>(new VolumePkg());
    p->opts_ = opts;
    p->path_ = file;
    p->readJsonFrom(file);
    p->resolveAll();
    return p;
}

std::shared_ptr<VolumePkg> VolumePkg::New(const fs::path& jsonFile)
{
    return load(jsonFile);
}

fs::path VolumePkg::path() const { return path_; }
std::string VolumePkg::name() const { return name_; }
void VolumePkg::setName(const std::string& v) { name_ = v; persistProjectState(); }
int VolumePkg::version() const { return version_; }

const std::vector<vc::project::Entry>& VolumePkg::volumeEntries() const { return volumes_; }
const std::vector<vc::project::Entry>& VolumePkg::segmentEntries() const { return segments_; }
const std::vector<vc::project::Entry>& VolumePkg::normalGridEntries() const { return normalGrids_; }

bool VolumePkg::addVolumeEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& e : volumes_) if (e.location == location) return false;
    volumes_.push_back({location, std::move(tags)});
    resolveVolumeEntry(volumes_.back());
    persistProjectState();
    return true;
}

bool VolumePkg::addSegmentsEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& e : segments_) if (e.location == location) return false;
    segments_.push_back({location, std::move(tags)});
    if (!outputSegments_) {
        outputSegments_ = location;
    }
    refreshSegmentations();
    persistProjectState();
    return true;
}

bool VolumePkg::addNormalGridEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& e : normalGrids_) if (e.location == location) return false;
    normalGrids_.push_back({location, std::move(tags)});
    resolveNormalGridEntry(normalGrids_.back());
    persistProjectState();
    return true;
}

bool VolumePkg::removeEntry(const std::string& location)
{
    auto eraseFrom = [&](std::vector<vc::project::Entry>& v) {
        auto it = std::find_if(v.begin(), v.end(),
                               [&](const auto& e) { return e.location == location; });
        if (it == v.end()) return false;
        v.erase(it);
        return true;
    };
    bool removed = false;
    if (eraseFrom(volumes_)) removed = true;
    if (eraseFrom(segments_)) removed = true;
    if (eraseFrom(normalGrids_)) removed = true;
    if (removed) {
        if (outputSegments_ && *outputSegments_ == location) outputSegments_.reset();
        resolveAll();
        persistProjectState();
    }
    return removed;
}

void VolumePkg::setOutputSegments(const std::string& location)
{
    outputSegments_ = location;
    refreshSegmentations();
    persistProjectState();
}

void VolumePkg::clearOutputSegments()
{
    outputSegments_.reset();
    persistProjectState();
}

bool VolumePkg::hasOutputSegments() const { return outputSegments_.has_value(); }

fs::path VolumePkg::outputSegmentsPath() const
{
    if (!outputSegments_) return {};
    if (vc::project::isLocationRemote(*outputSegments_)) return {};
    return vc::project::resolveLocalPath(*outputSegments_, path_.parent_path());
}

bool VolumePkg::hasVolumes() const { return !loadedVolumes_.empty(); }
bool VolumePkg::hasVolume(const std::string& id) const { return loadedVolumes_.count(id) > 0; }
std::size_t VolumePkg::numberOfVolumes() const { return loadedVolumes_.size(); }

std::vector<std::string> VolumePkg::volumeIDs() const
{
    std::vector<std::string> out;
    out.reserve(loadedVolumes_.size());
    for (const auto& [id, _] : loadedVolumes_) out.push_back(id);
    return out;
}

std::shared_ptr<Volume> VolumePkg::volume(const std::string& id)
{
    auto it = loadedVolumes_.find(id);
    if (it == loadedVolumes_.end()) return nullptr;
    return it->second;
}

std::shared_ptr<Volume> VolumePkg::volume()
{
    if (loadedVolumes_.empty()) return nullptr;
    return loadedVolumes_.begin()->second;
}

bool VolumePkg::addVolume(const std::shared_ptr<Volume>& volume)
{
    if (!volume) {
        Logger()->warn("Cannot add null volume to package");
        return false;
    }

    const auto id = volume->id();
    auto result = loadedVolumes_.emplace(id, volume);
    if (!result.second) {
        Logger()->warn("Volume '{}' already exists in package", id);
        return false;
    }

    const auto source = volume->isRemote()
        ? volume->remoteUrl()
        : volume->path().string();
    Logger()->info("Added external volume '{}' from '{}'", id, source);
    return true;
}

bool VolumePkg::addSingleVolume(const std::string& volumeDirName)
{
    if (volumeDirName.empty()) return false;
    for (const auto& e : volumes_) {
        if (vc::project::isLocationRemote(e.location)) continue;
        const auto base = vc::project::resolveLocalPath(e.location, path_.parent_path());
        const auto candidate = base / volumeDirName;
        if (!isSingleZarrVolumeDir(candidate)) continue;
        try {
            auto v = Volume::New(candidate);
            const auto id = v->id();
            if (loadedVolumes_.count(id) > 0) return false;
            loadedVolumes_.emplace(id, v);
            if (!e.tags.empty()) volumeTagsByID_[id] = e.tags;
            return true;
        } catch (const std::exception& ex) {
            Logger()->warn("addSingleVolume('{}'): {}", volumeDirName, ex.what());
            return false;
        }
    }
    return false;
}

bool VolumePkg::removeSingleVolume(const std::string& volumeIdOrDirName)
{
    if (loadedVolumes_.erase(volumeIdOrDirName) > 0) {
        volumeTagsByID_.erase(volumeIdOrDirName);
        return true;
    }
    for (auto it = loadedVolumes_.begin(); it != loadedVolumes_.end(); ++it) {
        if (it->second && it->second->path().filename().string() == volumeIdOrDirName) {
            const auto id = it->first;
            loadedVolumes_.erase(it);
            volumeTagsByID_.erase(id);
            return true;
        }
    }
    return false;
}

bool VolumePkg::reloadSingleVolume(const std::string& volumeId)
{
    auto it = loadedVolumes_.find(volumeId);
    if (it == loadedVolumes_.end() || !it->second) return false;
    const auto vp = it->second->path();
    loadedVolumes_.erase(it);
    volumeTagsByID_.erase(volumeId);
    try {
        auto v = Volume::New(vp);
        loadedVolumes_.emplace(v->id(), v);
        return true;
    } catch (const std::exception& ex) {
        Logger()->warn("reloadSingleVolume('{}'): {}", volumeId, ex.what());
        return false;
    }
}

bool VolumePkg::hasSegmentations() const
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    return !loadedSegmentations_.empty();
}

std::vector<std::string> VolumePkg::segmentationIDs() const
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    std::vector<std::string> out;
    out.reserve(loadedSegmentations_.size());
    for (const auto& [id, _] : loadedSegmentations_) out.push_back(id);
    return out;
}

std::shared_ptr<Segmentation> VolumePkg::segmentation(const std::string& id)
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    auto it = loadedSegmentations_.find(id);
    if (it == loadedSegmentations_.end()) return nullptr;
    return it->second;
}

void VolumePkg::removeSegmentation(const std::string& id)
{
    fs::path segPath;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) {
            throw std::runtime_error("Segmentation not found: " + id);
        }
        segPath = it->second->path();
        loadedSegmentations_.erase(it);
        segmentationTagsByID_.erase(id);
    }
    if (fs::exists(segPath)) fs::remove_all(segPath);
}

std::vector<fs::path> VolumePkg::normalGridPaths() const
{
    return resolvedNormalGridPaths_;
}

std::vector<fs::path> VolumePkg::normal3dZarrPaths() const
{
    std::vector<fs::path> out;
    for (const auto& [id, tags] : volumeTagsByID_) {
        if (std::find(tags.begin(), tags.end(), "normal3d") == tags.end()) continue;
        auto it = loadedVolumes_.find(id);
        if (it == loadedVolumes_.end()) continue;
        out.push_back(it->second->path());
    }
    return out;
}

std::vector<std::string> VolumePkg::volumeTags(const std::string& volumeId) const
{
    auto it = volumeTagsByID_.find(volumeId);
    if (it == volumeTagsByID_.end()) return {};
    return it->second;
}

std::vector<std::string> VolumePkg::segmentationTags(const std::string& segmentId) const
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    auto it = segmentationTagsByID_.find(segmentId);
    if (it == segmentationTagsByID_.end()) return {};
    return it->second;
}

bool VolumePkg::isSurfaceLoaded(const std::string& id) const
{
    std::shared_ptr<Segmentation> seg;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) return false;
        seg = it->second;
    }
    return seg->isSurfaceLoaded();
}

std::shared_ptr<QuadSurface> VolumePkg::loadSurface(const std::string& id)
{
    std::shared_ptr<Segmentation> seg;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) {
            Logger()->error("Cannot load surface - segmentation {} not found", id);
            return nullptr;
        }
        seg = it->second;
    }
    try {
        return seg->loadSurface();
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface for {}: {}", id, e.what());
    } catch (...) {
        Logger()->error("Failed to load surface for {}: unknown error", id);
    }
    return nullptr;
}

std::shared_ptr<QuadSurface> VolumePkg::getSurface(const std::string& id)
{
    std::shared_ptr<Segmentation> seg;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) return nullptr;
        seg = it->second;
    }
    return seg->getSurface();
}

bool VolumePkg::unloadSurface(const std::string& id)
{
    std::shared_ptr<Segmentation> seg;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) return false;
        seg = it->second;
    }
    seg->unloadSurface();
    return true;
}

std::vector<std::string> VolumePkg::getLoadedSurfaceIDs() const
{
    std::vector<std::shared_ptr<Segmentation>> snapshot;
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        snapshot.reserve(loadedSegmentations_.size());
        ids.reserve(loadedSegmentations_.size());
        for (const auto& [id, seg] : loadedSegmentations_) {
            ids.push_back(id);
            snapshot.push_back(seg);
        }
    }
    std::vector<std::string> out;
    out.reserve(snapshot.size());
    for (size_t i = 0; i < snapshot.size(); ++i) {
        if (snapshot[i]->isSurfaceLoaded()) out.push_back(ids[i]);
    }
    return out;
}

void VolumePkg::unloadAllSurfaces()
{
    std::vector<std::shared_ptr<Segmentation>> snapshot;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        snapshot.reserve(loadedSegmentations_.size());
        for (auto& [id, seg] : loadedSegmentations_) snapshot.push_back(seg);
    }
    for (auto& seg : snapshot) seg->unloadSurface();
}

void VolumePkg::loadSurfacesBatch(const std::vector<std::string>& ids)
{
    std::vector<std::shared_ptr<Segmentation>> toLoad;
    toLoad.reserve(ids.size());
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        for (const auto& id : ids) {
            auto it = loadedSegmentations_.find(id);
            if (it == loadedSegmentations_.end()) continue;
            if (it->second->isSurfaceLoaded() || !it->second->canLoadSurface()) continue;
            toLoad.push_back(it->second);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (auto& seg : toLoad) {
        try {
            seg->loadSurface();
        } catch (const std::exception& e) {
            Logger()->error("Failed to load surface for {}: {}", seg->id(), e.what());
        }
    }
}

bool VolumePkg::isRemote() const
{
    auto anyRemote = [](const std::vector<vc::project::Entry>& v) {
        return std::any_of(v.begin(), v.end(),
                           [](const auto& e) { return vc::project::isLocationRemote(e.location); });
    };
    return anyRemote(volumes_) || anyRemote(normalGrids_);
}

bool VolumePkg::hasRemoteCacheRoot() const
{
    return !remoteCacheRoot_.empty();
}

std::string VolumePkg::remoteCacheRootOrEmpty() const
{
    return remoteCacheRoot_.string();
}

void VolumePkg::setRemoteCacheRoot(const fs::path& dir)
{
    remoteCacheRoot_ = dir;
    opts_.remoteCacheRoot = dir;
    persistProjectState();
}

void VolumePkg::save(const fs::path& target)
{
    writeJsonTo(target);
    path_ = target;
}

void VolumePkg::saveAutosave()
{
    const auto file = autosaveFile();
    if (file.empty()) return;
    writeJsonTo(file);
}

void VolumePkg::persistProjectState()
{
    saveAutosave();
    if (!path_.empty()) {
        writeJsonTo(path_);
    }
}

void VolumePkg::resolveAll()
{
    loadedVolumes_.clear();
    volumeTagsByID_.clear();
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        loadedSegmentations_.clear();
        segmentationTagsByID_.clear();
    }
    resolvedNormalGridPaths_.clear();
    for (const auto& e : volumes_) resolveVolumeEntry(e);

    const vc::project::Entry* selectedSegments = nullptr;
    if (loadFirstSegmentationDir_ && !loadFirstSegmentationDir_->empty()) {
        selectedSegments = findSegmentsEntryByDirectoryName(
            segments_, *loadFirstSegmentationDir_, path_.parent_path());
        if (!selectedSegments) {
            Logger()->warn("Requested load-first segmentation directory '{}' not available; using the selected segmentation directory.",
                           *loadFirstSegmentationDir_);
        }
    }
    if (!selectedSegments && outputSegments_) {
        selectedSegments = findSegmentsEntryByLocation(segments_, *outputSegments_, path_.parent_path());
    }
    if (!selectedSegments) {
        selectedSegments = firstLocalSegmentsEntry(segments_);
    }
    if (selectedSegments) {
        outputSegments_ = selectedSegments->location;
        resolveSegmentsEntry(*selectedSegments);
    }
    for (const auto& e : normalGrids_) resolveNormalGridEntry(e);
}

void VolumePkg::resolveVolumeEntry(const vc::project::Entry& e)
{
    if (vc::project::isLocationRemote(e.location)) {
        if (!isDirectRemoteZarrLocation(e.location)) {
            Logger()->warn("Skipping remote volume collection '{}': remote listing is not supported in this branch", e.location);
            return;
        }

        try {
            auto v = Volume::NewFromUrl(e.location, opts_.remoteCacheRoot, {});
            const auto id = v->id();
            if (loadedVolumes_.count(id) > 0) {
                Logger()->warn("Duplicate remote volume id '{}' from '{}', skipping", id, e.location);
                return;
            }
            loadedVolumes_.emplace(id, v);
            if (!e.tags.empty()) volumeTagsByID_[id] = e.tags;
            return;
        } catch (const std::exception& ex) {
            if (opts_.failOnRemoteError) {
                throw;
            }
            Logger()->warn("Failed to load remote zarr volume '{}': {}", e.location, ex.what());
        }

        return;
    }

    const auto path = vc::project::resolveLocalPath(e.location, path_.parent_path());
    if (!fs::exists(path)) {
        Logger()->warn("Skipping volume '{}': path does not exist", e.location);
        return;
    }
    auto loadOne = [&](const fs::path& vp) {
        try {
            auto v = Volume::New(vp);
            const auto id = v->id();
            if (loadedVolumes_.count(id) > 0) {
                Logger()->warn("Duplicate volume id '{}' from '{}', skipping", id, vp.string());
                return;
            }
            loadedVolumes_.emplace(id, v);
            if (!e.tags.empty()) volumeTagsByID_[id] = e.tags;
        } catch (const std::exception& ex) {
            Logger()->warn("Failed to load volume '{}': {}", vp.string(), ex.what());
        }
    };
    if (isSingleZarrVolumeDir(path)) {
        loadOne(path);
    } else {
        for (const auto& child : immediateSubdirs(path)) {
            if (isSingleZarrVolumeDir(child)) loadOne(child);
        }
    }
}

void VolumePkg::resolveSegmentsEntry(const vc::project::Entry& e)
{
    const auto path = vc::project::resolveLocalPath(e.location, path_.parent_path());
    if (!fs::exists(path)) {
        Logger()->warn("Skipping segments '{}': path does not exist", e.location);
        return;
    }
    auto loadOne = [&](const fs::path& sp) {
        try {
            auto s = Segmentation::New(sp);
            const auto id = s->id();
            std::lock_guard<std::mutex> lk(segmentsMutex_);
            if (loadedSegmentations_.count(id) > 0) {
                Logger()->warn("Duplicate segment id '{}' from '{}', skipping", id, sp.string());
                return;
            }
            loadedSegmentations_.emplace(id, s);
            if (!e.tags.empty()) segmentationTagsByID_[id] = e.tags;
        } catch (const std::exception& ex) {
            Logger()->warn("Failed to load segment '{}': {}", sp.string(), ex.what());
        }
    };
    if (isSegmentDir(path)) {
        loadOne(path);
    } else {
        for (const auto& child : immediateSubdirs(path)) {
            if (isSegmentDir(child)) {
                loadOne(child);
            } else if (looksLikeTifxyzDir(child)) {
                Logger()->warn("Skipping segment '{}': missing meta.json", child.string());
            }
        }
    }
}

void VolumePkg::setSegmentsChangedCallback(std::function<void()> cb)
{
    std::lock_guard<std::mutex> lk(segmentsMutex_);
    segmentsChangedCb_ = std::move(cb);
}

void VolumePkg::notifySegmentsChanged()
{
    std::function<void()> cb;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        cb = segmentsChangedCb_;
    }
    if (cb) cb();
}

void VolumePkg::resolveNormalGridEntry(const vc::project::Entry& e)
{
    if (vc::project::isLocationRemote(e.location)) {
        Logger()->warn("Remote normal_grid entry '{}' not yet supported.", e.location);
        return;
    }
    const auto path = vc::project::resolveLocalPath(e.location, path_.parent_path());
    if (!fs::exists(path)) {
        Logger()->warn("Skipping normal_grid '{}': path does not exist", e.location);
        return;
    }
    if (isNormalGridDir(path)) {
        resolvedNormalGridPaths_.push_back(path);
    } else {
        for (const auto& child : immediateSubdirs(path)) {
            if (isNormalGridDir(child)) resolvedNormalGridPaths_.push_back(child);
        }
    }
}

utils::Json VolumePkg::toJson() const
{
    auto j = utils::Json::object();
    j["name"] = name_;
    j["version"] = version_;
    j["volumes"] = entriesToJson(volumes_);
    j["segments"] = entriesToJson(segments_);
    j["normal_grids"] = entriesToJson(normalGrids_);
    if (!remoteCacheRoot_.empty()) j["remote_cache_root"] = remoteCacheRoot_.string();
    if (outputSegments_) j["output_segments"] = *outputSegments_;
    return j;
}

void VolumePkg::fromJson(const utils::Json& j)
{
    name_ = j.value("name", std::string("Untitled"));
    version_ = j.value("version", 1);
    if (j.contains("volumes")) volumes_ = entriesFromJson(j.at("volumes"));
    if (j.contains("segments")) segments_ = entriesFromJson(j.at("segments"));
    if (j.contains("normal_grids")) normalGrids_ = entriesFromJson(j.at("normal_grids"));
    if (j.contains("remote_cache_root")) {
        remoteCacheRoot_ = j.at("remote_cache_root").get_string();
        if (!remoteCacheRoot_.empty()) {
            opts_.remoteCacheRoot = remoteCacheRoot_;
        }
    }
    if (j.contains("output_segments")) outputSegments_ = j.at("output_segments").get_string();
}

void VolumePkg::writeJsonTo(const fs::path& target) const
{
    atomicWriteString(target, toJson().dump(2));
}

void VolumePkg::readJsonFrom(const fs::path& source)
{
    if (!fs::exists(source)) {
        throw std::runtime_error("project file not found: " + source.string());
    }
    auto j = utils::Json::parse_file(source);
    fromJson(j);
}

std::string VolumePkg::getVolpkgDirectory() const
{
    if (path_.empty()) return {};
    return path_.parent_path().string();
}

std::string VolumePkg::getSegmentationDirectory() const
{
    const auto p = outputSegmentsPath();
    if (p.empty()) return {};
    return p.filename().string();
}

std::vector<std::string> VolumePkg::getAvailableSegmentationDirectories() const
{
    std::vector<std::string> out;
    out.reserve(segments_.size());
    for (const auto& e : segments_) {
        if (vc::project::isLocationRemote(e.location)) continue;
        out.push_back(vc::project::resolveLocalPath(e.location, path_.parent_path()).filename().string());
    }
    return out;
}

std::vector<fs::path> VolumePkg::availableSegmentPaths() const
{
    std::vector<fs::path> out;
    out.reserve(segments_.size());
    for (const auto& e : segments_) {
        out.push_back(vc::project::resolveLocalPath(e.location, path_.parent_path()));
    }
    return out;
}

fs::path VolumePkg::findSegmentPathByName(const std::string& dirName) const
{
    for (const auto& e : segments_) {
        const auto p = vc::project::resolveLocalPath(e.location, path_.parent_path());
        if (p.filename().string() == dirName) return p;
    }
    return {};
}

void VolumePkg::setSegmentationDirectory(const std::string& dirName)
{
    if (const auto* entry = findSegmentsEntryByDirectoryName(segments_, dirName, path_.parent_path())) {
        setOutputSegments(entry->location);
        return;
    }
    Logger()->warn("setSegmentationDirectory('{}'): no matching segments entry", dirName);
}

void VolumePkg::refreshSegmentations()
{
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        loadedSegmentations_.clear();
        segmentationTagsByID_.clear();
    }

    const vc::project::Entry* selectedSegments = nullptr;
    if (outputSegments_) {
        selectedSegments = findSegmentsEntryByLocation(segments_, *outputSegments_, path_.parent_path());
    }
    if (!selectedSegments && loadFirstSegmentationDir_ && !loadFirstSegmentationDir_->empty()) {
        selectedSegments = findSegmentsEntryByDirectoryName(
            segments_, *loadFirstSegmentationDir_, path_.parent_path());
        if (!selectedSegments) {
            Logger()->warn("Requested load-first segmentation directory '{}' not available; using the selected segmentation directory.",
                           *loadFirstSegmentationDir_);
        }
    }
    if (!selectedSegments) {
        selectedSegments = firstLocalSegmentsEntry(segments_);
    }
    if (selectedSegments) {
        outputSegments_ = selectedSegments->location;
        resolveSegmentsEntry(*selectedSegments);
    }
}

bool VolumePkg::addSingleSegmentation(const std::string& id)
{
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        if (loadedSegmentations_.count(id) > 0) return false;
    }
    const auto outDir = outputSegmentsPath();
    if (outDir.empty()) return false;
    const auto segPath = outDir / id;
    if (!fs::is_directory(segPath)) return false;
    if (!fs::exists(segPath / "meta.json")) {
        if (looksLikeTifxyzDir(segPath)) {
            Logger()->warn("Skipping segment '{}': missing meta.json", segPath.string());
        }
        return false;
    }
    try {
        auto s = Segmentation::New(segPath);
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        loadedSegmentations_.emplace(s->id(), s);
        return true;
    } catch (const std::exception& ex) {
        Logger()->error("Failed to add segmentation {}: {}", id, ex.what());
        return false;
    }
}

bool VolumePkg::removeSingleSegmentation(const std::string& id)
{
    std::shared_ptr<Segmentation> seg;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = loadedSegmentations_.find(id);
        if (it == loadedSegmentations_.end()) return false;
        seg = it->second;
        loadedSegmentations_.erase(it);
        segmentationTagsByID_.erase(id);
    }
    seg->unloadSurface();
    return true;
}

bool VolumePkg::reloadSingleSegmentation(const std::string& id)
{
    const auto outDir = outputSegmentsPath();
    if (outDir.empty() || !fs::is_directory(outDir / id)) return false;
    removeSingleSegmentation(id);
    return addSingleSegmentation(id);
}
