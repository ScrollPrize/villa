#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/QuadSurface.hpp"

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

void replaceFile(const fs::path& source, const fs::path& destination)
{
#if defined(_WIN32)
    if (!::MoveFileExW(source.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code ec(static_cast<int>(::GetLastError()),
                                 std::system_category());
        throw fs::filesystem_error(
            "cannot replace project file", source, destination, ec);
    }
#else
    fs::rename(source, destination);
#endif
}

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
        if (e.is_directory() && hasZarrMarkerAtRoot(e.path())) return true;
    }
    return false;
}

bool isSegmentDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    return Segmentation::checkDir(dir);
}

bool isNormalGridDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    // A streaming cache dir is a valid store even before any grid files have
    // been fetched — NormalGridVolume pulls them on demand via the marker URL.
    if (fs::is_regular_file(dir / vc::core::util::kNormalGridsRemoteMarker)) return true;
    return fs::is_directory(dir / "xy")
        && fs::is_directory(dir / "xz")
        && fs::is_directory(dir / "yz")
        && fs::exists(dir / "metadata.json");
}

bool isDirectRemoteZarrLocation(std::string location)
{
    location = vc::parseRemoteVolumeSpec(location).sourceUrl;
    location = asciiLower(std::move(location));
    const auto query = location.find('?');
    if (query != std::string::npos) location.erase(query);
    while (!location.empty() && location.back() == '/') location.pop_back();
    constexpr std::string_view suffix = ".zarr";
    return location.size() >= suffix.size()
        && location.compare(location.size() - suffix.size(), suffix.size(), suffix) == 0;
}

constexpr const char* kDirectRemoteZarrRequired =
    "remote Zarr volume locations must point directly to a .zarr root; "
    "collection listing is not supported";

std::string validateRemoteVolumeLocation(
    const std::string& location,
    bool requireDirectZarr)
{
    const auto schemeEnd = location.find("://");
    if (schemeEnd == std::string::npos) {
        return "Remote URL is missing scheme separator (expected '://').";
    }
    if (location.size() <= schemeEnd + 3) {
        return "Remote URL is missing host/bucket after scheme.";
    }
    try {
        (void)vc::parseRemoteVolumeSpec(location);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    if (requireDirectZarr && !isDirectRemoteZarrLocation(location))
        return kDirectRemoteZarrRequired;
    return {};
}

std::string tagValueWithPrefix(const std::vector<std::string>& tags, std::string_view prefix)
{
    for (const auto& tag : tags) {
        if (tag.rfind(prefix, 0) == 0) {
            return tag.substr(prefix.size());
        }
    }
    return {};
}

utils::Json metadataFromVolumeEntryTags(const std::vector<std::string>& tags)
{
    auto metadata = utils::Json::object();
    const auto voxelSizeTag = tagValueWithPrefix(tags, "vc-open-data-voxel-size-um:");
    if (!voxelSizeTag.empty()) {
        try {
            const double voxelSize = std::stod(voxelSizeTag);
            if (voxelSize > 0.0) {
                metadata["voxelsize"] = voxelSize;
            }
        } catch (...) {
        }
    }
    const auto nameTag = tagValueWithPrefix(tags, "vc-open-data-name:");
    if (!nameTag.empty())
        metadata["name"] = nameTag;
    return metadata;
}

bool samePersistedVolumeIdentity(const std::string& a, const std::string& b)
{
    if (a == b)
        return true;
    if (!vc::project::isLocationRemote(a) || !vc::project::isLocationRemote(b))
        return false;
    try {
        const auto aSpec = vc::parseRemoteVolumeSpec(a);
        const auto bSpec = vc::parseRemoteVolumeSpec(b);
        if (!aSpec.hasBaseScaleSelector && !bSpec.hasBaseScaleSelector)
            return false; // Preserve legacy exact-string native deduplication.
        return aSpec.sourceUrl == bSpec.sourceUrl &&
               aSpec.baseScaleLevel == bSpec.baseScaleLevel;
    } catch (...) {
        return false;
    }
}

bool loadedVolumeMatchesLocation(
    const std::shared_ptr<Volume>& volume,
    const std::string& location)
{
    if (!volume)
        return false;
    if (volume->isRemote())
        return samePersistedVolumeIdentity(volume->remoteLocator(), location);
    if (vc::project::isLocationRemote(location))
        return false;
    return volume->path().lexically_normal() ==
           vc::project::resolveLocalPath(location).lexically_normal();
}

std::vector<fs::path> immediateSubdirs(const fs::path& dir)
{
    std::vector<fs::path> out;
    if (!fs::is_directory(dir)) return out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_directory()) continue;
        const auto name = e.path().filename().string();
        if (name.empty() || name[0] == '.' || name == ".tmp") continue;
        if (name.find(".tmp-") != std::string::npos || name.ends_with(".previous")) continue;
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
#if defined(_WIN32)
    const char* home = std::getenv("USERPROFILE");
    if (home == nullptr || home[0] == '\0') return {};
    return fs::path(home) / ".VC3D";
#else
    const struct passwd* pw = getpwuid(geteuid());
    if (pw == nullptr || pw->pw_dir == nullptr || pw->pw_dir[0] == '\0') return {};
    return fs::path(pw->pw_dir) / ".VC3D";
#endif
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
    replaceFile(tmp, target);
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
        return validateRemoteVolumeLocation(location, false);
    }

    const auto path = resolveLocalPath(location);
    std::error_code ec;
    if (!fs::exists(path, ec)) return "Path does not exist: " + path.string();
    if (!fs::is_directory(path, ec)) return "Path is not a directory: " + path.string();

    switch (category) {
        case Category::Volumes:
            if (isSingleZarrVolumeDir(path)) return {};
            if (anyImmediateSubdir(path, &isSingleZarrVolumeDir)) return {};
            return "Not a zarr volume and contains no zarr volumes (expected volume metadata plus chunk-level .zarray or zarr.json).";
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

std::string validateSingleVolumeLocation(const std::string& location)
{
    if (location.empty()) return "Location is empty.";
    if (isLocationRemote(location))
        return validateRemoteVolumeLocation(location, true);

    const auto path = resolveLocalPath(location);
    std::error_code ec;
    const bool exists = fs::exists(path, ec);
    if (ec) return "Could not inspect path: " + ec.message();
    if (!exists) return "Path does not exist: " + path.string();
    const bool directory = fs::is_directory(path, ec);
    if (ec) return "Could not inspect path: " + ec.message();
    if (!directory) return "Path is not a directory: " + path.string();
    try {
        if (!isSingleZarrVolumeDir(path)) {
            return "Not a zarr volume (expected volume metadata plus chunk-level "
                   ".zarray or zarr.json).";
        }
    } catch (const fs::filesystem_error& error) {
        return "Could not inspect path: " + std::string(error.what());
    }
    return {};
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

std::shared_ptr<VolumePkg> VolumePkg::newEmpty(
    const vc::project::LoadOptions& opts)
{
    auto pkg = std::shared_ptr<VolumePkg>(new VolumePkg());
    pkg->opts_ = opts;
    pkg->remoteCacheRoot_ = opts.remoteCacheRoot;
    return pkg;
}

std::shared_ptr<VolumePkg> VolumePkg::newDetached(
    const vc::project::LoadOptions& opts)
{
    auto pkg = newEmpty(opts);
    pkg->automaticPersistence_ = false;
    return pkg;
}

std::shared_ptr<VolumePkg> VolumePkg::load(const fs::path& jsonFile,
                                           const vc::project::LoadOptions& opts)
{
    auto p = std::shared_ptr<VolumePkg>(new VolumePkg());
    p->opts_ = opts;
    p->path_ = jsonFile;
    p->readJsonFrom(jsonFile);
    if (!opts.deferResolution)
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
    if (!opts.deferResolution)
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
const std::vector<vc::project::Entry>& VolumePkg::lasagnaDatasetEntries() const { return lasagnaDatasets_; }

bool VolumePkg::addVolumeEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    std::string persistedLocation = location;
    std::optional<vc::RemoteVolumeSpec> inputSpec;
    if (vc::project::isLocationRemote(location)) {
        inputSpec = vc::parseRemoteVolumeSpec(location);
        if (inputSpec->hasBaseScaleSelector)
            persistedLocation = inputSpec->portableLocator;
    }
    for (const auto& e : volumes_) {
        if (inputSpec && inputSpec->hasBaseScaleSelector &&
            inputSpec->baseScaleLevel == 0 &&
            vc::project::isLocationRemote(e.location)) {
            try {
                const auto existing = vc::parseRemoteVolumeSpec(e.location);
                if (existing.baseScaleLevel == 0 &&
                    existing.sourceUrl == inputSpec->sourceUrl)
                    return false;
            } catch (...) {
            }
        }
        if (samePersistedVolumeIdentity(e.location, persistedLocation)) return false;
    }
    volumes_.push_back({persistedLocation, std::move(tags)});
    if (!opts_.deferResolution)
        resolveVolumeEntry(volumes_.back());
    persistProjectState();
    return true;
}

VolumePkg::AttachVolumeResult VolumePkg::attachPreparedVolume(
    const std::string& location,
    std::vector<std::string> tags,
    const std::shared_ptr<Volume>& volume,
    const fs::path& remoteCacheRoot)
{
    if (location.empty() || !volume)
        throw std::invalid_argument("volume location and prepared volume are required");

    std::string persistedLocation = location;
    if (vc::project::isLocationRemote(location)) {
        const auto spec = vc::parseRemoteVolumeSpec(location);
        if (spec.hasBaseScaleSelector)
            persistedLocation = spec.portableLocator;
    }

    auto entry = std::find_if(
        volumes_.begin(), volumes_.end(), [&](const auto& candidate) {
            return samePersistedVolumeIdentity(
                candidate.location, persistedLocation);
        });
    const bool entryExists = entry != volumes_.end();
    const std::string volumeId = volume->id();
    auto loaded = loadedVolumes_.find(volumeId);
    if (loaded != loadedVolumes_.end() &&
        !loadedVolumeMatchesLocation(loaded->second, persistedLocation)) {
        return AttachVolumeResult::VolumeIdConflict;
    }

    const bool insertVolume = loaded == loadedVolumes_.end();
    const bool updateCacheRoot =
        remoteCacheRoot_.empty() && !remoteCacheRoot.empty();
    if (entryExists && !insertVolume && !updateCacheRoot)
        return AttachVolumeResult::AlreadyAttached;

    const fs::path previousCacheRoot = remoteCacheRoot_;
    const fs::path previousOptionCacheRoot = opts_.remoteCacheRoot;
    const auto previousTags = volumeTagsByID_.find(volumeId);
    const std::optional<std::vector<std::string>> savedTags =
        previousTags == volumeTagsByID_.end()
            ? std::nullopt
            : std::optional(previousTags->second);

    if (!entryExists)
        volumes_.push_back({persistedLocation, std::move(tags)});
    if (insertVolume)
        loadedVolumes_.emplace(volumeId, volume);

    const auto& appliedTags = entryExists ? entry->tags : volumes_.back().tags;
    if (!appliedTags.empty())
        volumeTagsByID_[volumeId] = appliedTags;
    if (updateCacheRoot) {
        remoteCacheRoot_ = remoteCacheRoot;
        opts_.remoteCacheRoot = remoteCacheRoot;
    }

    try {
        persistProjectState();
    } catch (...) {
        if (!entryExists)
            volumes_.pop_back();
        if (insertVolume)
            loadedVolumes_.erase(volumeId);
        if (savedTags)
            volumeTagsByID_[volumeId] = *savedTags;
        else
            volumeTagsByID_.erase(volumeId);
        remoteCacheRoot_ = previousCacheRoot;
        opts_.remoteCacheRoot = previousOptionCacheRoot;
        throw;
    }

    return entryExists ? AttachVolumeResult::AlreadyAttached
                       : AttachVolumeResult::Attached;
}

bool VolumePkg::reconcileVolumeEntryTags(
    const std::string& location,
    const std::vector<std::string>& tags,
    const std::vector<std::string>& singletonPrefixes)
{
    for (auto& entry : volumes_) {
        if (!samePersistedVolumeIdentity(entry.location, location))
            continue;
        auto reconciled = entry.tags;
        for (const auto& prefix : singletonPrefixes) {
            reconciled.erase(
                std::remove_if(reconciled.begin(), reconciled.end(), [&](const std::string& tag) {
                    return tag.rfind(prefix, 0) == 0;
                }),
                reconciled.end());
        }
        for (const auto& tag : tags) {
            if (!tag.empty() &&
                std::find(reconciled.begin(), reconciled.end(), tag) == reconciled.end())
                reconciled.push_back(tag);
        }
        if (reconciled == entry.tags)
            return false;
        entry.tags = std::move(reconciled);

        // Deferred catalog loads have nothing to refresh. For an already
        // loaded volume, reopen its exact portable view only once.
        for (auto it = loadedVolumes_.begin(); it != loadedVolumes_.end(); ++it) {
            const auto& volume = it->second;
            if (!volume || !volume->isRemote() ||
                !samePersistedVolumeIdentity(volume->remoteLocator(), entry.location))
                continue;
            try {
                auto refreshed = Volume::NewFromUrl(
                    entry.location,
                    opts_.remoteCacheRoot.empty()
                        ? volume->remoteCacheRoot()
                        : opts_.remoteCacheRoot,
                    volume->remoteAuth(),
                    metadataFromVolumeEntryTags(entry.tags));
                const auto oldId = it->first;
                const auto newId = refreshed->id();
                if (newId != oldId && loadedVolumes_.count(newId) != 0) {
                    Logger()->warn(
                        "Reconciled remote volume '{}' would collide with loaded id '{}'; keeping the existing view",
                        entry.location, newId);
                    volumeTagsByID_[oldId] = entry.tags;
                    persistProjectState();
                    return true;
                }
                loadedVolumes_.erase(it);
                loadedVolumes_[newId] = std::move(refreshed);
                volumeTagsByID_.erase(oldId);
                volumeTagsByID_[newId] = entry.tags;
            } catch (const std::exception& ex) {
                Logger()->warn("Failed to refresh reconciled remote volume '{}': {}",
                               entry.location, ex.what());
            }
            break;
        }
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::mergeVolumeEntryTags(const std::string& location, const std::vector<std::string>& tags)
{
    if (location.empty() || tags.empty()) return false;
    for (auto& e : volumes_) {
        if (!samePersistedVolumeIdentity(e.location, location)) continue;
        bool changed = false;
        for (const auto& tag : tags) {
            if (tag.empty()) continue;
            if (std::find(e.tags.begin(), e.tags.end(), tag) == e.tags.end()) {
                e.tags.push_back(tag);
                changed = true;
            }
        }
        if (!changed) return false;

        for (auto it = loadedVolumes_.begin(); it != loadedVolumes_.end(); ++it) {
            const auto id = it->first;
            const auto& volume = it->second;
            if (!volume) continue;
            if (volume->isRemote() &&
                samePersistedVolumeIdentity(volume->remoteLocator(), location)) {
                auto metadata = metadataFromVolumeEntryTags(e.tags);
                if (!metadata.empty()) {
                    try {
                        auto refreshed = Volume::NewFromUrl(
                            e.location,
                            opts_.remoteCacheRoot.empty()
                                ? volume->remoteCacheRoot()
                                : opts_.remoteCacheRoot,
                            volume->remoteAuth(), metadata);
                        const auto refreshedId = refreshed->id();
                        if (refreshedId != id && loadedVolumes_.count(refreshedId) == 0) {
                            loadedVolumes_.erase(it);
                            loadedVolumes_.emplace(refreshedId, refreshed);
                            volumeTagsByID_.erase(id);
                            volumeTagsByID_[refreshedId] = e.tags;
                        } else if (refreshedId == id) {
                            it->second = refreshed;
                            volumeTagsByID_[id] = e.tags;
                        } else {
                            Logger()->warn(
                                "Remote volume metadata for '{}' resolves to duplicate id '{}'; keeping existing loaded volume",
                                location,
                                refreshedId);
                            volumeTagsByID_[id] = e.tags;
                        }
                    } catch (const std::exception& ex) {
                        Logger()->warn("Failed to refresh remote volume manifest metadata '{}': {}", location, ex.what());
                        volumeTagsByID_[id] = e.tags;
                    }
                } else {
                    volumeTagsByID_[id] = e.tags;
                }
                break;
            }
        }
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::addSegmentsEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& e : segments_) if (e.location == location) return false;
    segments_.push_back({location, std::move(tags)});
    if (!outputSegments_) {
        outputSegments_ = location;
    }
    if (!opts_.deferResolution)
        refreshSegmentations();
    persistProjectState();
    return true;
}

bool VolumePkg::reconcileSegmentsEntryTags(
    const std::string& location,
    const std::vector<std::string>& tags,
    const std::vector<std::string>& singletonPrefixes)
{
    for (auto& entry : segments_) {
        if (entry.location != location)
            continue;
        auto reconciled = entry.tags;
        for (const auto& prefix : singletonPrefixes) {
            reconciled.erase(
                std::remove_if(reconciled.begin(), reconciled.end(), [&](const auto& tag) {
                    return tag.rfind(prefix, 0) == 0;
                }),
                reconciled.end());
        }
        for (const auto& tag : tags) {
            if (!tag.empty() &&
                std::find(reconciled.begin(), reconciled.end(), tag) == reconciled.end())
                reconciled.push_back(tag);
        }
        if (reconciled == entry.tags)
            return false;
        entry.tags = std::move(reconciled);
        if (!opts_.deferResolution)
            refreshSegmentations();
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::relocateSegmentsEntry(const std::string& oldLocation,
                                      const std::string& newLocation)
{
    if (oldLocation.empty() || newLocation.empty() || oldLocation == newLocation)
        return false;
    if (std::any_of(segments_.begin(), segments_.end(), [&](const auto& entry) {
            return entry.location == newLocation;
        }))
        return false;
    for (auto& entry : segments_) {
        if (entry.location != oldLocation)
            continue;
        entry.location = newLocation;
        if (outputSegments_ && *outputSegments_ == oldLocation)
            outputSegments_ = newLocation;
        if (!opts_.deferResolution)
            refreshSegmentations();
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::addNormalGridEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& e : normalGrids_) if (e.location == location) return false;
    normalGrids_.push_back({location, std::move(tags)});
    if (!opts_.deferResolution)
        resolveNormalGridEntry(normalGrids_.back());
    persistProjectState();
    return true;
}

bool VolumePkg::reconcileNormalGridEntryTags(
    const std::string& location,
    const std::vector<std::string>& tags,
    const std::vector<std::string>& singletonPrefixes)
{
    for (auto& entry : normalGrids_) {
        if (entry.location != location)
            continue;
        auto reconciled = entry.tags;
        for (const auto& prefix : singletonPrefixes) {
            reconciled.erase(
                std::remove_if(reconciled.begin(), reconciled.end(), [&](const auto& tag) {
                    return tag.rfind(prefix, 0) == 0;
                }),
                reconciled.end());
        }
        for (const auto& tag : tags) {
            if (!tag.empty() &&
                std::find(reconciled.begin(), reconciled.end(), tag) == reconciled.end())
                reconciled.push_back(tag);
        }
        if (reconciled == entry.tags)
            return false;
        entry.tags = std::move(reconciled);
        if (!opts_.deferResolution) {
            resolvedNormalGridPaths_.clear();
            for (const auto& normalGrid : normalGrids_)
                resolveNormalGridEntry(normalGrid);
        }
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::relocateNormalGridEntry(const std::string& oldLocation,
                                        const std::string& newLocation)
{
    if (oldLocation.empty() || newLocation.empty() || oldLocation == newLocation)
        return false;
    if (std::any_of(normalGrids_.begin(), normalGrids_.end(), [&](const auto& entry) {
            return entry.location == newLocation;
        }))
        return false;
    for (auto& entry : normalGrids_) {
        if (entry.location != oldLocation)
            continue;
        entry.location = newLocation;
        if (!opts_.deferResolution) {
            resolvedNormalGridPaths_.clear();
            for (const auto& normalGrid : normalGrids_)
                resolveNormalGridEntry(normalGrid);
        }
        persistProjectState();
        return true;
    }
    return false;
}

bool VolumePkg::addLasagnaDatasetEntry(const std::string& location,
                                       std::vector<std::string> tags)
{
    if (location.empty()) return false;
    for (const auto& entry : lasagnaDatasets_)
        if (entry.location == location) return false;
    lasagnaDatasets_.push_back({location, std::move(tags)});
    persistProjectState();
    return true;
}

bool VolumePkg::reconcileLasagnaDatasetEntryTags(
    const std::string& location,
    const std::vector<std::string>& tags,
    const std::vector<std::string>& singletonPrefixes)
{
    for (auto& entry : lasagnaDatasets_) {
        if (entry.location != location) continue;
        auto reconciled = entry.tags;
        for (const auto& prefix : singletonPrefixes) {
            reconciled.erase(
                std::remove_if(reconciled.begin(), reconciled.end(), [&](const auto& tag) {
                    return tag.rfind(prefix, 0) == 0;
                }),
                reconciled.end());
        }
        for (const auto& tag : tags) {
            if (!tag.empty() &&
                std::find(reconciled.begin(), reconciled.end(), tag) == reconciled.end())
                reconciled.push_back(tag);
        }
        if (reconciled == entry.tags) return false;
        entry.tags = std::move(reconciled);
        persistProjectState();
        return true;
    }
    return false;
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
    if (eraseFrom(lasagnaDatasets_)) removed = true;
    if (removed) {
        if (outputSegments_ && *outputSegments_ == location) outputSegments_.reset();
        if (!opts_.deferResolution) resolveAll();
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

std::string VolumePkg::selectedLasagnaDataset() const
{
    return selectedLasagnaDataset_.value_or(std::string{});
}

void VolumePkg::setSelectedLasagnaDataset(std::string location)
{
    if (location.empty()) {
        clearSelectedLasagnaDataset();
        return;
    }
    selectedLasagnaDataset_ = std::move(location);
    persistProjectState();
}

void VolumePkg::clearSelectedLasagnaDataset()
{
    if (!selectedLasagnaDataset_) return;
    selectedLasagnaDataset_.reset();
    persistProjectState();
}

fs::path VolumePkg::selectedLasagnaDatasetPath() const
{
    if (!selectedLasagnaDataset_) return {};
    if (vc::project::isLocationRemote(*selectedLasagnaDataset_)) return {};
    return vc::project::resolveLocalPath(*selectedLasagnaDataset_, path_.parent_path());
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

bool VolumePkg::hasLoadedVolumeEntry(const std::string& location) const
{
    for (const auto& [id, volume] : loadedVolumes_) {
        (void)id;
        if (volume && volume->isRemote() &&
            samePersistedVolumeIdentity(volume->remoteLocator(), location))
            return true;
    }
    return false;
}

std::shared_ptr<Volume> VolumePkg::volume(const std::string& id) const
{
    auto it = loadedVolumes_.find(id);
    if (it == loadedVolumes_.end()) return nullptr;
    return it->second;
}

std::shared_ptr<Volume> VolumePkg::volume() const
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
    auto surf = seg->loadSurface();
    if (surf) {
        surf->backupRoot = path_.parent_path();
    }
    return surf;
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
    auto surf = seg->getSurface();
    if (surf) {
        surf->backupRoot = path_.parent_path();
    }
    return surf;
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
        // Also drop surfaces retained for other segmentation directories.
        for (auto& [location, segs] : segmentationsByLocation_) {
            for (auto& [id, seg] : segs) snapshot.push_back(seg);
        }
    }
    for (auto& seg : snapshot) {
        if (seg) seg->unloadSurface();
    }
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
    if (!automaticPersistence_) return;
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
        segmentationsByLocation_.clear();
        activeSegmentsLocation_.clear();
        segmentationTagsByID_.clear();
    }
    resolvedNormalGridPaths_.clear();
    struct RemoteVolumeResult {
        std::shared_ptr<Volume> volume;
        std::string error;
    };

    std::vector<std::size_t> remoteIndices;
    remoteIndices.reserve(volumes_.size());
    for (std::size_t i = 0; i < volumes_.size(); ++i) {
        if (vc::project::isLocationRemote(volumes_[i].location)) {
            remoteIndices.push_back(i);
        }
    }

    std::vector<RemoteVolumeResult> remoteResults(volumes_.size());
    if (!remoteIndices.empty()) {
        const auto remoteCacheRoot = opts_.remoteCacheRoot;
        auto loadRemote = [this, &remoteResults, remoteCacheRoot](std::size_t i) {
            const auto& entry = volumes_[i];
            try {
                if (!isDirectRemoteZarrLocation(entry.location)) {
                    remoteResults[i] = {nullptr, kDirectRemoteZarrRequired};
                    return;
                }
                remoteResults[i] = {
                    Volume::NewFromUrl(
                        entry.location, remoteCacheRoot, {},
                        metadataFromVolumeEntryTags(entry.tags)),
                    {}};
            } catch (const std::exception& ex) {
                remoteResults[i] = {nullptr, ex.what()};
            } catch (...) {
                remoteResults[i] = {nullptr, "unknown remote volume error"};
            }
        };

        std::atomic<std::size_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const auto index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= remoteIndices.size()) return;
                loadRemote(remoteIndices[index]);
            }
        };

        const auto hardware = std::thread::hardware_concurrency();
        const std::size_t workerCount = std::min<std::size_t>(
            remoteIndices.size(),
            std::max<std::size_t>(
                1, std::min<std::size_t>(hardware == 0 ? 4 : hardware, 8)));
        std::vector<std::future<void>> workers;
        workers.reserve(workerCount - 1);
        for (std::size_t i = 1; i < workerCount; ++i) {
            try {
                workers.emplace_back(std::async(std::launch::async, worker));
            } catch (const std::system_error& ex) {
                Logger()->warn(
                    "Could not start all remote volume workers; continuing "
                    "with {} worker(s): {}",
                    workers.size() + 1, ex.what());
                break;
            }
        }
        worker();
        for (auto& future : workers) future.get();
    }

    for (std::size_t i = 0; i < volumes_.size(); ++i) {
        const auto& entry = volumes_[i];
        if (!vc::project::isLocationRemote(entry.location)) {
            resolveVolumeEntry(entry);
            continue;
        }
        auto resolved = std::move(remoteResults[i]);
        if (!resolved.volume) {
            if (opts_.failOnRemoteError) {
                throw std::runtime_error(
                    "Failed to load remote zarr volume '" + entry.location +
                    "': " + resolved.error);
            }
            Logger()->warn("Failed to load remote zarr volume '{}': {}",
                           entry.location, resolved.error);
            continue;
        }
        const auto id = resolved.volume->id();
        if (loadedVolumes_.count(id) > 0) {
            Logger()->warn("Duplicate remote volume id '{}' from '{}', skipping",
                           id, entry.location);
            continue;
        }
        loadedVolumes_.emplace(id, std::move(resolved.volume));
        if (!entry.tags.empty()) volumeTagsByID_[id] = entry.tags;
    }

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
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        activeSegmentsLocation_ = selectedSegments->location;
        segmentationsByLocation_[activeSegmentsLocation_] = loadedSegmentations_;
    }
    for (const auto& e : normalGrids_) resolveNormalGridEntry(e);
}

void VolumePkg::resolveDeferredEntries()
{
    opts_.deferResolution = false;
    resolveAll();
}

void VolumePkg::resolveVolumeEntry(const vc::project::Entry& e)
{
    if (vc::project::isLocationRemote(e.location)) {
        try {
            if (!isDirectRemoteZarrLocation(e.location)) {
                Logger()->warn("Skipping remote volume location '{}': {}",
                               e.location, kDirectRemoteZarrRequired);
                return;
            }
            auto v = Volume::NewFromUrl(
                e.location,
                opts_.remoteCacheRoot,
                {},
                metadataFromVolumeEntryTags(e.tags));
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
    // Reuse Segmentation objects retained from an earlier visit to this entry
    // so surfaces they already loaded stay available. The directory is still
    // rescanned, so segments added or removed on disk are picked up.
    std::map<std::string, std::shared_ptr<Segmentation>> retainedByPath;
    {
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        auto it = segmentationsByLocation_.find(e.location);
        if (it != segmentationsByLocation_.end()) {
            for (const auto& [id, seg] : it->second) {
                if (seg) retainedByPath[seg->path().string()] = seg;
            }
        }
    }
    auto loadOne = [&](const fs::path& sp) {
        try {
            std::shared_ptr<Segmentation> s;
            if (auto retained = retainedByPath.find(sp.string());
                retained != retainedByPath.end()) {
                s = retained->second;
            } else {
                s = Segmentation::New(sp);
            }
            const auto id = s->id();
            std::lock_guard<std::mutex> lk(segmentsMutex_);
            if (loadedSegmentations_.count(id) > 0) {
                Logger()->warn("Duplicate segment id '{}' from '{}', skipping", id, sp.string());
                return;
            }
            loadedSegmentations_.emplace(id, s);
            auto tags = e.tags;
            const auto& metadata = s->metadata();
            if (metadata.is_object()) {
                auto replaceCoordinateTag = [&](const char* metadataKey,
                                                const char* tagPrefix) {
                    if (!metadata.contains(metadataKey))
                        return;
                    std::string value;
                    const auto& field = metadata[metadataKey];
                    if (field.is_string())
                        value = field.get_string();
                    else if (field.is_number_integer())
                        value = std::to_string(field.get_int());
                    if (value.empty())
                        return;
                    tags.erase(
                        std::remove_if(tags.begin(), tags.end(), [&](const auto& tag) {
                            return tag.rfind(tagPrefix, 0) == 0;
                        }),
                        tags.end());
                    tags.push_back(std::string(tagPrefix) + value);
                };
                replaceCoordinateTag(
                    "vc_open_data_coordinate_space",
                    "vc-open-data-coordinate-space:");
                replaceCoordinateTag(
                    "vc_open_data_source_coordinate_level",
                    "vc-open-data-source-coordinate-level:");
            }
            if (!tags.empty()) segmentationTagsByID_[id] = std::move(tags);
        } catch (const std::exception& ex) {
            Logger()->warn("Failed to load segment '{}': {}", sp.string(), ex.what());
        }
    };
    const bool aggregateOpenDataView =
        std::find(e.tags.begin(), e.tags.end(),
                  "vc-open-data-segment-aggregate") != e.tags.end();
    if (aggregateOpenDataView && !isSegmentDir(path)) {
        struct Candidate {
            fs::path path;
            int rank = 0;
        };
        std::vector<fs::path> candidatePaths;
        for (const auto& child : immediateSubdirs(path)) {
            if (isSegmentDir(child)) {
                candidatePaths.push_back(child);
                continue;
            }
            for (const auto& grandchild : immediateSubdirs(child)) {
                if (isSegmentDir(grandchild)) {
                    candidatePaths.push_back(grandchild);
                }
            }
        }
        std::sort(candidatePaths.begin(), candidatePaths.end());

        std::map<std::string, Candidate> byLineage;
        for (const auto& candidatePath : candidatePaths) {
            try {
                const auto originPath = candidatePath / "catalog-origin.json";
                if (fs::is_regular_file(originPath)) {
                    const auto origin = utils::Json::parse_file(originPath);
                    if (origin.value("cache_state", std::string{}) ==
                        "orphaned") {
                        continue;
                    }
                }
                auto candidate = Segmentation::New(candidatePath);
                const auto& metadata = candidate->metadata();
                std::string lineage = metadata.value(
                    "vc_open_data_catalog_segment_lineage_id",
                    std::string{});
                if (lineage.empty()) {
                    lineage = metadata.value(
                        "vc_open_data_segment_long_id", std::string{});
                }
                if (lineage.empty()) lineage = candidate->id();

                const auto representation = metadata.value(
                    "vc_open_data_representation", std::string{});
                int rank = 20;
                if (representation == "published-transformed") rank = 30;
                if (representation == "generated-native-transform") rank = 10;
                auto [it, inserted] = byLineage.emplace(
                    lineage, Candidate{candidatePath, rank});
                if (!inserted && rank > it->second.rank) {
                    it->second = Candidate{candidatePath, rank};
                }
            } catch (const std::exception& ex) {
                Logger()->warn("Failed to inspect aggregate segment '{}': {}",
                               candidatePath.string(), ex.what());
            }
        }
        for (const auto& [lineage, candidate] : byLineage) {
            (void)lineage;
            loadOne(candidate.path);
        }
    } else if (isSegmentDir(path)) {
        loadOne(path);
    } else {
        for (const auto& child : immediateSubdirs(path)) {
            if (isSegmentDir(child)) loadOne(child);
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
    j["lasagna_datasets"] = entriesToJson(lasagnaDatasets_);
    if (!remoteCacheRoot_.empty()) j["remote_cache_root"] = remoteCacheRoot_.string();
    if (outputSegments_) j["output_segments"] = *outputSegments_;
    if (selectedLasagnaDataset_) j["selected_lasagna_dataset"] = *selectedLasagnaDataset_;
    return j;
}

void VolumePkg::fromJson(const utils::Json& j)
{
    name_ = j.value("name", std::string("Untitled"));
    version_ = j.value("version", 1);
    if (j.contains("volumes")) volumes_ = entriesFromJson(j.at("volumes"));
    if (j.contains("segments")) segments_ = entriesFromJson(j.at("segments"));
    if (j.contains("normal_grids")) normalGrids_ = entriesFromJson(j.at("normal_grids"));
    if (j.contains("lasagna_datasets"))
        lasagnaDatasets_ = entriesFromJson(j.at("lasagna_datasets"));
    if (j.contains("remote_cache_root")) {
        remoteCacheRoot_ = j.at("remote_cache_root").get_string();
        if (!remoteCacheRoot_.empty()) {
            opts_.remoteCacheRoot = remoteCacheRoot_;
        }
    }
    if (j.contains("output_segments")) outputSegments_ = j.at("output_segments").get_string();
    if (j.contains("selected_lasagna_dataset")) {
        selectedLasagnaDataset_ = j.at("selected_lasagna_dataset").get_string();
        if (selectedLasagnaDataset_->empty()) selectedLasagnaDataset_.reset();
    }
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
        // Retain the outgoing directory's segmentations so switching back to
        // it reuses them (and their loaded surfaces) without hitting disk.
        if (!activeSegmentsLocation_.empty()) {
            segmentationsByLocation_[activeSegmentsLocation_] = loadedSegmentations_;
        }
        activeSegmentsLocation_.clear();
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
        std::lock_guard<std::mutex> lk(segmentsMutex_);
        activeSegmentsLocation_ = selectedSegments->location;
        segmentationsByLocation_[activeSegmentsLocation_] = loadedSegmentations_;
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
