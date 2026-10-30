#include "vc/core/types/VolumePkg.hpp"


#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <future>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc_reader/sftp.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include <vc_reader/zarr.hpp>

namespace fs = std::filesystem;


namespace vc::project {

bool hasEntryTag(const Entry& entry, std::string_view tag)
{
    return std::find(entry.tags.begin(), entry.tags.end(), tag) != entry.tags.end();
}

bool usesAnonymousRemoteAuth(const Entry& entry)
{
    return hasEntryTag(entry, kAnonymousRemoteAuthTag);
}

bool isFiberLasagnaEntry(const Entry& entry)
{
    return hasEntryTag(entry, kFiberLasagnaTag);
}

bool isLocationRemote(const std::string& location)
{
    if (vcr::sftp::isSftpUrl(location)) return true;
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

bool isSingleZarrVolumeDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    if (vcr::zarr::isLocalZarrArray(dir))
        return true;
    bool meta = fs::exists(dir / "meta.json")
             || fs::exists(dir / "metadata.json")
             || vcr::zarr::isLocalZarrNode(dir);
    if (!meta) return false;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_directory() && vcr::zarr::isLocalZarrNode(e.path())) return true;
    }
    return false;
}

bool isNormalGridDir(const fs::path& dir)
{
    if (!fs::is_directory(dir)) return false;
    // A streaming cache dir is a valid store even before any grid files have
    // been fetched — NormalGridVolume pulls them on demand via the marker URL.
    if (fs::is_regular_file(dir / vc::core::util::kNormalGridsRemoteMarker)) return true;
    return vc::core::util::hasNormalGridFiles(dir);
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

std::shared_ptr<Volume> openRemoteVolumeEntry(
    const vc::project::Entry& entry,
    const vcr::catalog::OpenDataManifest* catalog,
    const vc::HttpAuth& auth = {})
{
    const bool anonymous = vc::project::usesAnonymousRemoteAuth(entry);
    return Volume::NewFromUrl(
        entry.location,
        anonymous ? vc::HttpAuth{} : auth,
        vc::project::catalogVolumeMetadata(catalog, entry.location),
        !anonymous);
}

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

// One entry per volume: a working level (#vc-base-scale) does not make a
// second one. Note that this deliberately disagrees with
// sameAttachmentLocation() for remote locators: two selector-free locators
// are deduped there when they resolve to the same source URL, and kept
// distinct here unless the strings match. Pre-existing, and left alone, but it
// means attach/match and persisted-entry dedupe can reach different answers
// for the same pair.
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
        return aSpec.sourceUrl == bSpec.sourceUrl;
    } catch (...) {
        return false;
    }
}

fs::path absoluteLocalPath(
    const std::string& location,
    const fs::path& projectDirectory)
{
    auto path = vc::project::resolveLocalPath(location, projectDirectory);
    if (path.is_relative())
        path = fs::absolute(path);
    return path.lexically_normal();
}

// Do two already-absolute, lexically normalized paths name the same thing?
// A lexical comparison cannot see through a symlink, and the two sides do not
// always arrive in the same spelling: load() keeps path_ verbatim and the open
// dialog does not resolve links. fs::equivalent needs both paths to exist and
// reports through the error_code, so the lexical answer stays authoritative for
// locations that have not been created yet.
bool samePathOnDisk(const fs::path& a, const fs::path& b)
{
    if (a == b)
        return true;
    std::error_code ec;
    return fs::equivalent(a, b, ec);
}

bool sameAttachmentLocation(
    const std::string& a,
    const std::string& b,
    const fs::path& projectDirectory)
{
    const bool aRemote = vc::project::isLocationRemote(a);
    const bool bRemote = vc::project::isLocationRemote(b);
    if (aRemote != bRemote)
        return false;
    if (!aRemote) {
        return samePathOnDisk(
            absoluteLocalPath(a, projectDirectory),
            absoluteLocalPath(b, projectDirectory));
    }
    try {
        const auto aSpec = vc::parseRemoteVolumeSpec(a);
        const auto bSpec = vc::parseRemoteVolumeSpec(b);
        return aSpec.sourceUrl == bSpec.sourceUrl &&
               aSpec.baseScaleLevel == bSpec.baseScaleLevel;
    } catch (...) {
        return false;
    }
}

bool entryBacksAttachmentLocation(
    const std::string& entryLocation,
    const std::string& attachmentLocation,
    const fs::path& projectDirectory)
{
    if (sameAttachmentLocation(
            entryLocation, attachmentLocation, projectDirectory)) {
        return true;
    }
    if (vc::project::isLocationRemote(entryLocation) ||
        vc::project::isLocationRemote(attachmentLocation)) {
        return false;
    }
    const auto entryPath = absoluteLocalPath(entryLocation, projectDirectory);
    const auto attachmentPath =
        absoluteLocalPath(attachmentLocation, projectDirectory);
    try {
        return fs::is_directory(entryPath) &&
               !isSingleZarrVolumeDir(entryPath) &&
               samePathOnDisk(attachmentPath.parent_path(), entryPath) &&
               isSingleZarrVolumeDir(attachmentPath);
    } catch (const fs::filesystem_error&) {
        return false;
    }
}

bool loadedVolumeMatchesLocation(
    const std::shared_ptr<Volume>& volume,
    const std::string& location,
    const fs::path& projectDirectory)
{
    if (!volume)
        return false;
    if (volume->isRemote())
        return sameAttachmentLocation(
            volume->remoteLocator(), location, projectDirectory);
    if (vc::project::isLocationRemote(location))
        return false;
    auto loadedPath = volume->path();
    if (loadedPath.is_relative())
        loadedPath = fs::absolute(loadedPath);
    return samePathOnDisk(
        loadedPath.lexically_normal(),
        absoluteLocalPath(location, projectDirectory));
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

// Where a loaded volume was opened from, as an entry location names it.
std::string loadedVolumeLocation(const Volume& volume)
{
    return volume.isRemote() ? volume.remoteLocator() : volume.path().string();
}

nlohmann::json boundsJson(const Rect3D& bounds, bool enabled)
{
    return {{"enabled", enabled},
            {"low_xyz", {bounds.low[0], bounds.low[1], bounds.low[2]}},
            {"high_xyz", {bounds.high[0], bounds.high[1], bounds.high[2]}}};
}

std::optional<Rect3D> boundsFromJson(const nlohmann::json& saved)
{
    const auto low = saved.value("low_xyz", std::vector<double>{});
    const auto high = saved.value("high_xyz", std::vector<double>{});
    if (low.size() != 3 || high.size() != 3) return std::nullopt;
    Rect3D bounds;
    for (int axis = 0; axis < 3; ++axis) {
        bounds.low[axis] = std::min(low[axis], high[axis]);
        bounds.high[axis] = std::max(low[axis], high[axis]);
        if (!std::isfinite(bounds.low[axis]) || !std::isfinite(bounds.high[axis])) return std::nullopt;
    }
    return bounds;
}

}

namespace vc::project {

std::optional<vcr::catalog::OpenDataVolumeView> catalogVolumeView(const vcr::catalog::OpenDataManifest* catalog,
                                                                  const std::string& location)
{
    if (!catalog || !isLocationRemote(location)) return std::nullopt;
    try {
        const auto spec = vc::parseRemoteVolumeSpec(location);
        return catalog->volumeAt(spec.sourceUrl, spec.baseScaleLevel);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

utils::Json catalogVolumeMetadata(const vcr::catalog::OpenDataManifest* catalog, const std::string& location)
{
    auto metadata = utils::Json::object();
    const auto view = catalogVolumeView(catalog, location);
    if (!view) return metadata;
    const auto* volume = catalog->findSample(view->sampleId)->findVolume(view->volumeId);
    if (volume->pixelSizeUm && std::isfinite(*volume->pixelSizeUm) && *volume->pixelSizeUm > 0)
        metadata["voxelsize"] = std::ldexp(*volume->pixelSizeUm, view->level);
    std::string name = volume->suffix.empty() ? volume->id : volume->suffix;
    if (!view->source) name = view->artifact->type + " of " + name;
    if (view->level) name += " [L" + std::to_string(view->level) + "]";
    metadata["name"] = name;
    // The catalog volume's own zarr is that volume at every level; a derived
    // dataset is a volume of its own on its grid.
    if (view->source) metadata["uuid"] = view->volumeId;
    return metadata;
}

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
VolumePkg::~VolumePkg() = default;

std::shared_ptr<VolumePkg> VolumePkg::open(const fs::path& path, const vc::project::LoadOptions& opts)
{
    auto p = std::shared_ptr<VolumePkg>(new VolumePkg());
    p->document_ = vc::project::ProjectDocument::load(path);
    p->path_ = p->document_.path();
    p->opts_ = opts;
    p->readDocument();
    if (!opts.deferResolution) p->resolveAll();
    return p;
}

bool VolumePkg::writable() const { return document_.writable(); }
std::string VolumePkg::name() const { return name_; }
void VolumePkg::setName(const std::string& v)
{
    auto data = document_.data();
    data["name"] = v;
    document_.replace(std::move(data));
    name_ = v;
}

void VolumePkg::readDocument()
{
    const auto& data = document_.data();
    name_ = data.at("name").get<std::string>();
    const auto entries = [&](const char* key) {
        std::vector<vc::project::Entry> out;
        for (const auto& entry : data.at(key))
            out.push_back({entry.at("location"), entry.at("tags").get<std::vector<std::string>>()});
        return out;
    };
    volumes_ = entries("volumes");
    normalGrids_ = entries("normal_grids");
    lasagnaDatasets_ = entries("lasagna_datasets");
    const auto& view = data.at("view");
    const auto selection = [&](const char* key) -> std::optional<std::string> {
        const auto value = view.value(key, std::string{});
        return value.empty() ? std::nullopt : std::optional(value);
    };
    selectedLasagnaDataset_ = selection("selected_lasagna_dataset");
    selectedFiberInferenceDataset_ = selection("selected_fiber_inference_dataset");
    currentVolumeId_ = view.value("current_volume", std::string{});
    focusBounds_.reset();
    focusBoundsEnabled_ = false;
    if (const auto saved = view.find("focus_bounds"); saved != view.end() && saved->is_object()) {
        focusBounds_ = boundsFromJson(*saved);
        focusBoundsEnabled_ = focusBounds_ && saved->value("enabled", false);
    }
}

std::string VolumePkg::entryOwner(const std::string&) const { return {}; }

namespace {
// Keep resources inside the project tree portable; external paths stay absolute.
std::string portableResource(const std::string& location, const fs::path& directory) {
    if (location.empty() || vc::project::isLocationRemote(location)) return location;
    const auto path = vc::project::resolveLocalPath(location);
    if (!path.is_absolute()) return location;
    const auto relative = path.lexically_normal().lexically_relative(directory);
    if (relative.empty() || *relative.begin() == "..") return location;
    return relative.generic_string();
}
std::vector<std::string> portableTags(std::vector<std::string> tags, const fs::path& directory) {
    for (auto& tag : tags)
        for (const std::string prefix : {"vc-lasagna-derived:", "vc-lasagna-manifest:"})
            if (tag.starts_with(prefix)) tag = prefix + portableResource(tag.substr(prefix.size()), directory);
    return tags;
}
const char* resourceKey(const std::string& kind) {
    if (kind == "volume") return "volumes";
    if (kind == "normal_grid") return "normal_grids";
    return "lasagna_datasets";
}
nlohmann::json entryJson(const vc::project::Entry& entry) {
    return {{"location", entry.location}, {"tags", entry.tags}};
}
}
bool VolumePkg::writeEntry(const char* kind, const vc::project::Entry& entry)
{
    auto data = document_.data();
    auto& entries = data[resourceKey(kind)];
    auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& e) { return e.at("location") == entry.location; });
    if (found != entries.end()) {
        if (*found == entryJson(entry)) return false;
        *found = entryJson(entry);
    } else entries.push_back(entryJson(entry));
    document_.replace(std::move(data));
    return true;
}
nlohmann::json VolumePkg::viewJson() const
{
    auto view = nlohmann::json::object();
    if (selectedLasagnaDataset_) view["selected_lasagna_dataset"] = *selectedLasagnaDataset_;
    if (selectedFiberInferenceDataset_) view["selected_fiber_inference_dataset"] = *selectedFiberInferenceDataset_;
    if (!currentVolumeId_.empty()) view["current_volume"] = currentVolumeId_;
    if (focusBounds_) view["focus_bounds"] = boundsJson(*focusBounds_, focusBoundsEnabled_);
    return view;
}
void VolumePkg::writeView()
{
    if (!writable()) return; // Read-only projects may still be explored.
    auto data = document_.data();
    data["view"] = viewJson();
    try { document_.replace(std::move(data)); }
    catch (...) { readDocument(); throw; }
}

bool VolumePkg::reload()
{
    const auto volumes = volumes_, grids = normalGrids_, lasagna = lasagnaDatasets_;
    document_ = vc::project::ProjectDocument::load(path_);
    readDocument();
    if (volumes_ == volumes && normalGrids_ == grids && lasagnaDatasets_ == lasagna)
        return false;
    if (!opts_.deferResolution)
        followEntries(volumes);
    return true;
}

void VolumePkg::followEntries(const std::vector<vc::project::Entry>& previous)
{
    const auto backingEntry = [&](const Volume& volume) -> const vc::project::Entry* {
        const auto location = loadedVolumeLocation(volume);
        for (const auto& entry : volumes_)
            if (entryBacksAttachmentLocation(entry.location, location, directory())) return &entry;
        return nullptr;
    };
    for (auto it = loadedVolumes_.begin(); it != loadedVolumes_.end();) {
        const auto* entry = it->second ? backingEntry(*it->second) : nullptr;
        if (!entry) {
            volumeTagsByID_.erase(it->first);
            it = loadedVolumes_.erase(it);
            continue;
        }
        if (entry->tags.empty()) volumeTagsByID_.erase(it->first);
        else volumeTagsByID_[it->first] = entry->tags;
        ++it;
    }
    for (const auto& entry : volumes_)
        if (std::none_of(previous.begin(), previous.end(),
                         [&](const auto& old) { return old.location == entry.location; }))
            resolveVolumeEntry(entry);
    resolvedNormalGridPaths_.clear();
    for (const auto& entry : normalGrids_) resolveNormalGridEntry(entry);
}

const std::vector<vc::project::Entry>& VolumePkg::volumeEntries() const { return volumes_; }
const std::vector<vc::project::Entry>& VolumePkg::normalGridEntries() const { return normalGrids_; }
std::vector<vc::project::Entry> VolumePkg::lasagnaDatasetEntries() const
{
    std::vector<vc::project::Entry> entries;
    std::copy_if(lasagnaDatasets_.begin(), lasagnaDatasets_.end(),
                 std::back_inserter(entries),
                 [](const auto& entry) {
                     return !vc::project::isFiberLasagnaEntry(entry);
                 });
    return entries;
}

std::vector<vc::project::Entry> VolumePkg::fiberInferenceDatasetEntries() const
{
    std::vector<vc::project::Entry> entries;
    std::copy_if(lasagnaDatasets_.begin(), lasagnaDatasets_.end(),
                 std::back_inserter(entries),
                 vc::project::isFiberLasagnaEntry);
    return entries;
}

const std::vector<vc::project::Entry>& VolumePkg::allLasagnaDatasetEntries() const
{
    return lasagnaDatasets_;
}

std::optional<vc::project::Entry>
VolumePkg::matchingVolumeEntry(const std::string& location) const
{
    auto entry = std::find_if(
        volumes_.begin(), volumes_.end(), [&](const auto& candidate) {
            return entryBacksAttachmentLocation(
                candidate.location, location, directory());
        });
    return entry == volumes_.end()
        ? std::nullopt
        : std::optional<vc::project::Entry>(*entry);
}

bool VolumePkg::addVolumeEntry(const std::string& location, std::vector<std::string> tags)
{
    if (location.empty()) return false;
    std::string persistedLocation = portableResource(location, directory());
    if (vc::project::isLocationRemote(location)) {
        const auto spec = vc::parseRemoteVolumeSpec(location);
        if (spec.hasBaseScaleSelector)
            persistedLocation = spec.portableLocator;
    }
    for (const auto& e : volumes_) {
        // The location as given too: an explicit level 0 is canonicalized
        // to a selector-free locator, which dedupes only by exact string.
        if (samePersistedVolumeIdentity(e.location, persistedLocation) ||
            samePersistedVolumeIdentity(e.location, location) ||
            (!vc::project::isLocationRemote(e.location) && !vc::project::isLocationRemote(persistedLocation) &&
             vc::project::resolveLocalPath(e.location, directory()).lexically_normal() == vc::project::resolveLocalPath(persistedLocation, directory()).lexically_normal())) return false;
    }
    vc::project::Entry entry{persistedLocation, portableTags(std::move(tags), directory())};
    writeEntry("volume", entry);
    volumes_.push_back(std::move(entry));
    if (!opts_.deferResolution)
        resolveVolumeEntry(volumes_.back());
    return true;
}

VolumePkg::AttachVolumeResult VolumePkg::attachPreparedVolume(
    const std::string& location,
    std::vector<std::string> tags,
    const std::shared_ptr<Volume>& volume)
{
    if (location.empty() || !volume)
        throw std::invalid_argument("volume location and prepared volume are required");

    std::string persistedLocation = portableResource(location, directory());
    if (vc::project::isLocationRemote(location)) {
        const auto spec = vc::parseRemoteVolumeSpec(location);
        if (spec.hasBaseScaleSelector)
            persistedLocation = spec.portableLocator;
    }
    if (!loadedVolumeMatchesLocation(
            volume, persistedLocation, directory())) {
        throw std::invalid_argument(
            "prepared volume does not match its attachment location");
    }

    auto entry = std::find_if(
        volumes_.begin(), volumes_.end(), [&](const auto& candidate) {
            return entryBacksAttachmentLocation(
                candidate.location,
                persistedLocation,
                directory());
        });
    const bool backingEntryExists = entry != volumes_.end();
    const std::string volumeId = volume->id();
    auto loaded = loadedVolumes_.find(volumeId);
    if (loaded != loadedVolumes_.end() &&
        !loadedVolumeMatchesLocation(
            loaded->second, persistedLocation, directory())) {
        return AttachVolumeResult::VolumeIdConflict;
    }

    const bool insertVolume = loaded == loadedVolumes_.end();
    const bool insertEntry = !backingEntryExists;
    if (!insertVolume && !insertEntry)
        return AttachVolumeResult::AlreadyAttached;

    if (insertEntry) {
        // Written first: a failed write leaves the project as it was.
        vc::project::Entry added{persistedLocation, portableTags(std::move(tags), directory())};
        writeEntry("volume", added);
        volumes_.push_back(std::move(added));
    }
    if (insertVolume)
        loadedVolumes_.emplace(volumeId, volume);

    if (backingEntryExists && !entry->tags.empty())
        volumeTagsByID_[volumeId] = entry->tags;
    else if (insertEntry && !volumes_.back().tags.empty())
        volumeTagsByID_[volumeId] = volumes_.back().tags;

    return backingEntryExists ? AttachVolumeResult::AlreadyAttached
                              : AttachVolumeResult::Attached;
}

bool VolumePkg::addNormalGridEntry(const std::string& requestedLocation, std::vector<std::string> tags)
{
    const auto location = portableResource(requestedLocation, directory());
    if (location.empty()) return false;
    for (const auto& e : normalGrids_) if (e.location == location) return false;
    vc::project::Entry entry{location, portableTags(std::move(tags), directory())};
    writeEntry("normal_grid", entry);
    normalGrids_.push_back(std::move(entry));
    if (!opts_.deferResolution)
        resolveNormalGridEntry(normalGrids_.back());
    return true;
}

bool VolumePkg::addLasagnaDatasetEntry(const std::string& requestedLocation,
                                       std::vector<std::string> tags)
{
    const auto location = portableResource(requestedLocation, directory());
    if (location.empty()) return false;
    for (const auto& entry : lasagnaDatasets_)
        if (entry.location == location) return false;
    vc::project::Entry entry{location, portableTags(std::move(tags), directory())};
    writeEntry("lasagna_dataset", entry);
    lasagnaDatasets_.push_back(std::move(entry));
    return true;
}

VolumePkg::AttachLasagnaResult VolumePkg::attachPreparedLasagnaDataset(
    const std::string& requestedManifestLocation,
    std::vector<std::string> manifestTags,
    bool fiberInference,
    const std::vector<PreparedVolumeAttachment>& preparedVolumes,
    bool updateSelection)
{
    const auto manifestLocation = portableResource(requestedManifestLocation, directory());
    if (manifestLocation.empty())
        throw std::invalid_argument("Lasagna manifest location is required");
    auto hasTag = [](const std::vector<std::string>& tags, std::string_view tag) {
        return std::find(tags.begin(), tags.end(), tag) != tags.end();
    };
    auto addUniqueTags = [](std::vector<std::string>& target, const std::vector<std::string>& source) {
        for (const auto& tag : source) {
            if (!tag.empty() && std::find(target.begin(), target.end(), tag) == target.end())
                target.push_back(tag);
        }
    };
    auto removeFiberTag = [](std::vector<std::string>& tags) {
        tags.erase(std::remove(tags.begin(), tags.end(), vc::project::kFiberLasagnaTag), tags.end());
    };
    removeFiberTag(manifestTags);
    if (fiberInference)
        manifestTags.emplace_back(vc::project::kFiberLasagnaTag);

    std::set<std::string> incomingLocations;
    std::vector<std::pair<std::string, std::string>> incomingIdLocations;
    for (const auto& prepared : preparedVolumes) {
        if (prepared.location.empty() || !prepared.volume) {
            throw std::invalid_argument(
                "prepared Lasagna volumes require a location and volume");
        }
        if (!incomingLocations.insert(prepared.location).second) {
            throw std::invalid_argument(
                "prepared Lasagna volumes contain duplicate source location: " +
                prepared.location);
        }
        const std::string id = prepared.volume->id();
        auto duplicateId = std::find_if(
            incomingIdLocations.begin(), incomingIdLocations.end(),
            [&](const auto& candidate) { return candidate.first == id; });
        if (duplicateId != incomingIdLocations.end()) {
            throw std::invalid_argument(
                "prepared Lasagna volumes contain duplicate volume id '" + id +
                "' for locations '" + duplicateId->second + "' and '" +
                prepared.location + "'");
        }
        incomingIdLocations.push_back({id, prepared.location});
        if (!loadedVolumeMatchesLocation(
                prepared.volume, prepared.location, directory())) {
            throw std::invalid_argument(
                "prepared Lasagna volume does not match its attachment location");
        }
        auto loaded = loadedVolumes_.find(prepared.volume->id());
        if (loaded != loadedVolumes_.end() &&
            !loadedVolumeMatchesLocation(
                loaded->second, prepared.location, directory())) {
            return AttachLasagnaResult::VolumeIdConflict;
        }
    }

    const auto oldLasagna = lasagnaDatasets_;
    const auto oldVolumes = volumes_;
    const auto oldLoaded = loadedVolumes_;
    const auto oldVolumeTags = volumeTagsByID_;
    const auto oldSelected = selectedLasagnaDataset_;
    const auto oldSelectedFiber = selectedFiberInferenceDataset_;
    bool changed = false;
    try {
        auto manifest = std::find_if(lasagnaDatasets_.begin(), lasagnaDatasets_.end(), [&](const auto& entry) {
            return entry.location == manifestLocation;
        });
        if (manifest == lasagnaDatasets_.end()) {
            lasagnaDatasets_.push_back({manifestLocation, manifestTags});
            manifest = std::prev(lasagnaDatasets_.end());
            changed = true;
        } else {
            auto tags = manifest->tags;
            removeFiberTag(tags);
            addUniqueTags(tags, manifestTags);
            if (tags != manifest->tags) {
                manifest->tags = std::move(tags);
                changed = true;
            }
        }

        if (updateSelection && fiberInference) {
            if (!selectedFiberInferenceDataset_) {
                selectedFiberInferenceDataset_ = manifestLocation;
                changed = true;
            }
            if (selectedLasagnaDataset_ == manifestLocation) {
                selectedLasagnaDataset_.reset();
                changed = true;
            }
        } else if (updateSelection) {
            if (!selectedLasagnaDataset_) {
                selectedLasagnaDataset_ = manifestLocation;
                changed = true;
            }
            if (selectedFiberInferenceDataset_ == manifestLocation) {
                selectedFiberInferenceDataset_.reset();
                changed = true;
            }
        }

        for (const auto& prepared : preparedVolumes) {
            std::string persistedLocation = portableResource(prepared.location, directory());
            const auto tags = portableTags(prepared.tags, directory());
            if (vc::project::isLocationRemote(prepared.location)) {
                const auto spec = vc::parseRemoteVolumeSpec(prepared.location);
                if (spec.hasBaseScaleSelector)
                    persistedLocation = spec.portableLocator;
            }
            auto entry = std::find_if(
                volumes_.begin(), volumes_.end(), [&](const auto& candidate) {
                    return entryBacksAttachmentLocation(
                        candidate.location, persistedLocation,
                        directory());
                });
            if (entry == volumes_.end()) {
                volumes_.push_back({persistedLocation, tags});
                entry = std::prev(volumes_.end());
                changed = true;
            } else {
                const auto oldTags = entry->tags;
                for (const auto& tag : tags) {
                    addUniqueTags(entry->tags, {tag});
                }
                if (entry->tags != oldTags) {
                        changed = true;
                }
            }
            auto loaded = loadedVolumes_.find(prepared.volume->id());
            if (loaded == loadedVolumes_.end()) {
                loadedVolumes_.emplace(prepared.volume->id(), prepared.volume);
                changed = true;
            } else if (loaded->second != prepared.volume &&
                       !loadedVolumeMatchesLocation(
                           loaded->second, persistedLocation,
                           directory())) {
                loaded->second = prepared.volume;
                changed = true;
            }
            volumeTagsByID_[prepared.volume->id()] = entry->tags;
        }
        if (!changed)
            return AttachLasagnaResult::AlreadyAttached;
        auto data = document_.data();
        data["volumes"] = nlohmann::json::array();
        data["lasagna_datasets"] = nlohmann::json::array();
        for (const auto& entry : volumes_) data["volumes"].push_back(entryJson(entry));
        for (const auto& entry : lasagnaDatasets_) data["lasagna_datasets"].push_back(entryJson(entry));
        data["view"] = viewJson();
        document_.replace(std::move(data));
    } catch (...) {
        lasagnaDatasets_ = oldLasagna;
        volumes_ = oldVolumes;
        loadedVolumes_ = oldLoaded;
        volumeTagsByID_ = oldVolumeTags;
        selectedLasagnaDataset_ = oldSelected;
        selectedFiberInferenceDataset_ = oldSelectedFiber;
        // Some of the writes may have landed: follow what the store holds.
        try { reload(); } catch (...) {}
        throw;
    }
    return AttachLasagnaResult::Attached;
}

bool VolumePkg::removeEntry(const std::string& requestedLocation)
{
    const auto location = portableResource(requestedLocation, directory());
    auto data = document_.data();
    bool removed = false;
    for (const auto* key : {"volumes", "normal_grids", "lasagna_datasets"}) {
        auto& entries = data[key];
        const auto count = entries.size();
        entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto& e) { return e.at("location") == location; }), entries.end());
        removed |= count != entries.size();
    }
    if (!removed) return false;
    for (const auto* key : {"selected_lasagna_dataset", "selected_fiber_inference_dataset"})
        if (data["view"].value(key, std::string{}) == location) data["view"].erase(key);
    document_.replace(std::move(data));
    const auto previous = volumes_;
    readDocument();
    if (selectedLasagnaDataset_ == location || selectedFiberInferenceDataset_ == location) {
        if (selectedLasagnaDataset_ == location) selectedLasagnaDataset_.reset();
        if (selectedFiberInferenceDataset_ == location) selectedFiberInferenceDataset_.reset();
        writeView();
    }
    if (!opts_.deferResolution)
        followEntries(previous);
    return true;
}

bool VolumePkg::setVolumeLevel(const std::string& volumeId, int level)
{
    const auto loaded = loadedVolumes_.find(volumeId);
    if (loaded == loadedVolumes_.end() || !loaded->second->isRemote())
        throw std::invalid_argument("only a loaded remote volume has a working level");
    const auto current = loaded->second->remoteLocator();
    const auto entry = std::find_if(volumes_.begin(), volumes_.end(), [&](const auto& candidate) {
        return entryBacksAttachmentLocation(candidate.location, current, directory());
    });
    if (entry == volumes_.end())
        throw std::invalid_argument("volume " + volumeId + " has no project entry");
    const auto source = vc::parseRemoteVolumeSpec(current).sourceUrl;
    auto updated = *entry;
    updated.location = vc::parseRemoteVolumeSpec(source + "#vc-base-scale=" + std::to_string(level)).portableLocator;
    if (updated.location == entry->location) return false;

    // Opened before the entry changes: a level that fails to open leaves the
    // project and the loaded volume as they were.
    auto reopened = openRemoteVolumeEntry(updated, opts_.catalog.get());
    if (reopened->id() != volumeId)
        throw std::runtime_error("volume " + volumeId + " opens as " + reopened->id() + " at level " +
                                 std::to_string(level));
    auto data = document_.data();
    for (auto& e : data["volumes"])
        if (e.at("location") == entry->location) e["location"] = updated.location;
    document_.replace(std::move(data));
    readDocument();
    loadedVolumes_[volumeId] = std::move(reopened);
    return true;
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
    selectedLasagnaDataset_ = portableResource(location, directory());
    writeView();
}

void VolumePkg::clearSelectedLasagnaDataset()
{
    if (!selectedLasagnaDataset_) return;
    selectedLasagnaDataset_.reset();
    writeView();
}

std::string VolumePkg::selectedFiberInferenceDataset() const
{
    return selectedFiberInferenceDataset_.value_or(std::string{});
}

void VolumePkg::setSelectedFiberInferenceDataset(std::string location)
{
    if (location.empty()) {
        clearSelectedFiberInferenceDataset();
        return;
    }
    selectedFiberInferenceDataset_ = portableResource(location, directory());
    writeView();
}

void VolumePkg::clearSelectedFiberInferenceDataset()
{
    if (!selectedFiberInferenceDataset_) return;
    selectedFiberInferenceDataset_.reset();
    writeView();
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

bool VolumePkg::addSingleVolume(const std::string& volumeDirName)
{
    if (volumeDirName.empty()) return false;
    for (const auto& e : volumes_) {
        if (vc::project::isLocationRemote(e.location)) continue;
        const auto base = vc::project::resolveLocalPath(e.location, directory());
        const auto candidate = base / volumeDirName;
        if (!isSingleZarrVolumeDir(candidate)) continue;
        try {
            auto v = Volume::New(candidate, vc::project::catalogVolumeMetadata(opts_.catalog.get(), e.location));
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

std::vector<fs::path> VolumePkg::normalGridPaths() const
{
    return resolvedNormalGridPaths_;
}

std::vector<std::string> VolumePkg::volumeTags(const std::string& volumeId) const
{
    auto it = volumeTagsByID_.find(volumeId);
    if (it == volumeTagsByID_.end()) return {};
    return it->second;
}

std::optional<vcr::catalog::OpenDataVolumeView> VolumePkg::catalogView(const std::string& volumeId) const
{
    const auto loaded = loadedVolumes_.find(volumeId);
    if (!opts_.catalog || loaded == loadedVolumes_.end() || !loaded->second->isRemote()) return std::nullopt;
    return opts_.catalog->volumeAt(loaded->second->remoteUrl(), loaded->second->baseScaleLevel());
}

bool VolumePkg::overlayCompatible(const std::string& baseId, const std::string& overlayId) const
{
    const auto base = catalogView(baseId);
    const auto overlay = catalogView(overlayId);
    return !base || !overlay || base->sameGrid(*overlay);
}

bool VolumePkg::isRemote() const
{
    auto anyRemote = [](const std::vector<vc::project::Entry>& v) {
        return std::any_of(v.begin(), v.end(),
                           [](const auto& e) { return vc::project::isLocationRemote(e.location); });
    };
    return anyRemote(volumes_) || anyRemote(normalGrids_);
}

void VolumePkg::resolveAll()
{
    loadedVolumes_.clear();
    volumeTagsByID_.clear();
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
        auto loadRemote = [this, &remoteResults](std::size_t i) {
            const auto& entry = volumes_[i];
            try {
                if (!isDirectRemoteZarrLocation(entry.location)) {
                    remoteResults[i] = {nullptr, kDirectRemoteZarrRequired};
                    return;
                }
                remoteResults[i] = {
                    openRemoteVolumeEntry(entry, opts_.catalog.get()), {}};
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
            auto v = openRemoteVolumeEntry(e, opts_.catalog.get());
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

    const auto path = vc::project::resolveLocalPath(e.location, directory());
    if (!fs::exists(path)) {
        Logger()->warn("Skipping volume '{}': path does not exist", e.location);
        return;
    }
    auto loadOne = [&](const fs::path& vp) {
        try {
            auto v = Volume::New(vp, vc::project::catalogVolumeMetadata(opts_.catalog.get(), e.location));
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

void VolumePkg::resolveNormalGridEntry(const vc::project::Entry& e)
{
    if (vc::project::isLocationRemote(e.location)) {
        Logger()->warn("Remote normal_grid entry '{}' not yet supported.", e.location);
        return;
    }
    const auto path = vc::project::resolveLocalPath(e.location, directory());
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

void VolumePkg::setFocusBounds(std::optional<Rect3D> bounds, bool enabled)
{
    if (bounds) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(bounds->low[axis]) || !std::isfinite(bounds->high[axis]))
                throw std::invalid_argument("Project focus bounds must be finite");
            if (bounds->low[axis] > bounds->high[axis])
                std::swap(bounds->low[axis], bounds->high[axis]);
        }
    }
    focusBounds_ = bounds;
    focusBoundsEnabled_ = enabled && bounds.has_value();
    writeView();
}

void VolumePkg::setCurrentVolumeId(const std::string& id)
{
    if (currentVolumeId_ == id) return;
    currentVolumeId_ = id;
    writeView();
}

vc::project::AnnotationVolume vc::project::annotationVolume(const VolumePkg* pkg, const Volume& volume)
{
    const auto positive = [](std::optional<double> value) {
        return value && std::isfinite(*value) && *value > 0 ? value : std::nullopt;
    };
    if (const auto view = pkg ? pkg->catalogView(volume.id()) : std::nullopt) {
        const auto* sample = pkg->catalog()->findSample(view->sampleId);
        return {{view->volumeId, std::ldexp(1.0, -view->level)},
                positive(sample->findVolume(view->volumeId)->pixelSizeUm)};
    }
    const double scale = std::ldexp(1.0, -volume.baseScaleLevel());
    return {{volume.id(), scale}, positive(volume.voxelSize() * scale)};
}
