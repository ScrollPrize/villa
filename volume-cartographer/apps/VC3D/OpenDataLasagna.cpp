#include "OpenDataLasagna.hpp"

#include "OpenDataNormalGrids.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"
#include "utils/zarr.hpp"

#include <QSaveFile>

#include <vc_reader/s3_listing.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace vc3d::opendata {
namespace {

// Write a marker file whole or not at all.
void writeMarker(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    QSaveFile output(QString::fromStdString(path.string()));
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(text.data(), qint64(text.size())) != qint64(text.size()) || !output.commit()) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

void validateGroupDescriptor(const vc::lasagna::LasagnaDatasetManifest& manifest,
                             const vc::lasagna::LasagnaChannelGroup& group)
{
    const auto array = vc::lasagna::openLasagnaChannelArray(manifest, group, 1);
    const auto& meta = array.metadata();
    if (meta.dtype != utils::ZarrDtype::uint8)
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be uint8");
    if (group.channels.size() != 1) {
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must describe exactly one 3D channel");
    }
    if (meta.shape.size() != 3 || meta.chunks.size() != 3) {
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be a 3D (Z,Y,X) Zarr");
    }
    if (!manifest.baseShapeZYX) return;
    const double spacing = static_cast<double>(group.scaleFactor()) *
                           manifest.sourceToBase;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto expected = static_cast<std::size_t>(std::ceil(
            static_cast<double>((*manifest.baseShapeZYX)[axis]) / spacing));
        const auto actual = meta.shape[axis];
        const auto padding = meta.chunks[axis];
        if (actual < expected || actual > expected + padding) {
            throw std::runtime_error("Lasagna channel group '" + group.name +
                                     "' shape is incompatible with base_shape_zyx");
        }
    }
}

bool compatibleBaseShape(
    const std::array<std::size_t, 3>& catalogShape,
    const std::array<std::size_t, 3>& manifestShape)
{
    for (std::size_t axis = 0; axis < catalogShape.size(); ++axis) {
        // Published volume shapes and Lasagna base shapes use extent and
        // inclusive-coordinate conventions, respectively. They can therefore
        // differ by one voxel while describing the same coordinate domain.
        const auto smaller = std::min(catalogShape[axis], manifestShape[axis]);
        const auto larger = std::max(catalogShape[axis], manifestShape[axis]);
        if (larger - smaller > 1) return false;
    }
    return true;
}

void validatePrepared(const OpenDataLasagnaInfo& info,
                      const std::filesystem::path& manifestPath)
{
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto& manifest = dataset.manifest();
    (void)validateOpenDataLasagnaManifest(info, manifest);
    std::set<std::string> scheduled;
    std::vector<std::future<void>> validations;
    for (const auto& group : manifest.groups) {
        if (scheduled.insert(group.relativeZarrKey).second) {
            const auto* groupPtr = &group;
            validations.push_back(std::async(std::launch::async, [&manifest, groupPtr]() {
                validateGroupDescriptor(manifest, *groupPtr);
            }));
        }
    }
    for (auto& validation : validations) {
        validation.get();
    }
}

int dyadicLevelForShapes(const std::array<std::size_t, 3>& baseShape,
                         const std::array<int, 3>& workingShape)
{
    if (std::any_of(workingShape.begin(), workingShape.end(),
                    [](int extent) { return extent <= 0; }))
        throw std::runtime_error(
            "Active volume shape must contain positive extents");
    const std::array<std::size_t, 3> working{
        static_cast<std::size_t>(workingShape[0]),
        static_cast<std::size_t>(workingShape[1]),
        static_cast<std::size_t>(workingShape[2])};
    const double scale = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
        working, baseShape, 5);
    const int level = static_cast<int>(std::llround(std::log2(scale)));
    if (level < 0 || level > 5)
        throw std::runtime_error(
            "Active volume shape does not identify a supported Lasagna "
            "coordinate scale (L0-L5)");
    return level;
}

} // namespace

std::string discoverOpenDataLasagnaManifestUrl(const std::string& artifactUrl)
{
    const auto prefix = vcr::s3::splitPrefixUrl(artifactUrl);
    if (!prefix)
        throw std::runtime_error("Lasagna artifact URL is not a listable prefix: " + artifactUrl);
    const auto page = vcr::s3::parseListPage(vc::httpGetString(vcr::s3::listUrl(*prefix, "/", {}, 100)));
    std::vector<std::string> keys;
    for (const auto& object : page.objects) {
        if (object.key.rfind(prefix->key, 0) != 0) continue;
        const auto relative = object.key.substr(prefix->key.size());
        if (!relative.empty() && relative.find('/') == std::string::npos &&
            relative.ends_with(".lasagna.json")) {
            keys.push_back(object.key);
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 1) {
        throw std::runtime_error(
            "Lasagna artifact must contain exactly one root .lasagna.json; found " +
            std::to_string(keys.size()));
    }
    return vcr::catalog::joinOpenDataUrl(prefix->origin, keys.front());
}

OpenDataLasagnaDatasetKind validateOpenDataLasagnaManifest(
    const OpenDataLasagnaInfo& info,
    const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (info.sourceCoordinateLevel < 0 ||
        info.sourceCoordinateLevel > 5) {
        throw std::runtime_error(
            "Lasagna artifact has missing or invalid parameters.level");
    }
    if (info.lasagnaVersionPresent && !info.lasagnaVersion) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.lasagna_version");
    }
    if (info.sourceToBasePresent && !info.sourceToBase) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.source_to_base");
    }
    if (manifest.version != 2) {
        throw std::runtime_error("Unsupported open-data Lasagna manifest version " +
                                 std::to_string(manifest.version));
    }
    if (info.lasagnaVersion && manifest.version != *info.lasagnaVersion) {
        throw std::runtime_error("Outer and inner Lasagna versions disagree");
    }
    if (info.sourceToBase &&
        std::abs(manifest.sourceToBase - *info.sourceToBase) > 1.0e-12) {
        throw std::runtime_error("Outer and inner source_to_base values disagree");
    }
    if (info.baseShapeZYX &&
        (!manifest.baseShapeZYX ||
         !compatibleBaseShape(*info.baseShapeZYX, *manifest.baseShapeZYX))) {
        throw std::runtime_error(
            "Lasagna base_shape_zyx does not match its parent volume");
    }

    if (!manifest.fiberPredictionPrefixes().empty()) {
        return OpenDataLasagnaDatasetKind::FiberInference;
    }
    if (manifest.hasNormalSource()) {
        return OpenDataLasagnaDatasetKind::Normal;
    }
    throw std::runtime_error(
        "Open-data Lasagna artifact contains neither normal Lasagna "
        "(grad_mag/nx/ny) nor fiber inference (presence/nx/ny) channels");
}

std::optional<vcr::catalog::OpenDataVolumeView> lasagnaCatalogView(const VolumePkg& pkg,
                                                                   const vc::project::Entry& entry)
{
    if (!pkg.catalog() || vc::project::isLocationRemote(entry.location)) return std::nullopt;
    const auto marker = vc::project::resolveLocalPath(entry.location, pkg.path().parent_path()).parent_path() /
                        vc::lasagna::kLasagnaRemoteMarker;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(marker, ec)) return std::nullopt;
    try {
        return pkg.catalog()->volumeAt(
            nlohmann::json::parse(std::ifstream(marker)).value("artifact_url", std::string{}), 0);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string lasagnaSourceManifestLocation(const VolumePkg& pkg, const vc::project::Entry& entry)
{
    const auto view = lasagnaCatalogView(pkg, entry);
    if (!view)
        return entry.location;
    return vcr::catalog::joinOpenDataUrl(
        view->artifact->url(), std::filesystem::path(entry.location).filename().string());
}

std::vector<OpenDataLasagnaInfo> lasagnaArtifacts(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume)
{
    std::vector<OpenDataLasagnaInfo> result;
    for (std::size_t artifactIndex = 0;
         artifactIndex < volume.artifacts.size(); ++artifactIndex) {
        const auto& artifact = volume.artifacts[artifactIndex];
        if (vcr::catalog::classifyDerivedRepresentation(artifact) != vcr::catalog::OpenDataRepresentationKind::Lasagna)
            continue;
        const auto url = artifact.url();
        if (url.empty()) continue;
        OpenDataLasagnaInfo info;
        info.sampleId = sampleId;
        info.volumeId = volume.id;
        info.artifactIndex = artifactIndex;
        info.artifactUrl = url;
        info.modelId = artifact.modelId.value_or(std::string{});
        if (!artifact.levelParameterPresent || !artifact.sourceCoordinateLevel)
            continue;
        info.sourceCoordinateLevel = *artifact.sourceCoordinateLevel;
        info.lasagnaVersionPresent = artifact.lasagnaVersionPresent;
        info.lasagnaVersion = artifact.lasagnaVersion;
        info.sourceToBasePresent = artifact.sourceToBasePresent;
        info.sourceToBase = artifact.sourceToBase;
        info.baseShapeZYX = volume.shapeZYX;
        const auto duplicate = std::find_if(result.begin(), result.end(), [&](const auto& item) {
            return item.artifactUrl == info.artifactUrl &&
                   item.modelId == info.modelId &&
                   item.sourceCoordinateLevel == info.sourceCoordinateLevel;
        });
        if (duplicate == result.end()) result.push_back(std::move(info));
    }
    return result;
}

std::filesystem::path lasagnaCacheDir(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataLasagnaInfo& info)
{
    std::ostringstream identity;
    identity << info.artifactUrl << '\n' << info.modelId << '\n'
             << info.lasagnaVersion.value_or(-1) << '\n'
             << std::setprecision(17) << info.sourceToBase.value_or(-1.0);
    if (info.baseShapeZYX) {
        for (const auto extent : *info.baseShapeZYX) identity << '\n' << extent;
    }
    return remoteCacheRoot / "open_data" / "lasagna" /
           vcr::catalog::safePathComponent(info.sampleId) / vcr::catalog::safePathComponent(info.volumeId) /
           cacheKey(identity.str());
}

std::filesystem::path prepareOpenDataLasagna(
    const OpenDataLasagnaInfo& info,
    const std::filesystem::path& remoteCacheRoot,
    std::string* errorOut)
{
    try {
        if (remoteCacheRoot.empty())
            throw std::runtime_error("No remote cache directory configured");
        const auto cacheRoot = std::filesystem::absolute(remoteCacheRoot).lexically_normal();
        const auto finalDir = lasagnaCacheDir(cacheRoot, info);
        const auto markerPath = finalDir / vc::lasagna::kLasagnaRemoteMarker;
        if (std::filesystem::is_regular_file(markerPath)) {
            auto marker = nlohmann::json::parse(std::ifstream(markerPath));
            if (marker.value("artifact_url", std::string{}) == info.artifactUrl &&
                marker.value("sample_id", std::string{}) == info.sampleId &&
                marker.value("volume_id", std::string{}) == info.volumeId &&
                marker.value("model_id", std::string{}) == info.modelId) {
                const auto manifest = finalDir /
                    marker.value("manifest_file", std::string{});
                if (std::filesystem::is_regular_file(manifest)) {
                    bool markerChanged = false;
                    if (!marker.value("anonymous", false)) {
                        marker["anonymous"] = true;
                        markerChanged = true;
                    }
                    if (marker.value("source_coordinate_level", -1) !=
                        info.sourceCoordinateLevel) {
                        marker["source_coordinate_level"] =
                            info.sourceCoordinateLevel;
                        markerChanged = true;
                    }
                    if (markerChanged) {
                        writeMarker(markerPath, marker.dump(2));
                    }
                    validatePrepared(info, manifest);
                    return manifest;
                }
            }
        }

        const auto manifestUrl = discoverOpenDataLasagnaManifestUrl(info.artifactUrl);
        const auto relativeName = manifestUrl.substr(manifestUrl.find_last_of('/') + 1);
        vc::core::util::RemoteFileCacheOptions cacheOptions;
        cacheOptions.cacheRoot = cacheRoot;
        cacheOptions.destination =
            (finalDir / relativeName).lexically_relative(cacheRoot);
        cacheOptions.discoverAwsCredentials = false;
        const auto cached = vc::core::util::cacheRemoteFile(
            manifestUrl, cacheOptions);
        nlohmann::json marker{
            {"version", 1},
            {"anonymous", true},
            {"artifact_url", info.artifactUrl},
            {"sample_id", info.sampleId},
            {"volume_id", info.volumeId},
            {"model_id", info.modelId},
            {"source_coordinate_level", info.sourceCoordinateLevel},
            {"manifest_file", relativeName},
        };
        if (info.lasagnaVersion)
            marker["lasagna_version"] = *info.lasagnaVersion;
        if (info.sourceToBase)
            marker["source_to_base"] = *info.sourceToBase;
        if (info.baseShapeZYX)
            marker["base_shape_zyx"] = *info.baseShapeZYX;
        writeMarker(
            finalDir / vc::lasagna::kLasagnaRemoteMarker, marker.dump(2));
        try {
            validatePrepared(info, cached.path);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(markerPath, ec);
            throw;
        }
        return cached.path;
    } catch (const std::exception& e) {
        if (errorOut) *errorOut = e.what();
        return {};
    }
}

int attachOpenDataLasagna(VolumePkg& pkg,
                          const vcr::catalog::OpenDataSample& sample,
                          const std::filesystem::path& remoteCacheRoot,
                          std::vector<std::string>* messages,
                          const OpenDataResourceSelection* selection)
{
    int attached = 0;
    std::set<std::string> expected;
    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size();
         ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        if (selection && !selection->allowsVolume(volume.id)) {
            continue;
        }
        const auto infos = lasagnaArtifacts(sample.id, volume);
        const bool hasLasagnaArtifact = std::any_of(
            volume.artifacts.begin(), volume.artifacts.end(), [](const auto& artifact) {
                return vcr::catalog::classifyDerivedRepresentation(artifact) ==
                       vcr::catalog::OpenDataRepresentationKind::Lasagna;
            });
        if (infos.empty() && hasLasagnaArtifact) {
            if (messages) messages->push_back(
                "Skipped Lasagna for " + volume.id +
                ": parameters.level is missing or invalid.");
            continue;
        }
        if (infos.empty()) continue;

        for (const auto& info : infos) {
            if (selection &&
                !selection->allowsRepresentation(
                    volumeIndex, info.artifactIndex,
                    vcr::catalog::OpenDataRepresentationKind::Lasagna, volume.id)) {
                continue;
            }
            std::string error;
            const auto manifest =
                prepareOpenDataLasagna(info, remoteCacheRoot, &error);
            if (manifest.empty()) {
                if (messages) {
                    messages->push_back(
                        "Failed to prepare Lasagna for " + volume.id +
                        " at coordinate level L" +
                        std::to_string(info.sourceCoordinateLevel) + ": " + error);
                }
                continue;
            }
            expected.insert(manifest.string());
            const std::vector<std::string> tags{std::string(vc::project::kAnonymousRemoteAuthTag)};
            try {
                const auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
                const auto kind = validateOpenDataLasagnaManifest(
                    info, dataset.manifest());
                const bool fiberInference =
                    kind == OpenDataLasagnaDatasetKind::FiberInference;
                std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
                for (auto& prepared :
                     vc::lasagna::prepareLasagnaProjectVolumes(
                         dataset, manifest.string())) {
                    volumes.push_back({
                        std::move(prepared.location), std::move(prepared.tags),
                        std::move(prepared.volume)});
                }
                const auto result = pkg.attachPreparedLasagnaDataset(
                    manifest.string(), tags, fiberInference, volumes);
                if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict)
                    throw std::runtime_error(
                        "a Lasagna volume id conflicts with the project");
            } catch (const std::exception& error) {
                if (messages) {
                    messages->push_back(
                        "Failed to attach Lasagna volumes for " + volume.id +
                        ": " + error.what());
                }
                continue;
            }
            ++attached;
        }
    }

    std::vector<std::string> stale;
    for (const auto& entry : pkg.allLasagnaDatasetEntries()) {
        const auto view = lasagnaCatalogView(pkg, entry);
        if (!view || view->sampleId != sample.id)
            continue;
        if (expected.count(entry.location) == 0) stale.push_back(entry.location);
    }
    for (const auto& location : stale) pkg.removeEntry(location);
    return attached;
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    const auto volume = pkg.volume(loadedVolumeId);
    if (!volume)
        throw std::runtime_error("Active volume is not loaded for Lasagna shape pairing");
    return resolveLasagnaForView(pkg, pkg.catalogView(loadedVolumeId), volume->shape());
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaForView(
    const VolumePkg& pkg,
    const std::optional<vcr::catalog::OpenDataVolumeView>& view,
    const std::optional<std::array<int, 3>>& workingShape)
{
    if (view) {
        const int activeLevel = view->level;
        std::vector<std::pair<vc::project::Entry, vcr::catalog::OpenDataVolumeView>> parentMatches;
        for (const auto& entry : pkg.lasagnaDatasetEntries()) {
            const auto cached = lasagnaCatalogView(pkg, entry);
            if (cached && cached->sampleId == view->sampleId && cached->volumeId == view->volumeId)
                parentMatches.emplace_back(entry, *cached);
        }

        if (!parentMatches.empty()) {
            if (workingShape) {
                const auto shapeManifestPath = vc::project::resolveLocalPath(
                    parentMatches.front().first.location, pkg.path().parent_path());
                const auto dataset =
                    vc::lasagna::LasagnaDataset::open(shapeManifestPath);
                if (!dataset.manifest().baseShapeZYX)
                    throw std::runtime_error(
                        "Lasagna manifest has no base_shape_zyx for coordinate pairing");
                if (dyadicLevelForShapes(*dataset.manifest().baseShapeZYX, *workingShape) != activeLevel)
                    throw std::runtime_error(
                        "Active volume shape disagrees with its catalog coordinate level");
            }

            std::vector<const std::pair<vc::project::Entry, vcr::catalog::OpenDataVolumeView>*> compatibleMatches;
            for (const auto& match : parentMatches)
                if (activeLevel == 0 || activeLevel == match.second.level)
                    compatibleMatches.push_back(&match);
            if (compatibleMatches.size() > 1)
                throw std::runtime_error(
                    "Multiple Lasagna datasets match the active catalog coordinate space");
            if (compatibleMatches.empty())
                throw std::runtime_error(
                    "No Lasagna dataset is published for the active catalog coordinate level");

            const auto& [entry, cached] = *compatibleMatches.front();
            return ResolvedOpenDataLasagna{
                vc::project::resolveLocalPath(entry.location, pkg.path().parent_path()),
                lasagnaSourceManifestLocation(pkg, entry),
                std::ldexp(1.0, activeLevel),
                cached.artifact->url(),
                true};
        }
        const auto* volume = pkg.catalog()->findSample(view->sampleId)->findVolume(view->volumeId);
        if (!lasagnaArtifacts(view->sampleId, *volume).empty()) {
            throw std::runtime_error(
                "The manifest declares Lasagna for the active catalog volume, "
                "but its validated local cache entry is unavailable");
        }
    }

    const auto manualLocation = pkg.selectedLasagnaDataset();
    if (manualLocation.empty()) return std::nullopt;
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = vc3d::remoteCachePathFs();
    const auto resolvedLocation = vc::project::isLocationRemote(manualLocation)
        ? manualLocation
        : vc::project::resolveLocalPath(
              manualLocation, pkg.path().parent_path()).string();
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
        resolvedLocation, options);
    return ResolvedOpenDataLasagna{
        dataset.manifest().manifestPath, manualLocation, 1.0, {}, false};
}

} // namespace vc3d::opendata
