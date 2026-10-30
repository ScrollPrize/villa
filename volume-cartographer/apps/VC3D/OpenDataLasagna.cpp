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
    const auto meta = vc::lasagna::openLasagnaGroupArray(group, nullptr).meta;
    if (meta.dtype != "|u1")
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
        const auto actual = static_cast<std::size_t>(meta.shape[axis]);
        const auto padding = static_cast<std::size_t>(meta.chunks[axis]);
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

OpenDataLasagnaDatasetKind validatePrepared(const vcr::catalog::OpenDataVolume& volume,
                                            const vcr::catalog::OpenDataArtifact& artifact,
                                            const vc::lasagna::LasagnaDataset& dataset)
{
    const auto& manifest = dataset.manifest();
    const auto kind = validateOpenDataLasagnaManifest(volume, artifact, manifest);
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
    return kind;
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

// The single root manifest in a public catalogue artifact directory.
std::string discoverManifestUrl(const std::string& artifactUrl)
{
    const auto prefix = vcr::s3::splitPrefixUrl(artifactUrl);
    if (!prefix || prefix->key.empty())
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

// The catalog Lasagna artifact a dataset entry caches (the artifact_url of its
// remote marker); nullopt for other entries.
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

std::filesystem::path lasagnaCacheDir(const std::filesystem::path& remoteCacheRoot,
                                      const std::string& sampleId,
                                      const vcr::catalog::OpenDataVolume& volume,
                                      const vcr::catalog::OpenDataArtifact& artifact)
{
    std::ostringstream identity;
    identity << artifact.url() << '\n' << artifact.modelId.value_or(std::string{}) << '\n'
             << artifact.lasagnaVersion.value_or(-1) << '\n'
             << std::setprecision(17) << artifact.sourceToBase.value_or(-1.0);
    if (volume.shapeZYX) {
        for (const auto extent : *volume.shapeZYX) identity << '\n' << extent;
    }
    return remoteCacheRoot / "open_data" / "lasagna" /
           vcr::catalog::safePathComponent(sampleId) / vcr::catalog::safePathComponent(volume.id) /
           cacheKey(identity.str());
}

struct CachedLasagna {
    std::filesystem::path manifest;
    vc::lasagna::LasagnaDataset dataset;
    OpenDataLasagnaDatasetKind kind;
};

// Download the artifact's small dataset manifest beside a remote marker naming
// it, and open and validate it with its Zarr descriptors.
CachedLasagna cacheLasagna(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const std::filesystem::path& remoteCacheRoot)
{
    if (remoteCacheRoot.empty())
        throw std::runtime_error("No remote cache directory configured");
    const auto artifactUrl = artifact.url();
    const auto modelId = artifact.modelId.value_or(std::string{});
    const int level = *artifact.sourceCoordinateLevel;
    const auto cacheRoot = std::filesystem::absolute(remoteCacheRoot).lexically_normal();
    const auto finalDir = lasagnaCacheDir(cacheRoot, sampleId, volume, artifact);
    const auto markerPath = finalDir / vc::lasagna::kLasagnaRemoteMarker;
    if (std::filesystem::is_regular_file(markerPath)) {
        auto marker = nlohmann::json::parse(std::ifstream(markerPath));
        if (marker.value("artifact_url", std::string{}) == artifactUrl &&
            marker.value("sample_id", std::string{}) == sampleId &&
            marker.value("volume_id", std::string{}) == volume.id &&
            marker.value("model_id", std::string{}) == modelId) {
            const auto manifest = finalDir / marker.value("manifest_file", std::string{});
            if (std::filesystem::is_regular_file(manifest)) {
                bool markerChanged = false;
                if (!marker.value("anonymous", false)) {
                    marker["anonymous"] = true;
                    markerChanged = true;
                }
                if (marker.value("source_coordinate_level", -1) != level) {
                    marker["source_coordinate_level"] = level;
                    markerChanged = true;
                }
                if (markerChanged) {
                    writeMarker(markerPath, marker.dump(2));
                }
                auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
                const auto kind = validatePrepared(volume, artifact, dataset);
                return {manifest, std::move(dataset), kind};
            }
        }
    }

    const auto manifestUrl = discoverManifestUrl(artifactUrl);
    const auto relativeName = manifestUrl.substr(manifestUrl.find_last_of('/') + 1);
    vc::core::util::RemoteFileCacheOptions cacheOptions;
    cacheOptions.cacheRoot = cacheRoot;
    cacheOptions.destination = (finalDir / relativeName).lexically_relative(cacheRoot);
    cacheOptions.discoverAwsCredentials = false;
    const auto cached = vc::core::util::cacheRemoteFile(manifestUrl, cacheOptions);
    nlohmann::json marker{
        {"version", 1},
        {"anonymous", true},
        {"artifact_url", artifactUrl},
        {"sample_id", sampleId},
        {"volume_id", volume.id},
        {"model_id", modelId},
        {"source_coordinate_level", level},
        {"manifest_file", relativeName},
    };
    if (artifact.lasagnaVersion)
        marker["lasagna_version"] = *artifact.lasagnaVersion;
    if (artifact.sourceToBase)
        marker["source_to_base"] = *artifact.sourceToBase;
    if (volume.shapeZYX)
        marker["base_shape_zyx"] = *volume.shapeZYX;
    writeMarker(markerPath, marker.dump(2));
    try {
        auto dataset = vc::lasagna::LasagnaDataset::open(cached.path);
        const auto kind = validatePrepared(volume, artifact, dataset);
        return {cached.path, std::move(dataset), kind};
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(markerPath, ec);
        throw;
    }
}

} // namespace

std::vector<std::size_t> lasagnaArtifacts(const vcr::catalog::OpenDataVolume& volume)
{
    std::vector<std::size_t> result;
    for (std::size_t index = 0; index < volume.artifacts.size(); ++index) {
        const auto& artifact = volume.artifacts[index];
        if (vcr::catalog::classifyDerivedRepresentation(artifact) != vcr::catalog::OpenDataRepresentationKind::Lasagna ||
            artifact.url().empty() || !artifact.levelParameterPresent || !artifact.sourceCoordinateLevel)
            continue;
        const bool duplicate = std::any_of(result.begin(), result.end(), [&](std::size_t other) {
            const auto& item = volume.artifacts[other];
            return item.url() == artifact.url() && item.modelId == artifact.modelId &&
                   item.sourceCoordinateLevel == artifact.sourceCoordinateLevel;
        });
        if (!duplicate) result.push_back(index);
    }
    return result;
}

OpenDataLasagnaDatasetKind validateOpenDataLasagnaManifest(
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (!artifact.sourceCoordinateLevel || *artifact.sourceCoordinateLevel < 0 ||
        *artifact.sourceCoordinateLevel > 5) {
        throw std::runtime_error(
            "Lasagna artifact has missing or invalid parameters.level");
    }
    if (artifact.lasagnaVersionPresent && !artifact.lasagnaVersion) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.lasagna_version");
    }
    if (artifact.sourceToBasePresent && !artifact.sourceToBase) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.source_to_base");
    }
    if (manifest.version != 2) {
        throw std::runtime_error("Unsupported open-data Lasagna manifest version " +
                                 std::to_string(manifest.version));
    }
    if (artifact.lasagnaVersion && manifest.version != *artifact.lasagnaVersion) {
        throw std::runtime_error("Outer and inner Lasagna versions disagree");
    }
    if (artifact.sourceToBase &&
        std::abs(manifest.sourceToBase - *artifact.sourceToBase) > 1.0e-12) {
        throw std::runtime_error("Outer and inner source_to_base values disagree");
    }
    if (volume.shapeZYX &&
        (!manifest.baseShapeZYX ||
         !compatibleBaseShape(*volume.shapeZYX, *manifest.baseShapeZYX))) {
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

std::string lasagnaSourceManifestLocation(const VolumePkg& pkg, const vc::project::Entry& entry)
{
    const auto view = lasagnaCatalogView(pkg, entry);
    if (!view)
        return entry.location;
    return vcr::catalog::joinOpenDataUrl(
        view->artifact->url(), std::filesystem::path(entry.location).filename().string());
}

OpenDataLasagnaPreparation prepareOpenDataLasagna(const vcr::catalog::OpenDataSample& sample,
                                                  const std::filesystem::path& remoteCacheRoot,
                                                  const OpenDataResourceSelection* selection)
{
    OpenDataLasagnaPreparation preparation;
    preparation.sampleId = sample.id;
    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size(); ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        const auto attachable = lasagnaArtifacts(volume);
        for (std::size_t artifactIndex = 0; artifactIndex < volume.artifacts.size(); ++artifactIndex) {
            const auto& artifact = volume.artifacts[artifactIndex];
            if (vcr::catalog::classifyDerivedRepresentation(artifact) !=
                    vcr::catalog::OpenDataRepresentationKind::Lasagna ||
                (selection && !selection->allowsRepresentation(volumeIndex, artifactIndex,
                                                               vcr::catalog::OpenDataRepresentationKind::Lasagna,
                                                               volume.id))) {
                continue;
            }
            if (std::find(attachable.begin(), attachable.end(), artifactIndex) == attachable.end()) {
                if (!artifact.url().empty() && (!artifact.levelParameterPresent || !artifact.sourceCoordinateLevel))
                    preparation.messages.push_back("Skipped Lasagna for " + volume.id +
                                                   ": parameters.level is missing or invalid.");
                continue;
            }
            preparation.artifactUrls.push_back(artifact.url());
            try {
                const auto cached = cacheLasagna(sample.id, volume, artifact, remoteCacheRoot);
                OpenDataLasagnaPreparation::Dataset dataset{
                    cached.manifest.string(), cached.kind == OpenDataLasagnaDatasetKind::FiberInference, {}};
                for (auto& prepared : vc::lasagna::prepareLasagnaProjectVolumes(cached.dataset, dataset.manifest)) {
                    dataset.volumes.push_back(
                        {std::move(prepared.location), std::move(prepared.tags), std::move(prepared.volume)});
                }
                preparation.datasets.push_back(std::move(dataset));
            } catch (const std::exception& error) {
                preparation.messages.push_back("Failed to prepare Lasagna for " + volume.id +
                                               " at coordinate level L" +
                                               std::to_string(*artifact.sourceCoordinateLevel) + ": " +
                                               error.what());
            }
        }
    }
    return preparation;
}

int attachOpenDataLasagna(VolumePkg& pkg, const OpenDataLasagnaPreparation& preparation,
                          std::vector<std::string>* messages)
{
    int attached = 0;
    std::set<std::string> expected;
    const std::vector<std::string> tags{std::string(vc::project::kAnonymousRemoteAuthTag)};
    for (const auto& dataset : preparation.datasets) {
        expected.insert(dataset.manifest);
        if (pkg.attachPreparedLasagnaDataset(dataset.manifest, tags, dataset.fiberInference, dataset.volumes) ==
            VolumePkg::AttachLasagnaResult::VolumeIdConflict) {
            if (messages)
                messages->push_back("Failed to attach Lasagna volumes of " + dataset.manifest +
                                    ": a Lasagna volume id conflicts with the project");
            continue;
        }
        ++attached;
    }

    std::vector<std::string> stale;
    for (const auto& entry : pkg.allLasagnaDatasetEntries()) {
        const auto view = lasagnaCatalogView(pkg, entry);
        if (view && view->sampleId == preparation.sampleId && !expected.count(entry.location) &&
            std::find(preparation.artifactUrls.begin(), preparation.artifactUrls.end(), view->artifact->url()) !=
                preparation.artifactUrls.end())
            stale.push_back(entry.location);
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
        if (!lasagnaArtifacts(*volume).empty()) {
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
