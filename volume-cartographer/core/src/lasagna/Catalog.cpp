#include "vc/lasagna/Catalog.hpp"
#include <vc_reader/zarr.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <set>
#include <stdexcept>

namespace vc::lasagna {
namespace {
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

CatalogDatasetKind validatePrepared(const vcr::catalog::OpenDataVolume& volume,
                                            const vcr::catalog::OpenDataArtifact& artifact,
                                            const vc::lasagna::LasagnaDataset& dataset)
{
    const auto& manifest = dataset.manifest();
    const auto kind = validateCatalogManifest(volume, artifact, manifest);
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

// The single root manifest in a public catalogue artifact directory.
std::string discoverManifestUrl(const std::string& artifactUrl)
{
    std::vector<std::string> keys;
    try {
        const auto access = vcr::zarr::resolveAccess(artifactUrl);
        std::string token;
        do {
            const auto page = vcr::zarr::listPage(access, artifactUrl, {}, token);
            for (const auto& object : page.objects)
                if (object.key.ends_with(".lasagna.json")) keys.push_back(object.key);
            token = page.nextContinuationToken;
        } while (!token.empty());
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("Lasagna artifact URL is not a listable prefix: " + artifactUrl + " (" + e.what() + ")");
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 1) {
        throw std::runtime_error(
            "Lasagna artifact must contain exactly one root .lasagna.json; found " +
            std::to_string(keys.size()));
    }
    return vcr::catalog::joinOpenDataUrl(artifactUrl, keys.front());
}


} // namespace

// Download the artifact's small dataset manifest beside a remote marker naming
// it, and open and validate it with its Zarr descriptors.
CachedCatalogDataset cacheCatalogDataset(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const std::filesystem::path& remoteCacheRoot)
{
    if (remoteCacheRoot.empty())
        throw std::runtime_error("No remote cache directory configured");
    const auto artifactUrl = artifact.url();
    const auto modelId = artifact.modelId.value_or(std::string{});
    if (!artifact.sourceCoordinateLevel || *artifact.sourceCoordinateLevel < 0 || *artifact.sourceCoordinateLevel > 5)
        throw std::runtime_error("Lasagna artifact has missing or invalid parameters.level");
    const int level = *artifact.sourceCoordinateLevel;
    const auto cacheRoot = std::filesystem::absolute(remoteCacheRoot).lexically_normal();
    const auto finalDir = vcr::zarr::decodedCachePath(cacheRoot, artifactUrl);
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
                    vcr::zarr::atomicWrite(markerPath, marker.dump(2));
                }
                auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
                const auto kind = validatePrepared(volume, artifact, dataset);
                return {manifest, std::move(dataset), kind};
            }
        }
    }

    // A manifest already cached beside its arrays needs no request.
    std::vector<std::filesystem::path> present;
    std::error_code listError;
    for (const auto& file : std::filesystem::directory_iterator(finalDir, listError))
        if (file.path().filename().string().ends_with(".lasagna.json"))
            present.push_back(file.path());
    std::string relativeName;
    if (present.size() == 1) {
        relativeName = present.front().filename().string();
    } else {
        const auto manifestUrl = discoverManifestUrl(artifactUrl);
        relativeName = manifestUrl.substr(manifestUrl.find_last_of('/') + 1);
        LasagnaDatasetOpenOptions options;
        options.remoteCacheRoot = cacheRoot;
        options.discoverAwsCredentials = false;
        (void)materializeLasagnaManifest(manifestUrl, options);
        if (!std::filesystem::is_regular_file(finalDir / relativeName))
            throw std::runtime_error("Lasagna manifest did not cache at " + (finalDir / relativeName).string());
    }
    const auto manifest = finalDir / relativeName;
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
    vcr::zarr::atomicWrite(markerPath, marker.dump(2));
    try {
        auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
        const auto kind = validatePrepared(volume, artifact, dataset);
        return {manifest, std::move(dataset), kind};
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(markerPath, ec);
        throw;
    }
}
std::vector<std::size_t> catalogArtifacts(const vcr::catalog::OpenDataVolume& volume)
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

CatalogDatasetKind validateCatalogManifest(
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
        return CatalogDatasetKind::FiberInference;
    }
    if (manifest.hasNormalSource()) {
        return CatalogDatasetKind::Normal;
    }
    throw std::runtime_error(
        "Open-data Lasagna artifact contains neither normal Lasagna "
        "(grad_mag/nx/ny) nor fiber inference (presence/nx/ny) channels");
}


} // namespace vc::lasagna
