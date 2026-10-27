#include "OpenDataSampleProject.hpp"

#include "OpenDataNormalGrids.hpp"
#include "OpenDataLasagna.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <optional>
#include <system_error>

namespace vc3d::opendata {
namespace {

// A catalog Zarr VC3D opens remotely.
bool isRemoteZarr(const vcr::catalog::OpenDataArtifact& artifact)
{
    return vcr::catalog::isZarrArtifact(artifact) && vc::project::isLocationRemote(artifact.url());
}

void preflightPredictionAndSource(const std::string& predictionUrl,
                                  const std::string& sourceUrl,
                                  int sourceLevel)
{
    auto prediction = vc::render::openHttpZarrPyramid(
        predictionUrl, vc::HttpAuth{}, 0); // explicit zero requests strict validation
    auto source = vc::render::openHttpZarrPyramid(
        sourceUrl, vc::HttpAuth{}, sourceLevel);
    if (prediction.shapes.empty() || source.shapes.empty())
        throw std::runtime_error("prediction/source descriptor is empty");
    if (prediction.shapes[0] != source.shapes[0])
        throw std::runtime_error("prediction /0 shape does not match source physical /" +
                                 std::to_string(sourceLevel));
    if (prediction.dtype != source.dtype)
        throw std::runtime_error("prediction/source dtype mismatch");
    if (!prediction.physicalLevelZeroTransformIsIdentity)
        throw std::runtime_error("prediction /0 coordinate scale must be identity");
}

bool hasVolumeEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.volumeEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

std::string volumeLabel(const vcr::catalog::OpenDataVolume& volume)
{
    return volume.id.empty() ? std::string("<unnamed volume>") : volume.id;
}

std::string volumeArtifactLabel(const vcr::catalog::OpenDataVolume& volume,
                                const vcr::catalog::OpenDataArtifact& artifact)
{
    auto label = volumeLabel(volume);
    if (!artifact.type.empty()) {
        label += " (" + artifact.type + ")";
    }
    return label;
}

std::string segmentSourcePreferredVolumeId(
    const vcr::catalog::OpenDataSample& sample,
    const std::vector<std::string>& supportedVolumeIds)
{
    if (supportedVolumeIds.empty() || sample.segments.empty()) {
        return {};
    }

    std::map<std::string, std::size_t> segmentCountsByVolumeId;
    for (const auto& segment : sample.segments) {
        if (const auto volumeId = segment.sourceVolumeId(); !volumeId.empty()) {
            ++segmentCountsByVolumeId[volumeId];
        }
    }

    std::string preferredId;
    std::size_t preferredCount = 0;
    for (const auto& volumeId : supportedVolumeIds) {
        const auto it = segmentCountsByVolumeId.find(volumeId);
        const std::size_t count =
            it == segmentCountsByVolumeId.end() ? 0 : it->second;
        if (count > preferredCount) {
            preferredId = volumeId;
            preferredCount = count;
        }
    }

    return preferredCount == 0 ? std::string{} : preferredId;
}

std::filesystem::path sampleProjectCachePath(
    const std::filesystem::path& remoteCacheRoot,
    const vcr::catalog::OpenDataSample& sample)
{
    return remoteCacheRoot / "open_data" / "projects" /
           (vcr::catalog::safePathComponent(sample.id.empty() ? "sample" : sample.id) + ".volpkg.json");
}

bool isNonEmptyFile(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
           std::filesystem::file_size(path, ec) > 0 &&
           !ec;
}

} // namespace

std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const vcr::catalog::OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut,
    const OpenDataSampleProgressCallback& progressCallback,
    const OpenDataResourceSelection* selection,
    const OpenDataNewProject* newProject)
{
    auto result = OpenDataSampleProjectResult{};
    std::shared_ptr<VolumePkg> pkg;
    if (newProject && (newProject->path.empty() || newProject->name.empty()))
        throw std::invalid_argument("New project requires a name and destination.");
    const auto cachedProjectPath = (remoteCacheRoot.empty() || newProject)
        ? std::filesystem::path{}
        : sampleProjectCachePath(remoteCacheRoot, sample);
    // Catalog loads remain unresolved until normal-grid paths have been
    // reconciled.
    vc::project::LoadOptions loadOptions;
    loadOptions.catalog = vc3d::opendata::cachedCatalog();
    loadOptions.deferResolution = true;

    if (!cachedProjectPath.empty() && isNonEmptyFile(cachedProjectPath)) {
        try {
            pkg = VolumePkg::load(cachedProjectPath, loadOptions);
            result.messages.push_back("Loaded cached sample project: " +
                                      cachedProjectPath.string());
        } catch (const std::exception& e) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (!pkg) {
        pkg = newProject ? VolumePkg::newDetached(loadOptions) : VolumePkg::newEmpty(loadOptions);
    }
    pkg->setName(newProject ? newProject->name :
                 (sample.id.empty() ? "Open Data Sample" : sample.id));
    auto attachResult = attachOpenDataSampleVolumes(*pkg, sample, selection);
    result.supportedVolumes = attachResult.supportedVolumes;
    result.attachedVolumeEntries = attachResult.attachedVolumeEntries;
    result.skippedVolumes = attachResult.skippedVolumes;
    result.failedVolumes = attachResult.failedVolumes;
    result.preferredVolumeId = attachResult.preferredVolumeId;
    result.messages.insert(result.messages.end(),
                           attachResult.messages.begin(),
                           attachResult.messages.end());
    if (!remoteCacheRoot.empty()) {
        result.attachedNormalGrids =
            attachOpenDataNormalGrids(*pkg, sample, remoteCacheRoot, selection);
        if (result.attachedNormalGrids > 0) {
            result.messages.push_back("Attached " +
                                      std::to_string(result.attachedNormalGrids) +
                                      " streaming normal grid store(s).");
        }
        auto lasagna = prepareOpenDataLasagna(sample, remoteCacheRoot, selection);
        result.messages.insert(result.messages.end(), lasagna.messages.begin(), lasagna.messages.end());
        result.attachedLasagnaDatasets = attachOpenDataLasagna(*pkg, lasagna, &result.messages);
        if (result.attachedLasagnaDatasets > 0) {
            result.messages.push_back(
                "Attached " + std::to_string(result.attachedLasagnaDatasets) +
                " manifest-backed Lasagna dataset(s).");
        }
    }
    if (newProject) {
        // Detached preparation must not overwrite the active project's autosave.
        // Reload the saved JSON to activate ordinary persistence and exercise
        // exactly the same entry resolution as reopening it later.

        pkg->save(newProject->path);
        pkg = VolumePkg::load(newProject->path, loadOptions);
    }
    if (pkg->entryResolutionDeferred()) {
        if (progressCallback) {
            try { progressCallback({"resolving-volumes"}); } catch (...) {}
        }
        pkg->resolveDeferredEntries();
        if (progressCallback) {
            try { progressCallback({"project-ready"}); } catch (...) {}
        }
        for (const auto& entry : pkg->volumeEntries()) {
            const auto view = vc::project::catalogVolumeView(pkg->catalog().get(), entry.location);
            if (!view || view->sampleId != sample.id || pkg->hasLoadedVolumeEntry(entry.location)) {
                continue;
            }
            ++result.failedVolumes;
            result.messages.push_back(
                "Catalog volume failed remote open/base-level validation: " +
                entry.location + ".");
        }
    }

    if (!cachedProjectPath.empty()) {
        try {
            pkg->save(cachedProjectPath);
        } catch (const std::exception& e) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (resultOut) {
        *resultOut = std::move(result);
    }
    return pkg;
}

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const vcr::catalog::OpenDataSample& sample,
    const OpenDataResourceSelection* selection)
{
    OpenDataSampleProjectResult result;
    std::vector<std::string> supportedVolumeIds;

    struct PredictionCandidate {
        const vcr::catalog::OpenDataVolume* volume = nullptr;
        const vcr::catalog::OpenDataArtifact* prediction = nullptr;
        std::string predictionUrl;
        std::string sourceUrl;
        int sourceCoordinateLevel = 0;
    };
    std::vector<PredictionCandidate> predictions;
    struct VirtualSourceCandidate {
        const vcr::catalog::OpenDataVolume* volume = nullptr;
        std::string sourceUrl;
        int sourceCoordinateLevel = 0;
        std::string representationLabel;
    };
    std::vector<VirtualSourceCandidate> virtualSources;
    std::vector<std::string> attachedLocations;

    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size();
         ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        // Skip an excluded volume and all of its artifacts and counters.
        if (selection && !selection->allowsVolume(volume.id)) {
            continue;
        }
        if (volume.artifacts.empty()) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": no volume artifact.");
            continue;
        }

        bool foundSupportedArtifact = false;
        const auto* preferredSource = vcr::catalog::preferredVolumeArtifact(volume);
        std::string preferredSourceUrl;
        if (preferredSource && !vcr::catalog::classifyDerivedRepresentation(*preferredSource) &&
            isRemoteZarr(*preferredSource)) {
            preferredSourceUrl = preferredSource->url();
        }
        for (std::size_t artifactIndex = 0;
             artifactIndex < volume.artifacts.size(); ++artifactIndex) {
            const auto& artifact = volume.artifacts[artifactIndex];
            const std::string url = artifact.url();
            if (vcr::catalog::classifyDerivedRepresentation(artifact) ==
                vcr::catalog::OpenDataRepresentationKind::Lasagna) {
                if (selection &&
                    !selection->allowsRepresentation(
                        volumeIndex, artifactIndex,
                        vcr::catalog::OpenDataRepresentationKind::Lasagna, volume.id)) {
                    continue;
                }
                if (!artifact.levelParameterPresent ||
                    !artifact.sourceCoordinateLevel) {
                    result.messages.push_back(
                        "Skipped coordinate view for " +
                        volumeArtifactLabel(volume, artifact) +
                        ": parameters.level is missing or invalid.");
                    continue;
                }
                if (*artifact.sourceCoordinateLevel > 0) {
                    virtualSources.push_back(VirtualSourceCandidate{
                        &volume,
                        preferredSourceUrl,
                        *artifact.sourceCoordinateLevel,
                        volumeArtifactLabel(volume, artifact)});
                }
                continue;
            }
            if (!isRemoteZarr(artifact)) {
                continue;
            }

            // Source volumes use only the volume filter; predictions must also
            // pass the representation and kind filters.
            if (selection) {
                if (const auto reprKind =
                        vcr::catalog::classifyDerivedRepresentation(artifact)) {
                    if (!selection->allowsRepresentation(
                            volumeIndex, artifactIndex, *reprKind, volume.id)) {
                        continue;
                    }
                } else if (selection->rawVolumeIds &&
                           std::find(selection->rawVolumeIds->begin(),
                                     selection->rawVolumeIds->end(), volume.id) ==
                               selection->rawVolumeIds->end()) {
                    continue;
                }
            }

            if (!foundSupportedArtifact && !volume.id.empty()) {
                supportedVolumeIds.push_back(volume.id);
            }
            foundSupportedArtifact = true;
            ++result.supportedVolumes;
            if (result.preferredVolumeId.empty()) {
                result.preferredVolumeId = volume.id;
            }

            const auto label = volumeArtifactLabel(volume, artifact);
            // Every origin the catalog admits is public-read; the entry keeps
            // that transport policy so it never consults ambient AWS
            // credentials. The catalog names the volume when it opens.
            const std::vector<std::string> tags{std::string(vc::project::kAnonymousRemoteAuthTag)};
            if (hasVolumeEntry(pkg, url)) {
                ++result.skippedVolumes;
                result.messages.push_back("Skipped " + label + ": already attached.");
                if (!pkg.entryResolutionDeferred() &&
                    !pkg.hasLoadedVolumeEntry(url)) {
                    ++result.failedVolumes;
                    result.messages.push_back(
                        "Attached entry remains unavailable after remote validation: " +
                        label + ".");
                }
            } else {
                try {
                    if (pkg.addVolumeEntry(url, tags)) {
                        ++result.attachedVolumeEntries;
                        if (!pkg.entryResolutionDeferred() &&
                            !pkg.hasLoadedVolumeEntry(url)) {
                            ++result.failedVolumes;
                            result.messages.push_back(
                                "Persisted " + label +
                                " but remote open/base-level validation failed.");
                        }
                    } else {
                        ++result.failedVolumes;
                        result.messages.push_back("Failed to attach " + label + ".");
                    }
                } catch (const std::exception& e) {
                    ++result.failedVolumes;
                    result.messages.push_back("Failed to attach " + label + ": " + e.what());
                } catch (...) {
                    ++result.failedVolumes;
                    result.messages.push_back("Failed to attach " + label + ": unknown error.");
                }
            }
            if (hasVolumeEntry(pkg, url))
                attachedLocations.push_back(url);

            const bool prediction = vcr::catalog::classifyDerivedRepresentation(artifact) ==
                                    vcr::catalog::OpenDataRepresentationKind::Prediction;
            if (prediction && artifact.sourceCoordinateLevel) {
                predictions.push_back(PredictionCandidate{
                    &volume, &artifact, url, preferredSourceUrl, *artifact.sourceCoordinateLevel});
            } else if (prediction) {
                result.messages.push_back(
                    "Kept " + label +
                    " coordinate-unspecified: parameters.level is missing or invalid.");
            }
        }

        if (!foundSupportedArtifact) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": unsupported volume artifact.");
        }
    }

    // Why a representation cannot be paired with its source volume's grid.
    const auto pairingProblem = [&](const vcr::catalog::OpenDataVolume& volume,
                                    const std::string& sourceUrl,
                                    bool sourceMustBeAttached,
                                    const std::string& label) -> std::optional<std::string> {
        if (sourceUrl.empty() ||
            (sourceMustBeAttached &&
             std::find(attachedLocations.begin(), attachedLocations.end(), sourceUrl) ==
                 attachedLocations.end())) {
            return "Skipped virtual source for " + label +
                   ": preferred source Zarr is unavailable or was not attached.";
        }
        if (!volume.pixelSizeUm || !std::isfinite(*volume.pixelSizeUm) || *volume.pixelSizeUm <= 0.0) {
            return "Skipped coordinate pairing for " + label + ": source volume has no positive voxel size.";
        }
        return std::nullopt;
    };

    for (const auto& candidate : predictions) {
        const auto& volume = *candidate.volume;
        const auto& prediction = *candidate.prediction;
        const auto predictionLabel = volumeArtifactLabel(volume, prediction);
        const int level = candidate.sourceCoordinateLevel;
        if (const auto problem = pairingProblem(volume, candidate.sourceUrl,
                                                !(selection && selection->rawVolumeIds), predictionLabel)) {
            ++result.failedVolumes;
            result.messages.push_back(*problem);
            continue;
        }

        try {
            preflightPredictionAndSource(
                candidate.predictionUrl, candidate.sourceUrl, level);
        } catch (const std::exception& e) {
            ++result.failedVolumes;
            result.messages.push_back(
                "Skipped coordinate pairing for " + predictionLabel +
                " against source L" + std::to_string(level) + ": " + e.what());
            continue;
        }

        if (level == 0)
            continue;

        virtualSources.push_back(VirtualSourceCandidate{
            &volume, candidate.sourceUrl, level, predictionLabel});
    }

    std::vector<std::string> attachedVirtualLocators;
    for (const auto& candidate : virtualSources) {
        // An explicit source selection must not grow additional scan entries.
        // Prediction coordinate metadata remains available without attaching a
        // rebased source; native L0 Lasagna resolution is supported directly.
        if (selection && selection->rawVolumeIds)
            continue;
        const int level = candidate.sourceCoordinateLevel;
        if (const auto problem = pairingProblem(*candidate.volume, candidate.sourceUrl, true,
                                                candidate.representationLabel)) {
            ++result.failedVolumes;
            result.messages.push_back(*problem);
            continue;
        }

        const auto virtualLocator = vc::parseRemoteVolumeSpec(
            candidate.sourceUrl + "#vc-base-scale=" +
            std::to_string(level)).portableLocator;
        if (std::find(attachedVirtualLocators.begin(), attachedVirtualLocators.end(),
                      virtualLocator) != attachedVirtualLocators.end()) {
            continue;
        }
        attachedVirtualLocators.push_back(virtualLocator);

        const std::vector<std::string> tags{std::string(vc::project::kAnonymousRemoteAuthTag)};
        if (hasVolumeEntry(pkg, virtualLocator)) {
            ++result.skippedVolumes;
        } else if (pkg.addVolumeEntry(virtualLocator, tags)) {
            ++result.attachedVolumeEntries;
            result.messages.push_back(
                "Attached rebased source view " + virtualLocator + ".");
            if (!pkg.entryResolutionDeferred() &&
                !pkg.hasLoadedVolumeEntry(virtualLocator)) {
                ++result.failedVolumes;
                result.messages.push_back(
                    "Persisted rebased source view but remote open/base-level validation failed: " +
                    virtualLocator + ".");
            }
        } else {
            ++result.failedVolumes;
            result.messages.push_back(
                "Failed to persist rebased source view " + virtualLocator + ".");
        }
    }

    if (const auto segmentPreferredId =
            segmentSourcePreferredVolumeId(sample, supportedVolumeIds);
        !segmentPreferredId.empty()) {
        result.preferredVolumeId = segmentPreferredId;
    }

    return result;
}

} // namespace vc3d::opendata
