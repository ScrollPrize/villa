#include "OpenDataLasagna.hpp"

#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/Catalog.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include "vc_reader/zarr.hpp"

namespace vc3d::opendata {
namespace {

// Write a marker file whole or not at all.

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

// The catalog Lasagna artifact a dataset entry caches (the artifact_url of its
// remote marker); nullopt for other entries.
std::optional<vcr::catalog::OpenDataVolumeView> lasagnaCatalogView(
    const std::filesystem::path& projectPath, const std::shared_ptr<const vcr::catalog::OpenDataManifest>& catalog, const vc::project::Entry& entry)
{
    if (!catalog || vcr::zarr::isRemote(entry.location)) return std::nullopt;
    const auto marker = vc::project::resolveLocalPath(entry.location, projectPath.parent_path()).parent_path() /
                        vc::lasagna::kLasagnaRemoteMarker;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(marker, ec)) return std::nullopt;
    try {
        return catalog->volumeAt(nlohmann::json::parse(std::ifstream(marker)).value("artifact_url", std::string{}), 0);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<vcr::catalog::OpenDataVolumeView> lasagnaCatalogView(const VolumePkg& pkg, const vc::project::Entry& entry)
{
    return lasagnaCatalogView(pkg.path(), pkg.catalog(), entry);
}

} // namespace

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
        const auto attachable = vc::lasagna::catalogArtifacts(volume);
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
                const auto cached = vc::lasagna::cacheCatalogDataset(sample.id, volume, artifact, remoteCacheRoot);
                OpenDataLasagnaPreparation::Dataset dataset{
                    cached.manifest.string(), cached.kind == vc::lasagna::CatalogDatasetKind::FiberInference, {}};
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
    return resolveLasagnaSnapshot({pkg.path(), pkg.lasagnaDatasetEntries(), pkg.selectedLasagnaDataset(), pkg.catalog(), view, workingShape});
}

LasagnaResolutionSnapshot snapshotLasagnaForVolume(const VolumePkg& pkg, const std::string& volumeId)
{
    const auto volume = pkg.volume(volumeId);
    if (!volume)
        throw std::runtime_error("Active volume is unavailable for Lasagna");
    return {pkg.path(), pkg.lasagnaDatasetEntries(), pkg.selectedLasagnaDataset(), pkg.catalog(), pkg.catalogView(volumeId), volume->shape()};
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaSnapshot(const LasagnaResolutionSnapshot& snapshot)
{
    const auto& view = snapshot.view;
    const auto& workingShape = snapshot.workingShape;
    if (view) {
        const int activeLevel = view->level;
        std::vector<std::pair<vc::project::Entry, vcr::catalog::OpenDataVolumeView>> parentMatches;
        for (const auto& entry : snapshot.entries) {
            const auto cached = lasagnaCatalogView(snapshot.projectPath, snapshot.catalog, entry);
            if (cached && cached->sampleId == view->sampleId && cached->volumeId == view->volumeId)
                parentMatches.emplace_back(entry, *cached);
        }

        if (!parentMatches.empty()) {
            if (workingShape) {
                const auto shapeManifestPath =
                    vc::project::resolveLocalPath(parentMatches.front().first.location, snapshot.projectPath.parent_path());
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
                vc::project::resolveLocalPath(entry.location, snapshot.projectPath.parent_path()),
                vcr::catalog::joinOpenDataUrl(cached.artifact->url(), std::filesystem::path(entry.location).filename().string()),
                std::ldexp(1.0, activeLevel),
                cached.artifact->url(),
                true};
        }
        const auto* volume = snapshot.catalog->findSample(view->sampleId)->findVolume(view->volumeId);
        if (volume && !vc::lasagna::catalogArtifacts(*volume).empty()) {
            throw std::runtime_error(
                "The manifest declares Lasagna for the active catalog volume, "
                "but its validated local cache entry is unavailable");
        }
    }

    const auto manualLocation = snapshot.manualLocation;
    if (manualLocation.empty()) return std::nullopt;
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = vc3d::remoteCachePathFs();
    const auto resolvedLocation = vcr::zarr::isRemote(manualLocation)
        ? manualLocation
        : vc::project::resolveLocalPath(manualLocation, snapshot.projectPath.parent_path()).string();
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
        resolvedLocation, options);
    return ResolvedOpenDataLasagna{
        dataset.manifest().manifestPath, manualLocation, 1.0, {}, false};
}

} // namespace vc3d::opendata
