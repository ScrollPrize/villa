#pragma once

#include "vc/lasagna/Manifest.hpp"

#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QJsonObject>

#include <array>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

class VolumePkg;

// What a Spiral service reads of the VC3D project (spiral-fitting's
// fit_session.SpiralProject). The project file need not be readable on the
// service's host, so every session request carries this block instead.
namespace vc3d::spiral {

// What the project knows exactly of one store volume; unset where it does not.
struct ProjectVolumeFacts {
    std::optional<double> voxelSizeUm;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::optional<bool> zTopToBottom;
    std::optional<bool> leftHandedCoordinates;
};

// The facts of `pkg`'s loaded volumes by the store volume their annotations
// are kept in (vc::project::annotationVolume): every working level of a
// catalog volume is one entry. A catalog volume gives its orientation and
// full-resolution shape; any other volume its shape only when it is open at
// its own full resolution. The service picks the entry of the volume its
// inputs name, so entries no fit uses (a Lasagna channel) are harmless.
std::map<std::string, ProjectVolumeFacts> projectVolumeFacts(const VolumePkg& pkg);

// The project's Lasagna dataset as a Spiral fit reads it: a local one's
// identity (lasagnaIdentity; empty for a remote one, which the service
// identifies by its location's URL), the OME-Zarr store of each of its nx, ny and grad_mag
// channels (by fit input: normal_x, normal_y, gradient_magnitude; a path or
// a remote location) and the array of each, which must be one name; the
// arrays' downsampling relative to the dataset's base volume (source_to_base
// x 2^scaledown) and that volume's shape. The fit reads the packs a Spiral
// cache holds for the identity, not the stores. `error` says why a fit cannot
// read it: an unreadable dataset, or channels sharing a multichannel array.
// `remoteGroups` are the channel groups of the remote stores, by fit input.
struct ProjectLasagna {
    std::string location;
    std::string identity;
    std::map<std::string, std::string> stores;
    std::string group;
    double baseVoxelsPerArrayVoxel = 1;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::string error;
    std::map<std::string, vc::lasagna::LasagnaChannelGroup> remoteGroups;
};

// `lasagna` as a service on this host that builds missing packs from local
// arrays reads it: each remote store replaced by the store directory its
// read-through cache keeps the array in (vc::lasagna::
// lasagnaChannelArrayCacheDirectory, <store>/<group>). The service reads them
// only to build a pack, after cacheLasagnaChannelArray has filled them.
ProjectLasagna withCachedStores(ProjectLasagna lasagna);

// The remote groups of `lasagna` a service with the Spiral cache
// `spiralCache` would build packs from: the inputs of each pack kind (normals:
// normal_x and normal_y; grad_mag: gradient_magnitude) the cache holds no pack
// of. The cache keys packs by identity, so a dataset without one here (a
// remote manifest, which the service identifies by its URL) needs all of them.
// `kinds`, when given, limits them to the pack kinds the build reads (the
// service's POST /session/lasagna-packs).
std::vector<vc::lasagna::LasagnaChannelGroup> lasagnaGroupsToCache(
    const ProjectLasagna& lasagna, const std::filesystem::path& spiralCache,
    const std::optional<std::set<std::string>>& kinds = std::nullopt);

// The identity a Spiral cache keys a local Lasagna dataset's packs by
// (spiral-fitting's lasagna_source.manifest_identity): the run_uuid of the
// manifest's provenance record when that record names this manifest (or
// none), else "sha256-" and the SHA-256 of the manifest file's bytes.
// `provenance` is the record's text, empty when the manifest names none.
std::string lasagnaIdentity(const std::string& manifestBytes, const std::string& manifestName,
                            const std::string& provenance);

ProjectLasagna projectLasagna(const vc::lasagna::LasagnaDatasetManifest& manifest,
                              const std::string& location, const std::string& identity);
// The project's selected Lasagna dataset, local or remote (remote files are
// cached under `remoteCacheRoot`); nullopt when it selects none.
std::optional<ProjectLasagna> projectLasagna(const VolumePkg& pkg,
                                             const std::filesystem::path& remoteCacheRoot);

// The project's selected fiber-inference dataset: its identity, keying the
// fiber-direction samples a Spiral cache holds for it (empty for a remote
// dataset, identified by its URL), and its base shape;
// `error` says why it cannot be read. nullopt when the project selects none.
struct ProjectDataset {
    std::string location;
    std::string identity;
    // The HTTP(S) manifest a local copy of a remote dataset was copied from,
    // for a service host without the copy; empty otherwise.
    std::string remoteManifest;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    std::string error;
};
// The HTTP(S) manifest a local copy of a remote dataset (opened beside its
// lasagna-remote.json marker) was copied from; empty for any other manifest.
std::string remoteManifestUrl(const vc::lasagna::LasagnaDatasetManifest& manifest);
std::optional<ProjectDataset> projectFiberInference(const VolumePkg& pkg,
                                                    const std::filesystem::path& remoteCacheRoot);

// The request's "project" block: `name`, `lasagna`, `fiber_inference`, and per
// volume its facts, where both shapes are known the Lasagna arrays'
// downsampling relative to its voxels and the fiber-inference dataset's base
// voxels per its voxel (`fiber_scale`), and, when `umbilicusStore` is given,
// the umbilicus that store holds for it (points ZYX in the volume's own
// voxels). Throws when an umbilicus cannot be read.
QJsonObject projectBlock(const std::string& name,
                         const std::map<std::string, ProjectVolumeFacts>& volumes,
                         const std::optional<ProjectLasagna>& lasagna,
                         const std::optional<ProjectDataset>& fiberInference,
                         vc::objects::ObjectStoreSession* session,
                         const std::optional<vc::objects::AttachmentIdentity>& umbilicusStore);

} // namespace vc3d::spiral
