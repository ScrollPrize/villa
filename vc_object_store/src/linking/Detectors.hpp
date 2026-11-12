#pragma once
// The detect phase of every detector over one StoredGeometry snapshot. The
// standalone entry points (ObjectLinks.hpp) publish each detection in a write
// of its own; a link_objects request (LinkObjects.hpp) publishes several in
// one checked write.
#include "linking/StoredGeometry.hpp"
#include <atomic>
#include <filesystem>

namespace vc::object_links::detail {
Detection detectPointCollectionsToSurfaces(StoredGeometry& inputs, const PointCollectionSurfaceParams& params,
                                           const std::atomic_bool* cancel);
Detection detectSurfaceOverlaps(StoredGeometry& inputs, const SurfaceOverlapParams& params, const std::atomic_bool* cancel);
Detection detectTracksToTracks(StoredGeometry& inputs, const TrackTrackParams& params, const std::atomic_bool* cancel);
Detection detectFibersToSurfaces(StoredGeometry& inputs, const std::filesystem::path& ctVolume,
                                 const FiberSurfaceParams& params, const std::atomic_bool* cancel);
Detection detectTracksToSurfaces(StoredGeometry& inputs, const std::filesystem::path& ctVolume,
                                 const TrackSurfaceParams& params, const std::atomic_bool* cancel);
} // namespace vc::object_links::detail
