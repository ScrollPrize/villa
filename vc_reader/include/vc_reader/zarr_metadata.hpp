#pragma once
// What is known about a volume: its store's sidecar documents and the Open Data catalog.

#include "vc_reader/zarr.hpp"

namespace vcr::zarr {

// ---- store metadata -------------------------------------------------------------------

// Micrometers per voxel of level 0, from a store's sidecar document (meta.json or
// metadata.json, as published). By exact path, since these documents also record
// unrelated scales. In order: a top-level voxelsize (or voxel_size_um, voxelSizeUm,
// pixel_size_um, pixelSizeUm, resolution_um); scan.voxelsize; the acquisition record
// [scan.]tomo.acquisition.detector.samplePixelSize (millimeters); and for a derived
// store that record under source.metadata scaled by 2^source.resolution, which is then
// required. Only positive finite numbers count.
std::optional<double> voxelSizeUm(const Json& document);

// What a store's sidecar documents say about its volume, each field from the first
// document stating it (meta.json before metadata.json).
struct StoreMetadata {
    std::optional<double> voxelSizeUm;         // level 0, see voxelSizeUm()
    std::optional<double> energyKeV;           // acquisition.energy
    std::optional<double> detectorDistanceMm;  // acquisition.sampleDetectorDistance
    std::string beamline, scanName, acquisitionDate;  // beamline, scanRadix or scan_name, date
    // zarr_export: the source values the stored range spans, and the Z range kept.
    std::optional<std::array<double, 2>> valueWindow;
    std::optional<std::array<std::int64_t, 2>> zCrop;
    std::string sampleId, volumeId, scanId, name, createdAt, dataFormat;  // from the Open Data catalog
    Json documents = Json::object();  // file name (.zattrs too) or "catalog" -> the document as published
};
// The metadata of the store at `location` (or a numbered level of one). A missing,
// unreadable or malformed document does not end the search; this never throws. The
// default reader reads local stores and anonymous remote ones.
StoreMetadata storeMetadata(const std::string& location, const MetadataReader& read = {});
inline std::optional<double> storeVoxelSizeUm(const std::string& location, const MetadataReader& read = {})
{
    return storeMetadata(location, read).voxelSizeUm;
}
// Everything known about the volume at `location`: its sidecar documents and, for a store
// in a public channel, the Open Data catalog, one value per field with the store's
// documents winning. A remote store's result is gathered only the first time: it is kept
// with the volume in the disk caches (and for this process), and a kept copy is trusted
// like a chunk. A local store's documents are read at each call. `refresh` reads both
// sources as published now and replaces what was kept; if a source cannot be read, nothing
// kept changes and the kept metadata is returned when there is one.
StoreMetadata volumeMetadata(const std::string& location, const OpenOptions& options, bool refresh = false);
// Drops what this process remembers (not what is kept on disk).
void forgetVolumeMetadata();
Json storeMetadataJson(const StoreMetadata& metadata);
StoreMetadata storeMetadataFromJson(const Json& json);

}  // namespace vcr::zarr
