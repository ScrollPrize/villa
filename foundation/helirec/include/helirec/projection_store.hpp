#pragma once
// Frames kept in a multiscale Zarr store (helirec-projections-zarr, tools/helirec_projections_zarr.cpp):
//
//     STORE.zarr/<tile>/<scanNNNN>/{0,1,2,3}     (frame, row, col) at 1, 1/2, 1/4, 1/8 of the detector
//
// A scan group stands for the detector's files of that scan directory: RawReader takes its path where it takes
// a file, the frame index counts the group's frames, and `reduce` selects the level (a level that is declared and
// not stored is refused). The values of level l are what RawReader(reduce=l, decoder="openjph") gives for the
// files the store was made from (the downloader's decoder; the default OpenJPEG decoder rounds 1 % of the values
// of a Paris4 frame one count the other way), so a geometry and a calibration hold for both. Reading is vc_reader's.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "helirec/calibration/scan.hpp"

namespace helirec {

// true when `path` is a scan group of such a store (a directory with a zarr.json whose attributes say so)
bool is_projection_store_scan(const std::string& path);

// One level of a scan group, open for reading
struct ProjectionStoreLevel {
    struct Impl;
    std::shared_ptr<Impl> impl;
    std::int64_t frames = 0;
    int rows = 0, cols = 0;                // of the level
    int stored_rows = 0, stored_cols = 0;  // of the detector (level 0)
    int chunk_rows = 0;                    // rows of a chunk of the level
    int source_chunk_rows = 0;             // rows of a JPEG 2000 chunk of the files it was made from (0: not recorded)
    std::string dtype;                     // "uint16" or "float32"
    // rows [chunk chunk_rows, (chunk + 1) chunk_rows) of a frame -> out (chunk_rows x cols values; rows past the
    // frame's end are 0). Thread safe.
    void read_chunk(std::int64_t frame, int chunk, void* out) const;
};
// what a scan group says of itself without opening a level (reduce < 0), or with level `reduce` open
ProjectionStoreLevel open_projection_store(const std::string& scan_group, int reduce);

// The frames of `list` (detector files .../scanNNNN/<file>) as frames of the scan groups of `tile_group`
// (STORE.zarr/<tile>): the same frames in the same order, read from the store. Throws when the store does not
// hold one of them.
calibration::FrameList frames_in_projection_store(const calibration::FrameList& list, const std::string& tile_group);

}  // namespace helirec
