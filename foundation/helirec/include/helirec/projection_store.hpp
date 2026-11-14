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

// What a scan group holds of the detector's files of its scan directory, as runs of frames: the frames
// [source_first, source_first + count) of the file named `file` are the frames [index, index + count) of the
// group. A file may be held in several runs, or in part. Throws when `scan_group` is not a scan group of a store
// or does not record its source files.
struct ProjectionStoreRun {
    std::string file;
    std::int64_t source_first = 0, count = 0, index = 0;
};
std::vector<ProjectionStoreRun> projection_store_runs(const std::string& scan_group);

// The frames of `list` (detector files .../scanNNNN/<file>) as frames of the scan groups of `tile_group`
// (STORE.zarr/<tile>): the same frames in the same order, read from the store. Throws when the store does not
// hold one of them.
calibration::FrameList frames_in_projection_store(const calibration::FrameList& list, const std::string& tile_group);

// The sum of float images of a scan group (its darks or flats) at the level `reduce`, in double: the frames `frame`
// (indices in the group), each as it is, or with `dark` and `weight` (one per frame) as dark + (F - dark) weight
// (a flat taken to another ring current). rows x cols values of the level (returned in rows, cols if given).
std::vector<double> projection_store_sum(const std::string& scan_group, int reduce, const std::int64_t* frame, std::size_t n, const float* dark = nullptr,
                                         const double* weight = nullptr, int* rows = nullptr, int* cols = nullptr);

// ---------------------------------------------------------------- writing
// Creates what is missing of ROOT/<tile>/<scan> (the group, its four levels, the per-frame arrays, the attributes)
// and records level `reduce` as stored, from `source` (a folder or remote the frames came from), with values of
// `dtype` ("uint16" or "float32"). Levels recorded as stored before stay so. `spec` is JSON text:
//     {"tile", "scan", "role", "rows", "cols" (of the detector), "frame_first", "frames",
//      "tile_attributes": {...}, "files": [{"file", "source_first", "first", "count"}, ...],
//      "frame_values": {"angle_deg": [one per frame, null where unknown], ...}}
// Throws when the group exists for other frames or another detector. Arrays are blosc (bitshuffle, lz4,
// `clevel`), chunks of one frame by 128 rows by all columns, a shard per 100 frames. Call it from one thread and
// one process at a time; the writers below may then work in parallel.
void declare_projection_store_scan(const std::string& root, const std::string& spec, int reduce, const std::string& source,
                                   const std::string& dtype, int clevel = 1);

// Frames into one level of a declared scan group. Frames are numbered as in the detector's scan (the `first` of
// the files table). Writers of different shards (different hundreds of frames) may run in parallel, in threads
// or processes; a write of whole shards rewrites each once.
class ProjectionStoreWriter {
  public:
    ProjectionStoreWriter(const std::string& scan_group, int reduce);
    ~ProjectionStoreWriter();
    ProjectionStoreWriter(ProjectionStoreWriter&&) noexcept;
    int rows() const;
    int cols() const;
    const std::string& dtype() const;
    // whether frames [first, first + count) are stored (their first and last chunks: a write stores them together)
    bool holds(std::int64_t first, std::int64_t count) const;
    // data: (count, rows, cols) values of dtype(). verify: read back, frame by frame, and compared (throws)
    void write(std::int64_t first, std::int64_t count, const void* data, bool verify = true);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
