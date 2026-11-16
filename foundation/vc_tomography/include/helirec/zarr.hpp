#pragma once
// helirec's Zarr: v2 arrays, OME-Zarr pyramids, and the streaming OME-Zarr writer of reconstructed volumes.
//
// ZarrArray: one (z, y, x) uint8 or float32 array, C order, fill value 0, no filters; chunks raw, zlib or (uint8)
// lossless vc-delta3d ({"id": "vc-delta3d", "quant": 1}, the vc_reader / VC3D codec). Chunks that are entirely 0 are
// not stored (readers return the fill value). In a local directory (read and write), or at an http(s):// or s3://
// location (read only; s3://bucket/key is the bucket's public endpoint), optionally through a disk cache.
//
// OmeZarrWriter: one multiscale image per group (NGFF 0.4), axes z, y, x in micrometers.
//  * level 0 is written as slices arrive, each as the block of it that was computed (BlockSink: the rest of the slice
//    is 0), so a masked volume never exists as whole slices: only the chunks its blocks touch are held, and a z-chunk
//    is encoded and stored on worker threads once all its slices are in (any order of slices);
//  * optional further levels (2x mean pooling) are built from the stored level below when the writer is closed;
//  * coordinateTransformations place the array in the full volume: scale = voxel size, translation = origin
//    (z0, y0, x0) * voxel size (+ the pooling offset of coarser levels).
// Readable by zarr-python, tensorstore, napari, VC3D...
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "helirec/block_sink.hpp"
#include "helirec/common.hpp"

namespace helirec {

using ZarrIndex = std::array<std::int64_t, 3>;   // z, y, x

struct ZarrArrayMeta {
    ZarrIndex shape{0, 0, 0}, chunks{1, 1, 1};
    std::string dtype = "|u1";             // "|u1" or "<f4" (NumPy typestr)
    std::string compression = "none";      // none | zlib | vc-delta3d (uint8)
    int zlib_level = 1;
    std::string delta3d_preset = "fast";   // encoder preset (delta3d.hpp); not part of the stored metadata
    char separator = '/';                  // chunk keys z/y/x or z.y.x

    std::size_t itemsize() const { return dtype == "<f4" ? 4 : 1; }
    std::size_t chunk_bytes() const {
        return static_cast<std::size_t>(chunks[0]) * chunks[1] * chunks[2] * itemsize();
    }
    ZarrIndex grid() const {   // chunks along each axis
        return {(shape[0] + chunks[0] - 1) / chunks[0], (shape[1] + chunks[1] - 1) / chunks[1],
                (shape[2] + chunks[2] - 1) / chunks[2]};
    }
};

class ZarrArray {
  public:
    // writes the array's metadata (.zarray) in a new or existing local directory
    static ZarrArray create(const std::string& path, const ZarrArrayMeta& meta);
    // An existing array (of the kinds above): a local directory, or an http(s):// / s3:// location. cache_dir: a
    // remote array's chunks are kept there once read (and its metadata, for when the server cannot be reached).
    static ZarrArray open(const std::string& location, const std::string& cache_dir = {});

    const ZarrArrayMeta& meta() const { return meta_; }
    const std::string& path() const { return path_; }
    std::string chunk_path(std::int64_t iz, std::int64_t iy, std::int64_t ix) const;

    // One whole decoded chunk, meta().chunk_bytes() bytes in C order (edge chunks are stored full size, 0 beyond the
    // array). skip_empty: a chunk that is all 0 is not stored and a stored one is removed. Returns whether it was
    // stored. Thread-safe for distinct chunks.
    bool write_chunk(std::int64_t iz, std::int64_t iy, std::int64_t ix, const void* data, bool skip_empty = true) const;
    // false: the chunk is not stored (`out` is left as it is)
    bool read_chunk(std::int64_t iz, std::int64_t iy, std::int64_t ix, void* out) const;
    // the region [start, start + shape) in C order, 0 where no chunk is stored (chunks read on several threads)
    void read(const ZarrIndex& start, const ZarrIndex& shape, void* out) const;

  private:
    ZarrArray(std::string path, ZarrArrayMeta meta, std::string cache_dir = {});
    std::string path_, cache_dir_;
    ZarrArrayMeta meta_;
    bool remote_ = false;
};

// The levels of an OME-Zarr multiscale image (its first "multiscales" entry), or the one level of a plain array:
// voxel i of a level lies at scale * i + offset in level 0 voxels, per axis (z, y, x), from the datasets' scale and
// translation. A level that declares no translation is taken as block-reduced (offset (scale - 1) / 2: its voxels
// centred on the blocks of level 0 voxels they are the mean of).
struct OmeZarrLevel {
    int level = 0;
    std::string location;   // of the level's array (ZarrArray::open)
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
};
std::vector<OmeZarrLevel> ome_zarr_levels(const std::string& location, const std::string& cache_dir = {});

struct OmeZarrOptions {
    ZarrIndex shape{0, 0, 0};
    std::string dtype = "|u1";            // "|u1" or "<f4"
    ZarrIndex chunks{32, 256, 256};       // capped to the shape
    double voxel_size_um = 1.0;
    ZarrIndex origin{0, 0, 0};            // of the array in the full volume (z, y, x)
    int levels = 1;                       // pyramid levels (2x mean pooling above level 0)
    std::string compression = "none";     // none | zlib | vc-delta3d (uint8)
    int zlib_level = 1;
    std::string delta3d_preset = "fast";
    std::string name;                     // of the multiscale image (default: the directory's name)
    std::string attrs_json;               // a JSON object whose members join "multiscales" in the group's attributes
    bool overwrite = false;               // replace whatever is at the path
    int threads = 16;                     // chunk encoding / storing workers
    bool skip_empty = true;
};

class OmeZarrWriter : public BlockSink {
  public:
    OmeZarrWriter(const std::string& path, const OmeZarrOptions& options);
    ~OmeZarrWriter() override;
    OmeZarrWriter(const OmeZarrWriter&) = delete;
    OmeZarrWriter& operator=(const OmeZarrWriter&) = delete;

    // Slice z of level 0 as its block [x0, x0 + w) x [y0, y0 + h): `rows` (h, w) packed elements of the array's
    // dtype; the rest of the slice is 0 (w == 0 or h == 0: an empty slice). Each slice once, in any order.
    void block(std::int64_t z, int x0, int y0, int w, int h, const void* rows) override;
    // whole slices [z0, z0 + n): `data` (n, Y, X)
    void write(std::int64_t z0, std::int64_t n, const void* data);
    // Stores what is left (slices never written are 0) and builds the pyramid levels. progress(level, slabs): called
    // before a level is built; what it returns, when callable, is called after each of the level's z-chunks.
    void close(const std::function<std::function<void()>(int, std::int64_t)>& progress = {});

    // slices [z0, z1) of a level (whole y, x), from what is stored: call after close() (or for slices whose
    // z-chunks are complete)
    void read(int level, std::int64_t z0, std::int64_t z1, void* out) const;

    const std::string& path() const;
    const std::string& dtype() const;
    const ZarrIndex& shape() const;
    const ZarrIndex& chunks() const;
    int levels() const;
    const std::vector<ZarrIndex>& level_shapes() const;
    std::int64_t n_written() const;   // chunks stored / skipped as empty so far
    std::int64_t n_skipped() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
