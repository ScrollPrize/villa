#pragma once
// vc_tomography's Zarr output: the streaming OME-Zarr writer of reconstructed volumes. Arrays, chunk codecs, pyramid
// levels and metadata are vc_reader's (vcr::zarr::Writer and its pyramid functions); reading volumes goes through
// vc_reader as well (src/io/volume_levels.hpp).
//
// OmeZarrWriter: one multiscale image per group (NGFF 0.4), axes z, y, x in micrometers.
//  * level 0 is written as slices arrive, each as the block of it that was computed (BlockSink: the rest of the slice
//    is 0), so a masked volume never exists as whole slices: only the chunks its blocks touch are held, and a z-chunk
//    is encoded and stored on worker threads once all its slices are in (any order of slices);
//  * optional further levels (2x mean pooling: the mean of a block's voxels that lie in the level below; integers
//    rounded half to even) are built from the stored level below when the writer is closed;
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

#include "vc_tomography/block_sink.hpp"
#include "vc_tomography/common.hpp"

namespace vc_tomography {

using ZarrIndex = std::array<std::int64_t, 3>;   // z, y, x

struct OmeZarrOptions {
    ZarrIndex shape{0, 0, 0};
    std::string dtype = "|u1";            // "|u1" or "<f4"
    ZarrIndex chunks{32, 256, 256};       // capped to the shape
    double voxel_size_um = 1.0;
    ZarrIndex origin{0, 0, 0};            // of the array in the full volume (z, y, x)
    int levels = 1;                       // pyramid levels (2x mean pooling above level 0)
    std::string compression = "none";     // none | zlib | vc-delta3d (uint8; {"id": "vc-delta3d", "quant": 1})
    int zlib_level = 1;
    std::string delta3d_preset = "fast";  // the vc-delta3d predictor: fast, medium, slow, slower or delta (not stored)
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

}  // namespace vc_tomography
