#pragma once
// Layer composites (max, mean) along a tifxyz surface, from a reconstructed OME-Zarr block or straight from the raw
// projections (helirec-composite; Python helirec.band binds BandComposite).
//
// Geometry and sampling are vc_reader's (VC3D's): the surface is a grid of bicubic Hermite patches, normals are
// dP/du x dP/dv (u: increasing columns, v: increasing rows), and layer k is the trilinear sample at offset
// direction (start + k step) voxels along the unit normal. From the projections the layers' points are
// reconstructed where they lie (BandComposite: the exact backprojector at the pixels they need, no volume).
//
// Part of helirec_surfaces, which links vc_reader (C++20).
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "helirec/band.hpp"
#include "helirec/reconstruction.hpp"
#include "vc_reader/grid.hpp"
#include "vc_reader/tifxyz.hpp"
#include "vc_reader/zarr.hpp"

namespace helirec::surface {

// The offsets of n layers: direction (start + k step), k = 0 .. n - 1
std::vector<double> layer_offsets(int direction, double start, double step, int n);
// n slice indices spread over [z0, z1) (the centres of n equal parts), distinct and sorted
std::vector<int> spread(int z0, int z1, int n);

// One level of an OME-Zarr block placed in the full volume: level-0 voxel indices from the block's NGFF
// translation (its origin rounded to a whole voxel), the level's index i covering level-0 voxels
// [o + s i, o + s i + s - 1], s = 2^level.
class Volume {
  public:
    Volume(const std::string& path, int level = 0);
    const std::string& path() const { return path_; }
    int level() const { return level_; }
    int factor() const { return s_; }
    double voxel_um() const { return voxel_um_; }                     // of level 0
    const std::array<std::int64_t, 3>& origin0() const { return origin0_; }  // ZYX, level-0 voxels
    const std::array<std::int64_t, 3>& shape() const { return shape_; }      // ZYX of this level
    // the index (ZYX, real) of this level at a level-0 point (XYZ)
    std::array<double, 3> index(const double* xyz) const;
    // [lo, hi] of level-0 coordinates the sample centres cover, ZYX
    std::array<std::array<double, 3>, 2> bounds0() const;
    vcr::grid::Level grid_level();
    // slice z of this level, whole (Y, X), as float32
    std::vector<float> slice(std::int64_t z);

  private:
    std::string path_;
    int level_ = 0, s_ = 1;
    double voxel_um_ = 1;
    std::array<std::int64_t, 3> origin0_{}, shape_{};
    vcr::zarr::OpenedArray array_;
};

// [r0, r1) x [c0, c1) output pixels whose surface can reach the box [lo, hi] (level-0 ZYX), from the stored grid;
// pixel_scale: output pixels per full-resolution pixel (rows, columns). H, W: the canvas.
struct Window {
    int r0 = 0, r1 = 0, c0 = 0, c1 = 0;
    int H = 0, W = 0;
};
Window uv_window(const vcr::tifxyz::Surface& surface, const std::array<double, 3>& lo, const std::array<double, 3>& hi, double reach,
                 const std::array<double, 2>& pixel_scale);

// Positions and unit normals at the output pixels of a raster (vc_reader's tifxyz::Surface::resample), and which
// have both (the rest are 0)
struct Points {
    std::vector<double> xyz, normals;
    std::vector<std::uint8_t> ok;
    int width = 0, height = 0;
};
Points points(const vcr::tifxyz::Surface& surface, const vcr::grid::Raster& raster, vcr::grid::Interpolation interpolation);
vcr::grid::Raster raster_of(const vcr::tifxyz::Surface& surface, const Window& window, const std::array<double, 2>& pixel_scale);

// The layers (L, H, W) over a window of the canvas (vc_reader's grid::sample with its defaults), NaN where there is no value
std::vector<float> sample_layers(const vcr::tifxyz::Surface& surface, Volume& volume, const Window& window, const std::array<double, 2>& pixel_scale,
                                 const std::vector<double>& offsets, vcr::grid::Interpolation interpolation, int tile_rows = 256);

// Which side of the normal holds the material: +1 or -1, the side with the higher mean intensity at offsets
// 1 .. depth, from layers (L, H, W) that include offsets -depth .. depth; evidence: the profile (JSON object text)
struct Side {
    int direction = 1;
    std::string evidence;
};
Side side_from_layers(const std::vector<float>& layers, int H, int W, const std::vector<double>& offsets, int depth = 5);
Side material_side(const vcr::tifxyz::Surface& surface, Volume& volume, const Window& window, const std::array<double, 2>& pixel_scale,
                   vcr::grid::Interpolation interpolation, int depth = 5, int sub = 4);

// One output folder: max.tif, mean.tif (float32, NaN outside), their uint8 versions (.tif, .png), mask.png,
// layers/NN_offset+O.tif, xy_slices/ (float32 and png), surface_volume.zarr, composite.json (meta, a JSON object's
// text, with what was written added)
struct Slice {
    std::int64_t z = 0;
    std::vector<float> values;
    int ny = 0, nx = 0;
};
void write_set(const std::string& out, const std::vector<float>& layers, int H, int W, const std::vector<double>& offsets, const std::vector<std::uint8_t>& good,
               std::string meta, const std::array<double, 2>& u8_window, const std::vector<Slice>& slices, bool surface_zarr);

// --sweep KEY=V1,V2,... items as parameter sets (lists zipped position by position; a one-value list applies to
// every set; filters=CHAIN1;CHAIN2 sweeps whole chains): a JSON list of objects' text
std::string sweep_sets(const std::vector<std::string>& items);
// a set's folder name: db500_us_coeff2 ...
std::string set_name(const std::string& changes);

// Surface composites straight from the projections: the points' slices reconstructed through the band of pixels
// they need (the exact backprojector; whole-slice methods reconstruct the points' box slice by slice instead),
// sampled trilinearly (the whole-slice methods through vc_reader's sampling::sampleDense). Points are XYZ voxel coordinates of the volume: x, y = slice column and row, z = volume
// slice (view height h = z + lo). A pixel is kept when all its layers can be sampled inside [h_range) and the slice.
class BandComposite {
  public:
    // pts: (L, H, W, 3) XYZ; valid: (H, W); h_range: the view heights to reconstruct (none: what the points need);
    // n_sets: value sets (parameter sweeps)
    BandComposite(Reconstruction& recon, const float* pts, int L, int H, int W, const std::uint8_t* valid,
                  std::optional<std::array<int, 2>> h_range = std::nullopt, int n_sets = 1);
    ~BandComposite();
    bool band() const { return sampler_ != nullptr; }
    std::size_t n_points() const;
    const std::vector<std::uint8_t>& valid() const { return valid_; }  // (H, W)
    std::array<int, 2> h_range() const { return h_range_; }
    std::array<int, 4> box() const;  // x0, x1, y0, y1 of the pixels the points need
    // Reconstruct h_range in chunks; sweep: one filter chain per value set (JSON lists' text; empty: the
    // reconstructor's own); returns seconds per stage and "wall" (JSON object text)
    std::string run(int chunk = 64, bool timing = false, const std::vector<std::string>& sweep = {}, bool prefetch = true,
                    const std::function<void(int, int, double)>& progress = nullptr);
    std::vector<float> layers(int set = 0) const;  // (L, H, W), NaN where a pixel is not valid
    // max and mean over the layers (vc_reader's composite::reduce), (H, W) each, NaN where a pixel is not valid
    std::array<std::vector<float>, 2> composites(int set = 0) const;
    int L() const { return L_; }
    int H() const { return H_; }
    int W() const { return W_; }

  private:
    void run_slices(int chunk, const std::vector<std::string>& sweep, const std::function<void(int, int, double)>& progress);
    Reconstruction& r_;
    int L_ = 0, H_ = 0, W_ = 0, n_sets_ = 1;
    std::array<int, 2> h_range_{};
    std::vector<std::uint8_t> valid_;
    std::vector<std::int64_t> sel_, order_;
    std::vector<std::int32_t> zlo_;
    std::vector<float> fz_, y_, x_;
    std::unique_ptr<BandSampler> sampler_;
    std::array<int, 4> box_{};
    std::vector<float> vals_;  // whole-slice methods: (n_sets, points)
};

}  // namespace helirec::surface
