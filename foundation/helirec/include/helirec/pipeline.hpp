#pragma once
// End-to-end reconstruction of a chunk of xy slices from raw scan data:
//   read raw row windows -> grid (flat-field, weights, angular/vertical/horizontal splat)
//   -> luminosity equalisation -> divide -> mend -> filter chain (filters.hpp; before / after -log) -> -log
//   -> rebalance (half acquisition) -> ramp filter -> x balanced weights -> backprojection
// Every stage is a separate kernel call with a CPU and a CUDA implementation.
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "helirec/ghbp.hpp"
#include "helirec/helical.hpp"
#include "helirec/filters.hpp"
#include "helirec/kernels.hpp"
#include "helirec/raw_reader.hpp"

namespace helirec {

// One scan (main = index 0, then companions), with its calibration.
struct ScanData {
    ScanTrajectory traj;
    std::vector<double> control_A;      // per projection machine current
    FlatSeriesInfo flat_info;           // float32 angle of each flat series
    std::vector<float> dark;            // (rows, cols)
    std::vector<float> flats;           // (n_flats, rows, cols): (flat - dark) / (I_flat / 0.2)
    std::vector<float> weight;          // (rows, cols): weight map x horizontal apodisation
    std::shared_ptr<RawReader> reader;  // raw frames
    std::vector<int32_t> file_index, frame_index;  // per projection
};

struct ReconConfig {
    Device device = Device::CPU;
    // projection filters (filters.hpp), e.g. paganin + unsharp; physics for filters that need it
    FilterChain filters;
    double energy_kev = 0, distance_m = 0;
    int margin = -1;                    // <0: the chain's margin (filters' context rows)
    // gridding
    int n_grid = 0;                     // <=0: round(360 / mean step) of the main scan
    double angular_tolerance_steps = 5;
    float luminosity_repartition = 100.f;
    float mend_threshold = 1e-4f;       // mending domain: the merged columns the scans' weight maps cover
    float log_clip_min = 1e-6f, log_clip_max = 10.f;
    int block_angles = 500;
    // reconstruction
    float rot_center = 0.f;             // in the merged frame
    // backprojector: "ghbp" (reference), "fbp" (exact pixel-driven: bp_interp, bp_oversample), "sirt" / "mlem" /
    // "gridrec" (tomo_params: whole-slice methods, tomo.hpp) or "external"
    std::string backprojector = "ghbp";
    // fbp row interpolation: linear | cubic | lanczos3 | bspline3 | bspline5; rows upsampled (band-limited, FFT) by
    // bp_oversample first. Default oversample 2 + bspline3: within 0.1 % of ideal (sinc) interpolation for ~7 % more
    // time (linear alone loses ~10 % of the high-frequency contrast)
    std::string bp_interp = "bspline3";
    int bp_oversample = 2;
    std::shared_ptr<class ExternalBackprojector> external;  // backprojector "external" (e.g. Python)
    FilterParams tomo_params;            // backprojectors "sirt", "mlem", "gridrec" (tomo.hpp)
    int grid_size = 0;                  // <=0: as the reference (2 * rot_center + 1 rounding rule)
    int ghbp_legs = 8, ghbp_reduction_steps = 2;
    bool clip_outer_circle = true;
    double pixel_size_m = 0;            // voxel size (scale = 1 / voxel_cm)
    double axis_source_meters = 1e9;
    // Legacy reference mode: the inherited reference (Nabu / nabuxx) conventions, to reproduce existing
    // reconstructions: half-attenuation output scale (ramp pi / n_angles over pair-balanced full-turn weights),
    // nabuxx rebalance endpoints, integer ramp-padding reflection lround(2 rot_center), mending bounds = the main
    // weight map's first / last column read as left / right widths of the merged row, and the chunk-truncated
    // Paganin kernel. Default (false): attenuation in cm^-1, periodic opposite-view rebalance, reflection about
    // 2 rot_center, mending over the merged columns the scans cover, the fixed unit-DC Paganin kernel.
    // (helirec.recon pins every other parameter of the reference when this mode is selected.)
    bool legacy_reference = false;
};

// Optional intermediate outputs (filled when requested)
struct ChunkDebug {
    bool keep = false;
    std::vector<float> sino;      // (n_grid, DY, NX) -log sinogram before rebalance
    std::vector<float> weights;   // (n_grid, DY, NX) accumulated weights (cropped)
    std::vector<float> wbal;      // (n_grid, DY, NX) balanced weights
    std::vector<float> filtered;  // (DY, n_grid, NX) backprojector input
    // per-stage wall time (seconds); with timing on, the GPU is synchronised at stage boundaries
    bool timing = false;
    std::map<std::string, double> seconds;
    std::map<std::string, double> stats;  // sizes of the chunk (with timing on)
};

struct SliceRoi {   // [x0, x1) x [y0, y1) of the (N, N) slice
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;
};

// A backprojector outside the library (bindings: Python functions). Per slice it gets the rows on the
// reconstructor's device: ramp-filtered x balanced weights (filtered() == true), or the -log rows and the balanced
// weights (filtered() == false, for methods that filter / model the data themselves), and writes the ROI slice to
// `out` (host, (y1 - y0, x1 - x0)). `p`: the exact backprojector's geometry (angles: u = x cos a + y sin a + rot).
class ExternalBackprojector {
  public:
    virtual ~ExternalBackprojector() = default;
    virtual bool filtered() const = 0;
    virtual void slice(int view_height, const float* rows, const float* weights, int n_angles, int n_det,
                       const float* angles, const BackprojectParams& p, int grid_size, const SliceRoi& roi, Device device,
                       float* out) = 0;
};

class BandSampler;

class SliceReconstructor {
  public:
    SliceReconstructor(std::vector<ScanData> scans, ReconConfig cfg);
    ~SliceReconstructor();

    // reconstructable view heights [lo, hi) and info
    std::pair<int, int> doable() const;
    int margin() const;
    int n_grid() const;
    int grid_size() const;
    int merged_width() const;
    std::pair<int, int> mend_domain() const;   // [x0, x1) merged columns the mending works on
    const ReconConfig& config() const;

    // slices for view heights h_lo .. h_hi-1 -> out (h_hi - h_lo, N, N), or (n, y1 - y0, x1 - x0) with an ROI
    // (GHBP computes whole slices and crops; FBP computes only the ROI)
    void reconstruct(int h_lo, int h_hi, float* out, ChunkDebug* debug = nullptr, const SliceRoi* roi = nullptr);
    // same chunk, but only the band of pixels the sampler's points need, with the exact backprojector; the
    // sampler accumulates its point samples (call with contiguous, increasing chunks)
    // `sweep`: several filter chains (CUDA): everything before the filters is computed once per chunk and the
    // sampler's value set k receives chain k; the margin is the largest any chain needs (sweep_margin). Chains that
    // are one Fourier step at the intensity stage (e.g. paganin + unsharp) also share the forward FFT.
    // [next_lo, next_hi): the chunk that will follow; its raw windows are read in the background (CUDA)
    void reconstruct_band(int h_lo, int h_hi, BandSampler& band, ChunkDebug* debug = nullptr,
                          const std::vector<FilterChain>* sweep = nullptr, int next_lo = 0, int next_hi = 0);
    int sweep_margin(const std::vector<FilterChain>& chains) const;  // margin rows used for a sweep
    std::string describe(const FilterChain& chain) const;            // compiled steps, e.g. fft[paganin*unsharp]@intensity m20

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
