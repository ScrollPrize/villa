#pragma once
// End-to-end reconstruction of a chunk of xy slices from raw scan data:
//   read raw row windows -> grid (flat-field, weights, angular/vertical/horizontal splat)
//   -> luminosity equalisation -> divide -> mend -> filter chain (filters.hpp; before / after -log) -> -log
//   -> rebalance (half acquisition) -> ramp filter -> x balanced weights -> backprojection
// Every stage is a separate kernel call with a CPU and a CUDA implementation.
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "helirec/block_sink.hpp"
#include "helirec/ghbp.hpp"
#include "helirec/helical.hpp"
#include "helirec/filters.hpp"
#include "helirec/kernels.hpp"
#include "helirec/progress.hpp"
#include "helirec/raw_reader.hpp"
#include "helirec/sld.hpp"

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
    // scintillator light diffusion correction of the raw frames (sld.hpp); fraction 0: none
    SldParams sld;
    // gridding
    int n_grid = 0;                     // <=0: round(360 / mean step) of the main scan
    double angular_tolerance_steps = 5;
    float luminosity_repartition = 100.f;
    // fractional column offset of a scan in the merged frame: "lanczos3" (default), "cubic", or "linear" = the
    // reference's two-column splat, which averages neighbouring columns (cos(pi nu) at half a pixel). The legacy
    // reference mode always uses the splat.
    std::string x_interp = "lanczos3";
    // vertical position of a slice in a projection (a fractional detector row): "lanczos3" (default) or "cubic"
    // (6 / 4 rows; accumulate.hpp), or "linear" = the reference's two rows, a triangle kernel along z (sinc^2). The
    // legacy reference mode always interpolates linearly.
    std::string z_interp = "lanczos3";
    // angular gridding: "nearest" (default) = each projection whole in one bin and every slice backprojected at the
    // measured mean angle of each bin (helical.hpp nearest_angle_bins): no angular averaging; "linear" = the
    // reference's splat of each projection over the two grid angles around its own. Nearest needs a backprojector that
    // takes the angles of each slice (fbp, the whole-slice methods, bands): ghbp, whose plan fixes its angles, and the
    // legacy reference mode keep the splat (SliceReconstructor::angle_interp() tells which is in use).
    std::string angle_interp = "nearest";
    // continuous rotation: fraction of the angular step between projections swept during one exposure (0: none, or
    // step scans). > 0: that box blur along the angle is deconvolved in every sinogram (sino.hpp angular_fir).
    double exposure_arc = 0;
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
    // ramp filter of "ghbp", "fbp" and filtered "external": ramlak, or a windowed ramp (sino.hpp ramp_windowed)
    std::string ramp_filter_name = "ramlak";
    std::vector<float> ramp_filter_par;  // {p0, p1}; empty: 0.5, 8
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
    // weight map's first / last column read as left / right widths of the merged row, the chunk-truncated
    // Paganin kernel, the two-column splat of fractional column offsets, the luminosity window clipped to the
    // chunk and context rows gridded only from the projections of the chunk's own heights (the default reads the
    // window's 12 context rows either side and grids context rows from every projection that sees them). Default (false): attenuation in cm^-1, periodic opposite-view rebalance, reflection about
    // 2 rot_center, mending over the merged columns the scans cover, the fixed unit-DC Paganin kernel.
    // (helirec.recon pins every other parameter of the reference when this mode is selected.)
    bool legacy_reference = false;
};

// Optional intermediate outputs (filled when requested)
struct ChunkDebug {
    bool keep = false;
    std::vector<float> sino;      // (n_grid, DY, NX) -log sinogram before rebalance
    std::vector<float> weights;   // (n_grid, DY, NX) accumulated weights (cropped)
    std::vector<float> wbal;      // backprojector weights: balanced for FBP/gridrec, measured support for SIRT/MLEM
    std::vector<float> filtered;  // (DY, n_grid, NX) backprojector input
    std::vector<float> angles;    // (DY, n_grid) angle of each grid bin in each slice (radians)
    // per-stage wall time (seconds); with timing on, the GPU is synchronised at stage boundaries
    bool timing = false;
    std::map<std::string, double> seconds;
    std::map<std::string, double> stats;  // sizes of the chunk (with timing on)
};

struct SliceRoi {   // [x0, x1) x [y0, y1) of the (N, N) slice
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;
};

// A mask of a chunk's uint8 slices, possibly coarser than them (e.g. a pyramid level of a masked reference volume,
// reference.hpp): output voxel (s, y, x) (slice of the chunk, row and column of the ROI) is 0 where
// data[(s + oz) / fz, (y + oy) / fy, (x + ox) / fx] is 0, indices clamped to the mask.
struct OutputMask {
    const std::uint8_t* data = nullptr;   // (nz, ny, nx), host
    int nz = 0, ny = 0, nx = 0;
    int fz = 1, fy = 1, fx = 1;
    int oz = 0, oy = 0, ox = 0;
};

// Where the slices of a chunk go, float32 and / or uint8: host arrays (n, y1 - y0, x1 - x0), or sinks (block_sink.hpp,
// e.g. an OmeZarrWriter) that take each slice as the block of it that was computed, so that no whole slice is built:
// slice y of the chunk is the sink's slice sink_z + y, the block in the coordinates of the ROI. The uint8 version is
// to_uint8 (kernels.hpp) of each slice with the window [u8_lo, u8_hi], 0 outside the reconstruction disk with
// u8_zero_outside and where u8_mask is 0; on CUDA it is converted on the device, so a uint8-only chunk downloads
// one byte per pixel.
// Only the pixels that are needed are backprojected: those of `region` (slice coordinates; the rest of the output is
// 0), and with uint8 slices alone, those within the bounds of what u8_mask keeps, and for the exact backprojector
// only its tiles that hold a kept pixel (the others are 0 either way). The exact backprojector computes each pixel on
// its own, so the pixels computed are those of a whole-slice run; a chunk with none is not reconstructed at all.
struct SliceOutput {
    float* f32 = nullptr;
    std::uint8_t* u8 = nullptr;
    BlockSink* f32_sink = nullptr;
    BlockSink* u8_sink = nullptr;
    std::int64_t sink_z = 0;
    double u8_lo = 0.0, u8_hi = 1.0;
    bool u8_zero_outside = false;
    OutputMask u8_mask;
    std::optional<SliceRoi> region;
};

// A backprojector outside the library (bindings: Python functions). Per slice it gets the rows on the
// reconstructor's device: ramp-filtered x balanced weights (filtered() == true), or the -log rows and the balanced
// weights (filtered() == false, for methods that filter / model the data themselves); measured_only() instead
// requests measured support and values before hole filling / balancing. Writes the ROI slice to
// `out` (host, (y1 - y0, x1 - x0)). `p`: the exact backprojector's geometry (angles: u = x cos a + y sin a + rot).
class ExternalBackprojector {
  public:
    virtual ~ExternalBackprojector() = default;
    virtual bool filtered() const = 0;
    // Request measured-only gridding, with support preserved before any hole filling or rebalance.
    virtual bool measured_only() const { return false; }
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
    double exposure_arc_bins() const;          // the deconvolved exposure arc in grid angles (0: none)
    std::string angle_interp() const;          // the angular gridding in use: "nearest" or "linear"
    std::string z_interp() const;              // the vertical kernel in use
    const ReconConfig& config() const;

    // slices for view heights h_lo .. h_hi-1 -> out (h_hi - h_lo, N, N), or (n, y1 - y0, x1 - x0) with an ROI
    // (GHBP computes whole slices and crops; FBP computes only the ROI)
    // [next_lo, next_hi): the chunk that will follow; its raw windows are read in the background (CUDA)
    // progress: advanced by one per slice, with the step in hand ("read", "grid", "backproject") as its postfix
    void reconstruct(int h_lo, int h_hi, const SliceOutput& out, ChunkDebug* debug = nullptr, const SliceRoi* roi = nullptr,
                     int next_lo = 0, int next_hi = 0, ProgressBar* progress = nullptr);
    void reconstruct(int h_lo, int h_hi, float* out, ChunkDebug* debug = nullptr, const SliceRoi* roi = nullptr,
                     int next_lo = 0, int next_hi = 0) {
        SliceOutput o;
        o.f32 = out;
        reconstruct(h_lo, h_hi, o, debug, roi, next_lo, next_hi);
    }
    // same chunk, but only the band of pixels the sampler's points need, with the exact backprojector; the
    // sampler accumulates its point samples (call with contiguous, increasing chunks)
    // `sweep`: several filter chains (CUDA): everything before the filters is computed once per chunk and the
    // sampler's value set k receives chain k; the margin is the largest any chain needs (sweep_margin). Chains that
    // are one Fourier step at the intensity stage (e.g. paganin + unsharp) also share the forward FFT.
    void reconstruct_band(int h_lo, int h_hi, BandSampler& band, ChunkDebug* debug = nullptr,
                          const std::vector<FilterChain>* sweep = nullptr, int next_lo = 0, int next_hi = 0);
    int sweep_margin(const std::vector<FilterChain>& chains) const;  // margin rows used for a sweep
    std::string describe(const FilterChain& chain) const;            // compiled steps, e.g. fft[paganin*unsharp]@intensity m20

    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
