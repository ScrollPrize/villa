#include "vc_tomography/pipeline.hpp"

#include "common/interp_core.hpp"
#include "common/output_core.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <map>
#include <cmath>
#include <cstring>

#include "vc_tomography/accumulate.hpp"
#include "vc_tomography/band.hpp"
#include "vc_tomography/tomo.hpp"
#include "vc_tomography/fourier.hpp"
#include "vc_tomography/kernels.hpp"
#include "vc_tomography/sino.hpp"
#include "io/json.hpp"
#include "io/json_values.hpp"

#ifdef VC_TOMOGRAPHY_WITH_CUDA
#include "../cuda/cuda_utils.cuh"
#endif

namespace vc_tomography {

namespace {

// first / last column with column-sum >= thr * max (nabuxx get_weight_vanishing_margins_left_right)
std::pair<int, int> weight_vanishing_margins(const std::vector<float>& w, int rows, int cols, float thr) {
    std::vector<double> prof(cols, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) prof[c] += w[static_cast<std::size_t>(r) * cols + c];
    const double mx = *std::max_element(prof.begin(), prof.end());
    int first = 0, last = cols - 1;
    for (int c = 0; c < cols; ++c)
        if (prof[c] >= thr * mx) { first = c; break; }
    for (int c = cols - 1; c >= 0; --c)
        if (prof[c] >= thr * mx) { last = c; break; }
    return {first, last};
}

// angle_interp "views": the most turns the projections that see a chunk may span (the rows of its sinogram; the
// recordings of a line, two per turn, are then at most the rebalance's table: sino.hpp)
constexpr int kMaxViewTurns = 16;

// rows of the merged frame gridded at once (grid angles x rows of the stack): see Impl::run
constexpr int kBlockRows = 24000;

struct ScanChunk {  // host data of one scan for the current chunk
    ChunkScan cs;
    std::vector<uint16_t> raw;
    int win_rows = 0;
    std::vector<int32_t> win0;   // first raw row of each projection's window (row0 - vertical kernel context)
    std::vector<float> z_w;      // vertical kernel weights (z_kernel_weights)
};

std::vector<int32_t> window_rows(const ChunkScan& c, int z_pad) {
    std::vector<int32_t> w(c.row0);
    for (auto& r : w) r -= z_pad;
    return w;
}

// the vertical kernel of a view (z_w: host or device copy of ScanChunk::z_w)
// z_pad: context rows of the window; z_ctx: those the kernel needs (the rest covers the frames' roll)
void set_z_kernel(AccumScanView& v, int z_pad, int z_ctx, const float* z_w) {
    v.win_off = z_pad;
    v.z_taps = z_ctx > 0 ? 2 + 2 * z_ctx : 0;
    v.z_w = z_ctx > 0 ? z_w : nullptr;
}

// the roll of a scan's frames (ScanTrajectory::row_slope; slope, lut: host or device copies)
void set_roll(AccumScanView& v, const ScanData& sd, const float* slope, const float* lut, int lut_n) {
    if (sd.traj.row_slope.empty()) return;
    v.row_slope = slope;
    v.slope_col = sd.traj.slope_col;
    v.z_lut = v.z_taps > 0 ? lut : nullptr;
    v.z_lut_n = lut_n;
}

// the lateral registration of a scan's frames (ScanTrajectory::col_offset, col_slope; host or device copies)
// exact: the pipeline shifts columns exactly (x_interp = sinc); otherwise the lateral registration is not applied
void set_lateral(AccumScanView& v, const ScanData& sd, const ChunkScan& c, const float* offset, const float* slope, bool exact) {
    if (c.col_offset.empty() || !exact) return;
    v.col_offset = offset;
    v.col_slope = slope;
    v.slope_row = sd.traj.slope_row;
    v.x_exact = 1;   // (set_x_kernel leaves it off for a scan at a whole column offset)
}

// The detector map of a scan and the scales of its projections (ScanTrajectory::map_*, row_scale, col_scale), as
// planned by the reconstructor: `host` views the scan's own arrays (a device copy is the same with other pointers).
struct ScanMap {
    DetectorMapView host;
    std::vector<float> ev_f;       // map_ev in single precision (empty: no vertical terms)
    bool rows = false;             // a vertical part: terms with ev, or row_scale
    bool cols = false;             // a lateral part: terms with eu, or col_scale
    float reach = 0;               // the largest row displacement on the detector (rows)
    int col_guard = 3;             // AccumScanView::col_guard: 3 + the largest lateral displacement (columns, rounded up)
};

// map: the view of the scan's map in the backend's pointer space; row_scale, col_scale, lut: host or device copies.
// (after set_z_kernel)
void set_map(AccumScanView& v, const ScanData& sd, const ScanMap& m, const DetectorMapView& map, const float* row_scale,
             const float* col_scale, const float* lut, int lut_n) {
    v.col_guard = m.col_guard;
    if (!m.rows && !m.cols) return;
    v.map = map;
    v.map_rows = m.rows ? 1 : 0;
    v.map_cols = m.cols ? 1 : 0;
    v.row_scale = sd.traj.row_scale.empty() ? nullptr : row_scale;
    v.scale_row = sd.traj.scale_row;
    v.col_scale = sd.traj.col_scale.empty() ? nullptr : col_scale;
    v.scale_col = sd.traj.scale_col;
    if (m.rows) {
        v.z_lut = v.z_taps > 0 ? lut : nullptr;
        v.z_lut_n = lut_n;
    }
    if (m.cols) v.x_exact = 1;   // (the pipeline shifts columns exactly: checked by the reconstructor)
}

AccumScanView make_view(const ScanData& sd, const ScanChunk& ch, const uint16_t* raw, const float* dark,
                        const float* flats, const float* weight, const int32_t* row0, const float* fract_v,
                        const float* alpha, const float* cdiv, const float* wfac, const float* ffrac,
                        const int32_t* fi0, const int32_t* fi1, int x_kernel, bool x_exact) {
    AccumScanView v;
    v.n_proj = ch.cs.p1 - ch.cs.p0;
    v.det_rows = sd.traj.det_rows;
    v.det_cols = sd.traj.det_cols;
    v.win_rows = ch.win_rows;
    v.raw = raw; v.dark = dark; v.flats = flats; v.weight = weight;
    v.row0 = row0; v.fract_v = fract_v; v.alpha = alpha; v.current_div = cdiv; v.weight_factor = wfac;
    v.flat_frac = ffrac; v.flat_i0 = fi0; v.flat_i1 = fi1;
    v.sx = ch.cs.sx; v.ixs = ch.cs.ixs; v.fract_x = ch.cs.fract_x;
    set_x_kernel(v, x_kernel, x_exact);
    return v;
}

// noise splitting (ReconConfig::noise_split_scale): the view's share of it; salt: the scan's (Impl::split_salt)
void set_noise_split(AccumScanView& v, const ReconConfig& cfg, std::uint32_t salt, int first_projection) {
    if (cfg.noise_split_scale == 0.f) return;
    v.split_scale = cfg.noise_split_scale;
    v.split_gain = cfg.noise_split_gain;
    v.split_floor = cfg.noise_split_floor;
    v.split_key = cfg.noise_split_seed * 0x9e3779b1U ^ salt;
    float total = 1.f;
    for (int i = 0; i < 2; ++i) {
        v.split_amp[i] = std::sqrt(cfg.noise_split_smooth_share[i]);
        v.split_spacing[i] = cfg.noise_split_smooth_spacing[i];
        total += cfg.noise_split_smooth_share[i];
    }
    v.split_norm = 1.f / std::sqrt(total);
    v.split_flat_frames = cfg.noise_split_flat_frames;
    v.split_proj0 = first_projection;
}

void set_glare(AccumScanView& v, const ScintillatorGlare& g, const float* maps, const int32_t* i0, const int32_t* i1,
               const float* frac, int rows, const int32_t* row0) {
    v.glare = maps;
    v.glare_rows = rows;
    v.glare_cols = g.cols();
    v.glare_inv_bin = 1.f / static_cast<float>(g.bin());
    v.glare_i0 = i0;
    v.glare_i1 = i1;
    v.glare_frac = frac;
    v.glare_row0 = row0;
}

}  // namespace

struct SliceReconstructor::Impl {
    std::vector<ScanData> scans;
    ReconConfig cfg;
    std::vector<SpanInfo> spans;
    std::pair<int, int> doable;
    int margin = 0, n_grid = 0, NX = 0, grid_size = 0;
    int filter_margin = 0;   // margin of single-step filter chains (cfg.margin, or the chain's own)
    int lum_margin = 0;      // context rows of the luminosity window, added to the filters' in `margin`
    int x_kernel = 0;        // interp_core::Interp of fractional column offsets (legacy reference: Linear, the splat)
    bool x_exact = false;    // x_interp=sinc: the exact shift (measured pixels alone: the lanczos3 kernel, or col_wt)
    // band-dependent redundancy weights (PipelineConfig::sharpness_weighting)
    bool sharp = false;
    double sharp_f1 = 0, sharp_f2 = 0;
    std::vector<std::vector<float>> weight2;          // per scan: its weight map x the detector's squared response
    std::unique_ptr<FourierFilter> band2d, band1d;    // the high-pass on frames (H, NX) and along sinogram rows
    int band2d_h = -1;
    int z_kernel = 0, z_pad = 0;   // vertical kernel and its context rows either side of the window (linear: 0)
    int z_ctx = 0;                 // ... of which the kernel's own (the rest covers the frames' roll: row_slope)
    static constexpr int z_lut_n = 2048;
    std::vector<float> z_lut;      // the kernel's weights by fraction (z_kernel_lut), for scans with a roll
    std::vector<ScanMap> maps;     // per scan: its detector map and scales (ScanTrajectory::map_*, row_scale, col_scale)
    // nearest-bin angular gridding: each scan's bins (helical.hpp), the mean weight of each detector row (how much
    // a projection weighs in a slice), and the angles of the chunk's slices (DY, n_grid), -angle as fbp_angles
    bool nearest = false;
    std::vector<AngleBins> bins;
    std::vector<std::vector<double>> row_weight;
    std::vector<float> slice_fbp_angles;
    // The beam's divergence as the exact backprojector takes it (ReconConfig::fan_per_column): the fan [rad] of
    // every merged column; empty: a parallel beam.
    std::vector<float> fan;
    // One row per view (angle_interp "views": nearest-bin gridding without the wrap to one turn). The chunk's sinogram
    // has chunk_rows >= n_grid rows, row r holding the projections of the (unwrapped) bin row_base + r, so that views
    // a turn apart, which share a grid angle but were not recorded at the same angle, stay apart and are each
    // backprojected at their own. chunk_rows = n_grid otherwise. slice_opposite (DY, chunk_rows): the row whose
    // mirrored values pad each row for the ramp filter, the opposite view that weighs most in the slice.
    bool views = false;
    int chunk_rows = 0;
    long row_base = 0;
    std::vector<int32_t> slice_opposite;
    const float* fbp_angles_of(int y) const {
        return nearest ? slice_fbp_angles.data() + static_cast<std::size_t>(y) * chunk_rows : fbp_angles.data();
    }
    void set_slice_angles(const std::vector<ChunkScan*>& cs, int DY, int m);
    // exposure arc: taps of its inverse along the angle (empty: none), and the arc in grid angles
    std::vector<float> arc_taps;
    double arc_bins = 0;
    double recorded_step = 0;   // degrees between consecutive recorded projections of the main scan
    std::vector<float> arc_h;
    // context rows gridded from every projection that sees them (prepare_chunk_scan); the legacy reference grids
    // them from the projections of the chunk's own heights, as the reference does
    bool full_context = false, measured = false;
    // Per scan, the place across the beam it stands at (scans within a quarter of the detector's width of each other
    // in the merged frame and more than a quarter of its height apart stand at one), and whether some place holds
    // several scans: rows of scans above each other (a grid of circular scans). A height is then reconstructed from the scans that see it, one at each place at
    // least (doable_span with places), and a chunk may hold none of a scan's projections.
    std::vector<int> place;
    bool rows_of_scans = false;
    // per scan, the row of scans it belongs to (the first scan within a quarter of the detector's rows of its height)
    std::vector<int> scan_row;
    int mend_x0 = 0, mend_x1 = 0;   // mending domain [mend_x0, mend_x1) of the merged row
    // per scan: the merged column of its luminosity seam with the scans before it (the middle of the columns it
    // shares with the nearest earlier scan at its own height); -1: no earlier scan at its height (the first scan,
    // and the first of every further row of a grid), which is not taken to anyone's level
    std::vector<int> seam;
    LuminosityKernel lum = luminosity_kernel();
    // per-chunk invariants (built on first use)
    std::vector<float> angles;             // grid angles (radians)
    std::vector<float> fbp_angles;         // -angles: the exact backprojector's u = x cos - y sin + c matches GHBP's
                                           // h = rot + (j - c) cos A - (i - c) sin A (row i downwards)
    RampGeometry rg;
    BackprojectParams bp;
    GhbpParams gp;
    bool have_plan = false;
    GhbpPlan plan;
    std::unique_ptr<FourierFilter> ramp_filter;
    // filter chains, compiled (cached by their text form) and their FFT filters (plans kept, cached by slot+geometry)
    FilterContext fctx;
    // shared ownership: a running chunk / sweep keeps its pipelines alive even if the cache evicts them
    mutable std::map<std::string, std::shared_ptr<const FilterPipeline>> pipelines;
    // FFT filters by slot + geometry; cur_id: FourierFilterSpec::id of the coefficients loaded (never an address)
    struct FftEntry { int slot; int ph, pw, h, w, pad_top, pad_left; std::unique_ptr<FourierFilter> f; std::uint64_t cur_id; };
    std::vector<FftEntry> fft_cache;
    std::shared_ptr<const FilterPipeline> pipeline(const FilterChain& chain) const;
    FourierFilter* fft_for(int slot, const FourierFilterSpec& spec);
    // the steps of `stage` of chain P on frames R (L, H, NX); `m`: margin of single-step chains (run / sweep margin)
    void apply_filters(const FilterPipeline& P, FilterStage stage, float* R, int L, int H, int m, int W = 0);
    // scintillator glare (cfg.sld), per scan: the model, and the maps of the sampled projections by projection index
    // (kept while consecutive chunks need them)
    std::vector<std::unique_ptr<ScintillatorGlare>> glare;
    std::vector<std::map<int, std::vector<float>>> glare_maps;
    struct ChunkGlare {
        int rows = 0;
        std::vector<float> maps;        // (n_maps, rows, cols): the sampled projections bracketing the chunk's
        std::vector<int32_t> i0, i1;    // per chunk projection: the maps before / after it
        std::vector<int32_t> row0;      // BM18: retain only the rows this chunk samples, per map
        std::vector<float> frac;
    };
    ChunkGlare chunk_glare(std::size_t s, const ScanChunk& chunk);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    struct DevScan {
        cuda::DeviceBuffer<float> dark, flats, weight, weight2, map_ev_f;
        cuda::DeviceBuffer<int32_t> map_i, map_j;
        cuda::DeviceBuffer<double> map_eu, map_ev;
        DetectorMapView map;   // ScanMap::host with the device copies
    };
    std::vector<DevScan> dev;
    // persistent GPU work buffers (grow-only), pinned raw staging, copy stream
    std::vector<cuda::PinnedBuffer<uint16_t>> pinned;           // whole chunk raw windows, per scan
    std::vector<cuda::DeviceBuffer<uint16_t>> d_raw[2];         // per-block projection subsets (ring of 2), per scan
    std::vector<cuda::DeviceBuffer<int32_t>> d_slot[2];         // projection -> slot in d_raw, per scan
    cudaEvent_t ev_up[2] = {nullptr, nullptr}, ev_done[2] = {nullptr, nullptr};
    // read-ahead of the next chunk's raw windows (background thread, second pinned set)
    std::vector<cuda::PinnedBuffer<uint16_t>> pinned_next;
    // the chunk whose raw windows `pinned` holds (its 2m + 1 bottom rows are the next chunk's top rows)
    struct Held { int lo = 0, hi = 0, m = -1; std::vector<int> p0, p1; std::vector<std::vector<int32_t>> row0; };
    Held held;
    bool reuse_overlap = true;
    // Projections the raw buffers are sized for: the most any chunk of this height has. Growing a page-locked buffer
    // of GBs mid-run stalls every CUDA call for seconds, and chunks along a scan see different numbers of projections.
    int cap_rows = -1, cap_m = -1;
    std::vector<std::size_t> cap_proj;
    std::size_t raw_capacity(std::size_t s, int rows, int m) {
        if (rows != cap_rows || m != cap_m) {
            cap_proj.assign(scans.size(), 0);
            for (std::size_t i = 0; i < scans.size(); ++i)
                cap_proj[i] = static_cast<std::size_t>(max_chunk_projections(scans[i].traj, spans[i], doable.first, doable.second,
                                                                             rows, m, full_context, !measured, rows_of_scans));
            cap_rows = rows;
            cap_m = m;
        }
        return cap_proj[s];
    }
    std::thread pf_thread;
    int pf_lo = 0, pf_hi = 0, pf_m = -1;
    std::exception_ptr pf_err;
    void start_prefetch(int lo, int hi, int m);
    bool take_prefetch(int lo, int hi, int m);  // true: pinned now holds that chunk's raw windows
    ~Impl() {
        if (pf_thread.joinable()) pf_thread.join();
        if (cfg.device == Device::CUDA) release_device_scratch();   // (the kernels' thread_local scratch: see common.hpp)
    }
    cuda::DeviceBuffer<float> R, RW, Wp, Aha, Bha, Atot, Btot, S, W, row, wrow, pad, slice, arc_d;
    cuda::DeviceBuffer<float> R2, RW2, Wp2, Aha2, Bha2, Atot2, Btot2, W2, w2row, brow;   // (band-dependent weights)
    cuda::DeviceBuffer<float> Rr, RWr, R2r, RW2r;   // one row of scans' sums (rows of scans with the luminosity step)
    cuda::PinnedBuffer<float> slice_host;   // staging of the slices' download
    cuda::DeviceBuffer<std::uint8_t> slice_u8, slice_mask;
    cuda::PinnedBuffer<std::uint8_t> slice_u8_host;
    std::unique_ptr<cuda::Ghbp> ghbp;
    std::unique_ptr<cuda::Stream> copy;
#endif

    void setup_invariants();
    // grid the chunk's sinogram, rebalance, then for each filter chain k and slice y: ramp filter x balanced
    // weights -> sink(k, y, row, wrow) (row: (n_grid, NX) filtered sinogram, wrow: balanced weights; device pointers on
    // CUDA, host on CPU; with raw_rows, row is the unfiltered -log sinogram).
    // Several chains (CUDA only) share everything before the filters; `margin` must suit every chain.
    // margin_rows: context rows read either side of the chunk; filter_rows: the margin of single-step filter chains
    // (their horizontal padding; <= margin_rows, which may add the luminosity window's rows)
    void run(int h_lo, int h_hi, ChunkDebug* dbg, const std::vector<const FilterPipeline*>& sets, int margin_rows,
             int filter_rows, const std::function<void(int, int, float*, float*)>& sink, int next_lo = 0, int next_hi = 0);
    // The steps of run() written once for both backends (Ops: CpuOps or CudaOps below; pointers of its memory).
    struct GridBuffers {   // a block of grid angles (L, H, NX): sums, weights, and the luminosity step's scratch
        float *R, *RW, *R2, *RW2, *Rr, *RWr, *R2r, *RW2r, *Wp, *Wp2, *Aha, *Bha, *Atot, *Btot, *Aha2, *Bha2, *Atot2, *Btot2;
    };
    // grid angles [g0, g0 + L) of every scan into b.R / b.RW (and the second weighting's), the luminosity step, the
    // division, mending and the band combination. accumulate(s, R, RW, R2, RW2): scan s's projections added;
    // done(s): after scan s (skipped or added); ready(): after the sums are cleared, before any scan
    template <class Ops, class Accumulate, class Done, class Ready, class Tick>
    void grid_block(int L, int H, const std::vector<ScanChunk>& chunks, const GridBuffers& b, bool sharp, bool by_row,
                    const AccumParams& ap, bool filter_context, Accumulate&& accumulate, Done&& done, Ready&& ready, Tick&& tick);
    // the block's intensity filters, -log, the log filters
    template <class Ops, class Tick>
    void filter_block(const FilterPipeline& P, float* R, int L, int H, int fm, const AccumParams& ap, Tick&& tick);
    // the chunk's sinogram (NR, DY, NX) and weights balanced: holes filled, opposite views weighted
    template <class Ops>
    void balance(float* S, float* W, float* W2, int NR, int DY, int NX, bool sharp, bool measured_only);
    struct SliceBuffers {   // (NR, NX) rows of one slice, the padded rows, scratch
        float *row, *wrow, *w2row, *brow, *pad, *arc, *start, *start_w;
        const int32_t* opposite;   // slice_opposite in the backend's memory (views)
    };
    // the slices of the chunk for chain k: ramp filter x balanced weights per slice -> sink(k, y, row, wrow)
    template <class Ops, class Tick>
    void slices(int k, const float* S, const float* W, const float* W2, int NR, int DY, int NX, bool sharp, const RampGeometry& rg,
                const SliceBuffers& b, ChunkDebug* keep_filtered, bool measured_only,
                const std::function<void(int, int, float*, float*)>& sink, Tick&& tick);
    bool raw_rows = false;  // sinks get the unfiltered rows (external backprojectors that filter themselves)
    // An external backprojector that starts from the slice's FBP (ExternalBackprojector::fbp_start; 0: none). run()
    // then makes, beside the rows the sink gets, the rows the exact backprojector takes of that slice, from a copy
    // mended and balanced as backprojector "fbp"'s are: start_rows, set for the sink's call (device pointer on CUDA).
    int fbp_start = 0;
    const float* start_rows = nullptr;
    std::vector<std::uint32_t> split_salt;   // noise splitting: a number of each scan (of its first file's name)
    std::vector<float> start_h, start_wh, start_img_h;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::DeviceBuffer<float> start_d, start_wd, start_img_d;
#endif
    ProgressBar* progress = nullptr;   // of the chunk in hand: run() names its steps
    void step(const char* name) {
        if (progress) progress->set_postfix(name);
    }
    std::vector<float> up_h;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::DeviceBuffer<float> up_d;
#endif
    // rows for the exact backprojector: upsampled by bp_oversample, then B-spline coefficients for bspline3/5 (else the
    // rows themselves)
    const float* fbp_rows(const float* row) {
        const int os = cfg.bp_oversample, deg = interp_core::spline_degree(bp.interp);
        if (os <= 1 && !deg) return row;
        const std::size_t n = static_cast<std::size_t>(chunk_rows) * NX * os;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (cfg.device == Device::CUDA) {
            up_d.ensure(n);
            if (os > 1) cuda::upsample_rows(row, up_d.data(), chunk_rows, NX, os);
            if (deg) cuda::bspline_prefilter(os > 1 ? up_d.data() : row, up_d.data(), chunk_rows, NX * os, deg);
            return up_d.data();
        }
#endif
        up_h.resize(n);
        if (os > 1) cpu::upsample_rows(row, up_h.data(), chunk_rows, NX, os);
        if (deg) cpu::bspline_prefilter(os > 1 ? up_h.data() : row, up_h.data(), chunk_rows, NX * os, deg);
        return up_h.data();
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::DeviceBuffer<float> Icache;   // sweeps: pre-filter intensity of the chunk, or (one Fourier step) its spectra
    cuda::DeviceBuffer<float> d_z_lut;
    cuda::DeviceBuffer<int32_t> d_opposite;   // slice_opposite
#endif
};

SliceReconstructor::SliceReconstructor(std::vector<ScanData> scans, ReconConfig cfg) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    if (scans.empty()) throw Error("SliceReconstructor: no scans");
    I.scans = std::move(scans);
    // A pixel without a flat signal (its flat is not above its dark: a detector's time stamp, a blank row, a dead
    // pixel) has no flat-fielded value, so it carries no weight, whatever the weight map gave it.
    for (auto& s : I.scans) {
        const std::size_t plane = static_cast<std::size_t>(s.traj.det_rows) * s.traj.det_cols;
        if (!plane || s.weight.size() != plane || s.flats.size() % plane) continue;   // (the sizes are checked below)
        for (std::size_t f = 0; f < s.flats.size() / plane; ++f)
            for (std::size_t i = 0; i < plane; ++i)
                if (!(s.flats[f * plane + i] > 0.f) || !std::isfinite(s.flats[f * plane + i])) s.weight[i] = 0.f;
    }
    I.cfg = std::move(cfg);
    if (is_tomo_method(I.cfg.backprojector)) {  // a whole-slice method, run as an external backprojector
        I.cfg.external = make_tomo_backprojector(I.cfg.backprojector, I.cfg.tomo_params);
        I.cfg.backprojector = "external";
    }
    I.fbp_start = I.cfg.external ? I.cfg.external->fbp_start() : 0;
    if (I.cfg.noise_split_scale != 0.f) {
        if (I.cfg.legacy_reference) throw Error("the legacy reference mode has no noise splitting");
        if (!std::isfinite(I.cfg.noise_split_scale) || !(I.cfg.noise_split_gain >= 0.f) || !(I.cfg.noise_split_floor >= 0.f) ||
            !(I.cfg.noise_split_gain + I.cfg.noise_split_floor > 0.f) || !std::isfinite(I.cfg.noise_split_gain + I.cfg.noise_split_floor))
            throw Error("noise splitting: a finite scale and a noise variance gain x level + floor that is not zero");
        if (!(I.cfg.noise_split_flat_frames >= 0.f) || !std::isfinite(I.cfg.noise_split_flat_frames))
            throw Error("noise splitting: the frames of a flat must not be negative");
        for (int i = 0; i < 2; ++i)
            if (!(I.cfg.noise_split_smooth_share[i] >= 0.f) || !std::isfinite(I.cfg.noise_split_smooth_share[i]) ||
                (I.cfg.noise_split_smooth_share[i] > 0.f && !(I.cfg.noise_split_smooth_spacing[i] >= 1.f && I.cfg.noise_split_smooth_spacing[i] <= 1024.f)))
                throw Error("noise splitting: a smooth field has a share of the variance that is not negative and a lattice of 1 to 1024 pixels");
        for (const auto& s : I.scans) {
            if (s.reader && s.reader->reduce() != 0) throw Error("noise splitting is of the frames as stored, not of reduced ones");
            std::uint32_t h = 2166136261U;   // FNV-1a of the first file's name without its folder
            if (s.reader && s.reader->n_files()) {
                const std::string& f = s.reader->file(0);
                for (std::size_t i = f.find_last_of('/') == std::string::npos ? 0 : f.find_last_of('/') + 1; i < f.size(); ++i)
                    h = (h ^ static_cast<unsigned char>(f[i])) * 16777619U;
            }
            I.split_salt.push_back(h);
        }
    }
    if (I.fbp_start && I.cfg.legacy_reference) throw Error("the legacy reference mode has no FBP start for a whole-slice method (init=fbp)");
    for (const auto& s : I.scans)
        if (s.traj.n() != s.control_A.size() || s.traj.n() != s.file_index.size() || s.traj.n() != s.frame_index.size() ||
            s.traj.n() != s.traj.z_translation_m.size() ||
            (!s.traj.lateral_correction_m.empty() && s.traj.lateral_correction_m.size() != s.traj.n()) ||
            (!s.traj.row_offset.empty() && s.traj.row_offset.size() != s.traj.n()) ||
            (!s.traj.row_slope.empty() && s.traj.row_slope.size() != s.traj.n()) ||
            (!s.traj.col_offset.empty() && s.traj.col_offset.size() != s.traj.n()) ||
            (!s.traj.col_slope.empty() && s.traj.col_slope.size() != s.traj.n()) ||
            (!s.traj.col_scale.empty() && s.traj.col_scale.size() != s.traj.n()) ||
            (!s.traj.row_scale.empty() && s.traj.row_scale.size() != s.traj.n()))
            throw Error("ScanData: per-projection arrays have inconsistent lengths");
    for (const auto& s : I.scans)
        if (s.traj.map.j.size() != s.traj.map.i.size() || s.traj.map.eu.size() != s.traj.map.i.size() ||
            s.traj.map.ev.size() != s.traj.map.i.size())
            throw Error("ScanData: the detector map's terms (map_i, map_j, map_eu, map_ev) have inconsistent lengths");
    if (I.scans.empty() || I.scans[0].traj.n() < 2) throw Error("SliceReconstructor: no scan");
    I.recorded_step = recorded_angle_step(I.scans[0].traj.angles_deg);
    // the scans' places across the beam: one per scan, or several scans above each other at one
    I.place.assign(I.scans.size(), 0);
    for (std::size_t s = 0, places = 0; s < I.scans.size(); ++s) {
        const auto column = [&](std::size_t k) { return static_cast<double>(I.scans[k].traj.shift_x) - I.scans[k].traj.x_readjustment; };
        const auto above = [&](std::size_t q, std::size_t k) {  // at one place across the beam, at another height
            return std::fabs(column(q) - column(k)) < I.scans[k].traj.det_cols / 4.0 &&
                   std::fabs(I.scans[q].traj.z_readjustment - I.scans[k].traj.z_readjustment) > I.scans[k].traj.det_rows / 4.0;
        };
        I.place[s] = static_cast<int>(places);
        for (std::size_t q = 0; q < s; ++q)
            if (above(q, s)) I.place[s] = I.place[q];
        if (I.place[s] == static_cast<int>(places)) ++places;
        else I.rows_of_scans = true;
    }
    if (I.rows_of_scans) {
        if (I.cfg.legacy_reference) throw Error("legacy reference mode merges scans side by side, not scans above each other");
    }
    std::pair<int, int> whole_doable{0, 0};
    if (I.cfg.angle_subsets != 1) {
        if (I.cfg.legacy_reference) throw Error("legacy reference mode has no projection subsets");
        if (I.cfg.angle_subsets < 1 || I.cfg.angle_subset < 0 || I.cfg.angle_subset >= I.cfg.angle_subsets)
            throw Error("angle_subset must be in [0, angle_subsets)");
        std::vector<SpanInfo> whole;
        for (const auto& s : I.scans) whole.push_back(make_span_info(s.traj, I.cfg.angular_tolerance_steps));
        whole_doable = doable_span(whole, I.place);
        const int full = grid_angle_count(I.scans[0].traj.angles_deg);
        const double phase = angle_grid_phase(I.scans[0].traj.angles_deg, full);
        for (auto& s : I.scans) {
            const std::vector<int32_t> keep = angle_subset(s.traj.angles_deg, full, phase, I.cfg.angle_subsets, I.cfg.angle_subset);
            if (keep.size() < 2) throw Error("angle_subset: no projections left");
            auto take = [&](auto& v) {
                if (v.empty()) return;
                std::decay_t<decltype(v)> out(keep.size());
                for (std::size_t i = 0; i < keep.size(); ++i) out[i] = v[keep[i]];
                v = std::move(out);
            };
            take(s.traj.angles_deg);
            take(s.traj.z_translation_m);
            take(s.traj.lateral_correction_m);
            take(s.traj.row_offset);
            take(s.traj.row_slope);
            take(s.traj.col_offset);
            take(s.traj.col_slope);
            take(s.traj.col_scale);
            take(s.traj.row_scale);
            take(s.control_A);
            take(s.file_index);
            take(s.frame_index);
        }
    }
    for (const auto& s : I.scans) {
        const std::size_t plane = static_cast<std::size_t>(s.traj.det_rows) * s.traj.det_cols;
        if (s.dark.size() != plane || s.weight.size() != plane || s.flats.size() != plane * s.flat_info.angles_deg.size())
            throw Error("ScanData: calibration arrays have wrong sizes");
        I.spans.push_back(make_span_info(s.traj, I.cfg.angular_tolerance_steps));
    }
    I.doable = doable_span(I.spans, I.place);
    if (I.cfg.angle_subsets != 1) {   // (a subset's wider angular tolerance may reach a few heights further)
        I.doable = {std::max(I.doable.first, whole_doable.first), std::min(I.doable.second, whole_doable.second)};
        if (I.doable != whole_doable) throw Error("angle_subset: the subset cannot reconstruct every height of the scan");
    }
    const auto& m = I.scans[0].traj;
    I.fctx = FilterContext{I.cfg.energy_kev, I.cfg.distance_m, I.cfg.pixel_size_m, m.det_rows, m.det_cols,
                           I.cfg.legacy_reference};
    I.margin = I.filter_margin = I.cfg.margin >= 0 ? I.cfg.margin : I.pipeline(I.cfg.filters)->margin_rows();
    // Sequential operations need the sum of their supports: filter context must itself
    // have a full luminosity window. Keep this separate from each filter's kernel support.
    // (The legacy reference clips the luminosity window to the chunk, as the reference does.)
    if (!I.cfg.legacy_reference || (I.cfg.external && I.cfg.external->measured_only())) {
        I.lum_margin = I.scans.size() >= 2 && I.cfg.luminosity_repartition != 0.f ? I.lum.radius : 0;
        I.margin = std::max(I.margin, I.pipeline(I.cfg.filters)->margin_rows() + I.lum_margin);
    }
    I.measured = I.cfg.external && I.cfg.external->measured_only();
    I.full_context = I.measured || !I.cfg.legacy_reference;
    I.x_exact = !I.cfg.legacy_reference && I.cfg.x_interp == "sinc";
    I.x_kernel = I.cfg.legacy_reference ? static_cast<int>(interp_core::Linear)
                 : I.x_exact ? static_cast<int>(interp_core::Lanczos3) : interp_from_name(I.cfg.x_interp);
    if (I.x_kernel != interp_core::Linear && I.x_kernel != interp_core::Cubic && I.x_kernel != interp_core::Lanczos3)
        throw Error("x_interp must be sinc, lanczos3, cubic or linear");
    I.z_kernel = I.cfg.legacy_reference ? static_cast<int>(interp_core::Linear) : z_kernel_from_name(I.cfg.z_interp);
    I.z_ctx = z_kernel_context(I.z_kernel);
    // (one row of the window before the stack at least: a slice below the detector's last row pair takes the value
    // of its last row but one (padded_row), which lies one row before the window of a projection whose first row is
    // the detector's last)
    I.z_pad = std::max(I.z_ctx, 1);
    {   // rolled frames: rows of context for the largest row offset on the detector, and the kernel's table
        float reach = 0.f;
        for (const auto& s : I.scans) {
            if (I.cfg.legacy_reference && (!s.traj.row_slope.empty() || !s.traj.col_offset.empty() || !s.traj.col_slope.empty()))
                throw Error("legacy reference mode has no frame registration");
            if (I.cfg.legacy_reference && (!s.traj.map.i.empty() || !s.traj.col_scale.empty() || !s.traj.row_scale.empty()))
                throw Error("legacy reference mode has no detector map and no frame scales");
            const float span = std::max(std::fabs(s.traj.slope_col), std::fabs(static_cast<float>(s.traj.det_cols - 1) - s.traj.slope_col));
            float roll = 0.f, slope = 0.f;
            for (float v : s.traj.row_slope) {
                roll = std::max(roll, std::fabs(v) * span);
                slope = std::max(slope, std::fabs(v));
            }
            if (roll > 16.f) throw Error("row_slope: a row offset of more than 16 rows across the detector is not a roll");
            // the scan's detector map and scales: their parts, the passes of the inversion, the rows they reach
            ScanMap sm;
            const auto& t = s.traj;
            float col_scale = 0.f, row_scale = 0.f;
            for (float v : t.col_scale) col_scale = std::max(col_scale, std::fabs(v));
            for (float v : t.row_scale) row_scale = std::max(row_scale, std::fabs(v));
            sm.host = t.map.view();
            sm.host.det_rows = t.det_rows;
            sm.host.det_cols = t.det_cols;
            const DetectorMapPlan plan = detector_map_plan(sm.host, slope, col_scale, row_scale);
            sm.host.passes = plan.passes;
            sm.host.settle = plan.settle;
            const bool ev = std::any_of(t.map.ev.begin(), t.map.ev.end(), [](double v) { return v != 0; });
            const bool eu = std::any_of(t.map.eu.begin(), t.map.eu.end(), [](double v) { return v != 0; });
            if (ev) sm.ev_f.assign(t.map.ev.begin(), t.map.ev.end());
            sm.rows = ev || !t.row_scale.empty();
            sm.cols = eu || !t.col_scale.empty();
            if (sm.cols && !I.x_exact)
                throw Error("detector map / col_scale: the lateral part is applied where the columns are read exactly "
                            "(x_interp = sinc), not with x_interp = " + I.cfg.x_interp);
            {   // how far a sample's detector column can lie from its ideal one: the map, the scale, the registration
                const float cols = std::max(std::fabs(t.scale_col), std::fabs(static_cast<float>(t.det_cols - 1) - t.scale_col));
                const float rows = std::max(std::fabs(t.slope_row), std::fabs(static_cast<float>(t.det_rows - 1) - t.slope_row));
                float lateral = (eu ? static_cast<float>(plan.max_eu) : 0.f) + col_scale * cols, registration = 0.f;
                for (std::size_t i = 0; i < t.col_offset.size(); ++i)
                    registration = std::max(registration, std::fabs(t.col_offset[i]) + (i < t.col_slope.size() ? std::fabs(t.col_slope[i]) * rows : 0.f));
                lateral += registration;
                if (I.measured && !(lateral <= 256.f))
                    throw Error("a lateral displacement of more than 256 columns (detector map, col_scale, col_offset, col_slope) "
                                "is not a registration of the frames");
                sm.col_guard = 3 + static_cast<int>(std::ceil(lateral));
            }
            if (sm.rows) {
                const float rows = std::max(std::fabs(t.scale_row), std::fabs(static_cast<float>(t.det_rows - 1) - t.scale_row));
                sm.reach = (ev ? static_cast<float>(plan.max_ev) : 0.f) + row_scale * rows;
                if (!(sm.reach <= 16.f))
                    throw Error("detector map / row_scale: a row displacement of more than 16 rows is not a detector's distortion");
            }
            I.maps.push_back(std::move(sm));
            reach = std::max(reach, roll + I.maps.back().reach);
        }
        for (auto& m : I.maps) m.host.ev_f = m.ev_f.empty() ? nullptr : m.ev_f.data();
        if (reach > 0.f) {
            I.z_pad += static_cast<int>(std::ceil(reach)) + 1;
            I.z_lut = z_kernel_lut(I.z_kernel, Impl::z_lut_n);
        }
    }
    if (!I.cfg.sharpness_weighting.empty() && !I.cfg.legacy_reference && !I.measured) {
        auto numbers = [](const std::string& text, const char* what) {
            std::vector<double> v;
            if (!io::reals_of(text, ':', v)) throw Error(std::string(what) + ": numbers separated by ':' are expected ('" + text + "')");
            return v;
        };
        const auto blur = numbers(I.cfg.sharpness_weighting, "sharpness_weighting");
        const auto band = numbers(I.cfg.sharp_band, "sharp_band");
        if (band.size() != 2) throw Error("sharp_band: f1:f2 (cycles per pixel) is expected");
        I.sharp_f1 = band[0];
        I.sharp_f2 = band[1];
        band_filter_spec(1, 1, 16, I.sharp_f1, I.sharp_f2);   // (checks the band)
        for (const auto& s : I.scans) {
            auto w2 = blur_response(blur, I.cfg.sharp_f, s.traj.det_rows, s.traj.det_cols);
            for (std::size_t i = 0; i < w2.size(); ++i) w2[i] *= s.weight[i];
            I.weight2.push_back(std::move(w2));
        }
        I.sharp = true;
    }
    I.n_grid = I.cfg.n_grid > 0 ? I.cfg.n_grid : grid_angle_count(m.angles_deg);
    if (I.cfg.angle_interp != "linear" && I.cfg.angle_interp != "nearest" && I.cfg.angle_interp != "views")
        throw Error("angle_interp must be linear, nearest or views");
    // (the ghbp plan fixes its angles: it cannot take those of each slice, and keeps the splat)
    I.nearest = !I.cfg.legacy_reference && I.cfg.angle_interp != "linear" && I.cfg.backprojector != "ghbp";
    // (a row per view needs a backprojector that takes any number of rows at any angles: the exact one; the
    // whole-slice methods keep one row per grid angle)
    I.views = I.nearest && I.cfg.angle_interp == "views" && I.cfg.backprojector == "fbp";
    I.chunk_rows = I.n_grid;
    if (I.nearest) {
        const double phase = angle_grid_phase(m.angles_deg, I.n_grid);
        for (const auto& s : I.scans) {
            I.bins.push_back(nearest_angle_bins(s.traj.angles_deg, I.n_grid, phase));
            std::vector<double> prof(s.traj.det_rows, 0.0);
            for (int r = 0; r < s.traj.det_rows; ++r) {
                for (int c = 0; c < s.traj.det_cols; ++c) prof[r] += s.weight[static_cast<std::size_t>(r) * s.traj.det_cols + c];
                prof[r] /= s.traj.det_cols;
            }
            I.row_weight.push_back(std::move(prof));
        }
    }
    if (I.cfg.exposure_arc != 0) {
        if (I.cfg.legacy_reference) throw Error("legacy reference mode has no exposure arc correction");
        if (!(I.cfg.exposure_arc > 0 && I.cfg.exposure_arc <= 1)) throw Error("exposure_arc must be in [0, 1]");
        // Disabled with a row per view (angle_interp views): the filter runs along the rows of one turn of grid
        // angles, and has not been carried over to rows that are views of several turns (the neighbours of a slice's
        // first and last views are then other turns' rows or none). exposure_arc_bins() is 0 there.
        if (!I.views) {
            I.arc_bins = I.cfg.exposure_arc * I.recorded_step * I.n_grid / 360.0;
            I.arc_taps = arc_deblur_taps(I.arc_bins);
        }
    }
    // merged width: main width + ceil(max |companion offset|)
    float max_off = 0.f;
    for (const auto& s : I.scans) max_off = std::max(max_off, std::abs(static_cast<float>(s.traj.shift_x) - s.traj.x_readjustment - static_cast<float>(I.scans[0].traj.shift_x) + I.scans[0].traj.x_readjustment));
    I.NX = m.det_cols + static_cast<int>(std::ceil(max_off));
    // GHBP grid: reference rule  w = round_half_even(2 * rot_center), start = round(rot - w/2 ...) -> w + 1
    if (I.cfg.grid_size > 0) I.grid_size = I.cfg.grid_size;
    else I.grid_size = static_cast<int>(std::nearbyint(2.0 * I.cfg.rot_center)) + 1;
    // mending domain. Each scan's usable columns: first / last column whose weight-map column sum reaches 1e-4 of the
    // maximum, placed in the merged frame (scan column c at merged c + shift_x - x_readjustment); the domain is the
    // merged span they cover. Legacy: the reference passes the main scan's first / last column indices as left /
    // right margin widths of the merged row ([first, NX - last); here [1, 845) of 3372).
    if (I.cfg.legacy_reference) {
        const auto [first, last] = weight_vanishing_margins(I.scans[0].weight, m.det_rows, m.det_cols, 1e-4f);
        I.mend_x0 = first;
        I.mend_x1 = I.NX - last;
    } else {
        I.mend_x0 = I.NX;
        I.mend_x1 = 0;
        for (const auto& s : I.scans) {
            const auto [first, last] = weight_vanishing_margins(s.weight, s.traj.det_rows, s.traj.det_cols, 1e-4f);
            const double off = static_cast<double>(s.traj.shift_x) - s.traj.x_readjustment;
            I.mend_x0 = std::min(I.mend_x0, static_cast<int>(std::ceil(off + first)));
            I.mend_x1 = std::max(I.mend_x1, static_cast<int>(std::floor(off + last)) + 1);
        }
        I.mend_x0 = std::max(0, I.mend_x0);
        I.mend_x1 = std::min(I.NX, I.mend_x1);
    }
    // The beam's divergence: the fan of every merged column, fan_per_column x (the scan's column there - its middle
    // column). Where several scans hold a merged column, the mean by the weight of their columns (the sums of their
    // weight maps' columns, each over its largest); a column none holds takes the nearest scan's line on.
    // (in the exact backprojector alone: backprojector "fbp", and the FBP start of a method that asks for it there)
    if (I.cfg.fan_per_column != 0 && !I.cfg.legacy_reference && (I.cfg.backprojector == "fbp" || I.fbp_start == 2)) {
        std::vector<double> sum(I.NX, 0.0), wsum(I.NX, 0.0);
        for (const auto& s : I.scans) {
            const int R = s.traj.det_rows, C = s.traj.det_cols;
            std::vector<double> col(C, 0.0);
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) col[c] += s.weight[static_cast<std::size_t>(r) * C + c];
            const double top = *std::max_element(col.begin(), col.end());
            const double off = static_cast<double>(s.traj.shift_x) - s.traj.x_readjustment;
            for (int x = 0; x < I.NX; ++x) {
                const double c = x - off;
                if (c < 0 || c > C - 1) continue;
                const int i = std::min(static_cast<int>(c), C - 1);
                const double a = c - i;
                const double w = top > 0 ? std::max(((1 - a) * col[i] + (a > 0 ? a * col[i + 1] : 0.0)) / top, 1e-6) : 1e-6;
                sum[x] += w * (c - 0.5 * (C - 1));
                wsum[x] += w;
            }
        }
        I.fan.resize(I.NX);
        for (int x = 0; x < I.NX; ++x) {
            double from_middle = 0;
            if (wsum[x] > 0) {
                from_middle = sum[x] / wsum[x];
            } else {
                double nearest = 1e300;
                for (const auto& s : I.scans) {
                    const double c = x - (static_cast<double>(s.traj.shift_x) - s.traj.x_readjustment);
                    const double away = c < 0 ? -c : std::max(0.0, c - (s.traj.det_cols - 1));
                    if (away < nearest) nearest = away, from_middle = c - 0.5 * (s.traj.det_cols - 1);
                }
            }
            I.fan[x] = static_cast<float>(I.cfg.fan_per_column * from_middle);
        }
    }
    // luminosity seams: det_cols/2 + lround(mean of (shift_x - x_readjustment)) of a scan and the nearest before it
    // at its own height (within a quarter of the detector's rows: the scans of one row of a grid; every scan where
    // there are no rows of scans). The first scan of a row has none: the rows' levels against each other are left
    // to the weights' crossfade over the heights two rows share.
    I.seam.assign(I.scans.size(), -1);
    I.scan_row.assign(I.scans.size(), 0);
    for (std::size_t s = 1; s < I.scans.size(); ++s) {
        const double p1 = I.scans[s].traj.shift_x - I.scans[s].traj.x_readjustment;
        double p0 = 0;
        bool found = false;
        I.scan_row[s] = static_cast<int>(s);
        for (std::size_t j = 0; j < s; ++j) {
            if (std::fabs(I.scans[j].traj.z_readjustment - I.scans[s].traj.z_readjustment) > I.scans[s].traj.det_rows / 4.0) continue;
            const double pj = I.scans[j].traj.shift_x - I.scans[j].traj.x_readjustment;
            if (!found) I.scan_row[s] = I.scan_row[j];
            if (!found || std::fabs(pj - p1) < std::fabs(p0 - p1)) p0 = pj, found = true;
        }
        if (found) I.seam[s] = m.det_cols / 2 + static_cast<int>(std::lround(0.5 * (p0 + p1)));
        // (a row's scans are summed apart and then added to the rows before: they must follow each other)
        if (I.rows_of_scans && I.cfg.luminosity_repartition != 0.f && I.scan_row[s] != I.scan_row[s - 1] && found)
            throw Error("scans above each other (a grid) with the luminosity step: the scans of one row must follow each other in the list of scans");
    }
    if (I.cfg.sld.enabled()) {
        if (I.cfg.legacy_reference) throw Error("legacy reference mode has no scintillator diffusion correction");
        if (I.cfg.sld.stride < 1) throw Error("sld: stride must be >= 1");
        for (auto& s : I.scans) {
            I.glare.push_back(std::make_unique<ScintillatorGlare>(s.traj.det_rows, s.traj.det_cols, I.cfg.sld));
            I.glare.back()->correct(s.flats.data(), static_cast<int>(s.flat_info.angles_deg.size()));
        }
        I.glare_maps.resize(I.scans.size());
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (I.cfg.device == Device::CUDA) {
        for (const auto& s : I.scans) {
            Impl::DevScan d;
            d.dark.upload(s.dark.data(), s.dark.size());
            d.flats.upload(s.flats.data(), s.flats.size());
            if (!I.z_lut.empty() && I.d_z_lut.size() == 0) I.d_z_lut.upload(I.z_lut.data(), I.z_lut.size());
            d.weight.upload(s.weight.data(), s.weight.size());
            if (I.sharp) d.weight2.upload(I.weight2[I.dev.size()].data(), I.weight2[I.dev.size()].size());
            const ScanMap& sm = I.maps[I.dev.size()];
            d.map = sm.host;
            if (sm.host.n > 0) {
                d.map_i.upload(s.traj.map.i.data(), s.traj.map.i.size());
                d.map_j.upload(s.traj.map.j.data(), s.traj.map.j.size());
                d.map_eu.upload(s.traj.map.eu.data(), s.traj.map.eu.size());
                d.map_ev.upload(s.traj.map.ev.data(), s.traj.map.ev.size());
                if (!sm.ev_f.empty()) d.map_ev_f.upload(sm.ev_f.data(), sm.ev_f.size());
                d.map.i = d.map_i.data();
                d.map.j = d.map_j.data();
                d.map.eu = d.map_eu.data();
                d.map.ev = d.map_ev.data();
                d.map.ev_f = sm.ev_f.empty() ? nullptr : d.map_ev_f.data();
            }
            I.dev.push_back(std::move(d));
        }
        VC_TOMOGRAPHY_CUDA_CHECK(cudaDeviceSynchronize());
    }
#else
    if (I.cfg.device == Device::CUDA) throw Error("built without CUDA");
#endif
}

SliceReconstructor::~SliceReconstructor() = default;

std::pair<int, int> SliceReconstructor::doable() const { return impl_->doable; }
int SliceReconstructor::margin() const { return impl_->margin; }
int SliceReconstructor::n_grid() const { return impl_->n_grid; }
std::string SliceReconstructor::info_json() const {
    const auto [lo, hi] = doable();
    const auto [mx0, mx1] = mend_domain();
    Json counts = Json::list();
    for (int n : projection_counts()) counts.add(Json::whole_number(n));
    const ReconConfig& c = config();
    return Json::dict()
        .set("doable", Json::list().add(Json::whole_number(lo)).add(Json::whole_number(hi)))
        .set("margin", Json::whole_number(margin()))
        .set("n_grid", Json::whole_number(n_grid()))
        .set("grid_size", Json::whole_number(grid_size()))
        .set("merged_width", Json::whole_number(merged_width()))
        .set("mend_domain", Json::list().add(Json::whole_number(mx0)).add(Json::whole_number(mx1)))
        .set("legacy_reference", Json::of_bool(c.legacy_reference))
        .set("exposure_arc_bins", Json::of(exposure_arc_bins()))
        .set("rolling_shutter", Json::of(rolling_shutter()))
        .set("rolling_shutter_from_middle", Json::of_bool(rolling_shutter() != 0.0 && c.rolling_shutter_from_middle))
        .set("fan_per_column", Json::of(fan_per_column()))
        .set("angle_interp", Json::of(angle_interp()))
        .set("z_interp", Json::of(z_interp()))
        .set("angle_subsets", Json::whole_number(c.angle_subsets))
        .set("angle_subset", Json::whole_number(c.angle_subset))
        .set("projections", counts)
        .text();
}

std::vector<int> SliceReconstructor::projection_counts() const {
    std::vector<int> n;
    for (const auto& s : impl_->scans) n.push_back(static_cast<int>(s.traj.n()));
    return n;
}
int SliceReconstructor::grid_size() const { return impl_->grid_size; }
int SliceReconstructor::merged_width() const { return impl_->NX; }
std::pair<int, int> SliceReconstructor::mend_domain() const { return {impl_->mend_x0, impl_->mend_x1}; }
const ReconConfig& SliceReconstructor::config() const { return impl_->cfg; }
double SliceReconstructor::exposure_arc_bins() const { return impl_->arc_bins; }
double SliceReconstructor::rolling_shutter() const { return impl_->nearest ? impl_->cfg.rolling_shutter : 0.0; }
double SliceReconstructor::fan_per_column() const { return impl_->fan.empty() ? 0.0 : impl_->cfg.fan_per_column; }
const std::vector<float>& SliceReconstructor::fan_of_columns() const { return impl_->fan; }
std::string SliceReconstructor::angle_interp() const { return impl_->views ? "views" : impl_->nearest ? "nearest" : "linear"; }
std::string SliceReconstructor::z_interp() const {
    return z_kernel_name(impl_->z_kernel);
}

// Angles of the chunk's slices under nearest-bin gridding: bin b of slice y is at (b + d) 2 pi / n_grid, d the mean
// residual of the projections it holds, each weighted as the gridding weighs it in that slice's row (the mean
// detector weight of the rows it is interpolated from, times its current factor). A bin no projection reaches takes
// the residual of the nearest filled bin before it (the first filled one's, for those ahead of it).
// Rolling shutter (ReconConfig::rolling_shutter): the row a slice is read at was recorded at another angle than the
// projection's by shutter_bins x (row - middle row), which is added to the projection's residual (the row's distance
// from the middle row, to either side, with ReconConfig::rolling_shutter_from_middle).
// With a row per view (views) the rows are the unwrapped bins row_base + r: the same per row, the angle taken modulo
// the turn, and for each row the opposite row for the ramp's padding (slice_opposite): of the rows an odd number of
// half turns away, the one that weighs most in the slice (the nearest, if several or none do).
void SliceReconstructor::Impl::set_slice_angles(const std::vector<ChunkScan*>& cs, int DY, int m) {
    const int NG = n_grid, NR = chunk_rows;
    // (bins per detector row: the scan's signed angular step, in bins, over the frame's rows)
    const auto& a0 = scans[0].traj.angles_deg;
    const double shutter_bins = cfg.rolling_shutter * (a0.back() >= a0.front() ? recorded_step : -recorded_step) * NG / 360.0;
    slice_fbp_angles.resize(static_cast<std::size_t>(DY) * NR);
    if (views) slice_opposite.resize(static_cast<std::size_t>(DY) * NR);
    std::vector<double> sw(NR), sd(NR);
    for (int y = 0; y < DY; ++y) {
        std::fill(sw.begin(), sw.end(), 0.0);
        std::fill(sd.begin(), sd.end(), 0.0);
        for (std::size_t s = 0; s < cs.size(); ++s) {
            const auto& c = *cs[s];
            const auto& prof = row_weight[s];
            const int rows = scans[s].traj.det_rows;
            for (int k = 0; k < c.p1 - c.p0; ++k) {
                const int r = c.row0[k] + m + y;
                const double a = c.fract_v[k];
                if (r < 0 || r > rows - 1 || (a > 0 && r + 1 > rows - 1)) continue;
                const double w = ((1 - a) * prof[r] + (a > 0 ? a * prof[r + 1] : 0.0)) * c.weight_factor[k];
                if (!(w > 0)) continue;
                sw[c.j0[k]] += w;
                const double from_middle = r + a - 0.5 * (rows - 1);
                sd[c.j0[k]] += w * (bins[s].residual[c.p0 + k] + (rows > 1 ? shutter_bins * (cfg.rolling_shutter_from_middle ? std::fabs(from_middle) : from_middle) / (rows - 1) : 0.0));
            }
        }
        int first = -1;
        for (int b = 0; b < NR && first < 0; ++b)
            if (sw[b] > 0) first = b;
        double d = first >= 0 ? sd[first] / sw[first] : 0.0;
        float* out = slice_fbp_angles.data() + static_cast<std::size_t>(y) * NR;
        if (!views) {
            for (int i = 0; i < NG; ++i) {   // from the first filled bin, around the circle
                const int b = first >= 0 ? (first + i) % NG : i;
                if (sw[b] > 0) d = sd[b] / sw[b];
                out[b] = -static_cast<float>((b + d) * (2 * M_PI / NG));
            }
            continue;
        }
        for (int b = 0; b < NR; ++b) {   // (the rows ahead of the first filled one take its residual)
            if (sw[b] > 0) d = sd[b] / sw[b];
            double f = std::fmod(static_cast<double>(row_base) + b + d, static_cast<double>(NG));
            if (f < 0) f += NG;
            out[b] = -static_cast<float>(f * (2 * M_PI / NG));
        }
        int32_t* opp = slice_opposite.data() + static_cast<std::size_t>(y) * NR;
        const int half = NG / 2;
        for (int b = 0; b < NR; ++b) {
            int best = -1;
            for (int c = b % NG >= half ? b % NG - half : b % NG + half; c < NR; c += NG) {
                if (best < 0) { best = c; continue; }
                const int db = std::abs(best - b), dc = std::abs(c - b);
                if (sw[c] > sw[best] || (sw[c] == sw[best] && dc < db)) best = c;
            }
            opp[b] = best;
        }
    }
}

void SliceReconstructor::Impl::setup_invariants() {
    if (!angles.empty()) return;
    const int NG = n_grid;
    angles.resize(NG);
    for (int i = 0; i < NG; ++i) angles[i] = static_cast<float>(i * (2 * M_PI / NG));
    fbp_angles.resize(NG);
    for (int i = 0; i < NG; ++i) fbp_angles[i] = -angles[i];
    const bool legacy = cfg.legacy_reference;
    rg = ramp_geometry(NX, NG, cfg.rot_center, 1, legacy);
    gp.grid_size = grid_size;
    gp.rot_center = cfg.rot_center;
    gp.scale_factor = static_cast<float>(1.0 / (cfg.pixel_size_m * 1e2));
    gp.legs = cfg.ghbp_legs;
    gp.reduction_steps = cfg.ghbp_reduction_steps;
    gp.clip_outer_circle = cfg.clip_outer_circle;
    gp.axis_source_meters = cfg.axis_source_meters;
    gp.voxel_size_microns = static_cast<float>(cfg.pixel_size_m * 1e6);
    if (cfg.backprojector != "ghbp" && cfg.backprojector != "fbp" && cfg.backprojector != "external")
        throw Error("unknown backprojector '" + cfg.backprojector + "' (ghbp, fbp, sirt, mlem, gridrec, external)");
    if (cfg.backprojector == "external" && !cfg.external) throw Error("backprojector 'external' without an implementation");
    if (cfg.bp_oversample < 1 || cfg.bp_oversample > 8) throw Error("bp_oversample must be 1..8");
    bp.rot_center = cfg.rot_center;
    bp.scale = gp.scale_factor;
    bp.center = 0.5f * (grid_size - 1);
    bp.clip_outer_circle = cfg.clip_outer_circle;
    bp.clip_radius = static_cast<float>(grid_size / 2);
    bp.interp = interp_from_name(cfg.bp_interp);
    bp.u_scale = static_cast<float>(cfg.bp_oversample);
    bp.fan = fan.empty() ? nullptr : fan.data();
    bp.n_fan = static_cast<int>(fan.size());
    // output units: the ramp's angular weight 2 pi / NG with pair-balanced full-turn weights gives attenuation; the
    // reference's pi / NG gives attenuation / (angular span / pi) = half of it
    bp.unit_gain = legacy ? static_cast<float>(M_PI / (std::fabs(static_cast<double>(fbp_angles[1]) - fbp_angles[0]) * NG)) : 1.f;
    const FourierFilterSpec rspec = ramp_filter_spec(rg, ramp_scale(NG, legacy), cfg.ramp_filter_name, cfg.ramp_filter_par);
    ramp_filter = make_fourier_filter(cfg.device, rspec);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (cfg.device == Device::CUDA) {
        copy = std::make_unique<cuda::Stream>();
        for (int r = 0; r < 2; ++r) {
            VC_TOMOGRAPHY_CUDA_CHECK(cudaEventCreateWithFlags(&ev_up[r], cudaEventDisableTiming));
            VC_TOMOGRAPHY_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done[r], cudaEventDisableTiming));
        }
        return;
    }
#endif
}

std::shared_ptr<const FilterPipeline> SliceReconstructor::Impl::pipeline(const FilterChain& chain) const {
    std::string key;
    for (const auto& f : chain) {
        key += f.type + "(";
        for (const auto& [k, v] : f.params.values) key += k + "=" + v + ";";
        key += ")";
    }
    auto it = pipelines.find(key);
    if (it == pipelines.end()) {
        if (pipelines.size() >= 64)   // drop the cached chains nobody else holds (live sweeps keep theirs)
            for (auto e = pipelines.begin(); e != pipelines.end();) e = e->second.use_count() == 1 ? pipelines.erase(e) : std::next(e);
        it = pipelines.emplace(key, std::make_shared<const FilterPipeline>(chain, fctx)).first;
    }
    return it->second;
}

FourierFilter* SliceReconstructor::Impl::fft_for(int slot, const FourierFilterSpec& spec) {
    for (auto& e : fft_cache)
        if (e.slot == slot && e.ph == spec.ph && e.pw == spec.pw && e.h == spec.h && e.w == spec.w &&
            e.pad_top == spec.pad_top && e.pad_left == spec.pad_left) {
            if (spec.id == 0 || e.cur_id != spec.id) {  // other coefficients, same geometry: swap them, keep the plans
                e.f->set_filter(spec);
                e.cur_id = spec.id;
            }
            return e.f.get();
        }
    if (fft_cache.size() >= 4) fft_cache.erase(fft_cache.begin());  // e.g. full chunks + the last one
    FftEntry e{slot, spec.ph, spec.pw, spec.h, spec.w, spec.pad_top, spec.pad_left, make_fourier_filter(cfg.device, spec), spec.id};
    fft_cache.push_back(std::move(e));
    return fft_cache.back().f.get();
}

void SliceReconstructor::Impl::apply_filters(const FilterPipeline& P, FilterStage stage, float* R, int L, int H, int m, int W) {
    const auto& steps = P.steps();
    if (!W) W = NX;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        if (steps[s].stage != stage) continue;
        if (steps[s].fourier) {
            const auto& spec = P.step_spec(s, H, W, steps.size() == 1 ? m : -1);
            fft_for(static_cast<int>(s), spec)->apply(R, R, L);
        } else {
            for (auto* f : steps[s].members) f->apply(R, L, H, W, cfg.device);
        }
    }
}

// Glare maps of scan s for the projections [c.p0, c.p1): those of every stride-th projection of the scan (and its
// last), computed from whole frames on first use.
SliceReconstructor::Impl::ChunkGlare SliceReconstructor::Impl::chunk_glare(std::size_t s, const ScanChunk& chunk) {
    const auto& c = chunk.cs;
    const auto& sd = scans[s];
    auto& g = *glare[s];
    auto& have = glare_maps[s];
    const int stride = cfg.sld.stride, last = static_cast<int>(sd.traj.n()) - 1;
    const int first = (c.p0 / stride) * stride;
    std::vector<int> samples;
    for (int p = first;; p += stride) {
        samples.push_back(std::min(p, last));
        if (samples.back() >= c.p1 - 1) break;
    }
    ChunkGlare cg;
    const int n = c.p1 - c.p0, n_maps = static_cast<int>(samples.size());
    cg.i0.resize(n);
    cg.i1.resize(n);
    cg.frac.resize(n);
    for (int k = 0; k < n; ++k) {
        const int p = c.p0 + k;
        const int i0 = std::min((p - first) / stride, n_maps - 1), i1 = std::min(i0 + 1, n_maps - 1);
        const int d = samples[i1] - samples[i0];
        cg.i0[k] = i0;
        cg.i1[k] = i1;
        cg.frac[k] = d > 0 ? static_cast<float>(p - samples[i0]) / static_cast<float>(d) : 0.f;
    }
    cg.rows = g.rows();
    if (cfg.sld.bm18) {
        // Full detector context for the FFT, but only chunk strips are retained
        // and uploaded. Memory does not grow by n_projections * detector_height.
        const int nr = sd.traj.det_rows, nc = sd.traj.det_cols;
        cg.row0.assign(n_maps, nr);
        std::vector<int> ends(n_maps, 0);
        for (int k = 0; k < n; ++k) {
            // Include the nearest boundary pair used by vertical extrapolation.
            const int lo = std::clamp(c.row0[k] - z_pad, 0, std::max(0, nr - 2));
            const int hi = std::clamp(c.row0[k] - z_pad + chunk.win_rows, std::min(2, nr), nr);
            for (int i : {cg.i0[k], cg.i1[k]}) {
                cg.row0[i] = std::min(cg.row0[i], lo);
                ends[i] = std::max(ends[i], hi);
            }
        }
        cg.rows = 1;
        for (int i = 0; i < n_maps; ++i) cg.rows = std::max(cg.rows, ends[i] - cg.row0[i]);
        const std::size_t strip = static_cast<std::size_t>(cg.rows) * nc;
        cg.maps.resize(static_cast<std::size_t>(n_maps) * strip, 0.f);
        std::vector<float> frame(static_cast<std::size_t>(nr) * nc);
        for (int i = 0; i < n_maps; ++i) {
            if (ends[i] <= cg.row0[i]) continue;
            const int p = samples[i];
            const int32_t fi = sd.file_index[p], fr = sd.frame_index[p];
            sd.reader->visit_frames(&fi, &fr, 1, 64, [&](std::size_t, int r0, int count, const uint16_t* raw) {
                g.bin_raw(raw, r0, count, sd.dark.data(), static_cast<float>(sd.control_A[p] / 0.2), frame.data());
            });
            g.filter(frame.data(), 1);
            std::copy_n(frame.data() + static_cast<std::size_t>(cg.row0[i]) * nc,
                        static_cast<std::size_t>(ends[i] - cg.row0[i]) * nc, cg.maps.data() + i * strip);
        }
        return cg;
    }
    for (auto it = have.begin(); it != have.end();)   // keep only what this chunk uses
        it = std::binary_search(samples.begin(), samples.end(), it->first) ? std::next(it) : have.erase(it);
    std::vector<int> todo;
    for (int p : samples)
        if (!have.count(p)) todo.push_back(p);
    const std::size_t cell = static_cast<std::size_t>(g.rows()) * g.cols();
    if (!todo.empty()) {   // the frames are binned as they are read (a few rows of cells at a time), then filtered
        const std::size_t n = todo.size();
        std::vector<float> out(n * cell);
        std::vector<int32_t> fi(n), fr(n);
        for (std::size_t i = 0; i < n; ++i) {
            fi[i] = sd.file_index[todo[i]];
            fr[i] = sd.frame_index[todo[i]];
        }
        sd.reader->visit_frames(fi.data(), fr.data(), n, 4 * g.bin(), [&](std::size_t i, int r0, int nr, const uint16_t* rows) {
            // (the divisor as prepare_chunk_scan's current_div)
            g.bin_raw(rows, r0, nr, sd.dark.data(), static_cast<float>(sd.control_A[todo[i]] / 0.2), out.data() + i * cell);
        });
        g.filter(out.data(), static_cast<int>(n));
        for (std::size_t i = 0; i < n; ++i) have[todo[i]].assign(out.begin() + i * cell, out.begin() + (i + 1) * cell);
    }
    cg.maps.resize(samples.size() * cell);
    for (std::size_t i = 0; i < samples.size(); ++i) std::copy(have[samples[i]].begin(), have[samples[i]].end(), cg.maps.begin() + i * cell);
    return cg;
}

#ifdef VC_TOMOGRAPHY_WITH_CUDA
void SliceReconstructor::Impl::start_prefetch(int lo, int hi, int m) {
    if (pf_thread.joinable()) pf_thread.join();
    pf_lo = lo; pf_hi = hi; pf_m = m; pf_err = nullptr;
    if (pinned_next.size() != scans.size()) pinned_next.resize(scans.size());
    const bool chained = reuse_overlap && held.m == m && held.hi == lo && held.p0.size() == scans.size();
    std::vector<std::size_t> cap(scans.size());
    for (std::size_t s = 0; s < scans.size(); ++s) cap[s] = raw_capacity(s, hi - lo, m);
    pf_thread = std::thread([this, lo, hi, m, chained, cap] {
        try {
            for (std::size_t s = 0; s < scans.size(); ++s) {
                const auto& sd = scans[s];
                const ChunkScan c = prepare_chunk_scan(sd.traj, spans[s], sd.control_A, sd.flat_info, lo, hi, m, n_grid,
                                                       0.2, full_context, !measured, rows_of_scans);
                const int n = c.p1 - c.p0, win = hi - lo + 2 * m + 1 + 2 * z_pad, cols = sd.traj.det_cols;
                const std::vector<int32_t> win0 = window_rows(c, z_pad);
                const std::size_t per = static_cast<std::size_t>(win) * cols;
                pinned_next[s].ensure(std::max(static_cast<std::size_t>(n), cap[s]) * per);
                uint16_t* dst = pinned_next[s].data();
                // projections shared with the held chunk: copy its bottom 2m + 1 rows, read only the new rows
                int a = c.p0, b = c.p0;  // shared range [a, b)
                const int dy_old = held.hi - held.lo, ov = 2 * m + 1 + 2 * z_pad, win_old = dy_old + ov;
                if (chained) {
                    a = std::max(c.p0, held.p0[s]);
                    b = std::max(a, std::min(c.p1, held.p1[s]));
                    for (int k = a; k < b; ++k)
                        if (c.row0[k - c.p0] != held.row0[s][k - held.p0[s]] + dy_old) { a = b = c.p0; break; }
                }
                const uint16_t* src = pinned[s].data();
                if (b > a) {
#pragma omp parallel for schedule(static) num_threads(8)
                    for (int k = a; k < b; ++k)
                        std::memcpy(dst + static_cast<std::size_t>(k - c.p0) * per,
                                    src + (static_cast<std::size_t>(k - held.p0[s]) * win_old + dy_old) * cols,
                                    static_cast<std::size_t>(ov) * cols * sizeof(uint16_t));
                    std::vector<int32_t> r0(b - a);
                    for (int k = a; k < b; ++k) r0[k - a] = win0[k - c.p0] + ov;
                    sd.reader->read_rows(sd.file_index.data() + a, sd.frame_index.data() + a, r0.data(), b - a,
                                         win - ov, dst + static_cast<std::size_t>(a - c.p0) * per + static_cast<std::size_t>(ov) * cols,
                                         0, per);
                }
                // the others: whole windows
                if (a > c.p0)
                    sd.reader->read_rows(sd.file_index.data() + c.p0, sd.frame_index.data() + c.p0, win0.data(), a - c.p0,
                                         win, dst);
                if (b < c.p1) {
                    const int s0 = std::max(b, c.p0);
                    sd.reader->read_rows(sd.file_index.data() + s0, sd.frame_index.data() + s0, win0.data() + (s0 - c.p0),
                                         c.p1 - s0, win, dst + static_cast<std::size_t>(s0 - c.p0) * per);
                }
            }
        } catch (...) {
            pf_err = std::current_exception();
        }
    });
}

bool SliceReconstructor::Impl::take_prefetch(int lo, int hi, int m) {
    if (!pf_thread.joinable()) return false;
    pf_thread.join();
    const bool match = pf_lo == lo && pf_hi == hi && pf_m == m && !pf_err;
    pf_m = -1;
    if (!match) return false;
    std::swap(pinned, pinned_next);
    return true;
}
#endif

namespace {

#define VC_TOMOGRAPHY_FORWARD(ns, name) \
    template <class... A>              \
    static decltype(auto) name(A&&... a) { return ns::name(std::forward<A>(a)...); }
#define VC_TOMOGRAPHY_BACKEND_OPS(ns)                                                                              \
    VC_TOMOGRAPHY_FORWARD(ns, add_into) VC_TOMOGRAPHY_FORWARD(ns, luminosity_sums)                                 \
    VC_TOMOGRAPHY_FORWARD(ns, luminosity_apply) VC_TOMOGRAPHY_FORWARD(ns, divide)                                  \
    VC_TOMOGRAPHY_FORWARD(ns, prepare_measured) VC_TOMOGRAPHY_FORWARD(ns, mend_null_weights)                       \
    VC_TOMOGRAPHY_FORWARD(ns, add_scaled) VC_TOMOGRAPHY_FORWARD(ns, minus_log) VC_TOMOGRAPHY_FORWARD(ns, multiply) \
    VC_TOMOGRAPHY_FORWARD(ns, multiply_add) VC_TOMOGRAPHY_FORWARD(ns, weight_difference)                           \
    VC_TOMOGRAPHY_FORWARD(ns, angular_fir) VC_TOMOGRAPHY_FORWARD(ns, ramp_crop) VC_TOMOGRAPHY_FORWARD(ns, rebalance) \
    VC_TOMOGRAPHY_FORWARD(ns, rebalance_turns) VC_TOMOGRAPHY_FORWARD(ns, turn_holes)                               \
    VC_TOMOGRAPHY_FORWARD(ns, fill_turn_holes) VC_TOMOGRAPHY_FORWARD(ns, balance_weights_turns)                    \
    VC_TOMOGRAPHY_FORWARD(ns, rebalance_holes) VC_TOMOGRAPHY_FORWARD(ns, fill_holes)                               \
    VC_TOMOGRAPHY_FORWARD(ns, balance_weights)

// run()'s array operations on host memory
struct CpuOps {
    VC_TOMOGRAPHY_BACKEND_OPS(cpu)
    static void zero(float* p, std::size_t n) { std::fill(p, p + n, 0.f); }
    static void copy(float* dst, const float* src, std::size_t n) { std::memcpy(dst, src, n * sizeof(float)); }
    static void to_host(float* dst, const float* src, std::size_t n) { copy(dst, src, n); }
    static void ramp_pad(const float* row, float* pad, const RampGeometry& g, const int32_t* opposite) { cpu::ramp_pad(row, pad, g, opposite); }
    // rows [m, m + DY) of each of the block's L angles (L, H, NX) -> angles [g0, g0 + L) of the sinogram (NR, DY, NX)
    static void crop(float* sino, const float* block, int g0, int L, int H, int DY, int m, int NX) {
        for (int j = 0; j < L; ++j)
            copy(sino + static_cast<std::size_t>(g0 + j) * DY * NX, block + (static_cast<std::size_t>(j) * H + m) * NX, static_cast<std::size_t>(DY) * NX);
    }
    // slice y of the sinogram (NR, DY, NX) -> rows (NR, NX)
    static void gather(float* rows, const float* sino, int y, int NR, int DY, int NX) {
        for (int i = 0; i < NR; ++i) copy(rows + static_cast<std::size_t>(i) * NX, sino + (static_cast<std::size_t>(i) * DY + y) * NX, NX);
    }
};

#ifdef VC_TOMOGRAPHY_WITH_CUDA
// the same on the device (the default stream)
struct CudaOps {
    VC_TOMOGRAPHY_BACKEND_OPS(cuda)
    static void zero(float* p, std::size_t n) { VC_TOMOGRAPHY_CUDA_CHECK(cudaMemsetAsync(p, 0, n * sizeof(float))); }
    static void copy(float* dst, const float* src, std::size_t n) {
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    static void to_host(float* dst, const float* src, std::size_t n) { VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost)); }
    static void ramp_pad(const float* row, float* pad, const RampGeometry& g, const int32_t* opposite) { cuda::ramp_pad(row, pad, g, nullptr, opposite); }
    static void crop(float* sino, const float* block, int g0, int L, int H, int DY, int m, int NX) {
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy2DAsync(sino + static_cast<std::size_t>(g0) * DY * NX, static_cast<std::size_t>(DY) * NX * sizeof(float),
                                                   block + static_cast<std::size_t>(m) * NX, static_cast<std::size_t>(H) * NX * sizeof(float),
                                                   static_cast<std::size_t>(DY) * NX * sizeof(float), L, cudaMemcpyDeviceToDevice));
    }
    static void gather(float* rows, const float* sino, int y, int NR, int DY, int NX) {
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy2DAsync(rows, NX * sizeof(float), sino + static_cast<std::size_t>(y) * NX,
                                                   static_cast<std::size_t>(DY) * NX * sizeof(float), NX * sizeof(float), NR, cudaMemcpyDeviceToDevice));
    }
};
#endif
#undef VC_TOMOGRAPHY_BACKEND_OPS
#undef VC_TOMOGRAPHY_FORWARD

}  // namespace

template <class Ops, class Accumulate, class Done, class Ready, class Tick>
void SliceReconstructor::Impl::grid_block(int L, int H, const std::vector<ScanChunk>& chunks, const GridBuffers& b, bool sharp, bool by_row,
                                          const AccumParams& ap, bool filter_context, Accumulate&& accumulate, Done&& done, Ready&& ready,
                                          Tick&& tick) {
    const std::size_t n_el = static_cast<std::size_t>(L) * H * NX;
    if (sharp) {
        Ops::zero(b.R2, n_el);
        Ops::zero(b.RW2, n_el);
    }
    Ops::zero(b.R, n_el);
    Ops::zero(b.RW, n_el);
    ready();
    // Rows of scans with the luminosity step: a row's scans are summed apart (their luminosity is compared among
    // themselves: another row holds other rows of the detector around a height), then added to the rows before.
    float* const aR = by_row ? b.Rr : b.R;
    float* const aRW = by_row ? b.RWr : b.RW;
    float* const aR2 = by_row ? b.R2r : b.R2;
    float* const aRW2 = by_row ? b.RW2r : b.RW2;
    // Each scan that has an earlier scan at its height is taken to the luminosity of the scans before it at its seam
    // with them: the sums there before and after it is added.
    const bool luminosity = cfg.luminosity_repartition != 0.f && chunks.size() >= 2;
    bool before = false;   // a scan of this chunk (of this row of scans) has been added
    int row_now = -1;
    const auto add_row = [&] {
        Ops::add_into(b.R, b.Rr, n_el);
        Ops::add_into(b.RW, b.RWr, n_el);
        if (sharp) Ops::add_into(b.R2, b.R2r, n_el), Ops::add_into(b.RW2, b.RW2r, n_el);
    };
    for (std::size_t s = 0; s < chunks.size(); ++s) {
        if (chunks[s].cs.p1 == chunks[s].cs.p0) {  // (a scan at another height: the chunk holds none of it)
            done(s);
            continue;
        }
        if (by_row && scan_row[s] != row_now) {   // another row of scans begins
            if (row_now >= 0) add_row();
            Ops::zero(b.Rr, n_el);
            Ops::zero(b.RWr, n_el);
            if (sharp) Ops::zero(b.R2r, n_el), Ops::zero(b.RW2r, n_el);
            row_now = scan_row[s], before = false;
        }
        const bool joins = luminosity && before && seam[s] >= 0;
        if (joins) {
            Ops::luminosity_sums(aR, aRW, L, H, NX, seam[s], this->lum, b.Aha, b.Bha);
            Ops::copy(b.Wp, aRW, n_el);
            if (sharp) {
                Ops::luminosity_sums(aR2, aRW2, L, H, NX, seam[s], this->lum, b.Aha2, b.Bha2);
                Ops::copy(b.Wp2, aRW2, n_el);
            }
            tick("luminosity");
        }
        accumulate(s, aR, aRW, sharp ? aR2 : nullptr, sharp ? aRW2 : nullptr);
        done(s);
        tick("accumulate");
        if (joins) {
            Ops::luminosity_sums(aR, aRW, L, H, NX, seam[s], this->lum, b.Atot, b.Btot);
            Ops::luminosity_apply(aR, aRW, b.Wp, b.Aha, b.Bha, b.Atot, b.Btot, L, H, NX, cfg.luminosity_repartition);
        }
        if (sharp && joins) {
            Ops::luminosity_sums(aR2, aRW2, L, H, NX, seam[s], this->lum, b.Atot2, b.Btot2);
            Ops::luminosity_apply(aR2, aRW2, b.Wp2, b.Aha2, b.Bha2, b.Atot2, b.Btot2, L, H, NX, cfg.luminosity_repartition);
        }
        tick("luminosity");
        before = true;
    }
    if (by_row && row_now >= 0) add_row();
    Ops::divide(b.R, b.RW, n_el);
    if (ap.measured_only)
        Ops::prepare_measured(b.R, b.RW, L, H, NX, cfg.mend_threshold, filter_context, ap.log_values ? 0.f : 1.f);
    else if (cfg.mend_threshold > 0)
        Ops::mend_null_weights(b.R, b.RW, L, H, NX, cfg.mend_threshold, mend_x0, mend_x1);
    if (sharp) {   // the two weightings' intensities, combined by band: R + HP (R2 - R)
        Ops::divide(b.R2, b.RW2, n_el);
        if (cfg.mend_threshold > 0) Ops::mend_null_weights(b.R2, b.RW2, L, H, NX, cfg.mend_threshold, mend_x0, mend_x1);
        Ops::add_scaled(b.R2, b.R, -1.f, n_el);
        band2d->apply(b.R2, b.R2, L);
        Ops::add_scaled(b.R, b.R2, 1.f, n_el);
    }
    tick("divide_mend");
}

template <class Ops, class Tick>
void SliceReconstructor::Impl::filter_block(const FilterPipeline& P, float* R, int L, int H, int fm, const AccumParams& ap, Tick&& tick) {
    apply_filters(P, FilterStage::Intensity, R, L, H, fm);
    tick("filters");
    if (!ap.log_values) Ops::minus_log(R, static_cast<std::size_t>(L) * H * NX, cfg.log_clip_min, cfg.log_clip_max);
    apply_filters(P, FilterStage::Log, R, L, H, fm);
    tick("log");
}

template <class Ops>
void SliceReconstructor::Impl::balance(float* S, float* W, float* W2, int NR, int DY, int NX, bool sharp, bool measured_only) {
    if (views) Ops::rebalance_turns(S, W, NR, DY, NX, n_grid, cfg.rot_center);
    else if (!measured_only) Ops::rebalance(S, W, NR, DY, NX, cfg.rot_center, cfg.legacy_reference);
    if (!sharp) return;
    // the balanced weights of the second set (the sinogram's holes are filled already)
    if (views) {
        const TurnHoles holes = Ops::turn_holes(W2, NR, DY, NX, n_grid);
        if (!holes.all_empty) {
            Ops::fill_turn_holes(W2, holes, DY, NX);
            Ops::balance_weights_turns(W2, NR, DY, NX, n_grid, cfg.rot_center);
        }
    } else {
        const auto holes = Ops::rebalance_holes(W2, NR, DY, NX);
        if (holes.empty() || holes.front().first >= 0) {   // (every angle empty: nothing to do, as rebalance)
            Ops::fill_holes(W2, holes, NR, DY, NX);
            Ops::balance_weights(W2, NR, DY, NX, cfg.rot_center, cfg.legacy_reference);
        }
    }
}

template <class Ops, class Tick>
void SliceReconstructor::Impl::slices(int k, const float* S, const float* W, const float* W2, int NR, int DY, int NX, bool sharp,
                                      const RampGeometry& rg, const SliceBuffers& b, ChunkDebug* keep_filtered, bool measured_only,
                                      const std::function<void(int, int, float*, float*)>& sink, Tick&& tick) {
    const std::size_t rown = static_cast<std::size_t>(NR) * NX;
    for (int y = 0; y < DY; ++y) {
        Ops::gather(b.row, S, y, NR, DY, NX);
        Ops::gather(b.wrow, W, y, NR, DY, NX);
        if (sharp) Ops::gather(b.w2row, W2, y, NR, DY, NX);
        tick("ramp_gather");
        if (!arc_taps.empty()) {
            Ops::angular_fir(b.row, b.arc, NR, NX, arc_taps.data(), static_cast<int>(arc_taps.size()) - 1, measured_only ? b.wrow : nullptr);
            Ops::copy(b.row, b.arc, rown);
            tick("arc");
        }
        if (!raw_rows) {
            Ops::ramp_pad(b.row, b.pad, rg, views ? b.opposite + static_cast<std::size_t>(y) * NR : nullptr);
            ramp_filter->apply(b.pad, b.pad, NR);
            Ops::ramp_crop(b.pad, b.row, rg);
            if (sharp) {   // usual pair weights, and above the band the second set's: row w + HP (row (w2 - w))
                Ops::weight_difference(b.row, b.wrow, b.w2row, b.brow, rown);
                band1d->apply(b.brow, b.brow, NR);
                Ops::multiply_add(b.row, b.wrow, b.brow, rown);
            } else {
                Ops::multiply(b.row, b.wrow, rown);
            }
        }
        if (keep_filtered) Ops::to_host(&keep_filtered->filtered[static_cast<std::size_t>(y) * rown], b.row, rown);
        if (fbp_start && raw_rows) {   // the method's start: this slice's rows as the exact backprojector takes them
            Ops::copy(b.start, b.row, rown);
            Ops::copy(b.start_w, b.wrow, rown);
            if (cfg.mend_threshold > 0) Ops::mend_null_weights(b.start, b.start_w, NR, 1, NX, cfg.mend_threshold, mend_x0, mend_x1);
            Ops::rebalance(b.start, b.start_w, NR, 1, NX, cfg.rot_center, false);
            Ops::ramp_pad(b.start, b.pad, rg, nullptr);
            ramp_filter->apply(b.pad, b.pad, NR);
            Ops::ramp_crop(b.pad, b.start, rg);
            Ops::multiply(b.start, b.start_w, rown);
            start_rows = b.start;
        }
        tick("ramp");
        sink(k, y, b.row, b.wrow);
        start_rows = nullptr;
        tick("backproject");
    }
}

void SliceReconstructor::Impl::run(int h_lo, int h_hi, ChunkDebug* dbg, const std::vector<const FilterPipeline*>& sets,
                                   int margin_rows, int filter_rows,
                                   const std::function<void(int, int, float*, float*)>& sink, int next_lo, int next_hi) {
    auto& I = *this;
    if (h_lo < I.doable.first || h_hi > I.doable.second || h_hi <= h_lo)
        throw Error("reconstruct: heights outside the doable span");
    const int K = static_cast<int>(sets.size());
    if (K < 1) throw Error("reconstruct: no parameter set");
    const int DY = h_hi - h_lo, m = margin_rows, H = DY + 2 * m, NX = I.NX, NG = I.n_grid;
    const bool gpu = cfg.device == Device::CUDA;
    if (K > 1 && !gpu) throw Error("parameter sweeps need device='cuda'");
    const bool timing = dbg && dbg->timing;
    auto t_last = std::chrono::steady_clock::now();
    auto tick = [&](const char* stage) {  // adds the time since the previous tick to `stage`
        if (!timing) return;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (gpu) VC_TOMOGRAPHY_CUDA_CHECK(cudaDeviceSynchronize());
#endif
        const auto now = std::chrono::steady_clock::now();
        dbg->seconds[stage] += std::chrono::duration<double>(now - t_last).count();
        t_last = now;
    };
    setup_invariants();
    tick("setup");
    step("read");

    // ---------------------------------------------------------------- per-scan chunk geometry
    const bool measured_only = cfg.external && cfg.external->measured_only();
    const int fm = measured_only ? -1 : filter_rows;   // (measured data: every step its own margin)
    std::vector<ScanChunk> chunks(I.scans.size());
    for (std::size_t s = 0; s < I.scans.size(); ++s) {
        chunks[s].cs = prepare_chunk_scan(I.scans[s].traj, I.spans[s], I.scans[s].control_A, I.scans[s].flat_info,
                                          h_lo, h_hi, m, NG, 0.2, I.full_context, !measured_only, I.rows_of_scans);
        chunks[s].win_rows = H + 1 + 2 * I.z_pad;
        chunks[s].win0 = window_rows(chunks[s].cs, I.z_pad);
        chunks[s].z_w = z_kernel_weights(chunks[s].cs.fract_v, I.z_kernel);
        if (I.nearest && !I.views) {   // every projection whole in its bin
            auto& c = chunks[s].cs;
            for (int k = 0; k < c.p1 - c.p0; ++k) {
                const int b = I.bins[s].bin[c.p0 + k] % NG;
                c.j0[k] = b < 0 ? b + NG : b;
                c.alpha[k] = 0.f;
            }
        }
        if (timing) {
            const double np = chunks[s].cs.p1 - chunks[s].cs.p0;
            dbg->stats["n_proj_" + std::to_string(s)] = np;
            dbg->stats["raw_gb"] += np * chunks[s].win_rows * I.scans[s].traj.det_cols * 2 / 1e9;
        }
    }
    // the rows of the chunk's sinogram: the grid angles, or (views) the unwrapped bins of its projections
    I.chunk_rows = NG;
    if (I.views) {   // every projection whole in the row of its own bin
        long lo = 0, hi = -1;
        for (std::size_t s = 0; s < I.scans.size(); ++s) {
            const auto& c = chunks[s].cs;
            for (int k = c.p0; k < c.p1; ++k) {
                const long b = I.bins[s].bin[k];
                if (hi < lo) lo = hi = b;
                lo = std::min(lo, b);
                hi = std::max(hi, b);
            }
        }
        const long span = std::max<long>(hi - lo + 1, NG);
        if (span > static_cast<long>(kMaxViewTurns) * NG)
            throw Error("angle_interp views: the projections that see these slices span more than " + std::to_string(kMaxViewTurns) +
                        " turns (use angle_interp nearest)");
        I.row_base = lo;
        I.chunk_rows = static_cast<int>(span);
        for (std::size_t s = 0; s < I.scans.size(); ++s) {
            auto& c = chunks[s].cs;
            for (int k = 0; k < c.p1 - c.p0; ++k) {
                c.j0[k] = static_cast<int32_t>(I.bins[s].bin[c.p0 + k] - lo);
                c.alpha[k] = 0.f;
            }
        }
    }
    const int NR = I.chunk_rows;
    const bool views = I.views;
    RampGeometry rg = I.rg;   // (the chunk's rows)
    rg.n_proj = NR;
    if (I.nearest) {
        std::vector<ChunkScan*> cs;
        for (auto& ch : chunks) cs.push_back(&ch.cs);
        set_slice_angles(cs, DY, m);
    }
    if (dbg) {
        dbg->n_rows = NR;
        dbg->first_bin = views ? I.row_base : 0;
    }
    if (dbg && dbg->keep) {
        dbg->angles.resize(static_cast<std::size_t>(DY) * NR);
        for (int y = 0; y < DY; ++y)
            for (int i = 0; i < NR; ++i) dbg->angles[static_cast<std::size_t>(y) * NR + i] = -fbp_angles_of(y)[i];
    }
    if (timing) dbg->stats["sino_gb"] = 2.0 * NR * DY * NX * 4 / 1e9;
    tick("prepare");
    std::vector<ChunkGlare> cg(I.scans.size());
    if (cfg.sld.enabled()) {
        for (std::size_t s = 0; s < I.scans.size(); ++s)
            if (chunks[s].cs.p1 > chunks[s].cs.p0) cg[s] = chunk_glare(s, chunks[s]);
        tick("glare");
    }

    AccumParams ap;
    ap.H = H;
    ap.NX = NX;
    ap.measured_only = measured_only;
    const bool sharp = I.sharp && !measured_only;
    if (sharp) {
        auto make = [&](const FourierFilterSpec& sp) { return make_fourier_filter(gpu ? Device::CUDA : Device::CPU, sp); };
        if (!I.band2d || I.band2d_h != H) {
            I.band2d = make(band_filter_spec(2, H, NX, I.sharp_f1, I.sharp_f2));
            I.band2d_h = H;
        }
        if (!I.band1d) I.band1d = make(band_filter_spec(1, 1, NX, I.sharp_f1, I.sharp_f2));
    }
    // Keep phase retrieval and luminosity correction in intensity space. With neither,
    // interpolate line integrals, avoiding -log(mean(I)) != mean(-log(I)).
    ap.log_values = ap.measured_only && (scans.size() == 1 || cfg.luminosity_repartition == 0.f);
    for (const auto& step : sets[0]->steps())
        if (step.stage == FilterStage::Intensity) ap.log_values = false;
    ap.log_clip_min = cfg.log_clip_min;
    ap.log_clip_max = cfg.log_clip_max;
    const bool filter_context = !sets[0]->steps().empty();
    // Grid angles per block: cfg.block_angles at most, and fewer for a taller stack, so that a block's sums (a dozen
    // arrays of block angles x H rows x NX, with the raw windows and the rows' cache) stay about kBlockRows rows of
    // the merged frame whatever the chunk. (Measured on an RTX 5090, 102 rows of 3372 columns: 250 angles a block
    // take the time of 500 and 3.2 GB less of the device; 125 take 2.5 % longer.)
    const int BL = std::max(1, std::min(cfg.block_angles, std::max(16, kBlockRows / std::max(1, H))));
    const std::size_t blk = static_cast<std::size_t>(BL) * H * NX;
    const std::size_t sino_n = static_cast<std::size_t>(NR) * DY * NX;

    if (!gpu) {
        // ============================================================ CPU
        for (std::size_t s = 0; s < I.scans.size(); ++s) {
            const auto& sd = I.scans[s];
            auto& ch = chunks[s];
            const int n = ch.cs.p1 - ch.cs.p0;
            ch.raw.resize(static_cast<std::size_t>(n) * ch.win_rows * sd.traj.det_cols);
            if (!n) continue;  // (a scan at another height: the chunk holds none of it)
            sd.reader->read_rows(sd.file_index.data() + ch.cs.p0, sd.frame_index.data() + ch.cs.p0, ch.win0.data(),
                                 n, ch.win_rows, ch.raw.data());
        }
        tick("read_raw");
        step("grid");
        std::vector<float> S(sino_n), Wc(sino_n), Wc2(sharp ? sino_n : 0);
        const bool by_row = I.rows_of_scans && cfg.luminosity_repartition != 0.f;
        const std::size_t lum_n = static_cast<std::size_t>(BL) * H;
        std::vector<float> R(blk), RW(blk), Wp(blk), Aha(lum_n), Bha(lum_n), Atot(lum_n), Btot(lum_n);
        std::vector<float> R2(sharp ? blk : 0), RW2(sharp ? blk : 0), Wp2(sharp ? blk : 0), Aha2(sharp ? lum_n : 0), Bha2(Aha2.size()),
            Atot2(Aha2.size()), Btot2(Aha2.size());
        std::vector<float> Rr(by_row ? blk : 0), RWr(by_row ? blk : 0), R2r(by_row && sharp ? blk : 0), RW2r(by_row && sharp ? blk : 0);
        const GridBuffers gb{R.data(), RW.data(), R2.data(), RW2.data(), Rr.data(), RWr.data(), R2r.data(), RW2r.data(), Wp.data(), Wp2.data(),
                             Aha.data(), Bha.data(), Atot.data(), Btot.data(), Aha2.data(), Bha2.data(), Atot2.data(), Btot2.data()};
        for (int g0 = 0; g0 < NR; g0 += BL) {
            const int g1 = std::min(NR, g0 + BL), L = g1 - g0;
            const auto accumulate = [&](std::size_t s, float* aR, float* aRW, float* aR2, float* aRW2) {
                const auto& sd = I.scans[s];
                const auto& ch = chunks[s];
                const auto& c = ch.cs;
                const auto contrib = make_contributors(c.j0.data(), c.p1 - c.p0, NR, g0, g1, !I.nearest);
                auto view = make_view(sd, ch, ch.raw.data(), sd.dark.data(), sd.flats.data(), sd.weight.data(),
                                      c.row0.data(), c.fract_v.data(), c.alpha.data(), c.current_div.data(),
                                      c.weight_factor.data(), c.flat_frac.data(), c.flat_i0.data(), c.flat_i1.data(),
                                      I.x_kernel, I.x_exact);
                set_noise_split(view, cfg, cfg.noise_split_scale != 0.f ? I.split_salt[s] : 0U, c.p0);
                set_z_kernel(view, I.z_pad, I.z_ctx, ch.z_w.data());
                set_roll(view, sd, c.row_slope.data(), I.z_lut.data(), Impl::z_lut_n);
                set_lateral(view, sd, c, c.col_offset.data(), c.col_slope.data(), I.x_exact);
                set_map(view, sd, I.maps[s], I.maps[s].host, c.row_scale.data(), c.col_scale.data(), I.z_lut.data(), Impl::z_lut_n);
                if (cfg.sld.enabled())
                    set_glare(view, *I.glare[s], cg[s].maps.data(), cg[s].i0.data(), cg[s].i1.data(), cg[s].frac.data(),
                              cg[s].rows, cg[s].row0.empty() ? nullptr : cg[s].row0.data());
                cpu::accumulate(view, contrib, ap, aR, aRW, aR2 ? I.weight2[s].data() : nullptr, aR2, aRW2);
            };
            grid_block<CpuOps>(L, H, chunks, gb, sharp, by_row, ap, filter_context, accumulate, [](std::size_t) {}, [] {}, tick);
            filter_block<CpuOps>(*sets[0], R.data(), L, H, fm, ap, tick);
            CpuOps::crop(S.data(), R.data(), g0, L, H, DY, m, NX);
            CpuOps::crop(Wc.data(), RW.data(), g0, L, H, DY, m, NX);
            if (sharp) CpuOps::crop(Wc2.data(), RW2.data(), g0, L, H, DY, m, NX);
            tick("sino_crop");
        }
        for (auto& ch : chunks) std::vector<uint16_t>().swap(ch.raw);
        if (dbg && dbg->keep) { dbg->sino = S; dbg->weights = Wc; }
        balance<CpuOps>(S.data(), Wc.data(), Wc2.data(), NR, DY, NX, sharp, ap.measured_only);
        if (dbg && dbg->keep) { dbg->wbal = Wc; dbg->filtered.resize(static_cast<std::size_t>(DY) * NR * NX); }
        tick("rebalance");
        const std::size_t rown = static_cast<std::size_t>(NR) * NX;
        std::vector<float> row(rown), wrow(rown), w2row(sharp ? rown : 0), brow(sharp ? rown : 0), padded(static_cast<std::size_t>(NR) * rg.npad);
        arc_h.resize(arc_taps.empty() ? 0 : rown);
        if (I.fbp_start && raw_rows) I.start_h.resize(rown), I.start_wh.resize(rown);
        step("backproject");
        slices<CpuOps>(0, S.data(), Wc.data(), Wc2.data(), NR, DY, NX, sharp, rg,
                       SliceBuffers{row.data(), wrow.data(), w2row.data(), brow.data(), padded.data(), arc_h.data(), I.start_h.data(),
                                    I.start_wh.data(), I.slice_opposite.data()},
                       dbg && dbg->keep ? dbg : nullptr, ap.measured_only, sink, tick);
        return;
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    // ================================================================ CUDA
    // raw windows of the whole chunk -> pinned host memory; then, per block of grid angles, only the projections
    // contributing to that block go to the GPU (ring of two buffers: block b + 1 uploads while block b computes)
    const std::size_t nsc = I.scans.size();
    if (pinned.size() != nsc) pinned.resize(nsc);
    if (d_raw[0].size() != nsc)
        for (int r = 0; r < 2; ++r) { d_raw[r].resize(nsc); d_slot[r].resize(nsc); }
    const bool prefetched = take_prefetch(h_lo, h_hi, m);
    tick("read_wait");
    if (prefetched) step("grid");
    struct DevChunk {
        cuda::DeviceBuffer<int32_t> row0, fi0, fi1, gi0, gi1, grow0;
        cuda::DeviceBuffer<float> fract_v, alpha, cdiv, wfac, ffrac, glare, gfrac, z_w, slope, col_offset, col_slope, col_scale, row_scale;
    };
    std::vector<DevChunk> dc(nsc);
    std::vector<std::size_t> per(nsc);
    for (std::size_t s = 0; s < nsc; ++s) per[s] = static_cast<std::size_t>(chunks[s].win_rows) * I.scans[s].traj.det_cols;
    for (std::size_t s = 0; s < nsc; ++s) {
        const auto& c = chunks[s].cs;
        if (!prefetched) {
            pinned[s].ensure(std::max(static_cast<std::size_t>(c.p1 - c.p0), next_hi > next_lo ? raw_capacity(s, DY, m) : 0) * per[s]);
            tick("alloc");
        }
        dc[s].row0.upload(c.row0.data(), c.row0.size());
        dc[s].fi0.upload(c.flat_i0.data(), c.flat_i0.size());
        dc[s].fi1.upload(c.flat_i1.data(), c.flat_i1.size());
        dc[s].fract_v.upload(c.fract_v.data(), c.fract_v.size());
        if (!c.row_slope.empty()) dc[s].slope.upload(c.row_slope.data(), c.row_slope.size());
        if (!c.col_offset.empty()) {
            dc[s].col_offset.upload(c.col_offset.data(), c.col_offset.size());
            dc[s].col_slope.upload(c.col_slope.data(), c.col_slope.size());
        }
        if (!c.col_scale.empty()) dc[s].col_scale.upload(c.col_scale.data(), c.col_scale.size());
        if (!c.row_scale.empty()) dc[s].row_scale.upload(c.row_scale.data(), c.row_scale.size());
        dc[s].alpha.upload(c.alpha.data(), c.alpha.size());
        dc[s].cdiv.upload(c.current_div.data(), c.current_div.size());
        dc[s].wfac.upload(c.weight_factor.data(), c.weight_factor.size());
        dc[s].ffrac.upload(c.flat_frac.data(), c.flat_frac.size());
        if (!chunks[s].z_w.empty()) dc[s].z_w.upload(chunks[s].z_w.data(), chunks[s].z_w.size());
        if (cfg.sld.enabled()) {
            if (!cg[s].row0.empty()) dc[s].grow0.upload(cg[s].row0.data(), cg[s].row0.size());
            dc[s].glare.upload(cg[s].maps.data(), cg[s].maps.size());
            dc[s].gi0.upload(cg[s].i0.data(), cg[s].i0.size());
            dc[s].gi1.upload(cg[s].i1.data(), cg[s].i1.size());
            dc[s].gfrac.upload(cg[s].frac.data(), cg[s].frac.size());
        }
    }
    // block plans: contributors, contiguous projection ranges, projection -> slot maps
    struct BlockPlan {
        int g0, g1;
        std::vector<ContributorList> contrib;
        std::vector<std::vector<std::pair<int, int>>> ranges;
        std::vector<std::vector<int32_t>> slot;
        std::vector<std::size_t> count;
    };
    std::vector<BlockPlan> plans;
    std::vector<std::size_t> max_count(nsc, 0);
    for (int g0 = 0; g0 < NR; g0 += BL) {
        BlockPlan bpl;
        bpl.g0 = g0;
        bpl.g1 = std::min(NR, g0 + BL);
        for (std::size_t s = 0; s < nsc; ++s) {
            const auto& c = chunks[s].cs;
            const int n = c.p1 - c.p0;
            bpl.contrib.push_back(make_contributors(c.j0.data(), n, NR, bpl.g0, bpl.g1, !I.nearest));
            std::vector<char> used(n, 0);
            for (int32_t k : bpl.contrib.back().proj) used[k] = 1;
            std::vector<std::pair<int, int>> rg_;
            std::vector<int32_t> slot(n, -1);
            std::size_t cnt = 0;
            for (int k = 0; k < n;) {
                if (!used[k]) { ++k; continue; }
                const int a0 = k;
                while (k < n && used[k]) slot[k++] = static_cast<int32_t>(cnt++);
                rg_.push_back({a0, k});
            }
            bpl.ranges.push_back(std::move(rg_));
            bpl.slot.push_back(std::move(slot));
            bpl.count.push_back(cnt);
            max_count[s] = std::max(max_count[s], cnt);
        }
        plans.push_back(std::move(bpl));
    }
    for (std::size_t s = 0; s < nsc; ++s)
        for (int r = 0; r < 2; ++r) {
            d_raw[r][s].ensure(std::max<std::size_t>(1, max_count[s]) * per[s]);
            d_slot[r][s].ensure(static_cast<std::size_t>(chunks[s].cs.p1 - chunks[s].cs.p0));
        }
    held.lo = h_lo; held.hi = h_hi; held.m = m;
    held.p0.resize(nsc); held.p1.resize(nsc); held.row0.resize(nsc);
    for (std::size_t s = 0; s < nsc; ++s) {
        held.p0[s] = chunks[s].cs.p0; held.p1[s] = chunks[s].cs.p1;
        held.row0[s] = chunks[s].cs.row0;
    }
    // Not prefetched: a background thread reads the raw windows in the order the blocks need them, so the uploads and
    // the gridding start with the first block's projections (wait(b): block b's are in `pinned`).
    struct RawRead {
        std::mutex m;
        std::condition_variable cv;
        std::size_t done = 0;
        std::exception_ptr err;
        std::thread t;
        ~RawRead() {
            if (t.joinable()) t.join();
        }
        void wait(std::size_t b) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return done > b || err; });
            if (err) std::rethrow_exception(err);
        }
    } raw_read;
    if (!prefetched)
        raw_read.t = std::thread([&] {
            try {
                std::vector<std::vector<char>> got(nsc);
                for (std::size_t s = 0; s < nsc; ++s) got[s].assign(chunks[s].cs.p1 - chunks[s].cs.p0, 0);
                auto read = [&](std::size_t s, int a0, int e0) {   // the unread projections of [a0, e0)
                    const auto& sd = I.scans[s];
                    const auto& c = chunks[s].cs;
                    for (int k = a0; k < e0;) {
                        if (got[s][k]) { ++k; continue; }
                        const int k0 = k;
                        while (k < e0 && !got[s][k]) got[s][k++] = 1;
                        sd.reader->read_rows(sd.file_index.data() + c.p0 + k0, sd.frame_index.data() + c.p0 + k0,
                                             chunks[s].win0.data() + k0, k - k0, chunks[s].win_rows,
                                             pinned[s].data() + static_cast<std::size_t>(k0) * per[s]);
                    }
                };
                for (std::size_t b = 0; b <= plans.size(); ++b) {
                    for (std::size_t s = 0; s < nsc; ++s) {
                        if (b == plans.size()) read(s, 0, static_cast<int>(got[s].size()));   // (any no block uses)
                        else
                            for (auto [a0, e0] : plans[b].ranges[s]) read(s, a0, e0);
                    }
                    {
                        std::lock_guard<std::mutex> lock(raw_read.m);
                        raw_read.done = b + 1;
                    }
                    raw_read.cv.notify_all();
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(raw_read.m);
                    raw_read.err = std::current_exception();
                }
                raw_read.cv.notify_all();
            }
        });
    // the next chunk's raw windows are read while this one computes (its rows shared with this chunk are copied from
    // `pinned`: once that is complete)
    const bool want_next = next_hi > next_lo && next_lo >= I.doable.first && next_hi <= I.doable.second;
    if (want_next && prefetched) start_prefetch(next_lo, next_hi, m);
    cudaStream_t cs = copy->get();
    auto issue_upload = [&](std::size_t b) {
        const int r = static_cast<int>(b & 1);
        if (!prefetched) {
            raw_read.wait(b);
            tick("read_raw");
        }
        VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamWaitEvent(cs, ev_done[r], 0));  // ring slot free (block b - 2 done)
        for (std::size_t s = 0; s < nsc; ++s) {
            std::size_t off = 0;
            for (auto [a0, e0] : plans[b].ranges[s]) {
                const std::size_t q = static_cast<std::size_t>(e0 - a0);
                VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_raw[r][s].data() + off * per[s], pinned[s].data() + a0 * per[s],
                                                   q * per[s] * sizeof(uint16_t), cudaMemcpyHostToDevice, cs));
                off += q;
            }
            if (!plans[b].slot[s].empty())  // (a scan at another height: the chunk holds none of it)
                VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_slot[r][s].data(), plans[b].slot[s].data(),
                                                   plans[b].slot[s].size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs));
        }
        VC_TOMOGRAPHY_CUDA_CHECK(cudaEventRecord(ev_up[r], cs));
    };
    tick("upload_issue");
    R.ensure(blk); RW.ensure(blk); Wp.ensure(blk);
    if (sharp) {
        R2.ensure(blk); RW2.ensure(blk); Wp2.ensure(blk); W2.ensure(sino_n);
        Aha2.ensure(static_cast<std::size_t>(BL) * H); Bha2.ensure(Aha2.size()); Atot2.ensure(Aha2.size()); Btot2.ensure(Aha2.size());
    }
    Aha.ensure(static_cast<std::size_t>(BL) * H); Bha.ensure(Aha.size()); Atot.ensure(Aha.size()); Btot.ensure(Aha.size());
    // (as on the CPU: rows of scans with the luminosity step are summed row by row)
    const bool by_row = I.rows_of_scans && cfg.luminosity_repartition != 0.f;
    if (by_row) {
        Rr.ensure(blk); RWr.ensure(blk);
        if (sharp) R2r.ensure(blk), RW2r.ensure(blk);
    }
    const GridBuffers gb{R.data(), RW.data(), R2.data(), RW2.data(), Rr.data(), RWr.data(), R2r.data(), RW2r.data(), Wp.data(), Wp2.data(),
                         Aha.data(), Bha.data(), Atot.data(), Btot.data(), Aha2.data(), Bha2.data(), Atot2.data(), Btot2.data()};
    S.ensure(sino_n); W.ensure(sino_n);
    // sweeps: one Fourier step per chain -> cache the spectra (filter + inverse per chain); else the intensity
    bool spectral = K > 1;
    for (const auto* P : sets) spectral = spectral && P->single_fourier();
    FourierFilter* fwd = nullptr;
    if (spectral) {
        fwd = fft_for(0, sets[0]->step_spec(0, H, NX, fm));
        Icache.ensure(static_cast<std::size_t>(NR) * fwd->spectrum_floats());
    } else if (K > 1) {
        Icache.ensure(static_cast<std::size_t>(NR) * H * NX);
    }
    issue_upload(0);
    for (std::size_t b = 0; b < plans.size(); ++b) {
        const int r = static_cast<int>(b & 1);
        const int g0 = plans[b].g0, g1 = plans[b].g1, L = g1 - g0;
        const std::size_t n_el = static_cast<std::size_t>(L) * H * NX;
        if (b + 1 < plans.size()) issue_upload(b + 1);
        const auto accumulate = [&](std::size_t s, float* aR, float* aRW, float* aR2, float* aRW2) {
            const auto& sd = I.scans[s];
            const auto& ch = chunks[s];
            auto view = make_view(sd, ch, d_raw[r][s].data(), I.dev[s].dark.data(), I.dev[s].flats.data(),
                                  I.dev[s].weight.data(), dc[s].row0.data(), dc[s].fract_v.data(),
                                  dc[s].alpha.data(), dc[s].cdiv.data(), dc[s].wfac.data(), dc[s].ffrac.data(),
                                  dc[s].fi0.data(), dc[s].fi1.data(), I.x_kernel, I.x_exact);
            view.raw_slot = d_slot[r][s].data();
            set_noise_split(view, cfg, cfg.noise_split_scale != 0.f ? I.split_salt[s] : 0U, chunks[s].cs.p0);
            set_z_kernel(view, I.z_pad, I.z_ctx, dc[s].z_w.data());
            set_roll(view, sd, dc[s].slope.data(), I.d_z_lut.data(), Impl::z_lut_n);
            set_lateral(view, sd, chunks[s].cs, dc[s].col_offset.data(), dc[s].col_slope.data(), I.x_exact);
            set_map(view, sd, I.maps[s], I.dev[s].map, dc[s].row_scale.data(), dc[s].col_scale.data(), I.d_z_lut.data(), Impl::z_lut_n);
            if (cfg.sld.enabled())
                set_glare(view, *I.glare[s], dc[s].glare.data(), dc[s].gi0.data(), dc[s].gi1.data(), dc[s].gfrac.data(),
                          cg[s].rows, cg[s].row0.empty() ? nullptr : dc[s].grow0.data());
            cuda::accumulate(view, plans[b].contrib[s], ap, aR, aRW, nullptr, aR2 ? I.dev[s].weight2.data() : nullptr, aR2, aRW2);
        };
        // (the ring slot is free once the last scan's gridding of this block is queued)
        const auto done = [&](std::size_t s) {
            if (s + 1 == nsc) VC_TOMOGRAPHY_CUDA_CHECK(cudaEventRecord(ev_done[r], nullptr));
        };
        const auto ready = [&] {   // (block b's projections uploaded)
            VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamWaitEvent(nullptr, ev_up[r], 0));
            tick("upload_wait");
        };
        grid_block<CudaOps>(L, H, chunks, gb, sharp, by_row, ap, filter_context, accumulate, done, ready, tick);
        if (spectral) {  // sweep: keep the spectra of the padded intensity; filter / inverse / log / crop per chain below
            fwd = fft_for(0, sets[0]->step_spec(0, H, NX, fm));
            fwd->forward(R.data(), Icache.data() + static_cast<std::size_t>(g0) * fwd->spectrum_floats(), L);
            tick("filters");
        } else if (K > 1) {  // sweep: keep the intensity; the whole chain runs per chain below
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(Icache.data() + static_cast<std::size_t>(g0) * H * NX, R.data(),
                                               n_el * sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            filter_block<CudaOps>(*sets[0], R.data(), L, H, fm, ap, tick);
            CudaOps::crop(S.data(), R.data(), g0, L, H, DY, m, NX);   // rows [m, m + DY) of each grid angle -> the sinogram
        }
        CudaOps::crop(W.data(), RW.data(), g0, L, H, DY, m, NX);
        if (sharp) CudaOps::crop(W2.data(), RW2.data(), g0, L, H, DY, m, NX);
        tick("sino_crop");
    }
    if (!prefetched) {
        raw_read.wait(plans.size());
        raw_read.t.join();
        if (want_next) start_prefetch(next_lo, next_hi, m);
    }
    if (progress) {
        // the gridding and filtering queued above finish here rather than inside the first slice's download, so
        // that the step shown is the one running
        step("grid");
        VC_TOMOGRAPHY_CUDA_CHECK(cudaDeviceSynchronize());
        step("backproject");
    }
    const std::size_t rown = static_cast<std::size_t>(NR) * NX;
    row.ensure(rown); wrow.ensure(rown); pad.ensure(static_cast<std::size_t>(NR) * rg.npad);
    if (views) d_opposite.upload(I.slice_opposite.data(), I.slice_opposite.size());
    if (sharp) { w2row.ensure(rown); brow.ensure(rown); }
    if (!arc_taps.empty()) arc_d.ensure(rown);
    if (I.fbp_start && raw_rows) I.start_d.ensure(rown), I.start_wd.ensure(rown);
    const SliceBuffers sb{row.data(), wrow.data(), w2row.data(), brow.data(), pad.data(), arc_d.data(), I.start_d.data(), I.start_wd.data(),
                          d_opposite.data()};
    const auto chain_slices = [&](int k) {
        slices<CudaOps>(k, S.data(), W.data(), W2.data(), NR, DY, NX, sharp, rg, sb, dbg && dbg->keep && K == 1 ? dbg : nullptr,
                        ap.measured_only, sink, tick);
    };
    if (K == 1) {
        if (dbg && dbg->keep) {
            dbg->sino.resize(sino_n); dbg->weights.resize(sino_n);
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(dbg->sino.data(), S.data(), sino_n * sizeof(float), cudaMemcpyDeviceToHost));
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(dbg->weights.data(), W.data(), sino_n * sizeof(float), cudaMemcpyDeviceToHost));
        }
        balance<CudaOps>(S.data(), W.data(), W2.data(), NR, DY, NX, sharp, ap.measured_only);
        if (dbg && dbg->keep) {
            dbg->wbal.resize(sino_n);
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(dbg->wbal.data(), W.data(), sino_n * sizeof(float), cudaMemcpyDeviceToHost));
            dbg->filtered.resize(sino_n);
        }
        tick("rebalance");
        chain_slices(0);
        return;
    }
    // ---- sweep: shared weights, then per set: phase / log / crop from the cached intensity, slices
    std::vector<std::pair<int, int>> holes;
    TurnHoles tholes;
    bool all_empty = false;
    if (views) {
        tholes = cuda::turn_holes(W.data(), NR, DY, NX, NG);
        all_empty = tholes.all_empty;
        if (!all_empty) {   // (balanced weights are 0 where the weights are: fill_from_turns below reads either)
            cuda::fill_turn_holes(W.data(), tholes, DY, NX);
            cuda::balance_weights_turns(W.data(), NR, DY, NX, NG, cfg.rot_center);
            if (sharp) {
                cuda::fill_turn_holes(W2.data(), tholes, DY, NX);
                cuda::balance_weights_turns(W2.data(), NR, DY, NX, NG, cfg.rot_center);
            }
        }
    } else {
        holes = cuda::rebalance_holes(W.data(), NR, DY, NX);
        all_empty = !holes.empty() && holes.front().first < 0;
        if (!all_empty) {
            cuda::fill_holes(W.data(), holes, NR, DY, NX);
            cuda::balance_weights(W.data(), NR, DY, NX, cfg.rot_center, cfg.legacy_reference);
            if (sharp) {
                cuda::fill_holes(W2.data(), holes, NR, DY, NX);
                cuda::balance_weights(W2.data(), NR, DY, NX, cfg.rot_center, cfg.legacy_reference);
            }
        }
    }
    tick("rebalance");
    for (int k = 0; k < K; ++k) {
        for (int g0 = 0; g0 < NR; g0 += BL) {
            const int L = std::min(NR, g0 + BL) - g0;
            const std::size_t n_el = static_cast<std::size_t>(L) * H * NX;
            if (spectral) {
                FourierFilter* f = fft_for(0, sets[k]->step_spec(0, H, NX, fm));
                f->inverse(Icache.data() + static_cast<std::size_t>(g0) * f->spectrum_floats(), R.data(), L);
            } else {
                VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(R.data(), Icache.data() + static_cast<std::size_t>(g0) * H * NX,
                                                   n_el * sizeof(float), cudaMemcpyDeviceToDevice));
                apply_filters(*sets[k], FilterStage::Intensity, R.data(), L, H, fm);
            }
            tick("filters");
            cuda::minus_log(R.data(), n_el, cfg.log_clip_min, cfg.log_clip_max);
            apply_filters(*sets[k], FilterStage::Log, R.data(), L, H, fm);
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy2DAsync(S.data() + static_cast<std::size_t>(g0) * DY * NX, DY * NX * sizeof(float),
                                                 R.data() + static_cast<std::size_t>(m) * NX, static_cast<std::size_t>(H) * NX * sizeof(float),
                                                 static_cast<std::size_t>(DY) * NX * sizeof(float), L, cudaMemcpyDeviceToDevice));
            tick("log");
        }
        if (!all_empty && views) {
            cuda::fill_turn_holes(S.data(), tholes, DY, NX);
            cuda::fill_from_turns(S.data(), W.data(), NR, DY, NX, NG);
        } else if (!all_empty) {
            cuda::fill_holes(S.data(), holes, NR, DY, NX);
        }
        tick("rebalance");
        chain_slices(k);
    }
#endif
}

namespace {

// A block of an (ry, rx) slice: [x0, x0 + w) x [y0, y0 + h)
struct Block {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// zeros everywhere in the slice but the block
template <class T>
void clear_outside(T* slice, int ry, int rx, const Block& b) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < ry; ++y) {
        T* r = slice + static_cast<std::size_t>(y) * rx;
        if (y < b.y0 || y >= b.y0 + b.h) {
            std::memset(r, 0, static_cast<std::size_t>(rx) * sizeof(T));
        } else {
            std::memset(r, 0, static_cast<std::size_t>(b.x0) * sizeof(T));
            std::memset(r + b.x0 + b.w, 0, static_cast<std::size_t>(rx - b.x0 - b.w) * sizeof(T));
        }
    }
}

// the block's (h, w) packed rows into the slice
template <class T>
void place(T* slice, int rx, const Block& b, const T* rows) {
    for (int y = 0; y < b.h; ++y)
        std::memcpy(slice + static_cast<std::size_t>(b.y0 + y) * rx + b.x0, rows + static_cast<std::size_t>(y) * b.w,
                    static_cast<std::size_t>(b.w) * sizeof(T));
}

// The smallest block of an (ry, rx) slice holding every pixel the mask keeps in a chunk of n slices (w = 0: none).
Block mask_bounds(const OutputMask& m, int n, int ry, int rx) {
    auto index = [](int v, int o, int f, int size) { return std::min(std::max(v + o, 0) / f, size - 1); };
    // the mask elements the chunk reads
    const int k0 = index(0, m.oz, m.fz, m.nz), k1 = index(n - 1, m.oz, m.fz, m.nz);
    const int i0 = index(0, m.oy, m.fy, m.ny), i1 = index(ry - 1, m.oy, m.fy, m.ny);
    const int j0 = index(0, m.ox, m.fx, m.nx), j1 = index(rx - 1, m.ox, m.fx, m.nx);
    int ia = i1 + 1, ib = i0 - 1, ja = j1 + 1, jb = j0 - 1;
    for (int k = k0; k <= k1; ++k)
        for (int i = i0; i <= i1; ++i) {
            const std::uint8_t* row = m.data + (static_cast<std::size_t>(k) * m.ny + i) * m.nx;
            for (int j = j0; j <= j1; ++j)
                if (row[j]) {
                    ia = std::min(ia, i); ib = std::max(ib, i);
                    ja = std::min(ja, j); jb = std::max(jb, j);
                }
        }
    if (ib < ia) return {};
    // the pixels reading elements [a, b] of an axis: from the first one of a to the last one of b
    auto span = [](int a, int b, int first, int last, int o, int f, int size) {
        const int lo = a <= first ? 0 : a * f - o, hi = b >= last ? size : (b + 1) * f - o;
        return std::pair<int, int>(lo, hi);
    };
    const auto [y0, y1] = span(ia, ib, i0, i1, m.oy, m.fy, ry);
    const auto [x0, x1] = span(ja, jb, j0, j1, m.ox, m.fx, rx);
    return {x0, y0, x1 - x0, y1 - y0};
}

// The backprojector's tiles (kernels.hpp) of a (bh, bw) block that hold a pixel the mask slice keeps; the block's
// pixel (y, x) reads the mask element (mask_index(y, oy, fy, ny), mask_index(x, ox, fx, nx)) (as to_uint8).
void kept_tiles(const std::uint8_t* mask, int ny, int nx, int fy, int fx, int oy, int ox, int bh, int bw,
                std::vector<std::int32_t>& tiles) {
    const int T = kBackprojectTile, ntx = (bw + T - 1) / T, nty = (bh + T - 1) / T;
    const auto index = output_core::mask_index;
    tiles.clear();
    for (int ty = 0; ty < nty; ++ty) {
        const int i0 = index(ty * T, oy, fy, ny), i1 = index(std::min(bh, (ty + 1) * T) - 1, oy, fy, ny);
        for (int tx = 0; tx < ntx; ++tx) {
            const int j0 = index(tx * T, ox, fx, nx), j1 = index(std::min(bw, (tx + 1) * T) - 1, ox, fx, nx);
            bool kept = false;
            for (int i = i0; i <= i1 && !kept; ++i)
                kept = std::any_of(mask + static_cast<std::size_t>(i) * nx + j0, mask + static_cast<std::size_t>(i) * nx + j1 + 1,
                                   [](std::uint8_t v) { return v != 0; });
            if (kept) tiles.push_back(ty * ntx + tx);
        }
    }
}

}  // namespace

void SliceReconstructor::reconstruct(int h_lo, int h_hi, const SliceOutput& out, ChunkDebug* dbg, const SliceRoi* roi_in,
                                     int next_lo, int next_hi, ProgressBar* progress) {
    auto& I = *impl_;
    const int N = I.grid_size;
    const SliceRoi roi = roi_in ? *roi_in : SliceRoi{0, N, 0, N};
    if (!(0 <= roi.x0 && roi.x0 < roi.x1 && roi.x1 <= N && 0 <= roi.y0 && roi.y0 < roi.y1 && roi.y1 <= N))
        throw Error("reconstruct: ROI outside the slice");
    const bool want_f32 = out.f32 || out.f32_sink, want_u8 = out.u8 || out.u8_sink;
    if (!want_f32 && !want_u8) throw Error("reconstruct: no output");
    if (h_hi <= h_lo) throw Error("reconstruct: empty height range");
    const int n = h_hi - h_lo, rx = roi.x1 - roi.x0, ry = roi.y1 - roi.y0;
    const std::size_t plane = static_cast<std::size_t>(rx) * ry, N2 = static_cast<std::size_t>(N) * N;
    const OutputMask& mask = out.u8_mask;
    if (mask.data && (mask.nz < 1 || mask.ny < 1 || mask.nx < 1 || mask.fz < 1 || mask.fy < 1 || mask.fx < 1))
        throw Error("reconstruct: empty mask");
    // the block of the ROI that is computed (the rest is 0)
    Block B{0, 0, rx, ry};
    auto cut = [&](int x0, int x1, int y0, int y1) {   // to [x0, x1) x [y0, y1) of the ROI
        const int a = std::max(B.x0, x0), b = std::min(B.x0 + B.w, x1), c = std::max(B.y0, y0), d = std::min(B.y0 + B.h, y1);
        B = b > a && d > c ? Block{a, c, b - a, d - c} : Block{};
    };
    if (out.region) cut(out.region->x0 - roi.x0, out.region->x1 - roi.x0, out.region->y0 - roi.y0, out.region->y1 - roi.y0);
    if (mask.data && !want_f32 && B.w) {
        const Block kept = mask_bounds(mask, n, ry, rx);
        cut(kept.x0, kept.x0 + kept.w, kept.y0, kept.y0 + kept.h);
    }
    const bool partial = B.w != rx || B.h != ry;
    // slice y of the chunk to the sinks, as its block (null: an empty slice)
    auto to_sinks = [&](int y, const float* f32, const std::uint8_t* u8) {
        if (out.f32_sink) out.f32_sink->block(out.sink_z + y, B.x0, B.y0, f32 ? B.w : 0, f32 ? B.h : 0, f32);
        if (out.u8_sink) out.u8_sink->block(out.sink_z + y, B.x0, B.y0, u8 ? B.w : 0, u8 ? B.h : 0, u8);
    };
    if (!B.w) {   // nothing to compute
        if (out.f32) clear_outside(out.f32, n * ry, rx, Block{});
        if (out.u8) clear_outside(out.u8, n * ry, rx, Block{});
        for (int y = 0; y < n; ++y) to_sinks(y, nullptr, nullptr);
        if (progress) progress->update(static_cast<std::uint64_t>(n));
        return;
    }
    const int bw = B.w, bh = B.h;
    const std::size_t block_plane = static_cast<std::size_t>(bw) * bh;
    const SliceRoi computed{roi.x0 + B.x0, roi.x0 + B.x0 + bw, roi.y0 + B.y0, roi.y0 + B.y0 + bh};   // in the slice
    // uint8 conversion of the block; `mask_data`: the mask slice (host or device copy)
    const std::size_t mask_plane = static_cast<std::size_t>(mask.ny) * mask.nx;
    auto mask_slice = [&](int y) { return std::min(std::max(y + mask.oz, 0) / mask.fz, mask.nz - 1); };
    auto window = [&](const std::uint8_t* mask_data) {
        Uint8Window w;
        w.lo = out.u8_lo;
        w.hi = out.u8_hi;
        if (out.u8_zero_outside) { w.disk_n = N; w.x0 = computed.x0; w.y0 = computed.y0; }
        if (mask_data) {
            w.mask = mask_data;
            w.mask_ny = mask.ny; w.mask_nx = mask.nx;
            w.mask_fy = mask.fy; w.mask_fx = mask.fx;
            w.mask_oy = mask.oy + B.y0; w.mask_ox = mask.ox + B.x0;
        }
        return w;
    };
    I.setup_invariants();
    const bool use_ghbp = I.cfg.backprojector == "ghbp";
    if (use_ghbp && !I.have_plan) {
        I.plan = make_ghbp_plan(I.angles.data(), I.n_grid, I.NX, I.gp);
        I.have_plan = true;
    }
    BackprojectParams bp = I.bp;
    bp.x0 = computed.x0;
    bp.y0 = computed.y0;
    // uint8 slices alone under a mask: the exact backprojector computes only the tiles holding a pixel the mask
    // keeps (the others are 0 in the output either way). The list is that of the mask slice in hand.
    const bool mask_tiles = mask.data && !want_f32 && I.cfg.backprojector == "fbp";
    std::vector<std::int32_t> tiles;
    int tiles_of = -1;
    auto set_tiles = [&](int y) {   // false: the mask keeps nothing of slice y
        if (!mask_tiles) return true;
        if (mask_slice(y) != tiles_of) {
            tiles_of = mask_slice(y);
            kept_tiles(mask.data + tiles_of * mask_plane, mask.ny, mask.nx, mask.fy, mask.fx, mask.oy + B.y0,
                       mask.ox + B.x0, bh, bw, tiles);
        }
        bp.tiles = tiles.data();
        bp.n_tiles = static_cast<int>(tiles.size());
        return !tiles.empty();
    };
    const bool external = I.cfg.backprojector == "external";
    I.raw_rows = external && !I.cfg.external->filtered();
    const int nd = I.NX * I.cfg.bp_oversample;
    // (run() names the chunk's steps on the bar; cleared on every way out)
    struct Scope {
        Impl& I;
        Scope(Impl& impl, ProgressBar* p) : I(impl) { I.progress = p; }
        ~Scope() { I.progress = nullptr; I.raw_rows = false; }
    } scope(I, progress);
    auto f32_of = [&](int y) { return out.f32 + static_cast<std::size_t>(y) * plane; };
    auto u8_of = [&](int y) { return out.u8 + static_cast<std::size_t>(y) * plane; };
    // a block computed on the host (CPU kernels, external backprojectors): `compute` fills its (bh, bw) floats
    std::vector<float> host_block(out.f32 && !partial ? 0 : block_plane);
    std::vector<std::uint8_t> host_u8(want_u8 && !(out.u8 && !partial) ? block_plane : 0);
    auto host_sink = [&](int y, const std::function<void(float*)>& compute) {
        float* blk = out.f32 && !partial ? f32_of(y) : host_block.data();
        compute(blk);
        std::uint8_t* blk8 = nullptr;
        if (want_u8) {
            blk8 = out.u8 && !partial ? u8_of(y) : host_u8.data();
            cpu::to_uint8(blk, bw, blk8, bh, bw, window(mask.data ? mask.data + mask_slice(y) * mask_plane : nullptr));
        }
        if (partial) {
            if (out.f32) {
                clear_outside(f32_of(y), ry, rx, B);
                place(f32_of(y), rx, B, blk);
            }
            if (out.u8) {
                clear_outside(u8_of(y), ry, rx, B);
                place(u8_of(y), rx, B, blk8);
            }
        }
        to_sinks(y, blk, blk8);
        if (progress) progress->update(1);
    };
    // An external method's start (ExternalBackprojector::fbp_start): the exact FBP of slice y on the whole grid, in
    // attenuation per pixel, from the rows run() made of it (start_rows); given to the method for its slice() alone.
    struct Start {
        ExternalBackprojector* e = nullptr;
        ~Start() { if (e) e->set_start(nullptr); }
    };
    auto give_start = [&](int y, Start& held) {
        if (!I.start_rows) return;
        BackprojectParams sp = I.bp;   // (the whole grid; its fan table is there where the start carries the divergence)
        sp.scale = 1.f;
        const float* image = nullptr;
        if (I.cfg.device == Device::CPU) {
            I.start_img_h.resize(N2);
            cpu::backproject(I.fbp_rows(I.start_rows), I.fbp_angles_of(y), I.start_img_h.data(), I.chunk_rows, nd, N, N, sp);
            image = I.start_img_h.data();
        }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        else {
            I.start_img_d.ensure(N2);
            cuda::backproject(I.fbp_rows(I.start_rows), I.fbp_angles_of(y), I.start_img_d.data(), I.chunk_rows, nd, N, N, sp);
            image = I.start_img_d.data();
        }
#endif
        I.cfg.external->set_start(image);
        held.e = I.cfg.external.get();
    };
    if (I.cfg.device == Device::CPU) {
        std::vector<float> full(use_ghbp ? N2 : 0);
        const auto P = I.pipeline(I.cfg.filters);
        I.run(h_lo, h_hi, dbg, {P.get()}, I.margin, I.filter_margin, [&](int, int y, float* row, float* wrow) {
            host_sink(y, [&](float* dst) {
                if (external) {
                    Start held;
                    give_start(y, held);
                    I.cfg.external->slice(h_lo + y, row, wrow, I.chunk_rows, I.NX, I.fbp_angles_of(y), bp, N, computed, Device::CPU, dst);
                } else if (use_ghbp) {
                    cpu::ghbp(I.plan, row, full.data());
                    for (int r = 0; r < bh; ++r)
                        std::memcpy(dst + static_cast<std::size_t>(r) * bw,
                                    full.data() + static_cast<std::size_t>(computed.y0 + r) * N + computed.x0, bw * sizeof(float));
                } else if (set_tiles(y)) {
                    cpu::backproject(I.fbp_rows(row), I.fbp_angles_of(y), dst, I.chunk_rows, nd, bh, bw, bp);
                } else {
                    std::fill(dst, dst + block_plane, 0.f);
                }
            });
        });
        return;
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (use_ghbp && !I.ghbp) I.ghbp = std::make_unique<cuda::Ghbp>(I.plan);
    if (!external) I.slice.ensure(use_ghbp ? N2 : block_plane);
    if (!external && want_u8) I.slice_u8.ensure(block_plane);
    if (!external && want_u8 && mask.data) I.slice_mask.ensure(mask_plane);
    int mask_loaded = -1;   // the mask slice on the device
    const std::size_t at = static_cast<std::size_t>(B.y0) * rx + B.x0;   // the block in an output slice
    const auto P = I.pipeline(I.cfg.filters);
    I.run(h_lo, h_hi, dbg, {P.get()}, I.margin, I.filter_margin, [&](int, int y, float* row, float* wrow) {
        if (external) {
            host_sink(y, [&](float* dst) {
                Start held;
                give_start(y, held);
                I.cfg.external->slice(h_lo + y, row, wrow, I.chunk_rows, I.NX, I.fbp_angles_of(y), bp, N, computed, Device::CUDA, dst);
            });
            return;
        }
        // the block on the device: (bh, bw) rows at a pitch of `pitch` floats
        const float* dev = I.slice.data();
        std::size_t pitch = static_cast<std::size_t>(bw);
        if (use_ghbp) {
            I.ghbp->run(row, I.slice.data());
            dev += static_cast<std::size_t>(computed.y0) * N + computed.x0;
            pitch = static_cast<std::size_t>(N);
        } else if (set_tiles(y)) {
            cuda::backproject(I.fbp_rows(row), I.fbp_angles_of(y), I.slice.data(), I.chunk_rows, nd, bh, bw, bp);
        } else {   // nothing kept: an empty slice
            if (out.u8) clear_outside(u8_of(y), ry, rx, Block{});
            to_sinks(y, nullptr, nullptr);
            if (progress) progress->update(1);
            return;
        }
        if (want_u8) {   // converted before the float download, which then overlaps it
            if (mask.data && mask_slice(y) != mask_loaded) {
                mask_loaded = mask_slice(y);
                VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(I.slice_mask.data(), mask.data + mask_loaded * mask_plane, mask_plane,
                                                   cudaMemcpyHostToDevice));
            }
            cuda::to_uint8(dev, pitch, I.slice_u8.data(), bh, bw, window(mask.data ? I.slice_mask.data() : nullptr));
        }
        // the block: device -> page-locked staging -> the sinks as it is, the host arrays at its place in the slice
        const float* host = nullptr;
        const std::uint8_t* host8 = nullptr;
        if (want_f32) {
            host = cuda::download_block(dev, pitch, bw, bh, I.slice_host);
            if (out.f32) {
                if (partial) clear_outside(f32_of(y), ry, rx, B);
                cuda::copy_rows(f32_of(y) + at, host, bw, bh, rx);
            }
        }
        if (want_u8) {
            host8 = cuda::download_block(static_cast<const std::uint8_t*>(I.slice_u8.data()), bw, bw, bh, I.slice_u8_host);
            if (out.u8) {
                if (partial) clear_outside(u8_of(y), ry, rx, B);
                cuda::copy_rows(u8_of(y) + at, host8, bw, bh, rx);
            }
        }
        to_sinks(y, host, host8);
        if (progress) progress->update(1);
    }, next_lo, next_hi);
#endif
}

void SliceReconstructor::reconstruct_band(int h_lo, int h_hi, BandSampler& band, ChunkDebug* dbg,
                                          const std::vector<FilterChain>* sweep, int next_lo, int next_hi) {
    auto& I = *impl_;
    if (band.device() != I.cfg.device) throw Error("reconstruct_band: sampler and reconstructor devices differ");
    if (band.box().n_grid != I.grid_size) throw Error("reconstruct_band: sampler built for another slice size");
    const bool sw = sweep && !sweep->empty();
    std::vector<std::shared_ptr<const FilterPipeline>> held;   // owned for the whole run
    if (sw)
        for (const auto& c : *sweep) held.push_back(I.pipeline(c));
    else
        held.push_back(I.pipeline(I.cfg.filters));
    std::vector<const FilterPipeline*> sets;
    for (const auto& h : held) sets.push_back(h.get());
    if (static_cast<int>(sets.size()) != band.n_sets())
        throw Error("reconstruct_band: the sampler has " + std::to_string(band.n_sets()) + " value sets for " +
                    std::to_string(sets.size()) + " filter chains");
    I.setup_invariants();
    if (I.cfg.backprojector != "fbp")
        throw Error("band reconstruction evaluates the exact backprojector at band pixels: backprojector must be fbp "
                    "(use whole slices + sampling for '" + I.cfg.backprojector + "')");
    const int nd = I.NX * I.cfg.bp_oversample;
    // a sweep's single-step chains share one margin (the largest), so that they share the FFT geometry
    int fm = I.filter_margin;
    if (sw && I.cfg.margin < 0) {
        fm = 0;
        for (const auto& h : held) fm = std::max(fm, h->margin_rows());
    }
    I.run(h_lo, h_hi, dbg, sets, sw ? sweep_margin(*sweep) : I.margin, fm, [&](int k, int y, float* row, float*) {
        band.slice(h_lo + y, I.fbp_rows(row), I.chunk_rows, nd, I.fbp_angles_of(y), I.bp, k);
    }, next_lo, next_hi);
}

int SliceReconstructor::sweep_margin(const std::vector<FilterChain>& chains) const {
    const auto& I = *impl_;
    int m = 0;
    for (const auto& c : chains) m = std::max(m, I.pipeline(c)->margin_rows());
    return std::max(I.cfg.margin >= 0 ? I.cfg.margin : m, m + I.lum_margin);
}

std::string SliceReconstructor::describe(const FilterChain& chain) const { return impl_->pipeline(chain)->describe(); }

}  // namespace vc_tomography
