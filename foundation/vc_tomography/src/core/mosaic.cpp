// vc_tomography/mosaic.hpp: the canvas (a mosaic's tiles in one sinogram, one filtered backprojection) and a mosaic
// reconstructed so from its frames.
#include "vc_tomography/mosaic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>

#include "common/interp_core.hpp"
#include "io/json.hpp"
#include "vc_tomography/accumulate.hpp"
#include "vc_tomography/calibration/solve.hpp"
#include "vc_tomography/fourier.hpp"
#include "vc_tomography/kernels.hpp"
#include "vc_tomography/projection_store.hpp"
#include "vc_tomography/reconstruction.hpp"
#include "vc_tomography/sino.hpp"
#ifdef VC_TOMOGRAPHY_WITH_CUDA
#include "../cuda/cuda_utils.cuh"
#endif

namespace vc_tomography {

namespace {


// an array in the memory of a device: the host's for the CPU kernels, the GPU's for the CUDA ones
template <class T>
class Array {
  public:
    explicit Array(Device device) : dev_(device) {}
    void ensure(std::size_t n) {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (dev_ == Device::CUDA) {
            d_.ensure(n);
            return;
        }
#endif
        if (h_.size() < n) h_.resize(n);
    }
    T* data() {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (dev_ == Device::CUDA) return d_.data();
#endif
        return h_.data();
    }
    void zero(std::size_t n) {
        ensure(n);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (dev_ == Device::CUDA) {
            if (n) VC_TOMOGRAPHY_CUDA_CHECK(cudaMemset(d_.data(), 0, n * sizeof(T)));
            return;
        }
#endif
        std::fill(h_.begin(), h_.begin() + static_cast<std::ptrdiff_t>(n), T(0));
    }
    void put(const T* host, std::size_t n) {
        ensure(n);
        if (!n) return;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (dev_ == Device::CUDA) {
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(d_.data(), host, n * sizeof(T), cudaMemcpyHostToDevice));
            return;
        }
#endif
        std::memcpy(h_.data(), host, n * sizeof(T));
    }
    void get(T* host, std::size_t n, std::size_t from = 0) {
        if (!n) return;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (dev_ == Device::CUDA) {
            VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpy(host, d_.data() + from, n * sizeof(T), cudaMemcpyDeviceToHost));
            return;
        }
#endif
        std::memcpy(host, h_.data() + from, n * sizeof(T));
    }

  private:
    Device dev_;
    std::vector<T> h_;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    cuda::DeviceBuffer<T> d_;
#endif
};

const Json& member(const Json& j, const char* key) {
    const Json* v = j.find(key);
    if (!v) throw Error(std::string("mosaic: the geometry calibration has no \"") + key + "\"");
    return *v;
}

}  // namespace

// ---------------------------------------------------------------- the canvas

struct MosaicCanvas::Impl {
    explicit Impl(Device d) : G(d), W(d), ws(d), ww(d), gain(d), frac(d), pad(d), filt(d), up(d), img(d), first(d), source_views(d), bin_lower(d), column_offsets(d), measured_bins(d), brackets(d), poses(d), fans(d), targets(d), opposite(d), discs(d), ordinary(d), regional_weights(d), regional_image(d) {}
    CanvasConfig cfg;
    std::vector<CanvasTile> tiles;
    int NG = 0, size = 0, half = 0, n = 0, stride = 0, n_slices = 0;
    Array<float> G, W, ws, ww, gain, frac, pad, filt, up, img;
    Array<std::int32_t> first;
    Array<CanvasView> source_views;
    Array<int> bin_lower;
    Array<double> column_offsets, measured_bins;
    Array<CanvasBracket> brackets;
    Array<CanvasPose> poses;
    Array<CanvasFan> fans;
    Array<CanvasTarget> targets;
    Array<CanvasOpposite> opposite;
    Array<CanvasDisc> discs;
    Array<float> ordinary, regional_weights, regional_image;
    std::unique_ptr<FourierFilter> highpass;
    bool measured() const { return cfg.rebin_angles && (cfg.rebin_kernel == "measured" || cfg.rebin_kernel == "exposure"); }
    RampGeometry rg, rg_half;
    std::unique_ptr<FourierFilter> ramp;
    BackprojectParams bp;
    int degree = 0;
    std::vector<int> sub;            // the angles compared for the levels
    struct Kept {                    // of one tile in one slice, until the slice is made
        bool have = false;
        std::vector<float> signal;   // aligned weighted rows retained for regional FBP
        std::vector<float> w;        // (NG, stride): its weights as placed (shifted)
        std::vector<float> angles;   // original bin angles for the direct blend
        std::vector<float> sub_s;    // (sub, stride): value x weight as placed, of the angles compared
        std::vector<std::int32_t> first;
    };
    struct Made {
        bool done = false;
        int direction = 0;
        double phase = 0;  // common angular origin in radians
        int angle_sets = 0;
        std::vector<double> mean_angles;
        std::vector<CanvasOpposite> opposite;
        std::vector<double> levels;
        std::vector<float> angles;   // (NG / 2): the angle every folded row is backprojected at
        double spread = 0;
    };
    std::vector<std::vector<Kept>> kept;   // [slice][tile]
    std::vector<Made> made;

    float* slice_of(Array<float>& a, int y) { return a.data() + static_cast<std::size_t>(y) * NG * size; }
    std::vector<double> solve_levels(int y) const;
    CanvasRebin plan_rebin(const CanvasTile& tile, const float* angles, Made& slice,
                           const std::vector<int>& native_first, std::vector<int>& target_first);
    void direct_angles(int y);
    void make(int y);
    void filter(int y, float* sums, float* weights, Array<float>& filtered, CanvasDebug* debug);
    void project(int y, SliceRoi region, const float* filtered, Array<float>& image);
    void exposure_regions(int y, SliceRoi region);
};

MosaicCanvas::MosaicCanvas(CanvasConfig config, std::vector<CanvasTile> tiles) : impl_(std::make_unique<Impl>(config.device)) {
    Impl& I = *impl_;
    if (tiles.empty()) throw Error("mosaic canvas: no tiles");
    if (config.n_angles < 2 || config.n_angles % 2) throw Error("mosaic canvas: the angles of a turn must be an even number");
    if (!(config.pixel_size_m > 0)) throw Error("mosaic canvas: no pixel size");
    if (config.bp_oversample < 1 || config.bp_oversample > 8) throw Error("bp_oversample must be 1..8");
    if (config.anchor < 0 || config.anchor >= static_cast<int>(tiles.size())) throw Error("mosaic canvas: the anchor is not one of the tiles");
    if (config.rebin_kernel != "two_pass" && config.rebin_kernel != "linear" && config.rebin_kernel != "lanczos3" && config.rebin_kernel != "measured" && config.rebin_kernel != "exposure")
        throw Error("mosaic canvas: rebin_kernel expects two_pass, linear, lanczos3, measured or exposure");
    I.n = tiles[0].cols;
    double reach = 0;
    for (const CanvasTile& t : tiles) {
        if (!(std::isfinite(t.exposure_gate_rad) && t.exposure_gate_rad >= 0)) throw Error("mosaic canvas: invalid exposure gate");
        if (t.cols != I.n || t.cols < 2) throw Error("mosaic canvas: the tiles' rows must have one width");
        if (!t.fan.empty() && (static_cast<int>(t.fan.size()) != I.n ||
            !std::all_of(t.fan.begin(), t.fan.end(), [](float a) { return std::isfinite(a); }))) throw Error("mosaic canvas: invalid per-column fan");
        if (config.rebin_angles && config.rebin_kernel != "two_pass") {
            const double radius = std::hypot(t.ax, t.ay);
            for (std::size_t c = 1; c < t.fan.size(); ++c)
                if (radius * std::abs(static_cast<double>(t.fan[c]) - t.fan[c - 1]) >= 1)
                    throw Error("mosaic canvas: column fan does not guarantee an invertible detector mapping");
        }
        reach = std::max(reach, std::hypot(t.ax, t.ay) + std::max(t.rot_center, t.cols - 1 - t.rot_center));
    }
    I.cfg = std::move(config);
    I.tiles = std::move(tiles);
    I.NG = I.cfg.n_angles;
    I.half = I.cfg.half > 0 ? I.cfg.half : static_cast<int>(std::ceil(reach)) + 1;
    I.cfg.half = I.half;
    I.size = 2 * I.half + 1;
    I.stride = shift_row_stride(I.n);
    // the weight's fall at either end of a tile's rows
    std::vector<float> gain(static_cast<std::size_t>(I.n), 1.f);
    const int taper = std::min(I.cfg.taper, I.n / 2);
    for (int c = 0; c < taper; ++c) {
        const float g = static_cast<float>(0.5 - 0.5 * std::cos(M_PI * (c + 0.5) / taper));
        gain[static_cast<std::size_t>(c)] = g, gain[static_cast<std::size_t>(I.n - 1 - c)] = g;
    }
    I.gain.put(gain.data(), gain.size());
    // one ramp over rows as wide as the canvas, the axis at their middle; half the turn is backprojected
    I.rg = ramp_geometry(I.size, I.NG, static_cast<float>(I.half), 1, false);
    I.rg_half = I.rg;
    I.rg_half.n_proj = I.NG / 2;
    I.ramp = make_fourier_filter(I.cfg.device, ramp_filter_spec(I.rg, ramp_scale(I.NG, false), I.cfg.ramp_filter_name, I.cfg.ramp_filter_par));
    if (I.cfg.rebin_angles && I.cfg.rebin_kernel == "exposure" && I.tiles.size() > 1 &&
        std::all_of(I.tiles.begin(), I.tiles.end(), [](const CanvasTile& t) { return t.exposure_gate_rad > 0; })) {
        if (!(I.cfg.exposure_fade > 0)) throw Error("mosaic canvas: expected a positive regional taper");
        I.highpass = make_fourier_filter(I.cfg.device, band_filter_spec(1, 1, I.size, .04, .10));
    }
    I.bp.rot_center = static_cast<float>(I.half);
    I.bp.center = static_cast<float>(I.half);
    I.bp.scale = static_cast<float>(1.0 / (I.cfg.pixel_size_m * 1e2));
    I.bp.clip_outer_circle = false;
    I.bp.interp = interp_from_name(I.cfg.bp_interp);
    I.bp.u_scale = static_cast<float>(I.cfg.bp_oversample);
    I.degree = interp_core::spline_degree(I.bp.interp);
    for (int g = 0; g < I.NG; g += std::max(1, I.cfg.level_step)) I.sub.push_back(g);
}

MosaicCanvas::~MosaicCanvas() = default;
int MosaicCanvas::size() const { return impl_->size; }
bool MosaicCanvas::exposure_weighted() const { return bool(impl_->highpass); }
int MosaicCanvas::half() const { return impl_->half; }
const CanvasConfig& MosaicCanvas::config() const { return impl_->cfg; }

void MosaicCanvas::begin(int n_slices) {
    Impl& I = *impl_;
    if (n_slices < 1) throw Error("mosaic canvas: a chunk of no slices");
    I.n_slices = n_slices;
    const std::size_t all = static_cast<std::size_t>(n_slices) * I.NG * I.size;
    I.G.zero(all);
    I.W.zero(all);
    I.kept.assign(static_cast<std::size_t>(n_slices), std::vector<Impl::Kept>(I.tiles.size()));
    I.made.assign(static_cast<std::size_t>(n_slices), Impl::Made{});
}

void MosaicCanvas::add_view_angles(int y, const float* angles) {
    Impl& I = *impl_;
    if (!I.measured() || y < 0 || y >= I.n_slices) throw Error("mosaic: expected a measured-mode slice");
    auto& m = I.made[y];
    if (m.direction || m.done) throw Error("mosaic: supply target angles before adding rows");
    for (int r = 0; r < I.NG; ++r)
        if (!std::isfinite(angles[r])) throw Error("mosaic: nonfinite target angle");
    m.mean_angles.resize(I.NG);
    ++m.angle_sets;
    for (int r = 0; r < I.NG; ++r) {
        double& a = m.mean_angles[r];
        a = m.angle_sets == 1 ? angles[r] : a + std::remainder(angles[r] - a, 2 * M_PI) / m.angle_sets;
    }
}

CanvasRebin MosaicCanvas::Impl::plan_rebin(const CanvasTile& t, const float* angles, Made& slice,
                                          const std::vector<int>& native_first, std::vector<int>& target_first) {
    // A uniform physical grid shared by every tile. Work in bin units for
    // the periodic angular lookup. The second half is exactly conjugate.
    const bool native = this->measured();
    if (native && slice.mean_angles.empty()) slice.mean_angles.assign(angles, angles + NG);
    if (!slice.direction) {
        double travel = 0;
        for (int r = 1; r < NG; ++r)
            travel += std::remainder(static_cast<double>(angles[r]) - angles[r - 1], 2 * M_PI);
        if (NG == 2) travel = static_cast<double>(angles[1]) - angles[0];
        slice.direction = travel >= 0 ? 1 : -1;
        // Minimize the angular displacement of the first tile's views to the
        // uniform grid. Unwrap residuals against its first bin, including
        // acquisitions whose mean phase straddles the half-bin boundary.
        const double bin_width = slice.direction * 2 * M_PI / NG;
        double residual = 0;
        for (int r = 0; r < NG; ++r)
            residual += std::remainder((angles[r] - static_cast<double>(angles[0])) / bin_width - r, static_cast<double>(NG));
        slice.phase = angles[0] + bin_width * residual / NG;
    }
    const double step = slice.direction * 2 * M_PI / NG;
    const auto wrap = [this](double bin) {
        double v = std::fmod(bin, static_cast<double>(NG));
        if (v < 0) v += NG;
        return v < NG ? v : 0.0;
    };
    const auto bin_of = [&](double a, int r) {
        return a == static_cast<float>(slice.phase + r * step) ? double(r) : wrap((a - slice.phase) / step);
    };
    std::vector<double> measured(NG);
    std::vector<CanvasView> views(NG + 2);
    for (int r = 0; r < NG; ++r) {
        const double bin = bin_of(angles[r], r);
        measured[r] = bin;
        views[r + 1] = {bin, r};
        slice.spread = std::max(slice.spread, std::fabs(std::remainder(bin - (native ? bin_of(slice.mean_angles[r], r) : r), static_cast<double>(NG))));
    }
    std::sort(views.begin() + 1, views.end() - 1, [](CanvasView a, CanvasView b) {
        return a.bin < b.bin || (a.bin == b.bin && a.row < b.row);
    });
    views.front() = {views[NG].bin - NG, views[NG].row};
    views.back() = {views[1].bin + NG, views[1].row};
    std::vector<int> lower(NG);
    int k = 0;
    for (int r = 0; r < NG; ++r) {
        while (views[k + 1].bin <= r) ++k;
        lower[r] = k;
    }
    CanvasRebin grid{};
    const bool gather = cfg.rebin_kernel != "two_pass";
    if (gather || !t.fan.empty()) {
        source_views.put(views.data(), views.size());
        bin_lower.put(lower.data(), lower.size());
    }
    if (!gather && !t.fan.empty()) {
        measured_bins.put(measured.data(), measured.size());
        std::vector<double> fan_bins(t.fan.size());
        std::transform(t.fan.begin(), t.fan.end(), fan_bins.begin(), [&](float fan) { return wrap(fan / step); });
        column_offsets.put(fan_bins.data(), fan_bins.size());
        grid = {source_views.data(), bin_lower.data(), column_offsets.data(), measured_bins.data()};
    }

    std::vector<CanvasTarget> target(gather ? NG : 0);
    for (int r = 0; r < NG; ++r) {
        const double a = native ? slice.mean_angles[r] : slice.phase + r * step;
        const double origin = half - t.rot_center + t.ax * std::cos(a) + t.ay * std::sin(a);
        target_first[r] = static_cast<int>(std::floor(origin));
        if (gather) target[r] = {target_first[r], origin - target_first[r], native ? bin_of(a, r) : double(r)};
    }
    if (gather) {
        std::vector<CanvasPose> source(NG);
        for (int r = 0; r < NG; ++r) {
            const double cs = std::cos(static_cast<double>(angles[r])), sn = std::sin(static_cast<double>(angles[r]));
            const double projection = t.ax * cs + t.ay * sn;
            source[r] = {native_first[r], half - t.rot_center + projection - native_first[r],
                         projection, -t.ax * sn + t.ay * cs};
        }
        std::vector<CanvasFan> columns;
        columns.reserve(t.fan.size());
        for (double angle : t.fan)
            columns.push_back({angle / step, std::cos(angle) - 1, std::sin(angle)});
        poses.put(source.data(), NG);
        targets.put(target.data(), NG);
        fans.put(columns.data(), columns.size());
        return grid;
    }
    std::vector<CanvasBracket> br(NG);
    for (int r = 0; r < NG; ++r) {
        const CanvasView lo = views[lower[r]], hi = views[lower[r] + 1];
        br[r] = {lo.row, hi.row, target_first[r] - native_first[lo.row], target_first[r] - native_first[hi.row],
                 static_cast<float>((r - lo.bin) / (hi.bin - lo.bin))};
    }
    brackets.put(br.data(), br.size());
    return grid;
}

void MosaicCanvas::add(int tile, int y, const float* rows, const float* weights, const float* angles) {
    Impl& I = *impl_;
    if (tile < 0 || tile >= static_cast<int>(I.tiles.size()) || y < 0 || y >= I.n_slices) throw Error("mosaic canvas: no such tile or slice");
    if (I.made[static_cast<std::size_t>(y)].done) throw Error("mosaic canvas: the slice is made; nothing more is added to it");
    Impl::Kept& K = I.kept[static_cast<std::size_t>(y)][static_cast<std::size_t>(tile)];
    if (K.have) throw Error("mosaic canvas: a tile's rows of a slice are added once");
    const CanvasTile& t = I.tiles[static_cast<std::size_t>(tile)];
    const int NG = I.NG, n = I.n, stride = I.stride;
    // where the tile's column 0 falls on the canvas at every row's angle: whole columns, and the fraction its rows are delayed by
    K.first.resize(static_cast<std::size_t>(NG));
    std::vector<float> frac(NG);
    for (int r = 0; r < NG; ++r) {
        if (!std::isfinite(angles[r])) throw Error("mosaic canvas: nonfinite view angle");
        const double x = I.half - t.rot_center + t.ax * std::cos(static_cast<double>(angles[r])) + t.ay * std::sin(static_cast<double>(angles[r]));
        K.first[r] = static_cast<int>(std::floor(x));
        frac[r] = static_cast<float>(x - K.first[r]);
    }
    CanvasRebin grid{};
    if (I.cfg.rebin_angles) {
        const std::vector<int> native_first = K.first;
        grid = I.plan_rebin(t, angles, I.made[y], native_first, K.first);
    } else K.angles.assign(angles, angles + NG);
    I.first.put(K.first.data(), K.first.size());
    I.frac.put(frac.data(), frac.size());
    const std::size_t rown = static_cast<std::size_t>(NG) * stride;
    I.ws.ensure(rown);
    I.ww.ensure(rown);
    // Reuse the ramp's scratch for the native rows during angular rebinning.
    // The direct blend writes straight to the buffers consumed by place_rows.
    float* native_s = I.ws.data();
    float* native_w = I.ww.data();
    if (I.cfg.rebin_angles) {
        I.pad.ensure(rown);
        I.filt.ensure(rown);
        native_s = I.pad.data();
        native_w = I.filt.data();
    }
    const bool gather = I.cfg.rebin_angles && I.cfg.rebin_kernel != "two_pass";
    const CanvasGather rays{I.source_views.data(), I.bin_lower.data(), I.poses.data(), t.fan.empty() ? nullptr : I.fans.data(),
                            I.targets.data(), I.cfg.rebin_kernel == "lanczos3" || I.measured()};
    float* G = I.slice_of(I.G, y);
    float* W = I.slice_of(I.W, y);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (I.cfg.device == Device::CUDA) {
        if (grid.fan) cuda::rebin_tile_rows(grid, rows, weights, I.gain.data(), NG, n, stride, native_s, native_w);
        else cuda::tile_rows(rows, weights, I.gain.data(), NG, n, stride, native_s, native_w);
        cuda::shift_rows(native_s, NG, n, 0.f, nullptr, I.frac.data());
        cuda::shift_rows(native_w, NG, n, 0.f, nullptr, I.frac.data());
        if (gather) cuda::gather_rows(rays, native_s, native_w, NG, n, stride, I.ws.data(), I.ww.data());
        else if (I.cfg.rebin_angles) cuda::rebin_placed_rows(I.brackets.data(), native_s, native_w, NG, n, stride, I.ws.data(), I.ww.data());
        cuda::place_rows(I.ws.data(), I.first.data(), 1.f, false, NG, n, stride, G, I.size);
        cuda::place_rows(I.ww.data(), I.first.data(), 1.f, true, NG, n, stride, W, I.size);
    } else
#endif
    {
        if (grid.fan) cpu::rebin_tile_rows(grid, rows, weights, I.gain.data(), NG, n, stride, native_s, native_w);
        else cpu::tile_rows(rows, weights, I.gain.data(), NG, n, stride, native_s, native_w);
        cpu::shift_rows(native_s, NG, n, 0.f, I.frac.data());
        cpu::shift_rows(native_w, NG, n, 0.f, I.frac.data());
        if (gather) cpu::gather_rows(rays, native_s, native_w, NG, n, stride, I.ws.data(), I.ww.data());
        else if (I.cfg.rebin_angles) cpu::rebin_placed_rows(I.brackets.data(), native_s, native_w, NG, n, stride, I.ws.data(), I.ww.data());
        cpu::place_rows(I.ws.data(), I.first.data(), 1.f, false, NG, n, stride, G, I.size);
        cpu::place_rows(I.ww.data(), I.first.data(), 1.f, true, NG, n, stride, W, I.size);
    }
    // kept on the host until the slice is made: the weights as placed (a level is added through them, and the
    // angles are averaged by them), and of the angles compared the values
    if (I.highpass || !I.cfg.rebin_angles || (I.cfg.levels && I.tiles.size() > 1)) {
        K.w.resize(rown);
        I.ww.get(K.w.data(), rown);
    }
    if (!I.highpass && I.cfg.levels && I.tiles.size() > 1) {
        K.sub_s.resize(I.sub.size() * static_cast<std::size_t>(stride));
        for (std::size_t q = 0; q < I.sub.size(); ++q) I.ws.get(K.sub_s.data() + q * stride, static_cast<std::size_t>(stride), static_cast<std::size_t>(I.sub[q]) * stride);
    }
    if (I.highpass) {
        K.signal.resize(rown);
        I.ws.get(K.signal.data(), rown);
    }
    K.have = true;
}

// One level per tile: over the rays two tiles both recorded at one angle (weights above kShared on both), the
// weighted mean of the difference of their values is the difference of their levels; least squares over the pairs,
// each by the weight it was measured with, the anchor's level 0. A tile that shares no ray with another keeps 0.
std::vector<double> MosaicCanvas::Impl::solve_levels(int y) const {
    constexpr float kShared = 0.05f;   // a weight map is 1 where the flat is brightest
    constexpr int kLeast = 50;         // columns of an angle
    const std::size_t T = tiles.size();
    const std::vector<Kept>& K = kept[static_cast<std::size_t>(y)];
    std::vector<double> levels(T, 0.0);
    struct Pair {
        std::size_t i, j;
        double num = 0, den = 0;
    };
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < T; ++i)
        for (std::size_t j = i + 1; j < T; ++j)
            if (K[i].have && K[j].have && std::hypot(tiles[i].ax - tiles[j].ax, tiles[i].ay - tiles[j].ay) < 2.0 * n) pairs.push_back({i, j});
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(pairs.size()); ++p) {
        Pair& P = pairs[static_cast<std::size_t>(p)];
        const Kept &A = K[P.i], &B = K[P.j];
        for (std::size_t q = 0; q < sub.size(); ++q) {
            const std::size_t g = static_cast<std::size_t>(sub[q]);
            const int fa = A.first[g], fb = B.first[g], lo = std::max(fa, fb), hi = std::min(fa, fb) + n;
            if (hi - lo < kLeast) continue;
            const std::size_t row = g * stride;
            const float *wa = A.w.data() + row, *wb = B.w.data() + row;
            const float* sa = highpass ? A.signal.data() + row : A.sub_s.data() + q * stride;
            const float* sb = highpass ? B.signal.data() + row : B.sub_s.data() + q * stride;
            double num = 0, den = 0;
            int m = 0;
            for (int c = lo; c < hi; ++c) {
                const float a = wa[c - fa], b = wb[c - fb];
                if (!(a > kShared && b > kShared)) continue;
                const double w = static_cast<double>(a) * b;
                num += w * (static_cast<double>(sb[c - fb]) / b - static_cast<double>(sa[c - fa]) / a);
                den += w;
                ++m;
            }
            if (m >= kLeast) P.num += num, P.den += den;
        }
    }
    calibration::LinearProblem problem;
    problem.unknowns = T;
    for (const Pair& P : pairs) {
        if (!(P.den > 0)) continue;
        const double s = std::sqrt(P.den);
        const std::int64_t e = static_cast<std::int64_t>(problem.y.size());
        problem.row.push_back(e), problem.col.push_back(static_cast<std::int64_t>(P.i)), problem.value.push_back(s);    // level_i - level_j = mean(value_j - value_i)
        problem.row.push_back(e), problem.col.push_back(static_cast<std::int64_t>(P.j)), problem.value.push_back(-s);
        problem.y.push_back(s * P.num / P.den);
    }
    if (problem.y.empty()) return levels;
    problem.fixed.push_back(cfg.anchor);
    problem.fixed_value.push_back(0.0);
    const calibration::LinearSolution found = calibration::solve_linear(problem);
    if (found.usable && found.x.size() == T) levels = found.x;
    return levels;
}

void MosaicCanvas::Impl::direct_angles(int y) {
    Made& M = made[y];
    const auto& K = kept[y];
    const std::size_t T = tiles.size();
    // the angle of every row: the tiles' by the weight each has in the row; then of the two rows folded onto each other
    std::vector<double> sum_w(static_cast<std::size_t>(NG), 0.0), sum_a(static_cast<std::size_t>(NG), 0.0);
    std::vector<std::vector<double>> row_w(T);
    for (std::size_t k = 0; k < T; ++k) {
        if (!K[k].have) continue;
        row_w[k].assign(static_cast<std::size_t>(NG), 0.0);
        const float* w = K[k].w.data();
#pragma omp parallel for schedule(static)
        for (int r = 0; r < NG; ++r) {
            double s = 0;
            const float* row = w + static_cast<std::size_t>(r) * stride;
            for (int c = 0; c < n; ++c) s += row[c] > 0.f ? row[c] : 0.f;
            row_w[k][static_cast<std::size_t>(r)] = s;
        }
        for (int r = 0; r < NG; ++r) sum_w[static_cast<std::size_t>(r)] += row_w[k][static_cast<std::size_t>(r)], sum_a[static_cast<std::size_t>(r)] += row_w[k][static_cast<std::size_t>(r)] * K[k].angles[static_cast<std::size_t>(r)];
    }
    const std::size_t any = static_cast<std::size_t>(std::find_if(K.begin(), K.end(), [](const Kept& k) { return k.have; }) - K.begin());
    if (any == T) throw Error("mosaic canvas: a slice no tile was added to");
    const int H = NG / 2;
    const double step = 2 * M_PI / NG;
    M.angles.resize(static_cast<std::size_t>(H));
    std::vector<double> at(static_cast<std::size_t>(NG));
    for (int r = 0; r < NG; ++r) at[static_cast<std::size_t>(r)] = sum_w[static_cast<std::size_t>(r)] > 0 ? sum_a[static_cast<std::size_t>(r)] / sum_w[static_cast<std::size_t>(r)] : K[any].angles[static_cast<std::size_t>(r)];
    // (the rows' angles run one way round the turn, half a turn between row r and row r + H: its sign is the grid's)
    const double half_turn = at[static_cast<std::size_t>(H)] > at[0] ? M_PI : -M_PI;
    for (int r = 0; r < H; ++r) {
        const double wa = sum_w[static_cast<std::size_t>(r)], wb = sum_w[static_cast<std::size_t>(r + H)];
        const double a = at[static_cast<std::size_t>(r)], b = at[static_cast<std::size_t>(r + H)] - half_turn;
        M.angles[static_cast<std::size_t>(r)] = static_cast<float>(wa + wb > 0 ? (wa * a + wb * b) / (wa + wb) : a);
    }
    M.spread = 0;
    for (std::size_t k = 0; k < T; ++k) {
        if (!K[k].have) continue;
        for (int r = 0; r < NG; ++r) {
            if (!(row_w[k][static_cast<std::size_t>(r)] > 0)) continue;
            const double mine = K[k].angles[static_cast<std::size_t>(r)] - (r < H ? 0.0 : half_turn);
            M.spread = std::max(M.spread, std::fabs(mine - M.angles[static_cast<std::size_t>(r % H)]) / step);
        }
    }
}

static std::vector<CanvasOpposite> opposite_plan(const std::vector<double>& angles, int direction) {
    const int n = static_cast<int>(angles.size());
    const double step = direction * 2 * M_PI / n;
    std::vector<CanvasView> views(n);
    for (int r = 0; r < n; ++r) {
        double bin = std::fmod((angles[r] - angles[0]) / step, double(n));
        if (bin < 0) bin += n;
        views[r] = {bin, r};
    }
    std::sort(views.begin(), views.end(), [](CanvasView a, CanvasView b) { return a.bin < b.bin; });
    const auto at = [&](int i) {
        // The six-tap stencil may wrap more than once for very short turns.
        const int idx = (i % n + n) % n;
        return CanvasView{views[idx].bin + i - idx, views[idx].row};
    };
    std::vector<CanvasOpposite> plan(n);
    for (int i = 0; i < n; ++i) {
        auto& p = plan[views[i].row];
        // Periodic trapezoidal weight, including 1/2 for full-turn redundancy.
        p.quadrature = static_cast<float>(0.25 * (at(i + 1).bin - at(i - 1).bin));
        const double mirror = std::fmod(views[i].bin + 0.5 * n, double(n));
        const int lo = static_cast<int>(std::upper_bound(views.begin(), views.end(), mirror,
            [](double b, CanvasView v) { return b < v.bin; }) - views.begin()) - 1;
        const double t = (mirror - at(lo).bin) / (at(lo + 1).bin - at(lo).bin);
        p.fraction = static_cast<float>(t);
        p.lanczos = n >= 6 && p.fraction > 0 && p.fraction < 1;
        float norm = 0;
        for (int j = 0; j < 6; ++j) {
            const int k = lo + j - 2;
            const double gap = at(k).bin - at(k - 1).bin;
            if (j && (gap <= 0 || gap > 1.5)) p.lanczos = false;
            p.row[j] = at(k).row;
            p.coefficient[j] = interp_core::lanczos3(j - 2 - p.fraction);
            norm += p.coefficient[j];
        }
        for (float& coefficient : p.coefficient) coefficient /= norm;
    }
    return plan;
}

void MosaicCanvas::Impl::make(int y) {
    Made& M = made[static_cast<std::size_t>(y)];
    std::vector<Kept>& K = kept[static_cast<std::size_t>(y)];
    const std::size_t T = tiles.size(), rown = static_cast<std::size_t>(NG) * stride, all = static_cast<std::size_t>(NG) * size;
    float* Gy = slice_of(G, y);
    float* Wy = slice_of(W, y);
    M.levels.assign(T, 0.0);
    if (cfg.levels && T > 1) {
        M.levels = solve_levels(y);
        for (std::size_t k = 0; k < T; ++k) {
            if (!K[k].have || M.levels[k] == 0.0) continue;
            ww.put(K[k].w.data(), rown);
            first.put(K[k].first.data(), K[k].first.size());
            const float level = static_cast<float>(M.levels[k]);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
            if (cfg.device == Device::CUDA) cuda::place_rows(ww.data(), first.data(), level, true, NG, n, stride, Gy, size);
            else
#endif
                cpu::place_rows(ww.data(), first.data(), level, true, NG, n, stride, Gy, size);
        }
    }
    if (measured()) {
        if (!M.direction) throw Error("mosaic canvas: a slice no tile was added to");
        M.angles.assign(M.mean_angles.begin(), M.mean_angles.end());
        M.opposite = opposite_plan(M.mean_angles, M.direction);
        if (!highpass) K.clear();
        M.done = true;
        return;
    }
    if (cfg.rebin_angles) {
        if (!M.direction) throw Error("mosaic canvas: a slice no tile was added to");
        M.angles.resize(NG / 2);
        for (int r = 0; r < NG / 2; ++r) M.angles[r] = static_cast<float>(M.phase + r * (M.direction * 2 * M_PI / NG));
    } else direct_angles(y);
    K.clear();
    K.shrink_to_fit();
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (cfg.device == Device::CUDA) {
        cuda::fold_opposite(Gy, NG, size);
        cuda::fold_opposite(Wy, NG, size);
        cuda::divide(Gy, Wy, all);
        if (cfg.mend_threshold > 0) cuda::mend_null_weights(Gy, Wy, NG, 1, size, cfg.mend_threshold, 0, size);
    } else
#endif
    {
        cpu::fold_opposite(Gy, NG, size);
        cpu::fold_opposite(Wy, NG, size);
        cpu::divide(Gy, Wy, all);
        if (cfg.mend_threshold > 0) cpu::mend_null_weights(Gy, Wy, NG, 1, size, cfg.mend_threshold, 0, size);
    }
    M.done = true;
}

// Complete and ramp-filter one set of full-width weighted rays. The regional
// sums may occupy pad: completion consumes them before ramp padding reuses it.
void MosaicCanvas::Impl::filter(int y, float* Gy, float* Wy, Array<float>& filtered, CanvasDebug* debug) {
    const Made& M = made[y];
    const int bp_views = static_cast<int>(M.angles.size()), os = cfg.bp_oversample;
    const std::size_t all = static_cast<std::size_t>(NG) * size, bp_samples = static_cast<std::size_t>(bp_views) * size;
    pad.ensure(static_cast<std::size_t>(NG) * rg.npad);
    filtered.ensure(bp_samples);
    up.ensure(bp_samples * os);
    if (measured()) {
        opposite.put(M.opposite.data(), NG);
        float* completed = filtered.data();
        float* support = up.data();
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (cfg.device == Device::CUDA) {
            cuda::complete_views(opposite.data(), Gy, Wy, NG, size, completed, support);
            if (cfg.mend_threshold > 0) cuda::mend_null_weights(completed, support, NG, 1, size, cfg.mend_threshold, 0, size);
        } else
#endif
        {
            cpu::complete_views(opposite.data(), Gy, Wy, NG, size, completed, support);
            if (cfg.mend_threshold > 0) cpu::mend_null_weights(completed, support, NG, 1, size, cfg.mend_threshold, 0, size);
        }
    }
    if (debug) {
        debug->sino.resize(all), debug->weights.resize(all);
        W.get(debug->weights.data(), all, static_cast<std::size_t>(y) * all);
        if (measured()) {
            up.get(debug->sino.data(), all);
            std::size_t supported = 0, borrowed = 0;
            for (std::size_t i = 0; i < all; ++i) {
                if (debug->sino[i] <= cfg.mend_threshold) continue;
                ++supported;
                if (debug->weights[i] <= cfg.mend_threshold) ++borrowed;
            }
            debug->opposite_only_fraction = supported ? double(borrowed) / supported : 0;
            debug->weights.swap(debug->sino);
            filtered.get(debug->sino.data(), all);
        } else G.get(debug->sino.data(), all, static_cast<std::size_t>(y) * all);
        debug->angles = M.angles;
    }
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (cfg.device == Device::CUDA) {
        if (measured()) cuda::pad_measured(opposite.data(), filtered.data(), NG, size, rg.npad, rg.switch_pos, pad.data());
        else cuda::ramp_pad(Gy, pad.data(), rg);
        ramp->apply(pad.data(), pad.data(), bp_views);
        cuda::ramp_crop(pad.data(), filtered.data(), measured() ? rg : rg_half);
    } else
#endif
    {
        if (measured()) cpu::pad_measured(opposite.data(), filtered.data(), NG, size, rg.npad, rg.switch_pos, pad.data());
        else cpu::ramp_pad(Gy, pad.data(), rg);
        ramp->apply(pad.data(), pad.data(), bp_views);
        cpu::ramp_crop(pad.data(), filtered.data(), measured() ? rg : rg_half);
    }
    if (debug) {
        debug->filtered.resize(bp_samples);
        filtered.get(debug->filtered.data(), bp_samples);
    }
}

void MosaicCanvas::Impl::project(int y, SliceRoi R, const float* filtered, Array<float>& image) {
    const Made& M = made[y];
    const int bp_views = static_cast<int>(M.angles.size()), os = cfg.bp_oversample;
    const int ny = R.y1 - R.y0, nx = R.x1 - R.x0;
    image.ensure(static_cast<std::size_t>(ny) * nx);
    BackprojectParams bp = this->bp;
    bp.x0 = R.x0, bp.y0 = R.y0;
    const float* rows = nullptr;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (cfg.device == Device::CUDA) {
        rows = filtered;
        if (os > 1) {
            cuda::upsample_rows(filtered, up.data(), bp_views, size, os);
            rows = up.data();
        }
        if (degree) {
            cuda::bspline_prefilter(rows, up.data(), bp_views, size * os, degree);
            rows = up.data();
        }
        cuda::backproject(rows, M.angles.data(), image.data(), bp_views, size * os, ny, nx, bp);
    } else
#endif
    {
        rows = filtered;
        if (os > 1) {
            cpu::upsample_rows(filtered, up.data(), bp_views, size, os);
            rows = up.data();
        }
        if (degree) {
            cpu::bspline_prefilter(rows, up.data(), bp_views, size * os, degree);
            rows = up.data();
        }
        cpu::backproject(rows, M.angles.data(), image.data(), bp_views, size * os, ny, nx, bp);
    }
}

void MosaicCanvas::Impl::exposure_regions(int y, SliceRoi output) {
    const Made& M = made[y];
    const auto& aligned = kept[y];
    std::vector<CanvasDisc> footprints;
    std::vector<std::size_t> focuses;
    std::vector<SliceRoi> regions;
    for (std::size_t k = 0; k < tiles.size(); ++k) {
        if (!aligned[k].have) continue;
        const auto& t = tiles[k];
        const double radius = std::max(t.rot_center, t.cols - 1 - t.rot_center);
        const double x = half + t.ax, y0 = half + t.ay;
        const double dx = std::clamp(x, double(output.x0), double(output.x1 - 1)) - x;
        const double dy = std::clamp(y0, double(output.y0), double(output.y1 - 1)) - y0;
        if (dx * dx + dy * dy >= radius * radius) continue;
        focuses.push_back(k);
        footprints.push_back({x, y0, radius, std::min(radius, cfg.exposure_fade)});
        regions.push_back({std::max(output.x0, int(std::ceil(x - radius))), std::min(output.x1, int(std::floor(x + radius)) + 1),
                           std::max(output.y0, int(std::ceil(y0 - radius))), std::min(output.y1, int(std::floor(y0 + radius)) + 1)});
    }
    if (focuses.empty()) return;
    discs.put(footprints.data(), footprints.size());
    const auto blend = [&](int focus, SliceRoi region, const float* image) {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
        if (cfg.device == Device::CUDA)
            cuda::blend_region(discs.data(), footprints.size(), focus, region, output, image, img.data());
        else
#endif
            cpu::blend_region(discs.data(), footprints.size(), focus, region, output, image, img.data());
    };
    blend(-1, output, nullptr);
    const std::size_t samples = std::size_t(NG) * size;
    std::vector<double> sine(NG), cosine(NG);
    std::vector<float> quality(NG);
    for (int r = 0; r < NG; ++r) {
        sine[r] = std::sin(M.mean_angles[r]);
        cosine[r] = std::cos(M.mean_angles[r]);
    }
    for (std::size_t i = 0; i < focuses.size(); ++i) {
        const auto& center = tiles[focuses[i]];
        pad.zero(samples);
        regional_weights.zero(samples);
        for (std::size_t k = 0; k < tiles.size(); ++k) {
            const Kept& K = aligned[k];
            if (!K.have) continue;
            const auto& t = tiles[k];
            for (int r = 0; r < NG; ++r) {
                const double travel = t.exposure_gate_rad * ((center.ax - t.ax) * sine[r] - (center.ay - t.ay) * cosine[r]);
                const double q = travel * 0.5;
                quality[r] = float(.01 + .99 / (1. + q * q * q * q));
            }
            const float level = static_cast<float>(M.levels[k]);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
            if (cfg.device == Device::CUDA) {
                frac.put(quality.data(), NG);
                first.put(K.first.data(), NG);
                ws.put(K.signal.data(), K.signal.size());
                ww.put(K.w.data(), K.w.size());
                cuda::place_exposure(ws.data(), ww.data(), first.data(), frac.data(), level, NG, n, stride, pad.data(), regional_weights.data(), size);
            } else
#endif
                cpu::place_exposure(K.signal.data(), K.w.data(), K.first.data(), quality.data(), level, NG, n, stride, pad.data(), regional_weights.data(), size);
        }
        filter(y, pad.data(), regional_weights.data(), filt, nullptr);
        const auto add = [&](float scale) {
#ifdef VC_TOMOGRAPHY_WITH_CUDA
            if (cfg.device == Device::CUDA) cuda::add_scaled(filt.data(), ordinary.data(), scale, samples);
            else
#endif
                cpu::add_scaled(filt.data(), ordinary.data(), scale, samples);
        };
        add(-1.f);
        for (int r = 0; r < NG; r += 64) {
            float* rows = filt.data() + std::size_t(r) * size;
            highpass->apply(rows, rows, std::min(64, NG - r));
        }
        add(1.f);
        project(y, regions[i], filt.data(), regional_image);
        blend(int(i), regions[i], regional_image.data());
    }
}

void MosaicCanvas::slice(int y, float* out, const SliceRoi* roi, CanvasDebug* debug) {
    Impl& I = *impl_;
    if (y < 0 || y >= I.n_slices) throw Error("mosaic canvas: no such slice");
    if (!I.made[static_cast<std::size_t>(y)].done) I.make(y);
    const int size = I.size;
    const SliceRoi whole{0, size, 0, size};
    const SliceRoi R = roi ? *roi : whole;
    if (!(0 <= R.x0 && R.x0 < R.x1 && R.x1 <= size && 0 <= R.y0 && R.y0 < R.y1 && R.y1 <= size)) throw Error("mosaic canvas: the region is not inside the canvas");
    const int ny = R.y1 - R.y0, nx = R.x1 - R.x0;
    auto& baseline = I.highpass ? I.ordinary : I.filt;
    I.filter(y, I.slice_of(I.G, y), I.slice_of(I.W, y), baseline, debug);
    I.project(y, R, baseline.data(), I.img);
    if (I.highpass) I.exposure_regions(y, R);
    I.img.get(out, static_cast<std::size_t>(ny) * nx);
    // nothing is reconstructed outside the tiles' discs (from each axis to the far end of its rows)
#pragma omp parallel for schedule(static)
    for (int r = 0; r < ny; ++r) {
        std::vector<std::pair<int, int>> spans;
        const double Y = R.y0 + r - I.half;
        for (const CanvasTile& t : I.tiles) {
            const double radius = std::max(t.rot_center, t.cols - 1 - t.rot_center), dy = Y - t.ay;
            if (std::fabs(dy) > radius) continue;
            const double w = std::sqrt(radius * radius - dy * dy);
            spans.emplace_back(static_cast<int>(std::ceil(I.half + t.ax - w)) - R.x0, static_cast<int>(std::floor(I.half + t.ax + w)) + 1 - R.x0);
        }
        std::sort(spans.begin(), spans.end());
        float* row = out + static_cast<std::size_t>(r) * nx;
        int x = 0;
        for (const auto& [a, b] : spans) {
            if (a > x) std::fill(row + x, row + std::min(a, nx), 0.f);
            x = std::max(x, std::min(b, nx));
            if (x >= nx) break;
        }
        if (x < nx) std::fill(row + std::max(x, 0), row + nx, 0.f);
    }
}

const std::vector<double>& MosaicCanvas::levels(int y) const {
    if (y < 0 || y >= impl_->n_slices || !impl_->made[static_cast<std::size_t>(y)].done) throw Error("mosaic canvas: the slice is not made");
    return impl_->made[static_cast<std::size_t>(y)].levels;
}
double MosaicCanvas::angle_spread(int y) const {
    if (y < 0 || y >= impl_->n_slices || !impl_->made[static_cast<std::size_t>(y)].done) throw Error("mosaic canvas: the slice is not made");
    return impl_->made[static_cast<std::size_t>(y)].spread;
}

// ---------------------------------------------------------------- a mosaic from its frames

namespace {

// What a tile's own pipeline hands on per slice (an external backprojector that takes the rows as measured): the
// tile's rows go to the canvas, and its own slice is left empty.
class Collector : public ExternalBackprojector {
  public:
    MosaicCanvas* canvas = nullptr;
    int tile = 0, h_lo = 0, n_angles = 0, cols = 0;
    bool rebin_angles = false;
    bool uses_column_fan() const override { return rebin_angles; }
    bool filtered() const override { return false; }
    bool measured_only() const override { return true; }
    void slice(int view_height, const float* rows, const float* weights, int n_angles_, int n_det, const float* angles, const BackprojectParams&, int, const SliceRoi& roi,
               Device, float* out) override {
        if (n_angles_ != n_angles || n_det != cols) throw Error("mosaic: a tile's rows are not of the canvas' angles and columns");
        canvas->add(tile, view_height - h_lo, rows, weights, angles);
        std::fill(out, out + static_cast<std::size_t>(roi.y1 - roi.y0) * (roi.x1 - roi.x0), 0.f);
    }
};

}  // namespace

struct MosaicReconstruction::Impl {
    MosaicOptions options;
    std::string geometry;
    Json params;                     // the parameters asked for, over the defaults
    std::string tile_params;         // what every tile's pipeline is given
    struct Tile {
        MosaicTile pub;
        std::string projections;     // its group of the store
        std::optional<TileReconstruction> open;
        std::optional<calibration::DarksFlats> reduced;
    };
    std::vector<Tile> tiles;
    std::vector<MosaicTile> listing;  // every tile's `pub`, as tiles() gives them
    int anchor = 0, n_angles = 0, cols = 0, factor = 1;
    double row_origin = 0, pixel_size_m = 0;
    std::pair<int, int> doable{0, 0};
    std::shared_ptr<Collector> collector = std::make_shared<Collector>();
    std::unique_ptr<MosaicCanvas> canvas;
    std::vector<std::vector<double>> levels;
    std::vector<double> spread;

    TileReconstruction& open(std::size_t k) {
        Tile& t = tiles[k];
        if (t.open) return *t.open;
        TileOptions o;
        o.device = options.device;
        o.params = tile_params;
        if (static_cast<int>(k) != anchor || n_angles) o.row_origin = row_origin + t.pub.height;
        o.cache_dir = options.cache_dir;
        o.reduce = options.reduce;
        o.projections = t.projections;
        o.log = options.log;
        if (t.reduced) o.reduced = &*t.reduced;
        std::shared_ptr<Collector> c = collector;
        o.external_backprojector = [c](const FilterParams&) { return std::static_pointer_cast<ExternalBackprojector>(c); };
        t.open = tile_reconstructor(t.pub.master, geometry, t.pub.name, o);
        if (!t.reduced && !t.open->reduced.dark.empty()) t.reduced = t.open->reduced;   // (reduced once; a store's are read by the tile)
        return *t.open;
    }
    void close(std::size_t k) {
        if (!options.keep_tiles) tiles[k].open.reset();
    }
};

MosaicReconstruction::MosaicReconstruction(const std::string& projections, const std::string& geometry_text, MosaicOptions options) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    namespace fs = std::filesystem;
    I.options = std::move(options);
    I.collector->rebin_angles = I.options.rebin_angles;
    I.geometry = geometry_text;
    if (I.options.reduce < 0 || I.options.reduce > 3) throw Error("mosaic: reduce is 0, 1, 2 or 3");
    I.factor = 1 << I.options.reduce;
    const Json cal = Json::parse(geometry_text);
    const Json* model = cal.find("model");
    if (!model || !model->is(Json::String) || model->string != "joint") throw Error("mosaic: a joint geometry calibration (vc_tomography_calibration geometry's file) is needed");
    const Json& L = member(cal, "layout");
    const auto& masters = member(cal, "scans").array;
    const auto& names = member(L, "scans").array;
    const auto& places = member(member(cal, "parameters"), "scans").array;
    if (masters.size() != names.size() || masters.size() != places.size() || masters.empty()) throw Error("mosaic: the geometry calibration's scans, layout and parameters differ in number");
    I.anchor = static_cast<int>(member(L, "anchor").number);
    if (I.anchor < 0 || I.anchor >= static_cast<int>(masters.size())) throw Error("mosaic: the geometry calibration's anchor is not one of its scans");
    bool placed = masters.size() == 1;
    for (std::size_t k = 0; k < masters.size(); ++k) {
        Impl::Tile t;
        t.pub.name = member(names[k], "name").string;
        // the master: the calibration's, or the one of the same folders under `projections`
        const fs::path named(masters[k].string);
        fs::path local = fs::path(projections);
        {
            std::vector<fs::path> parts(named.begin(), named.end());
            for (std::size_t i = parts.size() > 3 ? parts.size() - 3 : 0; i < parts.size(); ++i) local /= parts[i];
        }
        t.pub.master = !projections.empty() && fs::exists(local) ? local.string() : named.string();
        if (!fs::exists(t.pub.master)) throw Error("mosaic: tile " + t.pub.name + "'s master is neither at " + named.string() + " nor at " + local.string());
        const auto& d = member(places[k], "d").array;
        if (d.size() != 2) throw Error("mosaic: a tile's place is two numbers");
        t.pub.ax = -d[0].number / I.factor, t.pub.ay = d[1].number / I.factor;
        t.pub.height = member(places[k], "height").number;
        if (static_cast<int>(k) != I.anchor && (d[0].number != 0 || d[1].number != 0)) placed = true;
        I.tiles.push_back(std::move(t));
    }
    if (!placed)
        throw Error("mosaic: the geometry calibration holds no place for its tiles (every d is 0); a whole-canvas reconstruction needs the tiles' places and "
                    "heights against the anchor, which vc_tomography_calibration geometry gives when it is run on all the tiles");
    // the parameters: the canvas takes the backprojector and the ramp's window, every tile's pipeline the rest
    I.params = Json::parse(update_parameters(default_parameters(), I.options.params.empty() ? "{}" : I.options.params));
    ReconConfig asked;
    {
        const Json* spec = I.params.find("backprojector");
        apply_backprojector(spec && spec->is(Json::String) ? spec->string : "fbp", asked);
        if (asked.backprojector != "fbp") throw Error("mosaic: the whole-canvas reconstruction is a filtered backprojection (backprojector=fbp(...))");
        if (member(I.params, "mode").string != "corrected") throw Error("mosaic: a mosaic is reconstructed in mode corrected");
    }
    // every tile's pipeline: the changes asked for; its rows as measured go to the canvas (an external backprojector),
    // one bin of one turn per row
    const auto for_tiles = [&](int n_grid) {
        Json t = Json::parse(I.options.params.empty() ? "{}" : I.options.params);
        if (!t.is(Json::Object)) throw Error("mosaic: the parameters are a JSON object");
        t.put("backprojector", Json::of(std::string("python")));
        t.put("angle_interp", Json::of(std::string("nearest")));
        if (n_grid) t.put("n_grid", Json::whole_number(n_grid));
        return t.text();
    };
    I.tile_params = for_tiles(0);
    if (!I.options.projection_store.empty())
        for (Impl::Tile& t : I.tiles) {
            const calibration::Scan scan = calibration::open_scan(t.pub.master);
            t.projections = projection_store_tile_group(I.options.projection_store, scan.sample_name);
        }
    // the anchor first: its numbering of the slices and its angles are everyone's
    {
        const std::size_t a = static_cast<std::size_t>(I.anchor);
        TileReconstruction& t = I.open(a);
        I.row_origin = t.row_origin - I.tiles[a].pub.height;
        I.n_angles = t.reconstructor->n_grid();
        I.cols = t.reconstructor->merged_width();
        I.pixel_size_m = t.reconstructor->config().pixel_size_m;
        if (I.n_angles % 2) throw Error("mosaic: the tiles' turn must have an even number of angles");
        I.tile_params = for_tiles(I.n_angles);
        I.tiles[a].open.reset();   // (opened again below with every tile's parameters)
    }
    I.collector->n_angles = I.n_angles, I.collector->cols = I.cols;
    std::vector<CanvasTile> on_canvas;
    for (std::size_t k = 0; k < I.tiles.size(); ++k) {
        TileReconstruction& t = I.open(k);
        MosaicTile& p = I.tiles[k].pub;
        p.rot_center = t.rot_center;
        p.doable = t.reconstructor->doable();
        if (t.reconstructor->n_grid() != I.n_angles || t.reconstructor->merged_width() != I.cols)
            throw Error("mosaic: tile " + p.name + " has other angles or columns than the anchor");
        I.doable = k == 0 ? p.doable : std::make_pair(std::max(I.doable.first, p.doable.first), std::min(I.doable.second, p.doable.second));
        // The physical acquisition step is independent of the requested reconstruction grid.
        if (t.scan.exposure_arc) {
            double fraction = *t.scan.exposure_arc;
            const Json* period = L.find("frame_period_s");
            if (period && period->is(Json::Number) && period->number > 0 && t.scan.count_time_s) {
                const double timed = *t.scan.count_time_s / period->number;
                if (timed > 0 && timed <= 1) fraction = timed;
            }
            p.exposure_gate_rad = fraction * recorded_angle_step(t.scan.unwrapped_angles_deg) * (M_PI / 180.);
        }
        on_canvas.push_back(CanvasTile{p.rot_center, p.ax, p.ay, I.cols, t.reconstructor->fan_of_columns(), p.exposure_gate_rad});
        I.close(k);
    }
    if (I.doable.second <= I.doable.first) throw Error("mosaic: no slice that every tile can give");
    CanvasConfig c;
    c.device = I.options.device;
    c.n_angles = I.n_angles;
    c.pixel_size_m = I.pixel_size_m;
    c.bp_interp = asked.bp_interp, c.bp_oversample = asked.bp_oversample;
    c.ramp_filter_name = asked.ramp_filter_name, c.ramp_filter_par = asked.ramp_filter_par;
    c.taper = std::max(0, I.options.taper / I.factor);
    c.levels = I.options.levels;
    c.exposure_fade = 600. / I.factor;
    c.anchor = I.anchor;
    c.rebin_angles = I.options.rebin_angles;
    c.rebin_kernel = I.options.rebin_kernel;
    if (const Json* v = I.params.find("mend_threshold"); v && v->is(Json::Number)) c.mend_threshold = static_cast<float>(v->number);
    I.canvas = std::make_unique<MosaicCanvas>(c, std::move(on_canvas));
    I.collector->canvas = I.canvas.get();
    for (const Impl::Tile& t : I.tiles) I.listing.push_back(t.pub);
}

MosaicReconstruction::~MosaicReconstruction() = default;
std::pair<int, int> MosaicReconstruction::doable() const { return impl_->doable; }
int MosaicReconstruction::size() const { return impl_->canvas->size(); }
double MosaicReconstruction::pixel_size_m() const { return impl_->pixel_size_m; }
const std::vector<MosaicTile>& MosaicReconstruction::tiles() const { return impl_->listing; }
const std::vector<std::vector<double>>& MosaicReconstruction::levels() const { return impl_->levels; }
const std::vector<double>& MosaicReconstruction::angle_spread() const { return impl_->spread; }

int MosaicReconstruction::max_chunk() const {
    const Impl& I = *impl_;
    if (!I.canvas->exposure_weighted()) return std::numeric_limits<int>::max();
    // Two full-width accumulators and the aligned signal/support of each tile per height.
    // A single native height can exceed the batch budget and is always processed.
    const std::size_t bytes = 2 * sizeof(float) * I.n_angles *
        (I.canvas->size() + I.tiles.size() * shift_row_stride(I.cols));
    constexpr std::size_t batch_budget = std::size_t(1) << 30;
    return static_cast<int>(std::max(std::size_t(1), batch_budget / bytes));
}

std::string MosaicReconstruction::info_json() const {
    const Impl& I = *impl_;
    Json tiles = Json::list();
    for (const Impl::Tile& t : I.tiles)
        tiles.add(Json::dict()
                      .set("name", Json::of(t.pub.name))
                      .set("master", Json::of(t.pub.master))
                      .set("ax", Json::of(t.pub.ax))
                      .set("ay", Json::of(t.pub.ay))
                      .set("height", Json::of(t.pub.height))
                      .set("exposure_gate_rad", Json::of(t.pub.exposure_gate_rad))
                      .set("rot_center", Json::of(t.pub.rot_center))
                      .set("doable", Json::list().add(Json::whole_number(t.pub.doable.first)).add(Json::whole_number(t.pub.doable.second))));
    const CanvasConfig& c = I.canvas->config();
    return Json::dict()
        .set("doable", Json::list().add(Json::whole_number(I.doable.first)).add(Json::whole_number(I.doable.second)))
        .set("size", Json::whole_number(I.canvas->size()))
        .set("half", Json::whole_number(I.canvas->half()))
        .set("n_angles", Json::whole_number(I.n_angles))
        .set("columns", Json::whole_number(I.cols))
        .set("pixel_size_m", Json::of(I.pixel_size_m))
        .set("reduce", Json::whole_number(I.options.reduce))
        .set("row_origin", Json::of(I.row_origin))
        .set("anchor", Json::of(I.tiles[static_cast<std::size_t>(I.anchor)].pub.name))
        .set("taper", Json::whole_number(c.taper))
        .set("levels", Json::of_bool(c.levels))
        .set("rebin_angles", Json::of_bool(c.rebin_angles))
        .set("rebin_kernel", Json::of(c.rebin_angles ? c.rebin_kernel : "direct"))
        .set("exposure_weighted", Json::of_bool(I.canvas->exposure_weighted()))
        .set("bp_interp", Json::of(c.bp_interp))
        .set("bp_oversample", Json::whole_number(c.bp_oversample))
        .set("ramp_filter_name", Json::of(c.ramp_filter_name))
        .set("tile_parameters", Json::parse(I.tile_params))
        .set("tiles", tiles)
        .text();
}

void MosaicReconstruction::reconstruct(int h_lo, int h_hi, float* out, const SliceRoi* roi, std::vector<CanvasDebug>* debug) {
    Impl& I = *impl_;
    if (h_lo < I.doable.first || h_hi > I.doable.second || h_hi <= h_lo) throw Error("mosaic: slices outside the doable span");
    const int size = I.canvas->size(), limit = std::min(max_chunk(), h_hi - h_lo);
    const std::size_t plane = roi ? static_cast<std::size_t>(roi->y1 - roi->y0) * (roi->x1 - roi->x0) : static_cast<std::size_t>(size) * size;
    if (debug) debug->assign(static_cast<std::size_t>(h_hi - h_lo), CanvasDebug{});
    I.levels.clear(), I.spread.clear();
    for (int lo = h_lo; lo < h_hi;) {
        const int n = std::min(limit, h_hi - lo);
        I.canvas->begin(n);
        if (I.options.rebin_angles && (I.options.rebin_kernel == "measured" || I.options.rebin_kernel == "exposure")) {
            for (std::size_t k = 0; k < I.tiles.size(); ++k) {
                const auto angles = I.open(k).reconstructor->nearest_view_angles(lo, lo + n);
                for (int y = 0; y < n; ++y)
                    I.canvas->add_view_angles(y, angles.data() + static_cast<std::size_t>(y) * I.n_angles);
                I.close(k);
            }
        }
        std::vector<float> own(static_cast<std::size_t>(n));   // a tile's own slices: one pixel each, left empty
        const SliceRoi one{0, 1, 0, 1};
        for (std::size_t k = 0; k < I.tiles.size(); ++k) {
            TileReconstruction& t = I.open(k);
            I.collector->tile = static_cast<int>(k), I.collector->h_lo = lo;
            t.reconstructor->reconstruct(lo, lo + n, own.data(), nullptr, &one);
            I.close(k);
        }
        for (int y = 0; y < n; ++y) {
            const auto index = static_cast<std::size_t>(lo - h_lo + y);
            I.canvas->slice(y, out + index * plane, roi, debug ? &(*debug)[index] : nullptr);
            I.levels.push_back(I.canvas->levels(y));
            I.spread.push_back(I.canvas->angle_spread(y));
        }
        lo += n;
    }
}

}  // namespace vc_tomography
