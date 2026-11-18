// vc_tomography/mosaic.hpp: the canvas (a mosaic's tiles in one sinogram, one filtered backprojection) and a mosaic
// reconstructed so from its frames.
#include "vc_tomography/mosaic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
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
    explicit Impl(Device d) : G(d), W(d), ws(d), ww(d), gain(d), frac(d), pad(d), filt(d), up(d), img(d), first(d) {}
    CanvasConfig cfg;
    std::vector<CanvasTile> tiles;
    int NG = 0, size = 0, half = 0, n = 0, stride = 0, n_slices = 0;
    Array<float> G, W, ws, ww, gain, frac, pad, filt, up, img;
    Array<std::int32_t> first;
    RampGeometry rg, rg_half;
    std::unique_ptr<FourierFilter> ramp;
    BackprojectParams bp;
    int degree = 0;
    std::vector<int> sub;            // the angles compared for the levels
    struct Kept {                    // of one tile in one slice, until the slice is made
        bool have = false;
        std::vector<float> w;        // (NG, stride): its weights as placed (shifted)
        std::vector<float> sub_s;    // (sub, stride): value x weight as placed, of the angles compared
        std::vector<std::int32_t> first;
        std::vector<float> angles;
    };
    struct Made {
        bool done = false;
        std::vector<double> levels;
        std::vector<float> angles;   // (NG / 2): the angle every folded row is backprojected at
        double spread = 0;
    };
    std::vector<std::vector<Kept>> kept;   // [slice][tile]
    std::vector<Made> made;

    float* slice_of(Array<float>& a, int y) { return a.data() + static_cast<std::size_t>(y) * NG * size; }
    std::vector<double> solve_levels(int y) const;
    void make(int y);
};

MosaicCanvas::MosaicCanvas(CanvasConfig config, std::vector<CanvasTile> tiles) : impl_(std::make_unique<Impl>(config.device)) {
    Impl& I = *impl_;
    if (tiles.empty()) throw Error("mosaic canvas: no tiles");
    if (config.n_angles < 2 || config.n_angles % 2) throw Error("mosaic canvas: the angles of a turn must be an even number");
    if (!(config.pixel_size_m > 0)) throw Error("mosaic canvas: no pixel size");
    if (config.bp_oversample < 1 || config.bp_oversample > 8) throw Error("bp_oversample must be 1..8");
    if (config.anchor < 0 || config.anchor >= static_cast<int>(tiles.size())) throw Error("mosaic canvas: the anchor is not one of the tiles");
    I.n = tiles[0].cols;
    double reach = 0;
    for (const CanvasTile& t : tiles) {
        if (t.cols != I.n || t.cols < 2) throw Error("mosaic canvas: the tiles' rows must have one width");
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
    I.rg_half = ramp_geometry(I.size, I.NG / 2, static_cast<float>(I.half), 1, false);
    I.ramp = make_fourier_filter(I.cfg.device, ramp_filter_spec(I.rg, ramp_scale(I.NG, false), I.cfg.ramp_filter_name, I.cfg.ramp_filter_par));
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

void MosaicCanvas::add(int tile, int y, const float* rows, const float* weights, const float* angles) {
    Impl& I = *impl_;
    if (tile < 0 || tile >= static_cast<int>(I.tiles.size()) || y < 0 || y >= I.n_slices) throw Error("mosaic canvas: no such tile or slice");
    Impl::Kept& K = I.kept[static_cast<std::size_t>(y)][static_cast<std::size_t>(tile)];
    if (K.have) throw Error("mosaic canvas: a tile's rows of a slice are added once");
    if (I.made[static_cast<std::size_t>(y)].done) throw Error("mosaic canvas: the slice is made; nothing more is added to it");
    const CanvasTile& t = I.tiles[static_cast<std::size_t>(tile)];
    const int NG = I.NG, n = I.n, stride = I.stride;
    // where the tile's column 0 falls on the canvas at every row's angle: whole columns, and the fraction its rows are delayed by
    K.first.resize(static_cast<std::size_t>(NG));
    K.angles.assign(angles, angles + NG);
    std::vector<float> frac(static_cast<std::size_t>(NG));
    for (int r = 0; r < NG; ++r) {
        const double a = angles[r], s = I.half - t.rot_center + t.ax * std::cos(a) + t.ay * std::sin(a), f = std::floor(s);
        K.first[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(f);
        frac[static_cast<std::size_t>(r)] = static_cast<float>(s - f);
    }
    I.first.put(K.first.data(), K.first.size());
    I.frac.put(frac.data(), frac.size());
    const std::size_t rown = static_cast<std::size_t>(NG) * stride;
    I.ws.ensure(rown);
    I.ww.ensure(rown);
    float* G = I.slice_of(I.G, y);
    float* W = I.slice_of(I.W, y);
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (I.cfg.device == Device::CUDA) {
        cuda::tile_rows(rows, weights, I.gain.data(), NG, n, stride, I.ws.data(), I.ww.data());
        cuda::shift_rows(I.ws.data(), NG, n, 0.f, nullptr, I.frac.data());
        cuda::shift_rows(I.ww.data(), NG, n, 0.f, nullptr, I.frac.data());
        cuda::place_rows(I.ws.data(), I.first.data(), 1.f, false, NG, n, stride, G, I.size);
        cuda::place_rows(I.ww.data(), I.first.data(), 1.f, true, NG, n, stride, W, I.size);
    } else
#endif
    {
        cpu::tile_rows(rows, weights, I.gain.data(), NG, n, stride, I.ws.data(), I.ww.data());
        cpu::shift_rows(I.ws.data(), NG, n, 0.f, I.frac.data());
        cpu::shift_rows(I.ww.data(), NG, n, 0.f, I.frac.data());
        cpu::place_rows(I.ws.data(), I.first.data(), 1.f, false, NG, n, stride, G, I.size);
        cpu::place_rows(I.ww.data(), I.first.data(), 1.f, true, NG, n, stride, W, I.size);
    }
    // kept on the host until the slice is made: the weights as placed (a level is added through them, and the
    // angles are averaged by them), and of the angles compared the values
    K.w.resize(rown);
    I.ww.get(K.w.data(), rown);
    if (I.cfg.levels && I.tiles.size() > 1) {
        K.sub_s.resize(I.sub.size() * static_cast<std::size_t>(stride));
        for (std::size_t q = 0; q < I.sub.size(); ++q) I.ws.get(K.sub_s.data() + q * stride, static_cast<std::size_t>(stride), static_cast<std::size_t>(I.sub[q]) * stride);
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
            const float *wa = A.w.data() + g * stride, *wb = B.w.data() + g * stride, *sa = A.sub_s.data() + q * stride, *sb = B.sub_s.data() + q * stride;
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

// The slice's sinogram as it is filtered: the tiles' levels added, the angle of every row, the two halves of the
// turn folded, the sums divided, what no tile recorded mended.
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

void MosaicCanvas::slice(int y, float* out, const SliceRoi* roi, CanvasDebug* debug) {
    Impl& I = *impl_;
    if (y < 0 || y >= I.n_slices) throw Error("mosaic canvas: no such slice");
    if (!I.made[static_cast<std::size_t>(y)].done) I.make(y);
    const Impl::Made& M = I.made[static_cast<std::size_t>(y)];
    const int NG = I.NG, H = NG / 2, size = I.size, os = I.cfg.bp_oversample;
    const SliceRoi whole{0, size, 0, size};
    const SliceRoi R = roi ? *roi : whole;
    if (!(0 <= R.x0 && R.x0 < R.x1 && R.x1 <= size && 0 <= R.y0 && R.y0 < R.y1 && R.y1 <= size)) throw Error("mosaic canvas: the region is not inside the canvas");
    const int ny = R.y1 - R.y0, nx = R.x1 - R.x0;
    const std::size_t all = static_cast<std::size_t>(NG) * size, half_rows = static_cast<std::size_t>(H) * size;
    float* Gy = I.slice_of(I.G, y);
    if (debug) {
        debug->sino.resize(all), debug->weights.resize(all);
        I.G.get(debug->sino.data(), all, static_cast<std::size_t>(y) * all);
        I.W.get(debug->weights.data(), all, static_cast<std::size_t>(y) * all);
        debug->angles = M.angles;
    }
    // every row is whole and the turn's second half repeats its first (mirrored): the rows of the whole turn are
    // padded (each from its opposite row, as any sinogram's: sino.hpp), the first half filtered and backprojected,
    // each row standing for itself and for its opposite
    I.pad.ensure(static_cast<std::size_t>(NG) * I.rg.npad);
    I.filt.ensure(half_rows);
    I.up.ensure(half_rows * os);
    I.img.ensure(static_cast<std::size_t>(ny) * nx);
    BackprojectParams bp = I.bp;
    bp.x0 = R.x0, bp.y0 = R.y0;
    const float* rows = nullptr;
#ifdef VC_TOMOGRAPHY_WITH_CUDA
    if (I.cfg.device == Device::CUDA) {
        cuda::ramp_pad(Gy, I.pad.data(), I.rg);
        I.ramp->apply(I.pad.data(), I.pad.data(), H);
        cuda::ramp_crop(I.pad.data(), I.filt.data(), I.rg_half);
        rows = I.filt.data();
        if (os > 1) {
            cuda::upsample_rows(I.filt.data(), I.up.data(), H, size, os);
            rows = I.up.data();
        }
        if (I.degree) {
            cuda::bspline_prefilter(rows, I.up.data(), H, size * os, I.degree);
            rows = I.up.data();
        }
        cuda::backproject(rows, M.angles.data(), I.img.data(), H, size * os, ny, nx, bp);
    } else
#endif
    {
        cpu::ramp_pad(Gy, I.pad.data(), I.rg);
        I.ramp->apply(I.pad.data(), I.pad.data(), H);
        cpu::ramp_crop(I.pad.data(), I.filt.data(), I.rg_half);
        rows = I.filt.data();
        if (os > 1) {
            cpu::upsample_rows(I.filt.data(), I.up.data(), H, size, os);
            rows = I.up.data();
        }
        if (I.degree) {
            cpu::bspline_prefilter(rows, I.up.data(), H, size * os, I.degree);
            rows = I.up.data();
        }
        cpu::backproject(rows, M.angles.data(), I.img.data(), H, size * os, ny, nx, bp);
    }
    if (debug) {
        debug->filtered.resize(half_rows);
        I.filt.get(debug->filtered.data(), half_rows);
    }
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
        on_canvas.push_back(CanvasTile{p.rot_center, p.ax, p.ay, I.cols});
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
    c.anchor = I.anchor;
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
    const int n = h_hi - h_lo, size = I.canvas->size();
    I.canvas->begin(n);
    std::vector<float> own(static_cast<std::size_t>(n));   // a tile's own slices: one pixel each, left empty
    const SliceRoi one{0, 1, 0, 1};
    for (std::size_t k = 0; k < I.tiles.size(); ++k) {
        TileReconstruction& t = I.open(k);
        I.collector->tile = static_cast<int>(k), I.collector->h_lo = h_lo;
        t.reconstructor->reconstruct(h_lo, h_hi, own.data(), nullptr, &one);
        I.close(k);
    }
    const std::size_t plane = roi ? static_cast<std::size_t>(roi->y1 - roi->y0) * (roi->x1 - roi->x0) : static_cast<std::size_t>(size) * size;
    if (debug) debug->assign(static_cast<std::size_t>(n), CanvasDebug{});
    I.levels.clear(), I.spread.clear();
    for (int y = 0; y < n; ++y) {
        I.canvas->slice(y, out + static_cast<std::size_t>(y) * plane, roi, debug ? &(*debug)[static_cast<std::size_t>(y)] : nullptr);
        I.levels.push_back(I.canvas->levels(y));
        I.spread.push_back(I.canvas->angle_spread(y));
    }
}

}  // namespace vc_tomography
