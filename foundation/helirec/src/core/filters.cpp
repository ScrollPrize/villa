#include "helirec/filters.hpp"

#include <fftw3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

#include "../io/json.hpp"

namespace helirec {

namespace {

std::mutex& planner_mutex() { return fftw_planner_mutex(); }

using cplx = std::complex<double>;

// in-place 2D complex DFT (double), sign = FFTW_FORWARD / FFTW_BACKWARD (unnormalised)
void dft2(std::vector<cplx>& a, int n0, int n1, int sign) {
    fftw_plan plan;
    {
        std::lock_guard<std::mutex> lock(planner_mutex());
        plan = fftw_plan_dft_2d(n0, n1, reinterpret_cast<fftw_complex*>(a.data()),
                                reinterpret_cast<fftw_complex*>(a.data()), sign, FFTW_ESTIMATE);
    }
    fftw_execute(plan);
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_destroy_plan(plan);
}

std::vector<cplx> dft1(const std::vector<double>& g) {
    const int n = static_cast<int>(g.size());
    std::vector<cplx> a(g.begin(), g.end());
    fftw_plan plan;
    {
        std::lock_guard<std::mutex> lock(planner_mutex());
        plan = fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex*>(a.data()), reinterpret_cast<fftw_complex*>(a.data()),
                                FFTW_FORWARD, FFTW_ESTIMATE);
    }
    fftw_execute(plan);
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_destroy_plan(plan);
    return a;
}

int signed_index(int i, int n) { return i <= n / 2 ? i : i - n; }

// nabuxx periodic gaussian incl. in-place symmetrisation (paganin_filter.cpp:172-201)
std::vector<double> periodic_gaussian(int n, double sigma) {
    std::vector<double> g(n);
    for (int i = 0; i < n; ++i)
        g[i] = static_cast<double>(static_cast<float>(std::exp(-std::min(double(i) * i / (2 * sigma * sigma), 100.0))));
    double s = g[0];
    for (int k = 1; k < n; ++k) {
        g[k] += g[n - k];
        s += g[k];
    }
    for (auto& v : g) v /= s;
    return g;
}

bool is_log_method(const std::string& m) { return m == "log" || m == "laplacian"; }

void apply_unsharp(std::vector<cplx>& K, int n0, int n1, double sigma, double coeff, bool log_method) {
    const auto G0 = dft1(periodic_gaussian(n0, sigma));
    const auto G1 = dft1(periodic_gaussian(n1, sigma));
    for (int i = 0; i < n0; ++i) {
        const double f0 = 2 * M_PI * signed_index(i, n0) / n0;
        for (int j = 0; j < n1; ++j) {
            const double f1 = 2 * M_PI * signed_index(j, n1) / n1;
            const cplx g = G0[i] * G1[j];
            const cplx u = log_method ? 1.0 + coeff * ((2 - 2 * std::cos(f0)) + (2 - 2 * std::cos(f1))) * g
                                      : (1.0 + coeff) - coeff * g;
            K[static_cast<std::size_t>(i) * n1 + j] *= u;
        }
    }
}

std::vector<int> smooth_numbers(long limit, bool with_11_13, int min_pow2) {
    std::vector<int> out;
    for (long a = 1; a <= limit; a *= 2)
        for (long b = a; b <= limit; b *= 3)
            for (long c = b; c <= limit; c *= 5)
                for (long d = c; d <= limit; d *= 7) {
                    if (a < (1L << min_pow2)) continue;
                    out.push_back(static_cast<int>(d));
                    if (with_11_13) {
                        if (d * 11 <= limit) out.push_back(static_cast<int>(d * 11));
                        if (d * 13 <= limit) out.push_back(static_cast<int>(d * 13));
                    }
                }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

int roundfft(int n) {
    const auto v = smooth_numbers(std::max<long>(64, 40L * n), true, 0);
    for (int x : v)
        if (x > n) return x;
    throw Error("roundfft: no candidate");
}

int next_smooth_power(int n) {
    const auto v = smooth_numbers(std::max<long>(64, 4L * n), false, 0);
    for (int x : v)
        if (x >= n) return x;
    throw Error("next_smooth_power: no candidate");
}

// ================================================================== parameters
double FilterParams::num(const std::string& k) const {
    auto it = values.find(k);
    if (it == values.end()) throw Error("filter parameter '" + k + "' missing");
    try {
        return std::stod(it->second);
    } catch (...) {
        throw Error("filter parameter '" + k + "': not a number ('" + it->second + "')");
    }
}
double FilterParams::num(const std::string& k, double def) const { return has(k) ? num(k) : def; }
std::string FilterParams::str(const std::string& k, const std::string& def) const {
    auto it = values.find(k);
    return it == values.end() ? def : it->second;
}

void ProjectionFilter::multiply_response(std::vector<std::complex<double>>&, int, int, int) const {
    throw Error("filter '" + type + "' has no frequency response");
}
void ProjectionFilter::apply(float*, int, int, int, Device) { throw Error("filter '" + type + "' cannot be applied"); }

namespace {

void check_keys(const std::string& type, const FilterParams& p, std::set<std::string> allowed) {
    allowed.insert("stage");
    for (const auto& [k, v] : p.values)
        if (!allowed.count(k)) {
            std::string list;
            for (const auto& a : allowed) list += (list.empty() ? "" : ", ") + a;
            throw Error("filter '" + type + "': unknown parameter '" + k + "' (known: " + list + ")");
        }
}

FilterStage stage_of(const FilterParams& p) {
    const std::string s = p.str("stage", "intensity");
    if (s == "intensity") return FilterStage::Intensity;
    if (s == "log") return FilterStage::Log;
    throw Error("filter stage must be 'intensity' or 'log', not '" + s + "'");
}

// The padded FFT filter of a fused Fourier step for stacks (H, W) with margin m: rows padded to roundfft(H),
// columns to roundfft(W + 2 m), symmetric; coefficients = the product of the members' responses.
FourierFilterSpec fourier_step_spec(const std::vector<ProjectionFilter*>& members, int H, int W, int m) {
    if (H <= 2 * m) throw Error("filter: stack height must exceed 2 x margin");
    const int N0 = roundfft(H), N1 = roundfft(W + 2 * m);
    const int pos0 = -((N0 - H) / 2);
    const int pos1 = -m - (N1 - (W + 2 * m)) / 2;
    std::vector<cplx> K(static_cast<std::size_t>(N0) * N1, cplx(1.0, 0.0));
    for (const auto* f : members) f->multiply_response(K, N0, N1, m);
    FourierFilterSpec sp;
    sp.ndim = 2;
    sp.h = H;
    sp.w = W;
    sp.ph = N0;
    sp.pw = N1;
    sp.pad_top = -pos0;
    sp.pad_left = -pos1;
    sp.mode = PadMode::Symmetric;
    static std::atomic<std::uint64_t> next_id{1};
    sp.id = next_id++;
    const int nc = N1 / 2 + 1;
    sp.filter.resize(static_cast<std::size_t>(N0) * nc);
    sp.filter_imag.resize(sp.filter.size());
    for (int i = 0; i < N0; ++i)
        for (int j = 0; j < nc; ++j) {
            const cplx v = K[static_cast<std::size_t>(i) * N1 + j];
            sp.filter[static_cast<std::size_t>(i) * nc + j] = static_cast<float>(v.real());
            sp.filter_imag[static_cast<std::size_t>(i) * nc + j] = static_cast<float>(v.imag());
        }
    return sp;
}

// A response given on a fixed reference grid (Nr rows x N1 columns, so that it does not depend on the chunk height) as
// the spectrum of its real-space kernel truncated to rows [-m, m], normalised to unit sum (unit DC) and embedded in
// the step's (N0, N1) grid. `P` is consumed.
std::vector<cplx> fixed_kernel_spectrum(std::vector<cplx> P, int Nr, int N0, int N1, int m) {
    dft2(P, Nr, N1, FFTW_BACKWARD);
    std::vector<cplx> Q(static_cast<std::size_t>(N0) * N1, cplx(0));
    double sum = 0;
    for (int dy = -m; dy <= m; ++dy) {
        const std::size_t src = static_cast<std::size_t>((dy + Nr) % Nr) * N1, dst = static_cast<std::size_t>((dy + N0) % N0) * N1;
        for (int j = 0; j < N1; ++j) {
            Q[dst + j] = P[src + j].real();
            sum += P[src + j].real();
        }
    }
    for (auto& q : Q) q /= sum;
    dft2(Q, N0, N1, FFTW_FORWARD);
    return Q;
}

// ------------------------------------------------------------------ paganin
// nabuxx paganin_filter.cpp (pag_version 2): H = 1 / (1 + (2 - cos f0 - cos f1) pfact), pfact = 2 (l/px / 2 pi)^2,
// l = sqrt(lambda D delta/beta pi); real-space kernel truncated vertically to dy in [-m, m-1] (m = step margin)
class Paganin final : public ProjectionFilter {
  public:
    Paganin(const FilterParams& p, const FilterContext& c) : ctx_(c), db_(p.num("delta_beta")) {
        if (db_ < 0) throw Error("paganin: delta_beta must be >= 0");
    }
    bool fourier() const override { return true; }
    int own_margin() const {
        if (margin_ < 0) margin_ = margin_rows();
        return margin_;
    }
    float l_over_px() const {
        const float lam = 1.239842e-3f / static_cast<float>(ctx_.energy_kev);
        const float l = std::sqrt(lam * static_cast<float>(ctx_.distance_m * 1.0e6) * static_cast<float>(db_) *
                                  static_cast<float>(M_PI));
        return l / static_cast<float>(ctx_.pixel_size_m * 1.0e6);
    }
    // H = 1 / (1 + pfact (2 - cos f0 - cos f1)) = 1 / (1 - (pfact / 2) lap), lap: the 5-point Laplacian
    double pfact() const { return 2.0 * std::pow(static_cast<double>(l_over_px()) / (2 * M_PI), 2); }
    // nabu PaganinPhaseRetrieval(shape, margin=None): padded = next_power(2n), float64 filter,
    // conv kernel = irfft2(filter); margin = first row with |k[r,0]| < k[0,0]/1000
    int margin_rows() const override {
        if (db_ <= 0) return 0;
        const int det_rows = ctx_.det_rows, det_cols = ctx_.det_cols;
        const int nyp = next_smooth_power(2 * det_rows), nxp = next_smooth_power(2 * det_cols);
        const double px = ctx_.pixel_size_m * 1e6, D = ctx_.distance_m * 1e6, L = 1.23984199e-3 / ctx_.energy_kev;
        const int ncx = nxp / 2 + 1;
        std::vector<cplx> F(static_cast<std::size_t>(nyp) * ncx);
        for (int i = 0; i < nyp; ++i) {
            const double fy = signed_index(i, nyp) / (nyp * px);
            for (int j = 0; j < ncx; ++j) {
                const double fx = j / (nxp * px);
                F[static_cast<std::size_t>(i) * ncx + j] = 1.0 / (1 + db_ * L * D * M_PI * (fy * fy + fx * fx));
            }
        }
        std::vector<double> k(static_cast<std::size_t>(nyp) * nxp);
        fftw_plan plan;
        {
            std::lock_guard<std::mutex> lock(planner_mutex());
            plan = fftw_plan_dft_c2r_2d(nyp, nxp, reinterpret_cast<fftw_complex*>(F.data()), k.data(), FFTW_ESTIMATE);
        }
        fftw_execute(plan);
        {
            std::lock_guard<std::mutex> lock(planner_mutex());
            fftw_destroy_plan(plan);
        }
        const double vmax = k[0];
        for (int r = 0; r < nyp; ++r)
            if (std::abs(k[static_cast<std::size_t>(r) * nxp]) < vmax / 1e3) return r == 0 ? det_rows : r;
        return det_rows;
    }
    // the untruncated discrete-Laplacian response on an n0 x n1 grid
    std::vector<cplx> response(int n0, int n1) const {
        const double pfact = db_ > 0 ? this->pfact() : 0.0;
        std::vector<cplx> P(static_cast<std::size_t>(n0) * n1);
        for (int i = 0; i < n0; ++i) {
            const double c0 = std::cos(2 * M_PI * signed_index(i, n0) / n0);
            for (int j = 0; j < n1; ++j) {
                const double c1 = std::cos(2 * M_PI * signed_index(j, n1) / n1);
                P[static_cast<std::size_t>(i) * n1 + j] = 1.0 / (1.0 + (2 - c0 - c1) * pfact);
            }
        }
        return P;
    }
    void multiply_response(std::vector<cplx>& K, int N0, int N1, int margin) const override {
        if (db_ <= 0) return;
        if (!ctx_.legacy) {
            // the real-space kernel on a fixed reference height (independent of the chunk), rows [-m, m], unit sum,
            // embedded in the step's (N0, N1) grid
            const int m = own_margin(), Nr = next_smooth_power(std::max(64, 16 * m));
            if (N0 < 2 * m + 1) throw Error("paganin: padded stack height below the kernel support 2 x margin + 1");
            const auto Q = fixed_kernel_spectrum(response(Nr, N1), Nr, N0, N1, m);
            for (std::size_t k = 0; k < K.size(); ++k) K[k] *= Q[k];
            return;
        }
        // legacy (nabuxx pag_version 2): truncated on the chunk's own padded grid, DC not renormalised
        auto P = response(N0, N1);
        dft2(P, N0, N1, FFTW_BACKWARD);  // -> real-space kernel (x N0*N1)
        if (margin > 0)
            for (int i = margin; i < N0 - margin; ++i)
                std::fill(P.begin() + static_cast<std::ptrdiff_t>(i) * N1, P.begin() + static_cast<std::ptrdiff_t>(i + 1) * N1, cplx(0));
        dft2(P, N0, N1, FFTW_FORWARD);
        const double norm = 1.0 / (static_cast<double>(N0) * N1);
        for (std::size_t k = 0; k < K.size(); ++k) K[k] *= P[k] * norm;
    }

  private:
    FilterContext ctx_;
    double db_;
    mutable int margin_ = -1;
};

// ------------------------------------------------------------------ eikonal phase retrieval
// Local second-order closure (src/common/epr_core.hpp): the Paganin result, then `iters` iterations
// I += relax Paganin[measured - F(I)]. The Paganin filter is the fixed (unit-DC) kernel in every mode.
class Epr final : public ProjectionFilter {
  public:
    Epr(const FilterParams& p, const FilterContext& c)
        : db_(p.num("delta_beta")), iters_(static_cast<int>(p.num("iters", 3))), relax_(p.num("relax", 0.75)),
          max_shift_(p.num("max_shift", 12.5)), pag_(paganin_params(p), fixed_kernel(c)) {
        if (db_ <= 0) throw Error("epr: delta_beta must be > 0");
        if (iters_ < 0 || iters_ > 100) throw Error("epr: iters must be 0..100");
        if (!(relax_ > 0 && relax_ <= 1)) throw Error("epr: relax must be in (0, 1]");
        if (!(c.energy_kev > 0 && c.distance_m > 0 && c.pixel_size_m > 0))
            throw Error("epr: needs the energy, the propagation distance and the pixel size");
    }
    int margin_rows() const override {
        const int m = pag_.own_margin();
        return iters_ > 0 ? 2 * m + 4 : m;   // the estimate's support + one more for the correction, + the stencils
    }
    void apply(float* frames, int n, int H, int W, Device device) override {
        std::lock_guard<std::mutex> lock(mutex_);
        EprParams ep;
        ep.shift_px2 = 0.5 * pag_.pfact();
        ep.delta_beta = db_;
        ep.iters = iters_;
        ep.relax = static_cast<float>(relax_);
        ep.max_shift = static_cast<float>(max_shift_);
        FourierFilter& f = paganin_for(H, W, device);
        if (device == Device::CPU) { cpu::epr_frames(frames, n, H, W, ep, f); return; }
#ifdef HELIREC_WITH_CUDA
        cuda::epr_frames(frames, n, H, W, ep, f);
#else
        throw Error("built without CUDA");
#endif
    }

  private:
    static FilterParams paganin_params(const FilterParams& p) {
        FilterParams q;
        q.values["delta_beta"] = p.str("delta_beta", "");
        return q;
    }
    static FilterContext fixed_kernel(FilterContext c) {
        c.legacy = false;
        return c;
    }
    FourierFilter& paganin_for(int H, int W, Device device) {
        for (auto& e : filters_)
            if (e.H == H && e.W == W && e.device == device) return *e.f;
        if (filters_.size() >= 4) filters_.erase(filters_.begin());  // e.g. full chunks + the last one
        const FourierFilterSpec spec = fourier_step_spec({&pag_}, H, W, pag_.own_margin());
        Entry e{H, W, device, nullptr};
#ifdef HELIREC_WITH_CUDA
        if (device == Device::CUDA) e.f = cuda::make_fourier_filter(spec);
        else
#endif
            e.f = cpu::make_fourier_filter(spec);
        filters_.push_back(std::move(e));
        return *filters_.back().f;
    }
    struct Entry {
        int H, W;
        Device device;
        std::unique_ptr<FourierFilter> f;
    };
    double db_;
    int iters_;
    double relax_, max_shift_;
    Paganin pag_;
    std::vector<Entry> filters_;
    std::mutex mutex_;
};

// ------------------------------------------------------------------ mtf_paganin
// Phase retrieval and detector deblurring as one Wiener filter. The detector records the propagated intensity
// blurred by its point spread: I_det^ = (M / P) I^, with P the Paganin response (1 / P: the phase-contrast gain of a
// homogeneous object) and M the detector MTF. For white detector noise and an object spectrum that falls as P^2
// (~ k^-4 beyond the Paganin corner, the law of sharp interfaces) the Wiener estimate of I is
//     W = P D,   D = (1 + nsr) M / (M^2 + nsr)
// i.e. Paganin times a regularised inverse of the MTF: exact (1 / M) where M^2 >> nsr, at most (1 + nsr) / (2 sqrt nsr)
// (at M = sqrt nsr), falling to zero with M beyond. nsr is the noise-to-signal power ratio at low frequency. The
// factor 1 + nsr makes the DC gain one. Like the Paganin filter, the kernel is fixed: computed on its own grid,
// truncated to [-m, m] rows, unit sum, so the result does not depend on the chunk height.
//   mtf: Gaussian exp(-2 pi^2 sigma^2 nu^2) (sigma in pixels), or a radial table "nu:M/nu:M/..." (nu in cycles per
//   pixel, increasing; linear in between, M(0) = 1, held beyond the last entry).
class MtfPaganin final : public ProjectionFilter {
  public:
    MtfPaganin(const FilterParams& p, const FilterContext& c)
        : pag_(paganin_params(p), fixed_kernel(c)), sigma_(p.num("sigma", 0)), wiener_(p.has("nsr")) {
        if (wiener_) {
            nsr_ = p.num("nsr");
            if (!(nsr_ > 0)) throw Error("mtf_paganin: nsr must be > 0");
            if (p.has("target") || p.has("gain")) throw Error("mtf_paganin: nsr (the Wiener form) excludes target and gain");
        } else {
            target_ = p.num("target", 0.5);
            gain_ = p.num("gain", 12);
            if (!(target_ >= 0)) throw Error("mtf_paganin: target must be >= 0");
            if (!(gain_ > 1)) throw Error("mtf_paganin: gain must be > 1");
        }
        const std::string table = p.str("mtf", "");
        if (table.empty() == !(sigma_ > 0)) throw Error("mtf_paganin: give either sigma > 0 or an mtf table");
        if (!table.empty()) {
            std::istringstream in(table);
            std::string item;
            while (std::getline(in, item, '/')) {
                const auto colon = item.find(':');
                double nu = 0, m = 0;
                try {
                    if (colon == std::string::npos) throw Error("");
                    nu = std::stod(item.substr(0, colon));
                    m = std::stod(item.substr(colon + 1));
                } catch (...) {
                    throw Error("mtf_paganin: mtf table entries are nu:M, separated by '/' ('" + item + "')");
                }
                if (!(nu >= 0) || !(m > 0 && m <= 1) || (!table_.empty() && nu <= table_.back().first))
                    throw Error("mtf_paganin: mtf table needs increasing nu >= 0 and 0 < M <= 1");
                table_.push_back({nu, m});
            }
            if (table_.empty()) throw Error("mtf_paganin: empty mtf table");
            if (table_.front().first > 0) table_.insert(table_.begin(), {0.0, 1.0});
            else if (table_.front().second != 1.0) throw Error("mtf_paganin: M(0) must be 1");
        }
    }
    bool fourier() const override { return true; }
    double mtf(double nu) const {
        if (table_.empty()) return std::exp(-2 * M_PI * M_PI * sigma_ * sigma_ * nu * nu);
        if (nu >= table_.back().first) return table_.back().second;
        auto hi = std::upper_bound(table_.begin(), table_.end(), nu, [](double v, const auto& e) { return v < e.first; });
        const auto lo = hi - 1;
        return lo->second + (hi->second - lo->second) * (nu - lo->first) / (hi->first - lo->first);
    }
    // The deblur gain at radial frequency nu (cycles per pixel), M the detector MTF there; 1 at nu = 0.
    //   target form (default): T sqrt(1 + e^2) / sqrt(M^2 + e^2), T the Gaussian MTF of `target` pixels, e = 1 / gain:
    //     the blur is replaced by the target blur where M >> e, and the gain never exceeds ~gain. The response
    //     M x gain falls monotonically (no pass-band hump), so edges do not ring and the noise stays unstructured.
    //   Wiener form (nsr given): (1 + nsr) M / (M^2 + nsr), peaking at (1 + nsr) / (2 sqrt(nsr)) where M = sqrt(nsr).
    double deblur_gain(double nu, double m) const {
        if (wiener_) return (1 + nsr_) * m / (m * m + nsr_);
        const double e = 1 / gain_, t = std::exp(-2 * M_PI * M_PI * target_ * target_ * nu * nu);
        return t * std::sqrt((1 + e * e) / (m * m + e * e));
    }
    // D on an n0 x n1 grid (radial frequency in cycles per pixel)
    std::vector<cplx> deblur_response(int n0, int n1) const {
        std::vector<cplx> D(static_cast<std::size_t>(n0) * n1);
        for (int i = 0; i < n0; ++i) {
            const double f0 = static_cast<double>(signed_index(i, n0)) / n0;
            for (int j = 0; j < n1; ++j) {
                const double f1 = static_cast<double>(signed_index(j, n1)) / n1;
                const double nu = std::hypot(f0, f1);
                D[static_cast<std::size_t>(i) * n1 + j] = deblur_gain(nu, mtf(nu));
            }
        }
        return D;
    }
    // rows of D's real-space kernel: the last row whose central value reaches 1e-4 of the peak, + 1 (at most 64). The
    // Wiener form does not vanish at Nyquist, so its kernel decays slowly (~ r^-2): this truncation leaves the
    // response within about 3 % (nsr 0.01) along the rows near their Nyquist frequency, exact elsewhere.
    int deblur_margin() const {
        if (dmargin_ >= 0) return dmargin_;
        const int n = 256;
        auto D = deblur_response(n, n);
        dft2(D, n, n, FFTW_BACKWARD);
        const double peak = std::abs(D[0].real());
        int last = 0;
        for (int r = 1; r < n / 2; ++r)
            if (std::abs(D[static_cast<std::size_t>(r) * n].real()) >= peak / 1e4) last = r;
        return dmargin_ = std::min(last + 1, 64);
    }
    int margin_rows() const override { return pag_.own_margin() + deblur_margin(); }
    void multiply_response(std::vector<cplx>& K, int N0, int N1, int) const override {
        const int m = margin_rows(), Nr = next_smooth_power(std::max(64, 16 * m));
        if (N0 < 2 * m + 1) throw Error("mtf_paganin: padded stack height below the kernel support 2 x margin + 1");
        auto F = pag_.response(Nr, N1);
        const auto D = deblur_response(Nr, N1);
        for (std::size_t k = 0; k < F.size(); ++k) F[k] *= D[k];
        const auto Q = fixed_kernel_spectrum(std::move(F), Nr, N0, N1, m);
        for (std::size_t k = 0; k < K.size(); ++k) K[k] *= Q[k];
    }

  private:
    static FilterParams paganin_params(const FilterParams& p) {
        FilterParams q;
        q.values["delta_beta"] = p.str("delta_beta", "");
        return q;
    }
    static FilterContext fixed_kernel(FilterContext c) {
        c.legacy = false;
        return c;
    }
    Paganin pag_;
    double sigma_, nsr_ = 0, target_ = 0, gain_ = 0;
    bool wiener_;
    std::vector<std::pair<double, double>> table_;
    mutable int dmargin_ = -1;
};

// ------------------------------------------------------------------ wiener
// The minimum-error linear estimate of the object from measured tables, in place of paganin + unsharp or mtf_paganin.
// A recorded frame is the object's projection through the system's transfer T (detector blur x phase contrast x
// whatever the gridding does), plus noise; with S the signal-to-noise power ratio of the recording the estimate is
//     W = (1 / T) g,   g = S / (S + weight)           (weight 1: the Wiener filter)
// so the output shows the object through g: all of it where the data carry it, none where they do not. T and S are
// tables over the frequency magnitude nu (cycles per pixel) and, optionally, the frequency's angle theta from the
// row axis (degrees: 0 = along the rows of a frame, the scan's z; 90 = along the columns), read from a JSON file:
//     {"nu": [...], "theta": [...], "transfer": [[per theta] per nu], "snr": [[...]]}     ("theta" absent: radial,
//     "transfer": [per nu]). Linear between entries (give smooth, finely sampled tables: a kink is a long kernel);
//     below the first nu towards T = 1, g = 1 at nu = 0, as nu^2; held beyond the last nu and outside the thetas.
//     Several snr tables may be kept in one file (snr=NAME picks one).
// A filter of the frames that depends on (nu, theta) only is a filter of the volume that is symmetric about z, at
// the same frequencies: the tables can be measured in either (evaluation/wiener_table.py).
// target > 0: g x the MTF of a Gaussian of `target` pixels (a gentler cut-off: less ringing at edges).
// Like paganin's, the kernel is fixed: computed on its own grid, truncated to [-m, m] rows, unit sum.
class Wiener final : public ProjectionFilter {
  public:
    explicit Wiener(const FilterParams& p) : weight_(p.num("weight", 1)), target_(p.num("target", 0)) {
        if (!(weight_ >= 0)) throw Error("wiener: weight must be >= 0");
        if (!(target_ >= 0)) throw Error("wiener: target must be >= 0");
        const std::string path = p.str("table", ""), snr_name = p.str("snr", "snr");
        if (path.empty()) throw Error("wiener: table=FILE (JSON: nu, [theta,] transfer, snr) is required");
        std::ifstream in(path);
        if (!in) throw Error("wiener: cannot read the table '" + path + "'");
        std::stringstream text;
        text << in.rdbuf();
        const Json j = Json::parse(text.str());
        const auto numbers = [&](const Json* a, const std::string& what) {
            if (!a || !a->is(Json::Array)) throw Error("wiener: '" + what + "' (an array) is missing in " + path);
            std::vector<double> v;
            for (const auto& e : a->array) {
                if (!e.is(Json::Number)) throw Error("wiener: '" + what + "' must hold numbers");
                v.push_back(e.number);
            }
            return v;
        };
        nu_ = numbers(j.find("nu"), "nu");
        if (j.find("theta")) theta_ = numbers(j.find("theta"), "theta");
        const std::size_t nn = nu_.size(), nt = theta_.empty() ? 1 : theta_.size();
        if (nn == 0) throw Error("wiener: empty table");
        for (std::size_t i = 0; i < nn; ++i)
            if (!(nu_[i] > 0) || (i && nu_[i] <= nu_[i - 1])) throw Error("wiener: nu must be > 0 and increasing");
        for (std::size_t i = 0; i < theta_.size(); ++i)
            if (!(theta_[i] >= 0 && theta_[i] <= 90) || (i && theta_[i] <= theta_[i - 1]))
                throw Error("wiener: theta must be within 0..90 degrees and increasing");
        const auto grid = [&](const std::string& what) {
            const Json* a = j.find(what);
            if (!a || !a->is(Json::Array) || a->array.size() != nn)
                throw Error("wiener: '" + what + "' needs one entry per nu in " + path);
            std::vector<double> v;
            for (const auto& row : a->array) {
                if (theta_.empty() && row.is(Json::Number)) { v.push_back(row.number); continue; }
                const auto r = numbers(&row, what);
                if (r.size() != nt) throw Error("wiener: '" + what + "' needs one value per theta at every nu");
                v.insert(v.end(), r.begin(), r.end());
            }
            return v;
        };
        transfer_ = grid("transfer");
        const auto snr = grid(snr_name);
        gain_.resize(snr.size());
        for (std::size_t k = 0; k < snr.size(); ++k) {
            if (!(transfer_[k] > 0)) throw Error("wiener: the transfer must be > 0");
            if (!(snr[k] >= 0)) throw Error("wiener: the snr must be >= 0");
            gain_[k] = snr[k] + weight_ > 0 ? snr[k] / (snr[k] + weight_) : 1.0;
        }
    }
    bool fourier() const override { return true; }
    // W at the frequency (f0 along the rows, f1 along the columns), cycles per pixel
    double response_at(double f0, double f1) const {
        const double nu = std::hypot(f0, f1);
        if (nu <= 0) return 1.0;
        const std::size_t nt = theta_.empty() ? 1 : theta_.size();
        // theta: two columns of the tables and their weights
        std::size_t t0 = 0, t1 = 0;
        double wt = 0;
        if (nt > 1) {
            const double th = std::atan2(std::abs(f1), std::abs(f0)) * 180.0 / M_PI;
            if (th >= theta_.back()) t0 = t1 = nt - 1;
            else if (th > theta_.front()) {
                t1 = static_cast<std::size_t>(std::upper_bound(theta_.begin(), theta_.end(), th) - theta_.begin());
                t0 = t1 - 1;
                wt = (th - theta_[t0]) / (theta_[t1] - theta_[t0]);
            }
        }
        const auto at = [&](const std::vector<double>& v, std::size_t i) { return v[i * nt + t0] * (1 - wt) + v[i * nt + t1] * wt; };
        double T, g;
        if (nu >= nu_.back()) {
            T = at(transfer_, nu_.size() - 1);
            g = at(gain_, nu_.size() - 1);
        } else if (nu < nu_.front()) {
            const double w = (nu / nu_.front()) * (nu / nu_.front());   // even in nu: no cusp at 0 (a long kernel)
            T = 1 + (at(transfer_, 0) - 1) * w;
            g = 1 + (at(gain_, 0) - 1) * w;
        } else {
            const std::size_t i1 = static_cast<std::size_t>(std::upper_bound(nu_.begin(), nu_.end(), nu) - nu_.begin());
            const double w = (nu - nu_[i1 - 1]) / (nu_[i1] - nu_[i1 - 1]);
            T = at(transfer_, i1 - 1) * (1 - w) + at(transfer_, i1) * w;
            g = at(gain_, i1 - 1) * (1 - w) + at(gain_, i1) * w;
        }
        if (target_ > 0) g *= std::exp(-2 * M_PI * M_PI * target_ * target_ * nu * nu);
        return g / T;
    }
    std::vector<cplx> response(int n0, int n1) const {
        std::vector<cplx> R(static_cast<std::size_t>(n0) * n1);
        for (int i = 0; i < n0; ++i) {
            const double f0 = static_cast<double>(signed_index(i, n0)) / n0;
            for (int j = 0; j < n1; ++j)
                R[static_cast<std::size_t>(i) * n1 + j] = response_at(f0, static_cast<double>(signed_index(j, n1)) / n1);
        }
        return R;
    }
    // Rows of context. At a frequency f1 along the columns the kernel's profile over the rows is the transform of
    // W(., f1); m is where what is left of every such profile beyond +-m rows (in absolute value; six f1 from 0 to
    // Nyquist) falls below 2 % of the largest profile's sum, at most 64 rows. For a Paganin-like response that is
    // Paganin's own margin; where W does not vanish at the rows' Nyquist frequency the kernel has a slow alternating
    // tail, and the truncation leaves the response within about 2 % of its largest value near that frequency
    // (exact to a fraction of a percent below 0.1 cycles per pixel).
    int margin_rows() const override {
        if (margin_ >= 0) return margin_;
        const int n = 1024, lines = 6;
        std::vector<std::vector<double>> prof(lines, std::vector<double>(n));
        std::vector<double> w(n);
        double largest = 0;
        for (int line = 0; line < lines; ++line) {
            for (int i = 0; i < n; ++i) w[i] = response_at(static_cast<double>(signed_index(i, n)) / n, 0.1 * line);
            const auto k = dft1(w);                   // (W is even in f0: the forward transform is the kernel x n)
            double total = 0;
            for (int r = 0; r < n; ++r) total += prof[line][r] = std::abs(k[r].real());
            largest = std::max(largest, total);
        }
        int margin = 1;
        for (int line = 0; line < lines; ++line) {
            double tail = 0;
            for (int r = 1; r < n; ++r) tail += prof[line][r];
            int m = 0;
            while (m < 64 && tail > 2e-2 * largest) {
                ++m;
                tail -= prof[line][m] + prof[line][n - m];
            }
            margin = std::max(margin, m);
        }
        return margin_ = margin;
    }
    void multiply_response(std::vector<cplx>& K, int N0, int N1, int) const override {
        const int m = margin_rows(), Nr = next_smooth_power(std::max(64, 16 * m));
        if (N0 < 2 * m + 1) throw Error("wiener: padded stack height below the kernel support 2 x margin + 1");
        const auto Q = fixed_kernel_spectrum(response(Nr, N1), Nr, N0, N1, m);
        for (std::size_t k = 0; k < K.size(); ++k) K[k] *= Q[k];
    }

  private:
    double weight_, target_;
    std::vector<double> nu_, theta_, transfer_, gain_;   // tables (nu, theta), row-major
    mutable int margin_ = -1;
};

// ------------------------------------------------------------------ unsharp, gaussian
class Unsharp final : public ProjectionFilter {
  public:
    explicit Unsharp(const FilterParams& p)
        : sigma_(p.num("sigma")), coeff_(p.num("coeff")), method_(p.str("method", "gaussian")) {
        if (sigma_ <= 0) throw Error("unsharp: sigma must be > 0");
        if (method_ != "gaussian" && method_ != "imagej" && method_ != "log" && method_ != "laplacian")
            throw Error("unsharp: method must be gaussian or log");
    }
    bool fourier() const override { return true; }
    int margin_rows() const override { return coeff_ != 0 ? static_cast<int>(std::ceil(2 * 4 * sigma_ + 1)) : 0; }
    void multiply_response(std::vector<cplx>& K, int n0, int n1, int) const override {
        if (coeff_ != 0) apply_unsharp(K, n0, n1, sigma_, coeff_, is_log_method(method_));
    }

  private:
    double sigma_, coeff_;
    std::string method_;
};

class Gaussian final : public ProjectionFilter {
  public:
    explicit Gaussian(const FilterParams& p) : sigma_(p.num("sigma")) {
        if (sigma_ <= 0) throw Error("gaussian: sigma must be > 0");
    }
    bool fourier() const override { return true; }
    int margin_rows() const override { return static_cast<int>(std::ceil(2 * 4 * sigma_ + 1)); }
    void multiply_response(std::vector<cplx>& K, int n0, int n1, int) const override {
        const auto G0 = dft1(periodic_gaussian(n0, sigma_));
        const auto G1 = dft1(periodic_gaussian(n1, sigma_));
        for (int i = 0; i < n0; ++i)
            for (int j = 0; j < n1; ++j) K[static_cast<std::size_t>(i) * n1 + j] *= G0[i] * G1[j];
    }

  private:
    double sigma_;
};

// ------------------------------------------------------------------ median
class Median final : public ProjectionFilter {
  public:
    explicit Median(const FilterParams& p) : r_(static_cast<int>(p.num("radius", 1))) {
        if (r_ < 1 || r_ > 3) throw Error("median: radius must be 1..3");
    }
    int margin_rows() const override { return r_; }
    void apply(float* frames, int n, int H, int W, Device device) override {
        if (device == Device::CPU) { cpu::median_frames(frames, n, H, W, r_); return; }
#ifdef HELIREC_WITH_CUDA
        cuda::median_frames(frames, n, H, W, r_);
#else
        throw Error("built without CUDA");
#endif
    }

  private:
    int r_;
};

std::map<std::string, FilterFactory>& registry() {
    static std::map<std::string, FilterFactory> r = {
        {"paganin", [](const FilterParams& p, const FilterContext& c) -> std::unique_ptr<ProjectionFilter> {
             check_keys("paganin", p, {"delta_beta"});
             return std::make_unique<Paganin>(p, c);
         }},
        {"epr", [](const FilterParams& p, const FilterContext& c) -> std::unique_ptr<ProjectionFilter> {
             check_keys("epr", p, {"delta_beta", "iters", "relax", "max_shift"});
             if (stage_of(p) != FilterStage::Intensity) throw Error("epr: runs at the intensity stage");
             return std::make_unique<Epr>(p, c);
         }},
        {"mtf_paganin", [](const FilterParams& p, const FilterContext& c) -> std::unique_ptr<ProjectionFilter> {
             check_keys("mtf_paganin", p, {"delta_beta", "sigma", "mtf", "nsr", "target", "gain"});
             return std::make_unique<MtfPaganin>(p, c);
         }},
        {"wiener", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
             check_keys("wiener", p, {"table", "snr", "weight", "target"});
             return std::make_unique<Wiener>(p);
         }},
        {"unsharp", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
             check_keys("unsharp", p, {"sigma", "coeff", "method"});
             return std::make_unique<Unsharp>(p);
         }},
        {"gaussian", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
             check_keys("gaussian", p, {"sigma"});
             return std::make_unique<Gaussian>(p);
         }},
        {"median", [](const FilterParams& p, const FilterContext&) -> std::unique_ptr<ProjectionFilter> {
             check_keys("median", p, {"radius"});
             return std::make_unique<Median>(p);
         }},
    };
    return r;
}

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace

void register_filter(const std::string& type, FilterFactory make) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[type] = std::move(make);
}

std::unique_ptr<ProjectionFilter> make_filter(const FilterSpec& spec, const FilterContext& ctx) {
    FilterFactory f;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        auto it = registry().find(spec.type);
        if (it == registry().end()) {
            std::string list;
            for (const auto& [k, v] : registry()) list += (list.empty() ? "" : ", ") + k;
            throw Error("unknown filter type '" + spec.type + "' (known: " + list + ")");
        }
        f = it->second;
    }
    auto flt = f(spec.params, ctx);
    flt->type = spec.type;
    flt->stage = stage_of(spec.params);
    return flt;
}

std::vector<std::string> filter_types() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    std::vector<std::string> out;
    for (const auto& [k, v] : registry()) out.push_back(k);
    return out;
}

// ================================================================== pipeline
FilterPipeline::FilterPipeline(const FilterChain& chain, const FilterContext& ctx) {
    for (const auto& spec : chain) filters_.push_back(make_filter(spec, ctx));
    for (auto& f : filters_) {
        if (f->fourier() && !steps_.empty() && steps_.back().fourier && steps_.back().stage == f->stage) {
            steps_.back().members.push_back(f.get());
            // the fused kernel's support is the sum of its members' (legacy: the reference's max)
            steps_.back().margin = ctx.legacy ? std::max(steps_.back().margin, f->margin_rows())
                                              : steps_.back().margin + f->margin_rows();
        } else {
            steps_.push_back({f->stage, f->fourier(), {f.get()}, f->margin_rows()});
        }
    }
}

FilterPipeline::~FilterPipeline() = default;

int FilterPipeline::margin_rows() const {
    int m = 0;
    for (const auto& s : steps_) m += s.margin;
    return m;
}

bool FilterPipeline::single_fourier() const {
    return steps_.size() == 1 && steps_[0].fourier && steps_[0].stage == FilterStage::Intensity;
}

const FourierFilterSpec& FilterPipeline::step_spec(std::size_t s, int H, int W, int margin) const {
    if (s >= steps_.size() || !steps_[s].fourier) throw Error("FilterPipeline: not a Fourier step");
    const int m = margin >= 0 ? margin : steps_[s].margin;
    const std::vector<int> key{static_cast<int>(s), H, W, m};
    auto it = spec_cache_.find(key);
    if (it != spec_cache_.end()) return it->second;
    return spec_cache_.emplace(key, fourier_step_spec(steps_[s].members, H, W, m)).first->second;
}

std::string FilterPipeline::describe() const {
    std::ostringstream o;
    for (std::size_t s = 0; s < steps_.size(); ++s) {
        o << (s ? " -> " : "") << (steps_[s].fourier ? "fft[" : "[");
        for (std::size_t k = 0; k < steps_[s].members.size(); ++k) o << (k ? "*" : "") << steps_[s].members[k]->type;
        o << "]@" << (steps_[s].stage == FilterStage::Intensity ? "intensity" : "log") << " m" << steps_[s].margin;
    }
    return o.str();
}

}  // namespace helirec
