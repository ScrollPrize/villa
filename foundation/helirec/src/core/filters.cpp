#include "helirec/filters.hpp"

#include <fftw3.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>

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
    void multiply_response(std::vector<cplx>& K, int N0, int N1, int margin) const override {
        if (db_ <= 0) return;
        const double pfact = 2.0 * std::pow(static_cast<double>(l_over_px()) / (2 * M_PI), 2);
        auto response = [&](int n0, int n1) {   // the untruncated discrete-Laplacian response on an n0 x n1 grid
            std::vector<cplx> P(static_cast<std::size_t>(n0) * n1);
            for (int i = 0; i < n0; ++i) {
                const double c0 = std::cos(2 * M_PI * signed_index(i, n0) / n0);
                for (int j = 0; j < n1; ++j) {
                    const double c1 = std::cos(2 * M_PI * signed_index(j, n1) / n1);
                    P[static_cast<std::size_t>(i) * n1 + j] = 1.0 / (1.0 + (2 - c0 - c1) * pfact);
                }
            }
            return P;
        };
        if (!ctx_.legacy) {
            // the real-space kernel on a fixed reference height (independent of the chunk), rows [-m, m], unit sum,
            // embedded in the step's (N0, N1) grid
            const int m = own_margin(), Nr = next_smooth_power(std::max(64, 16 * m));
            if (N0 < 2 * m + 1) throw Error("paganin: padded stack height below the kernel support 2 x margin + 1");
            auto P = response(Nr, N1);
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
    if (H <= 2 * m) throw Error("filter: stack height must exceed 2 x margin");
    const int N0 = roundfft(H), N1 = roundfft(W + 2 * m);
    const int pos0 = -((N0 - H) / 2);
    const int pos1 = -m - (N1 - (W + 2 * m)) / 2;
    std::vector<cplx> K(static_cast<std::size_t>(N0) * N1, cplx(1.0, 0.0));
    for (const auto* f : steps_[s].members) f->multiply_response(K, N0, N1, m);
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
    return spec_cache_.emplace(key, std::move(sp)).first->second;
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
