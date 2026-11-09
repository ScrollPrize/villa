// CPU backend of the slice methods (src/common/tomo_algorithms.hpp) + gridrec.
#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "common/gridrec_tables.hpp"
#include "common/tomo_algorithms.hpp"
#include "common/tomo_core.hpp"
#include "helirec/tomo.hpp"

namespace helirec {

namespace tomo_gridrec {
// prolate spheroidal wave function: Legendre coefficients
const float kCoefs[11] = {0.5767616E+02f, -0.8931343E+02f, 0.4167596E+02f, -0.1053599E+02f, 0.1662374E+01f,
                          -0.1780527E-00f, 0.1372983E-01f, -0.7963169E-03f, 0.3593372E-04f, -0.1295941E-05f,
                          0.3817796E-07f};

// next power of 2 >= n_det (from 16), also large enough for the N x N image around the FFT origin: the
// de-apodisation table winv covers offsets -linv .. linv (linv = pdim / 2 - 1) and the image spans offsets
// -(N / 2) .. (N - 1) / 2 from its origin, so pdim >= N + 2 (even N) / N + 1 (odd N).
int pdim_for(int n_det, int N) {
    int p = 16;
    while (p < std::max(n_det, N + 2 - (N & 1))) p *= 2;
    return p;
}

// even N: the pipeline's centre (N - 1) / 2 is half a pixel before the integer FFT origin N / 2, so the image is
// sampled at offsets + 1/2: each gridded frequency sample (j cos, j sin) / pdim gets the shift's phase
float grid_shift(int N) { return (N & 1) ? 0.f : 0.5f; }

// wtbl (ltbl + 1): PSWF kernel table; winv (pdim - 1): its de-apodisation
void pswf_tables(int pdim, std::vector<float>& wtbl, std::vector<float>& winv) {
    const int linv = pdim / 2 - 1;
    wtbl.assign(kLtbl + 1, 0.f);
    winv.assign(2 * linv + 1, 0.f);
    const float fac = static_cast<float>(kLtbl) / (linv + 0.5f);
    const float polyz = tomo_core::legendre(kNt, kCoefs, 0.f);
    wtbl[0] = 1.f;
    for (int i = 1; i <= kLtbl; ++i) wtbl[i] = tomo_core::legendre(kNt, kCoefs, static_cast<float>(i) / kLtbl) / polyz;
    float norm = std::sqrt(static_cast<float>(M_PI) / 2 / kC / kLambda) / 1.2f;
    winv[linv] = norm / wtbl[0];
    for (int i = 1; i <= linv; ++i) {
        norm = -norm;
        winv[linv + i] = winv[linv - i] = norm / wtbl[static_cast<int>(std::round(i * fac))];
    }
}

// ramp-filter windows; x = j / pdim
float filter_value(const std::string& name, float x, int j, const std::vector<float>& p) {
    const float pi = static_cast<float>(M_PI), ax = std::fabs(2 * x);
    const float p0 = p.empty() ? 0.5f : p[0], p1 = p.size() > 1 ? p[1] : 8.f;
    if (name == "none") return 1.f;
    if (name == "shepp") return j == 0 ? 0.f : ax * (std::sin(pi * x) / (pi * x));
    if (name == "cosine") return ax * std::cos(pi * x);
    if (name == "hann") return ax * 0.5f * (1.f + std::cos(2 * pi * x / p0));
    if (name == "hamming") return ax * (0.54f + 0.46f * std::cos(2 * pi * x / p0));
    if (name == "ramlak") return ax;
    if (name == "parzen") return ax * std::pow(1.f - std::fabs(x) / p0, 3.f);
    if (name == "butterworth") return ax / (1.f + std::pow(x / p0, 2 * p1));
    throw Error("gridrec: unknown filter '" + name + "' (none, shepp, cosine, hann, hamming, ramlak, parzen, butterworth)");
}

// filter x phase factors, pdim / 2 entries; `scale` multiplies the normalisation pi / pd / dt
std::vector<std::complex<float>> filter_phase(int pdim, int n_angles, float center, const TomoArgs& a, float scale) {
    const int pd2 = pdim / 2;
    const float norm = static_cast<float>(M_PI) / pdim / n_angles * scale, rtmp1 = 2 * static_cast<float>(M_PI) * center / pdim;
    std::vector<std::complex<float>> A(pd2);
    for (int j = 0; j < pd2; ++j) {
        const float f = filter_value(a.filter_name, static_cast<float>(j) / pdim, j, a.filter_par);
        const float x = j * rtmp1;
        A[j] = f * std::complex<float>(std::cos(x), -std::sin(x)) * norm;
    }
    return A;
}

// angular span / pi: the normalisation assumes dt angles over pi; 360 deg (weighted, half-acquisition) data -> 2
float span_factor(const float* angles, int n) {
    if (n < 2) return 1.f;
    return std::fabs(angles[1] - angles[0]) * n / static_cast<float>(M_PI);
}
}  // namespace tomo_gridrec

namespace cpu {

namespace {
std::mutex& fftw_mutex() { return fftw_planner_mutex(); }

// gridrec for one slice of (weighted) rows; out N x N per-pixel image, row-major [iy][ix]
void gridrec_slice(const float* data, int NG, int ND, const float* angles, float center, int N, const TomoArgs& a,
                   float* out) {
    using namespace tomo_gridrec;
    using cf = std::complex<float>;
    const int pdim = pdim_for(ND, N), pd2 = pdim / 2, M2 = pd2;
    std::vector<float> wtbl, winv;
    pswf_tables(pdim, wtbl, winv);
    const auto filphase = filter_phase(pdim, NG, center, a, span_factor(angles, NG));
    const float sh = grid_shift(N), w0 = -2.f * static_cast<float>(M_PI) * sh / pdim;
    const float L2 = static_cast<float>(static_cast<int>(kC / M_PI));
    const int L = static_cast<int>(2 * kC / M_PI);
    const float tblspcg = 2.f * kLtbl / L;
    std::vector<cf> H(static_cast<std::size_t>(pdim) * pdim, cf(0));
    std::vector<cf> sino(pdim);
    fftwf_plan rev, fwd2;
    {
        std::lock_guard<std::mutex> lock(fftw_mutex());
        rev = fftwf_plan_dft_1d(pdim, reinterpret_cast<fftwf_complex*>(sino.data()), reinterpret_cast<fftwf_complex*>(sino.data()),
                                FFTW_BACKWARD, FFTW_ESTIMATE);
        fwd2 = fftwf_plan_dft_2d(pdim, pdim, reinterpret_cast<fftwf_complex*>(H.data()), reinterpret_cast<fftwf_complex*>(H.data()),
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    }
    std::vector<float> wu(8), wv(8);
    for (int p = 0; p < NG; ++p) {
        const float sp = std::sin(angles[p]), cp = std::cos(angles[p]);
        for (int j = 0; j < pdim; ++j) sino[j] = j < ND ? cf(data[static_cast<std::size_t>(p) * ND + j], 0.f) : cf(0);
        fftwf_execute(rev);
        const float wp = w0 * (cp + sp);
        for (int j = 1; j < pd2; ++j) {
            const cf ph = sh != 0.f ? cf(std::cos(wp * j), std::sin(wp * j)) : cf(1.f, 0.f);
            const cf c1 = filphase[j] * sino[j] * ph, c2 = std::conj(filphase[j]) * sino[pdim - j] * std::conj(ph);
            const float U = j * cp + M2, V = j * sp + M2;
            int iul = static_cast<int>(std::ceil(U - L2)), iuh = static_cast<int>(std::floor(U + L2));
            int ivl = static_cast<int>(std::ceil(V - L2)), ivh = static_cast<int>(std::floor(V + L2));
            iul = std::max(iul, 1); iuh = std::min(iuh, pdim - 1);
            ivl = std::max(ivl, 1); ivh = std::min(ivh, pdim - 1);
            for (int iv = ivl, k = 0; iv <= ivh; ++iv, ++k) wv[k] = wtbl[static_cast<int>(std::round(std::fabs(V - iv) * tblspcg))];
            for (int iu = iul, k = 0; iu <= iuh; ++iu, ++k) wu[k] = wtbl[static_cast<int>(std::round(std::fabs(U - iu) * tblspcg))];
            for (int iu = iul, k2 = 0; iu <= iuh; ++iu, ++k2)
                for (int iv = ivl, k = 0; iv <= ivh; ++iv, ++k) {
                    const float w = wu[k2] * wv[k];
                    H[static_cast<std::size_t>(iu) * pdim + iv] += w * c1;
                    H[static_cast<std::size_t>(pdim - iu) * pdim + (pdim - iv)] += w * c2;
                }
        }
    }
    fftwf_execute(fwd2);
    {
        std::lock_guard<std::mutex> lock(fftw_mutex());
        fftwf_destroy_plan(rev);
        fftwf_destroy_plan(fwd2);
    }
    // The image origin is the pipeline's centre (N - 1) / 2: pixel
    // o = (N - 1) / 2 for odd N; for even N the FFT origin is o = N / 2 and the half-pixel offset was applied as a
    // phase above (grid_shift). The de-apodisation table is centred on o, and H's U axis (cos) runs along x, V (sin)
    // along y (checked against the projector, tests/test_tomo.py).
    const int o = N / 2, linv = pd2 - 1;
#pragma omp parallel for schedule(static)
    for (int iy = 0; iy < N; ++iy)
        for (int ix = 0; ix < N; ++ix) {
            const int iu = ((ix - o) % pdim + pdim) % pdim, iv = ((iy - o) % pdim + pdim) % pdim;
            out[static_cast<std::size_t>(iy) * N + ix] =
                winv[linv + ix - o] * winv[linv + iy - o] * H[static_cast<std::size_t>(iu) * pdim + iv].real();
        }
}

struct CpuBackend final : TomoWorkspace {
    tomo_core::Geom g, gfull;
    TomoGeom key;
    int n_angles = 0;
    std::vector<float> angles;
    std::vector<tomo_core::AngleC> ak;
    std::map<int, std::vector<float>> bufs;
    std::vector<std::vector<int32_t>> keep;
    std::vector<float> l2, xT;
    int have_l2 = -1;    // projector the cached chord sums belong to
    int pj = tomo_core::Chord;
    int norm = 1;        // SIRT ray norm: 1 |a_i|^2, 2 row sum

    CpuBackend(const TomoGeom& t, const float* ang) : key(t), n_angles(t.n_angles), angles(ang, ang + t.n_angles) {
        gfull.N = t.N; gfull.n_det = t.n_det; gfull.c = 0.5f * (t.N - 1); gfull.rot = t.rot; gfull.R = t.R;
        gfull = tomo_core::box_geom(gfull, 0, t.N, 0, t.N);
        g = gfull;
        ak.resize(n_angles);
        for (int i = 0; i < n_angles; ++i) ak[i] = tomo_core::angle_constants(std::cos(ang[i]), std::sin(ang[i]));
    }
    bool matches(const TomoGeom& t, const float* ang) const {
        return t.n_angles == key.n_angles && t.n_det == key.n_det && t.N == key.N && t.rot == key.rot && t.R == key.R &&
               std::equal(angles.begin(), angles.end(), ang);
    }
    float* buf(int slot, std::size_t n) {
        auto& v = bufs[slot];
        if (v.size() < n) v.resize(n);
        return v.data();
    }
    const int32_t* ints(const std::vector<int32_t>& v) {
        keep.push_back(v);
        return keep.back().data();
    }
    static void transpose(const float* x, float* xT, int W, int H) {
#pragma omp parallel for schedule(static)
        for (int ix = 0; ix < W; ++ix)
            for (int iy = 0; iy < H; ++iy) xT[static_cast<std::size_t>(ix) * H + iy] = x[static_cast<std::size_t>(iy) * W + ix];
    }
    template <class E>
    void project(const float* x, const int32_t* sel, int n_sel, const E& epi) {
        const int ND = g.n_det;
        const float* xt = nullptr;
        if (n_sel >= 16) {
            xT.resize(static_cast<std::size_t>(g.W) * g.H);
            transpose(x, xT.data(), g.W, g.H);
            xt = xT.data();
        }
#pragma omp parallel for schedule(dynamic, 64)
        for (long q = 0; q < static_cast<long>(n_sel) * ND; ++q) {
            const int a = sel[q / ND], d = static_cast<int>(q % ND);
            const float v = pj == tomo_core::SF ? tomo_core::project_ray<true, false, tomo_core::SF>(x, xt, g, ak[a], d, nullptr)
                                                : tomo_core::project_ray<true, false>(x, xt, g, ak[a], d, nullptr);
            epi(static_cast<std::size_t>(a) * ND + d, v);
        }
    }
    const float* len2() {
        if (have_l2 != pj * 4 + norm) {
            const int ND = g.n_det;
            l2.resize(static_cast<std::size_t>(n_angles) * ND);
#pragma omp parallel for schedule(dynamic, 64)
            for (long q = 0; q < static_cast<long>(n_angles) * ND; ++q)
                pj == tomo_core::SF
                    ? (norm == 2 ? (void)tomo_core::project_ray<false, 2, tomo_core::SF>(nullptr, nullptr, gfull, ak[q / ND], static_cast<int>(q % ND), &l2[q])
                                 : (void)tomo_core::project_ray<false, 1, tomo_core::SF>(nullptr, nullptr, gfull, ak[q / ND], static_cast<int>(q % ND), &l2[q]))
                    : (norm == 2 ? (void)tomo_core::project_ray<false, 2>(nullptr, nullptr, gfull, ak[q / ND], static_cast<int>(q % ND), &l2[q])
                                 : (void)tomo_core::project_ray<false, 1>(nullptr, nullptr, gfull, ak[q / ND], static_cast<int>(q % ND), &l2[q]));
            have_l2 = pj * 4 + norm;
        }
        return l2.data();
    }
    void backproject(const float* y, float* x, const int32_t* sel, int n_sel, bool accumulate) {
        const int ND = g.n_det;
#pragma omp parallel for schedule(dynamic, 8)
        for (int r = 0; r < g.H; ++r)
            for (int q = 0; q < g.W; ++q) {
                const float X = g.ox + q - g.c, Y = g.oy + r - g.c;
                float s = 0.f;
                if (!g.clip || tomo_core::in_disk(X, Y, g))
                    for (int k = 0; k < n_sel; ++k) {
                        const int a = sel[k];
                        s += pj == tomo_core::SF ? tomo_core::backproject_pixel<tomo_core::SF>(y + static_cast<std::size_t>(a) * ND, g, X, Y, ak[a])
                                                 : tomo_core::backproject_pixel(y + static_cast<std::size_t>(a) * ND, g, X, Y, ak[a]);
                    }
                const std::size_t i = static_cast<std::size_t>(r) * g.W + q;
                x[i] = accumulate ? x[i] + s : s;
            }
    }
    template <class F>
    void each(std::size_t n, const F& f) {
#pragma omp parallel for schedule(static)
        for (long i = 0; i < static_cast<long>(n); ++i) f(static_cast<std::size_t>(i));
    }
    void fill(float* d, float v, std::size_t n) { std::fill(d, d + n, v); }
    void weighted_rows(const float* rows, const float* w, float* out, std::size_t n) {
#pragma omp parallel for schedule(static)
        for (long i = 0; i < static_cast<long>(n); ++i) out[i] = rows[i] * w[i];
    }
    void gridrec(const float* wrows, float* out, const TomoArgs& a) {
        gridrec_slice(wrows, n_angles, g.n_det, angles.data(), g.rot, g.N, a, out);
    }
};

CpuBackend& backend_for(const TomoGeom& t, const float* angles, std::unique_ptr<TomoWorkspace>* ws,
                        std::unique_ptr<TomoWorkspace>& local) {
    std::unique_ptr<TomoWorkspace>& slot = ws ? *ws : local;
    auto* b = dynamic_cast<CpuBackend*>(slot.get());
    if (!b || !b->matches(t, angles)) {
        slot = std::make_unique<CpuBackend>(t, angles);
        b = static_cast<CpuBackend*>(slot.get());
    }
    b->keep.clear();
    return *b;
}

TomoGeom whole(int n_angles, int n_det, int N, float rot) {
    TomoGeom t;
    t.n_angles = n_angles; t.n_det = n_det; t.N = N; t.rot = rot;
    return t;
}
}  // namespace

void tomo_reconstruct(const TomoArgs& a, const TomoGeom& t, const float* rows, const float* weights, const float* angles,
                      float* out, std::unique_ptr<TomoWorkspace>* ws) {
    std::unique_ptr<TomoWorkspace> local;
    tomo_algos::run(backend_for(t, angles, ws, local), a, rows, weights, out);
}

void chord_project(const float* img, float* sino, const float* angles, int n_angles, int n_det, int N, float rot, int projector) {
    std::unique_ptr<TomoWorkspace> local;
    auto& b = backend_for(whole(n_angles, n_det, N, rot), angles, nullptr, local);
    b.pj = projector;
    std::vector<int32_t> all(n_angles);
    for (int i = 0; i < n_angles; ++i) all[i] = i;
    b.project(img, all.data(), n_angles, tomo_algos::StoreEpi{sino});
}

void chord_backproject(const float* sino, float* img, const float* angles, int n_angles, int n_det, int N, float rot, int projector) {
    std::unique_ptr<TomoWorkspace> local;
    auto& b = backend_for(whole(n_angles, n_det, N, rot), angles, nullptr, local);
    b.pj = projector;
    std::vector<int32_t> all(n_angles);
    for (int i = 0; i < n_angles; ++i) all[i] = i;
    b.backproject(sino, img, all.data(), n_angles, false);
}

}  // namespace cpu
}  // namespace helirec
