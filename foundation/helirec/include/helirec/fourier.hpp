#pragma once
// Padded real-to-complex Fourier filtering, the common core of Paganin phase
// retrieval (2D) and the FBP ramp filter (1D along rows).
//
// For each frame (h, w) of a stack:
//   1. pad into (ph, pw), the data placed at (pad_top, pad_left); the border is
//      filled according to `mode` (numpy.pad semantics);
//   2. real FFT (2D, or 1D along each row if ndim == 1);
//   3. multiply by the real filter, shape (ph, pw/2+1) for 2D or (pw/2+1) for 1D;
//   4. inverse FFT, normalized by 1/(ph*pw) (2D) or 1/pw (1D);
//   5. crop back to (h, w) (out of place or in place).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

enum class PadMode {
    Zero,       // numpy 'constant' with 0
    Edge,       // numpy 'edge'      : a a a | a b c | c c c
    Reflect,    // numpy 'reflect'   : d c b | a b c d | c b a
    Symmetric,  // numpy 'symmetric' : c b a | a b c d | d c b
    Wrap,       // numpy 'wrap'      : periodic
};

PadMode pad_mode_from_string(const std::string& s);

struct FourierFilterSpec {
    int ndim = 2;               // 2: 2D filter per frame; 1: 1D filter per row
    int h = 0, w = 0;           // frame size
    int ph = 0, pw = 0;         // padded size (ph ignored for ndim == 1, rows are independent)
    int pad_top = 0, pad_left = 0;
    PadMode mode = PadMode::Edge;
    std::vector<float> filter;       // real part: (ph, pw/2+1) for ndim 2, (pw/2+1) for ndim 1
    std::vector<float> filter_imag;  // optional imaginary part (same shape); empty = real filter
    std::uint64_t id = 0;            // identity of these coefficients (unique per process; 0: none)
};

class FourierFilter {
  public:
    virtual ~FourierFilter() = default;
    // in: (n_frames, h, w) -> out: (n_frames, h, w); in == out allowed.
    // CPU: host pointers. CUDA: device pointers, `stream` is a cudaStream_t.
    virtual void apply(const float* in, float* out, int n_frames, void* stream = nullptr) = 0;
    const FourierFilterSpec& spec() const { return spec_; }
    // replace the filter coefficients by those of `s` (same geometry; the FFT plans are kept)
    virtual void set_filter(const FourierFilterSpec& s) {
        if (s.ndim != spec_.ndim || s.h != spec_.h || s.w != spec_.w || s.ph != spec_.ph || s.pw != spec_.pw ||
            s.pad_top != spec_.pad_top || s.pad_left != spec_.pad_left || s.mode != spec_.mode ||
            s.filter.size() != spec_.filter.size() || s.filter_imag.size() != spec_.filter_imag.size())
            throw Error("FourierFilter::set_filter: different filter geometry");
        spec_.filter = s.filter;
        spec_.filter_imag = s.filter_imag;
        on_filter_changed();
    }

    // Two-stage use (CUDA, for filters that differ only in their coefficients, e.g. parameter sweeps):
    // forward() stores the spectra of the padded frames (spectrum_floats() floats per frame, interleaved complex);
    // inverse() applies the current filter to stored spectra, inverse transforms and crops (spectra unchanged).
    virtual std::size_t spectrum_floats() const { throw Error("FourierFilter: no two-stage support"); }
    virtual void forward(const float*, float*, int, void* = nullptr) { throw Error("FourierFilter: no two-stage support"); }
    virtual void inverse(const float*, float*, int, void* = nullptr) { throw Error("FourierFilter: no two-stage support"); }

  protected:
    explicit FourierFilter(FourierFilterSpec s) : spec_(std::move(s)) {}
    virtual void on_filter_changed() {}
    FourierFilterSpec spec_;
};

void validate(const FourierFilterSpec& s);

// the one process-wide lock around FFTW plan creation / destruction (float and double; FFTW's planner is not thread
// safe), shared by every CPU FFT user (src/cpu/fourier_fftw.cpp)
std::mutex& fftw_planner_mutex();

namespace cpu {
std::unique_ptr<FourierFilter> make_fourier_filter(const FourierFilterSpec& spec);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
std::unique_ptr<FourierFilter> make_fourier_filter(const FourierFilterSpec& spec);
}
#endif

// Index of the source sample for padded coordinate i (may be outside [0, n)).
// Returns -1 for zero padding. Shared by the CPU and CUDA paths.
#ifdef __CUDACC__
__host__ __device__
#endif
inline int pad_source_index(int i, int n, PadMode mode) {
    if (i >= 0 && i < n) return i;
    switch (mode) {
        case PadMode::Zero: return -1;
        case PadMode::Edge: return i < 0 ? 0 : n - 1;
        case PadMode::Wrap: { int r = i % n; return r < 0 ? r + n : r; }
        case PadMode::Reflect: {
            if (n == 1) return 0;
            const int period = 2 * (n - 1);
            int r = i % period; if (r < 0) r += period;
            return r < n ? r : period - r;
        }
        case PadMode::Symmetric: {
            const int period = 2 * n;
            int r = i % period; if (r < 0) r += period;
            return r < n ? r : period - 1 - r;
        }
    }
    return -1;
}

}  // namespace helirec
