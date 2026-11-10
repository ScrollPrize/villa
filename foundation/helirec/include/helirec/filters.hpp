#pragma once
// Projection filters: a configurable chain applied to every gridded stack of a chunk (frames of H rows = the
// chunk's slices + 2 x margin, W = merged width), before and/or after -log.
//
//   chain = [{type: paganin, delta_beta: 1000}, {type: unsharp, sigma: 1.2, coeff: 4}, {type: median, radius: 1}]
//
// Each filter declares the vertical context it needs (margin rows) and the stage it runs at ("intensity":
// before -log, default; "log": after -log, before the crop to the chunk's rows). Linear shift-invariant filters
// ("Fourier" filters) only provide a frequency response; consecutive Fourier filters of one stage are fused into
// a single padded FFT filter (FourierFilter). Other filters run their own CPU / CUDA kernels, in chain order.
//
// Built-in types (register_filter adds more, e.g. the Python filters of the bindings):
//   paganin  delta_beta                      Paganin (discrete Laplacian response). The real-space kernel is fixed:
//                                            computed on its own grid, truncated to [-m, m] rows (m: its margin),
//                                            normalised to unit DC, so a constant stays constant and the result does
//                                            not depend on the chunk height. Legacy reference mode: the nabuxx kernel
//                                            (pag_version 2: truncated to rows [-m, m - 1] of the chunk's padded grid,
//                                            not renormalised: DC < 1, chunk-height dependent)
//   unsharp  sigma, coeff, method=gaussian   (1 + c) - c G   (method log: 1 + c Lap G)
//   gaussian sigma                           Gaussian blur (periodic, as unsharp's G)
//   median   radius (1..3)                   (2r+1)^2 median per frame, edges clamped
//   epr      delta_beta, iters=3,            eikonal phase retrieval (local second-order closure, src/common/
//            relax=0.75, max_shift=12.5      epr_core.hpp), in place of paganin: starts from the Paganin result
//                                            and iterates I += relax Paganin[measured - F(I)] against the
//                                            second-order model F of the propagated intensity. iters=0 is Paganin.
//                                            max_shift: limit of the modelled ray shift (pixels; <= 0: none).
//                                            Intensity stage only; margin 2 x Paganin's + 4 rows
//   mtf_paganin delta_beta, sigma | mtf,     phase retrieval and detector deblurring as one filter, in place of
//            target=0.5, gain=12 | nsr       paganin + unsharp: Paganin x D(M), M the detector MTF (Gaussian of `sigma`
//                                            pixels, or a radial table mtf="nu:M/nu:M/...", nu in cycles per pixel).
//                                            Default D = T sqrt(1 + e^2) / sqrt(M^2 + e^2), e = 1 / gain: the blur is
//                                            replaced by a Gaussian of `target` pixels (0: none), the gain capped at
//                                            ~gain. With nsr: the Wiener form D = (1 + nsr) M / (M^2 + nsr) (sharper
//                                            cut-off: rings at edges, band-limited noise). Fixed unit-DC kernel as
//                                            paganin's.
// Every filter accepts stage=intensity|log.
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "helirec/common.hpp"
#include "helirec/fourier.hpp"

namespace helirec {

// physics / geometry a filter may need
struct FilterContext {
    double energy_kev = 0, distance_m = 0, pixel_size_m = 0;
    int det_rows = 0, det_cols = 0;
    bool legacy = false;   // legacy reference mode: the reference's kernels (Paganin: truncated to the step margin)
};

// name -> value (as text); typed getters
struct FilterParams {
    std::map<std::string, std::string> values;
    bool has(const std::string& k) const { return values.count(k) != 0; }
    double num(const std::string& k) const;                 // required
    double num(const std::string& k, double def) const;
    std::string str(const std::string& k, const std::string& def) const;
};

struct FilterSpec {
    std::string type;
    FilterParams params;
};
using FilterChain = std::vector<FilterSpec>;

enum class FilterStage { Intensity, Log };

class ProjectionFilter {
  public:
    virtual ~ProjectionFilter() = default;
    std::string type;
    FilterStage stage = FilterStage::Intensity;
    virtual int margin_rows() const = 0;
    virtual bool fourier() const { return false; }
    // Fourier filters: K (n0 x n1, unshifted DFT layout) *= this filter's response; `margin`: margin rows of the
    // fused step (the Paganin kernel is truncated to it)
    virtual void multiply_response(std::vector<std::complex<double>>& K, int n0, int n1, int margin) const;
    // other filters: in place on frames (n_frames, H, W) in host (CPU) or device (CUDA) memory
    virtual void apply(float* frames, int n_frames, int H, int W, Device device);
};

using FilterFactory = std::function<std::unique_ptr<ProjectionFilter>(const FilterParams&, const FilterContext&)>;
void register_filter(const std::string& type, FilterFactory make);
std::unique_ptr<ProjectionFilter> make_filter(const FilterSpec& spec, const FilterContext& ctx);
std::vector<std::string> filter_types();

// smallest 2^a 3^b 5^c 7^d * {1, 11, 13} STRICTLY greater than n (nabuxx roundfft)
int roundfft(int n);
// smallest 2^a 3^b 5^c 7^d >= n (nabu generate_powers / _get_next_power)
int next_smooth_power(int n);

// A chain compiled into steps: consecutive Fourier filters of one stage form one step (one FFT filter).
class FilterPipeline {
  public:
    struct Step {
        FilterStage stage;
        bool fourier;
        std::vector<ProjectionFilter*> members;
        int margin;  // Fourier step: sum of its members' (the fused kernel's support; legacy reference mode: the
                     // max, as the reference); other: the filter's
    };
    FilterPipeline(const FilterChain& chain, const FilterContext& ctx);
    ~FilterPipeline();
    int margin_rows() const;          // rows of context the chain needs: sum over its steps
    bool single_fourier() const;      // one Fourier step at the intensity stage (sweeps share its forward FFT)
    const std::vector<Step>& steps() const { return steps_; }
    // FFT filter of Fourier step s for stacks (H, W); `margin` < 0: the step's own (else e.g. a sweep's common one).
    // Padding: rows to roundfft(H), columns to roundfft(W + 2 margin), symmetric. Cached by (s, H, W, margin)
    // for the pipeline's lifetime (references stay valid).
    const FourierFilterSpec& step_spec(std::size_t s, int H, int W, int margin = -1) const;
    std::string describe() const;

  private:
    std::vector<std::unique_ptr<ProjectionFilter>> filters_;
    std::vector<Step> steps_;
    mutable std::map<std::vector<int>, FourierFilterSpec> spec_cache_;
};

// Eikonal phase retrieval of frames (n_frames, H, W), in place: measured intensity -> contact-plane intensity.
// `paganin`: the Paganin filter for (H, W) frames with the same delta_beta (initial estimate and preconditioner).
struct EprParams {
    double shift_px2 = 0;    // A = delta_beta lambda L / (4 pi px^2): ray shift (pixels) per unit grad ln I
    double delta_beta = 0;
    int iters = 3;
    float relax = 0.75f;
    float max_shift = 12.5f; // pixels; <= 0: no limit
    float eps = 1e-8f;       // floor of intensities
};

namespace cpu {
void median_frames(float* frames, int n_frames, int H, int W, int radius);
void epr_frames(float* frames, int n_frames, int H, int W, const EprParams& p, FourierFilter& paganin);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void median_frames(float* frames, int n_frames, int H, int W, int radius, void* stream = nullptr);
void epr_frames(float* frames, int n_frames, int H, int W, const EprParams& p, FourierFilter& paganin,
                void* stream = nullptr);
}
#endif

}  // namespace helirec
