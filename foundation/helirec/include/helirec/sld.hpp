#pragma once
// Scintillator light diffusion (SLD) correction, after Mirone et al., "Eikonal phase retrieval" (J. Synchrotron
// Rad. 32, 2025, section 3.6) as run at BM18 (nabuxx diffusion digest).
//
// The detector records M = K * T: the image T blurred by the scintillator's point spread
//   K = (delta + f t) / (1 + f),   t(r) ~ exp(-r / l) / r  (r > 0, unit sum):
// a fraction f / (1 + f) of the light lands in a long tail of damping length l (pixels; ~100 at BM18).
// Deconvolved, up to the factor (1 + f) that cancels between projection and flat:
//   T / (1 + f) = M - G,   G = S * M,   S^ = f t^ / (1 + f t^)       (the "glare")
// S is smooth (scale l), so G is computed from frames box-binned by `bin` and sampled bilinearly. What the bins
// do not resolve of S is under f t^(pi / bin) of the fine-scale contrast (about 0.3 % of it for the defaults).
// Outside the detector the scintillator is taken as dark, or (lit_outside) as lit like the detector's nearest edge:
// the frame is then continued by its edge values over the padding. Where the beam is wider than the detector's
// field the second is nearer the truth: light from beyond the field reaches the pixels near the edge.
//
// In the reconstruction (SliceReconstructor, ReconConfig::sld) the glare of the normalised signal
// (raw - dark) / current is subtracted before the flat division: the flats are corrected once, and the glare of
// every `stride`-th projection is computed from its whole frame and interpolated linearly in between.
// The opt-in BM18 profile instead uses the unit-mass multi-tail PSF and full-resolution FFTs, with an
// illuminated edge extension followed by zero padding. Its maps are the complete signed correction
// M - K^-1*M, including high frequencies. Flats are clipped after correction; projections remain linear.
#include <cstdint>
#include <memory>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

struct SldParams {
    double fraction = 0;   // f: tail weight relative to the direct image; 0: no correction
    double length = 100;   // l: damping length of the tail (detector pixels)
    int bin = 8;           // box binning of the frames the glare is computed from
    int stride = 10;       // every stride-th projection gets its own glare map
    bool lit_outside = false;   // binned profile: beyond the detector the frame continues by its edge values (else dark)
    // BM18: full-resolution, unit-mass PSF, multiple tails and illuminated borders.
    // Python's BM18 profile defaults to bin=1, stride=1, borders=100 detector pixels.
    bool bm18 = false;
    std::vector<double> fractions, lengths;
    int border_x = 0, border_y = 0;
    bool enabled() const { return bm18 || fraction > 0 || !fractions.empty(); }
};

class ScintillatorGlare {
  public:
    ScintillatorGlare(int det_rows, int det_cols, const SldParams& p);
    ~ScintillatorGlare();
    int rows() const;   // glare map size: ceil(det / bin)
    int cols() const;
    int bin() const;
    // glare maps (n, rows, cols) of raw frames, S * bin((raw - dark) / div), in two steps: bin_raw() bins detector
    // rows [r0, r0 + nr) of one frame (`data`: those rows; r0 a multiple of bin(), nr a multiple of it or up to the
    // last row; thread safe) into the rows r0 / bin() ... of its binned frame `out` (rows, cols); filter() turns n
    // binned frames into their glare, in place
    void bin_raw(const uint16_t* data, int r0, int nr, const float* dark, float div, float* out) const;
    void filter(float* maps, int n);
    // glare maps (n, rows, cols) of float frames (n, det_rows, det_cols)
    void from_frames(const float* frames, int n, float* out);
    // in place: frames minus correction; legacy T/(1+f), BM18 full T clipped at zero (flats; tests)
    void correct(float* frames, int n);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
