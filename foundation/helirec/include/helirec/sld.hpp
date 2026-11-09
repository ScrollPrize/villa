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
// Outside the detector the scintillator is taken as dark.
//
// In the reconstruction (SliceReconstructor, ReconConfig::sld) the glare of the normalised signal
// (raw - dark) / current is subtracted before the flat division: the flats are corrected once, and the glare of
// every `stride`-th projection is computed from its whole frame and interpolated linearly in between.
#include <cstdint>
#include <memory>

#include "helirec/common.hpp"

namespace helirec {

struct SldParams {
    double fraction = 0;   // f: tail weight relative to the direct image; 0: no correction
    double length = 100;   // l: damping length of the tail (detector pixels)
    int bin = 8;           // box binning of the frames the glare is computed from
    int stride = 10;       // every stride-th projection gets its own glare map
    bool enabled() const { return fraction > 0; }
};

class ScintillatorGlare {
  public:
    ScintillatorGlare(int det_rows, int det_cols, const SldParams& p);
    ~ScintillatorGlare();
    int rows() const;   // glare map size: ceil(det / bin)
    int cols() const;
    int bin() const;
    // glare maps (n, rows, cols) of raw frames (n, det_rows, det_cols): S * bin((raw - dark) / div[i])
    void from_raw(const uint16_t* raw, const float* dark, const float* div, int n, float* out);
    // glare maps (n, rows, cols) of float frames (n, det_rows, det_cols)
    void from_frames(const float* frames, int n, float* out);
    // in place: frames (n, det_rows, det_cols) minus their glare, T / (1 + f)  (flats; tests)
    void correct(float* frames, int n);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace helirec
