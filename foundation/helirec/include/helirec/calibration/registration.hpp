#pragma once
// Displacement between two blocks of samples that hold the same content, to a fraction of a sample.
//
// The blocks are (n0, n1, n2) float arrays: for projections, (frame, detector row, detector column). The result
// is the displacement d for which a(x) matches b(x + d), found as the maximum of the band-passed
// cross-correlation, located continuously (Newton steps on the correlation's Fourier sum, started from its
// integer maximum). Every axis is treated alike; an axis of length 1 is left out.
//
// What the estimate assumes: one rigid displacement of the whole block, content that is sampled finely enough
// along every axis within the band kept, and no signal common to a and b that sits at another displacement
// (a pattern fixed to the detector does, if both blocks come from nearby detector positions: callers keep such
// pairs apart). It does not model rotation, scale or shear between the blocks.
#include <cstddef>

namespace helirec::calibration {

struct RegistrationOptions {
    // Band kept on axes 1 and 2, as Gaussian cut-offs in cycles per sample of the radial frequency: the high-pass
    // removes the beam's profile and the level of each frame, the low-pass the noise and whatever the frame axis
    // does not sample finely enough.
    double f_lo = 0.01, f_hi = 0.15;
    // Frequencies within this many bins of zero along axis 1 or along axis 2 are left out (when both axes exist):
    // stripes along the rows or the columns are fixed to the detector or the beam, not to the sample, and sit at
    // zero displacement along their own direction whatever the sample does.
    int stripe_notch = 2;
    double taper = 0.25;          // fraction of each axis tapered to zero at each end
    int search[3] = {3, 24, 24};  // the integer maximum is looked for within +- search samples
};

struct Registration {
    double shift[3] = {0, 0, 0};  // a(x) ~ b(x + shift)
    double correlation = 0;       // of the band-passed blocks at that shift, -1..1
    bool at_edge = false;         // the integer maximum lies on the border of the search window
    bool converged = false;
};

Registration register_blocks(const float* a, const float* b, int n0, int n1, int n2, const RegistrationOptions& opt = {});

// Column of a mirror axis. Block a holds columns [a0, a0 + n); block b was cut from columns [b0, b0 + n) and its
// columns reversed; shift_cols is their registration along the columns.
inline double mirror_axis(int a0, int b0, int n, double shift_cols) { return 0.5 * (a0 + b0 + n - 1 - shift_cols); }

// Synthetic checks of the estimator and of the conventions above; true when all pass.
bool registration_selftest(bool verbose);

}  // namespace helirec::calibration
