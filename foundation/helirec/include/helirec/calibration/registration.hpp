#pragma once
// Displacement between two blocks of samples that hold the same content, to a fraction of a sample.
//
// The blocks are float arrays over (frame, detector row, detector column). Block a has n[0] x n[1] x n[2] samples.
// Block b is cut with a margin on every side, (n[j] + 2 margin[j]) samples per axis, so that it can be displaced
// by a fraction of a sample exactly (a phase ramp on the spectrum of the padded block). The result is the
// displacement d for which a(x) matches b(x + margin + d): the maximum of the band-passed cross-correlation of a
// with the displaced b, both under one window, repeated until no displacement is left. Two blocks correlated
// under one window without that repetition are pulled towards zero displacement, by more the longer the
// content's correlation is against the block (16 % along 16 frames in the self test).
//
// What the estimate assumes: one rigid displacement of the whole block, content that is sampled finely enough
// along every axis within the band kept, and no signal common to a and b that sits at another displacement
// (a pattern fixed to the detector does, if both blocks come from nearby detector positions: callers keep such
// pairs apart). It does not model rotation, scale or shear between the blocks, except a known displacement along
// the rows per frame (row_shift), which is removed from b before anything else.
#include <cstddef>
#include <memory>

#include "helirec/common.hpp"

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
    double taper = 0.25;          // fraction of each axis of block a tapered to zero at each end
    int search[3] = {3, 24, 24};  // the integer maximum is looked for within +- search samples
    int max_evaluations = 12;
    double tolerance = 1e-3;      // stop when the displacement left is below this [samples]
    int threads = 0;              // FFT threads (0: up to 8)
};

struct Registration {
    double shift[3] = {0, 0, 0};  // a(x) ~ b(x + margin + shift)
    // Non-zero: the blocks are displaced by about this many whole samples. b must be cut again that much further
    // and the registration repeated; `shift` is then only the first estimate.
    int whole[3] = {0, 0, 0};
    double correlation = 0;       // of the band-passed blocks at that shift, -1..1
    // Second derivatives of that correlation with respect to the displacement (frames, rows, columns) at the shift:
    // how sharply each direction is determined, and how they are tied. Zero for an axis that is not registered.
    double curvature[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    bool at_edge = false;         // the integer maximum lies on the border of the search window
    bool converged = false;
    int evaluations = 0;
};

// One registration geometry (block sizes, margins, band) on one device, kept for many pairs of blocks: the
// transforms' plans, the band weights and all work memory are made once.
//
// row_shift: null, or one value per frame of b (n[0] + 2 margin[0]): frame q of b is first displaced along the
// rows to b'(q, v) = b(q, v + row_shift[q]). margin[1] must exceed the largest of them by a few rows.
// An axis with margin < 2 is not displaced: its estimate keeps the pull towards zero.
//
// CPU (FFTW) and CUDA (cuFFT) run the same estimator on the same arithmetic; results differ by float rounding.
// The blocks are host arrays on both. staging_a() / staging_b() return host memory of the blocks' sizes that the
// device takes fastest (page-locked for CUDA): filling those and passing them to run() saves a copy.
class Registrar {
  public:
    Registrar(const int n[3], const int margin[3], const RegistrationOptions& opt = {}, Device device = Device::CPU);
    ~Registrar();
    Registrar(const Registrar&) = delete;
    Registrar& operator=(const Registrar&) = delete;
    bool fits(const int n[3], const int margin[3], Device device) const;
    float* staging_a();
    float* staging_b();
    Registration run(const float* a, const float* b, const double* row_shift = nullptr);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One pair of blocks (a Registrar made and dropped)
Registration register_blocks(const float* a, const float* b, const int n[3], const int margin[3], const double* row_shift = nullptr,
                             const RegistrationOptions& opt = {}, Device device = Device::CPU);

// Column of a mirror axis. Block a holds columns [a0, a0 + n); block b was cut from columns [b0, b0 + n) and its
// columns reversed; shift_cols is their registration along the columns.
inline double mirror_axis(int a0, int b0, int n, double shift_cols) { return 0.5 * (a0 + b0 + n - 1 - shift_cols); }

// Synthetic checks of the estimator and of the conventions above; true when all pass.
bool registration_selftest(bool verbose, Device device = Device::CPU);

}  // namespace helirec::calibration
