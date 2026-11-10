#pragma once
// Cross-spectra of two volumes on cubic blocks, in polar frequency bins: what a 3D Fourier shell correlation and the
// measures derived from it need (correlation, spectral signal-to-noise ratio, the linear filter between the two
// volumes, each one's power), kept apart for in-plane and axial frequencies.
//
// Volume a is the one under test on its own grid; volume b is the comparison: another reconstruction on the same
// grid (e.g. the other half of a projection split), or a finer reference resampled on the block (reference.hpp
// RegisteredVolume) with `oversample` samples per voxel of a along each axis. Only the frequencies of a's grid are
// used of b: cropping its transform is an ideal low-pass and decimation, so nothing above a's Nyquist frequency
// aliases into the comparison, and no anti-alias filter has to be chosen.
//
// Per block:
//   1. each volume has its window-weighted mean removed and is multiplied by a separable Hann window (the same
//      continuous window on both grids), then transformed (real 3D FFT);
//   2. optionally b is registered on a: the shift maximising their cross-correlation, whitened per shell and
//      restricted to frequencies below align_band (integer search within max_shift voxels, then Newton iterations on
//      the continuous shift), is removed from b as a phase ramp;
//   3. |A|^2, |B|^2 and Re(A conj B) are summed in bins (shell, theta): shells of |k| of width 0.5 / shells cycles per
//      voxel, out to the corners of the cube (n_shells() of them: frequencies along an axis end at shell `shells`),
//      and thetas equal bins of the angle of k from the z axis (0: along z, the axial direction; 90 degrees: in the
//      xy plane). Frequencies with a component at the Nyquist index are left out.
// The sums are normalised so that the sum over all bins of aa is the mean square of the windowed a - mean, relative
// to the window's (and bb that of b), but for the Nyquist frequencies left out: tables add over blocks. curves() reduces tables to curves over the shells.
//
// CPU (FFTW, host pointers) and CUDA (cuFFT, device pointers) implementations with the same semantics.
#include <array>
#include <memory>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {

struct BlockSpectraSpec {
    int n = 128;              // block edge (voxels of a); even
    int oversample = 1;       // b has (n * oversample)^3 samples: sample (k, j, i) at voxel (k, j, i) / oversample of a
    int shells = 16;          // shells up to 0.5 cycles per voxel
    int thetas = 6;           // polar-angle bins of 90 / thetas degrees
    bool align = true;        // register b on a by a sub-voxel shift
    float align_band = 0.25f; // cycles per voxel: the frequencies the shift is found from
    int max_shift = 4;        // voxels, per axis

    int n_shells() const;                        // shells in all: ceil(shells * sqrt 3)
    int n_bins() const { return n_shells() * thetas; }
};
void validate(const BlockSpectraSpec& s);

struct BlockSpectra {
    std::vector<double> aa, bb, ab;       // (n_shells, thetas) sums of |A|^2, |B|^2, Re(A conj B)
    std::array<double, 3> shift{0, 0, 0}; // (z, y, x), voxels of a: displacement of b's content from a's, removed
    double peak = 0;                      // whitened band-limited correlation at that shift (0 when not aligned)
    double mean_a = 0, mean_b = 0;        // window-weighted means and variances of the two blocks
    double var_a = 0, var_b = 0;
};

class BlockCorrelator {
  public:
    virtual ~BlockCorrelator() = default;
    // a: (n, n, n); b: (n * oversample)^3; C order (z, y, x). shift: use this shift (z, y, x) instead of finding
    // one (the registration found with another volume a of the same block). CUDA: device pointers.
    virtual void compute(const float* a, const float* b, BlockSpectra& out, const double* shift = nullptr) = 0;
    const BlockSpectraSpec& spec() const { return spec_; }
    // frequencies counted in each bin (n_shells, thetas): the same for every block
    const std::vector<double>& counts() const { return counts_; }

  protected:
    explicit BlockCorrelator(const BlockSpectraSpec& s);
    BlockSpectraSpec spec_;
    std::vector<double> counts_;
};

namespace cpu {
std::unique_ptr<BlockCorrelator> make_block_correlator(const BlockSpectraSpec& spec);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
std::unique_ptr<BlockCorrelator> make_block_correlator(const BlockSpectraSpec& spec);
}
#endif

// Curves over the shells from tables (n_shells, thetas) of sums (one block's, or the sums over many), restricted to
// polar angles [theta_lo, theta_hi] degrees: whole bins inside the interval (0, 90: every direction, the shell
// correlation; 60, 90: in-plane; 0, 30: axial).
struct ShellCurves {
    std::vector<double> frequency;     // shell centres, cycles per voxel
    std::vector<double> count;         // frequencies summed
    std::vector<double> correlation;   // ab / sqrt(aa bb)
    // The correlation with b of the best linear estimate of b from a that weighs each bin (shell, theta) on its
    // own: sqrt(sum_bins (ab^2 / aa) / sum_bins bb). No filter of a that is constant over each bin changes it (the
    // plain correlation of a shell mixes directions of unequal signal-to-noise ratio, and rises when a filter
    // attenuates the poorer ones, e.g. a blur along z): the measure to compare reconstructions with.
    std::vector<double> matched;
    // signal-to-noise power ratio of a implied by c = max(correlation, 0):
    std::vector<double> snr_split;     // c / (1 - c): b has the same signal and independent noise of the same power
                                       // (two halves of one scan; the whole scan then has twice this)
    std::vector<double> snr_reference; // c^2 / (1 - c^2): b is noiseless (a lower bound when it is not)
    std::vector<double> filter;        // ab / aa: the least-squares linear filter taking a to b
    std::vector<double> power_a, power_b;   // mean power per frequency
};
ShellCurves shell_curves(const BlockSpectraSpec& spec, const std::vector<double>& counts, const std::vector<double>& aa,
                         const std::vector<double>& bb, const std::vector<double>& ab, double theta_lo = 0,
                         double theta_hi = 90);

// ------------------------------------------------------------------------------------------ resampling
// A (z, y, x) float32 volume sampled on a regular grid of points given by an affine map: out[k, j, i] (nz, ny, nx) is
// the volume interpolated at (z, y, x) = m (k, j, i, 1), m a 3 x 4 matrix in row-major order (rows z, y, x). interp:
// "linear", "cubic" (Keys) or "lanczos3"; the volume is 0 outside. CUDA: device pointers.
int resample_interp_from_name(const std::string& name);
namespace cpu {
void resample_affine(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                     int interp);
}
#ifdef HELIREC_WITH_CUDA
namespace cuda {
void resample_affine(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                     int interp, void* stream = nullptr);
// the same from a volume in host memory (uploaded for the call); out: device
void resample_affine_host(const float* src, int sz, int sy, int sx, const double* m, float* out, int nz, int ny, int nx,
                          int interp);
}
#endif

// ------------------------------------------------------------------------------------------ shared (implementations)
namespace block_spectra {
// The registration of step 2 from the cross-spectrum A conj B on the box of low frequencies |k| <= kb per axis
// (kz, ky in [-kb, kb], kx in [0, kb]; (2 kb + 1, 2 kb + 1, kb + 1) interleaved complex, normalised as the tables)
// and the tables of the two powers: shift (z, y, x) as BlockSpectra::shift, and the correlation there.
int align_box(const BlockSpectraSpec& s);   // kb
void align(const BlockSpectraSpec& s, const float* box, const std::vector<double>& aa, const std::vector<double>& bb,
           std::array<double, 3>& shift, double& peak);
std::vector<double> bin_counts(const BlockSpectraSpec& s);
std::vector<double> theta_tan2(const BlockSpectraSpec& s);   // tan^2 of the upper edge of each theta bin but the last
std::vector<float> hann(int n, int oversample);              // the window at the n * oversample sample positions
}  // namespace block_spectra

}  // namespace helirec
