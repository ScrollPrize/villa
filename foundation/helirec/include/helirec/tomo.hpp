#pragma once
// Whole-slice reconstruction methods, CPU and CUDA, one slice at a time. Each is a pipeline backprojector
// (ReconConfig.backprojector = "sirt" | "mlem" | "gridrec", parameters in ReconConfig.tomo_params) receiving the
// slice's rows on the reconstructor's device.
//
//   sirt     simultaneous iterative reconstruction: x += relax A^T((b - Ax) / norm) / A^T 1      num_iter, relax, norm
//   mlem     maximum-likelihood EM: x *= A^T(b / Ax) / A^T 1                                     num_iter
//   gridrec  Fourier gridding with a prolate spheroidal kernel; filters none, shepp, cosine, hann, hamming,
//            ramlak, parzen, butterworth (filter_name, filter_par)
//
// Data: the -log rows (n_angles x n_det) and the half-acquisition balanced weights. SIRT / MLEM use the rays with
// weight > 0 (360 deg data, each ray at its measured position); gridrec reconstructs the weighted rows (each pair of
// opposite rays sums to one full projection). Projector (tomo_core.hpp): exact chord lengths at the cell centre
// (chord) or the separable footprint integrated over the cell (sf). Image: the pipeline's N x N grid; unknowns: the
// pixels inside the reconstruction disk (radius R). Initial image (sirt / mlem): init_value, or init=gridrec.
// The whole slice is always reconstructed; the pipeline crops the requested ROI afterwards.
#include <memory>
#include <string>
#include <vector>

#include "helirec/common.hpp"
#include "helirec/filters.hpp"
#include "helirec/pipeline.hpp"

namespace helirec {

struct TomoArgs {
    std::string algorithm = "sirt";
    int num_iter = 1;
    float relax = 1.f;                       // sirt relaxation
    int norm = 2;                            // sirt ray normalisation: 2 row sum (stable for relax < 2 with either
                                             // projector), 1 |a_i|^2 (unstable with sf, whose small footprint-tail
                                             // weights shrink |a_i|^2 below the row sum)
    std::string filter_name = "shepp";       // gridrec
    std::vector<float> filter_par = {0.5f, 8.f};
    std::string init = "constant";           // constant | gridrec
    float init_value = 1e-6f;
    int projector = 1;                       // 0 chord (line through the cell centre), 1 sf (separable footprint)
};

// slice geometry: the N x N grid
struct TomoGeom {
    int n_angles = 0, n_det = 0, N = 0;
    float rot = 0.f;
    float R = 1e30f;                         // reconstruction disk radius (pixels outside are not unknowns)
};

// per-geometry state reused across slices (buffers, angle tables, chord sums, FFT plans)
struct TomoWorkspace {
    virtual ~TomoWorkspace() = default;
};
TomoArgs tomo_args(const std::string& algorithm, const FilterParams& p);   // name + text parameters
std::vector<std::string> tomo_algorithms();

// one slice: rows (n_angles, n_det) -log sinogram, weights (n_angles, n_det) balanced weights; angles: the exact
// backprojector's (u = x cos a + y sin a + rot); out: the N x N image (row-major) in per-pixel units.
// ws (optional): workspace kept between calls with the same geometry.
namespace cpu {
void tomo_reconstruct(const TomoArgs& a, const TomoGeom& g, const float* rows, const float* weights,
                       const float* angles, float* out, std::unique_ptr<TomoWorkspace>* ws = nullptr);
void chord_project(const float* img, float* sino, const float* angles, int n_angles, int n_det, int N, float rot,
                   int projector = 0);
void chord_backproject(const float* sino, float* img, const float* angles, int n_angles, int n_det, int N, float rot,
                       int projector = 0);
}  // namespace cpu
#ifdef HELIREC_WITH_CUDA
namespace cuda {
// device rows / weights / out, host angles
void tomo_reconstruct(const TomoArgs& a, const TomoGeom& g, const float* rows, const float* weights,
                       const float* angles_host, float* out, std::unique_ptr<TomoWorkspace>* ws = nullptr);
void chord_project(const float* img, float* sino, const float* angles_host, int n_angles, int n_det, int N, float rot,
                   int projector = 0);
void chord_backproject(const float* sino, float* img, const float* angles_host, int n_angles, int n_det, int N, float rot,
                       int projector = 0);
}  // namespace cuda
#endif

// the pipeline backprojectors "sirt", "mlem", "gridrec"
bool is_tomo_method(const std::string& backprojector);
std::shared_ptr<ExternalBackprojector> make_tomo_backprojector(const std::string& algorithm, const FilterParams& p);

}  // namespace helirec
