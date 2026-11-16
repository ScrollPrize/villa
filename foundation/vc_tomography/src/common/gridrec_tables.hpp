#pragma once
// gridrec tables, shared by the CPU and CUDA gridrec (defined in src/cpu/tomo.cpp)
#include <complex>
#include <vector>

#include "vc_tomography/tomo.hpp"

namespace vc_tomography::tomo_gridrec {
constexpr float kC = 7.0f;   // bandwidth parameter of the prolate spheroidal kernel
constexpr int kNt = 20, kLtbl = 512;
int pdim_for(int n_det, int N);
float grid_shift(int N);   // image offset (pixels, both axes) of the pipeline centre from the FFT origin
int kernel_cells();        // width of the gridding kernel in grid cells
void pswf_tables(int pdim, std::vector<float>& wtbl, std::vector<float>& winv);
std::vector<std::complex<float>> filter_phase(int pdim, int n_angles, float center, const TomoArgs& a, float scale);
float span_factor(const float* angles, int n);
}  // namespace vc_tomography::tomo_gridrec
