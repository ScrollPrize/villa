#pragma once
// gridrec tables, shared by the CPU and CUDA gridrec (defined in src/cpu/tomo.cpp)
#include <complex>
#include <vector>

#include "helirec/tomo.hpp"

namespace helirec::tomo_gridrec {
constexpr float kC = 7.0f, kLambda = 0.99998546f;
constexpr int kNt = 20, kLtbl = 512;
int pdim_for(int n_det, int N);
float grid_shift(int N);   // image offset (pixels, both axes) of the pipeline centre from the FFT origin
void pswf_tables(int pdim, std::vector<float>& wtbl, std::vector<float>& winv);
std::vector<std::complex<float>> filter_phase(int pdim, int n_angles, float center, const TomoArgs& a, float scale);
float span_factor(const float* angles, int n);
}  // namespace helirec::tomo_gridrec
