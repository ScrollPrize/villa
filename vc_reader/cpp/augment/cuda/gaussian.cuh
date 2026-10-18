#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
void blur(View<float>, const Grid&, const std::vector<double>&, double, Boundary, bool);
void blur(View<double>, const Grid&, const std::vector<double>&, double, Boundary, bool);
// Separate input/output views must not overlap. The input is preserved.
void blur_into(View<float>, View<float>, const Grid&, const std::vector<double>&, double, Boundary, bool);
void blur_into(View<double>, View<double>, const Grid&, const std::vector<double>&, double, Boundary, bool);
} // namespace vcr::augment::cuda
