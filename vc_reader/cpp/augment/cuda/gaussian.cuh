#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
void blur(View<float>, const Grid&, const std::vector<double>&, double, bool, bool);
void blur(View<double>, const Grid&, const std::vector<double>&, double, bool, bool);
} // namespace vcr::augment::cuda
