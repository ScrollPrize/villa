#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
void blur(Buffer<float>&, const Grid&, const std::vector<double>&, double, bool, bool);
void blur(Buffer<double>&, const Grid&, const std::vector<double>&, double, bool, bool);
} // namespace vcr::augment::cuda
