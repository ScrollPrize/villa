#pragma once
#include "../spatial.hpp"
namespace vcr::augment::cuda {
Array resample_cuda(const Array&, const Parameters&, bool, double, bool, std::size_t);
Array resize_cuda(const Array&, const Shape&, bool, bool);
} // namespace vcr::augment::cuda
