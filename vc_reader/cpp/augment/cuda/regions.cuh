#pragma once
#include "label_values.cuh"
namespace vcr::augment::cuda {
DeviceArray regions_cuda(const DeviceArray&, std::size_t, const std::vector<std::vector<double>>&, Session&);
}
