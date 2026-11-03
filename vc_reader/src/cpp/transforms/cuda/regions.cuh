#pragma once
#include "label_values.cuh"
namespace vcr::transforms::cuda {
DeviceArray regions_cuda(const DeviceArray&, std::size_t, const std::vector<std::vector<double>>&, Session&);
}
