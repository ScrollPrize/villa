#pragma once
#include "resident.cuh"
namespace vcr::augment::cuda {
DeviceArray resample_cuda(const DeviceArray&, const Parameters&, bool, double, bool, const double*);
DeviceArray resize_cuda(const DeviceArray&, const Shape&, bool, bool);
void geometry_cuda(Sample&, const Parameters&, Session&);
}
