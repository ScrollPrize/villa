#pragma once
#include "resident.cuh"
namespace vcr::augment::cuda {
DeviceArray resample_cuda(const DeviceArray&, const Parameters&, bool, double, bool, const double*);
DeviceArray resize_cuda(const DeviceArray&, const Shape&, bool, bool);
void geometry_cuda(Sample&, const Parameters&, Session&);
// Jacobian mapping of resampled dense directional/local-coordinate fields.
void directional_cuda(DeviceArray&, const Target&, const Parameters&, const double* field, Session&);
// Resident (N, D) points and vectors; outside="drop" reads back the kept count.
void sparse_cuda(DeviceArray&, const Target&, const Parameters&, const double* field, Session&);
}
