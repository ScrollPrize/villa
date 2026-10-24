#pragma once
#include "resident.cuh"
namespace vcr::transforms::cuda {
Parameters sample_elastic(const Elastic&, const Sample&, Random&, bool record);
}
