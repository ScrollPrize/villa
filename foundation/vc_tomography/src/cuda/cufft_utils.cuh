#pragma once
#include <cufft.h>

#include <string>

#include "vc_tomography/common.hpp"

namespace vc_tomography::cuda {

inline void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error ") + std::to_string(static_cast<int>(r)) + " in " + what);
}

}  // namespace vc_tomography::cuda
