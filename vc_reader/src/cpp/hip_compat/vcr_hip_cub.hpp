// CUB and libcu++ names for VC_READER_ENABLE_HIP builds. The <cub/...> and
// <cuda/std/...> stand-ins in this directory include this header.
//
// hipCUB mirrors the CUB API, so cub:: names resolve to hipcub::. CUB's custom
// key decomposers return ::cuda::std::tuple references; the rocPRIM backend of
// hipCUB requires ::rocprim::tuple references, so tie/tuple map to rocPRIM's.
#pragma once

#include "cuda_runtime.h"
#include <functional>
#include <hipcub/hipcub.hpp>
#include <rocprim/types/tuple.hpp>

namespace cub = hipcub;

namespace cuda::std {
using ::std::plus;
using ::rocprim::tie;
using ::rocprim::tuple;
}
