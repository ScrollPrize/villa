// Included first in every HIP translation unit of a VC_READER_ENABLE_HIP build.
//
// vc_reader marks host/device helpers when __CUDACC__ is defined; the HIP
// compiler defines __HIPCC__ instead, so define __CUDACC__ for these units.
// (HIP's own headers consult __CUDACC__ only for NVIDIA platforms.)
#pragma once
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif
#include "cuda_runtime.h"
