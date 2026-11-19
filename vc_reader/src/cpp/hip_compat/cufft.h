// HIP stand-in for <cufft.h> in VC_READER_ENABLE_HIP builds: maps the cuFFT
// names vc_reader uses to hipFFT (see cuda_runtime.h in this directory).
#pragma once

#include "cuda_runtime.h"
#include <hipfft/hipfft.h>

#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftComplex hipfftComplex
#define cufftDoubleComplex hipfftDoubleComplex
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_Z2Z HIPFFT_Z2Z
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define cufftCreate hipfftCreate
#define cufftDestroy hipfftDestroy
#define cufftSetStream hipfftSetStream
#define cufftMakePlanMany64 hipfftMakePlanMany64
#define cufftExecC2C hipfftExecC2C
#define cufftExecZ2Z hipfftExecZ2Z
