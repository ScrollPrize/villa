#pragma once
#ifndef VC_READER_CUDA
#error "Link a CUDA-enabled vc_reader::reader and vc_reader::transforms build to use this header"
#endif
#include "vc_reader/native/cuda_buffer.hpp"
#include "vc_reader/native/transforms/cuda/device_io.hpp"
