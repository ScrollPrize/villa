#pragma once
// Shared types for helirec kernels.
//
// Conventions
//  * All image data is float32, C-contiguous, row-major.
//  * A projection stack has shape (n_proj, n_rows, n_cols); a sinogram stack
//    (n_slices, n_angles, n_cols); a slice (ny, nx).
//  * Every stage exists as  helirec::cpu::<stage>  and (when built with CUDA)
//    helirec::cuda::<stage>  with the same semantics. CUDA entry points take
//    DEVICE pointers and an opaque stream (cudaStream_t passed as void*), so
//    stages can be chained on the GPU without host round trips.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace helirec {

enum class Device { CPU, CUDA };

inline const char* device_name(Device d) { return d == Device::CPU ? "cpu" : "cuda"; }

// true when compiled with CUDA support and at least one device is usable
bool cuda_available();
// Page-locked host buffers (raw windows, download staging) are kept for reuse when a reconstructor is destroyed
// (page-locking costs ~0.2 s/GB); this frees the ones not in use. No-op without CUDA.
void release_pinned_memory();

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace helirec
