#pragma once
// Shared types for vc_tomography kernels.
//
// Conventions
//  * All image data is float32, C-contiguous, row-major.
//  * A projection stack has shape (n_proj, n_rows, n_cols); a sinogram stack
//    (n_slices, n_angles, n_cols); a slice (ny, nx).
//  * Every stage exists as  vc_tomography::cpu::<stage>  and (when built with CUDA)
//    vc_tomography::cuda::<stage>  with the same semantics. CUDA entry points take
//    DEVICE pointers and an opaque stream (cudaStream_t passed as void*), so
//    stages can be chained on the GPU without host round trips.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vc_tomography {

enum class Device { CPU, CUDA };

inline const char* device_name(Device d) { return d == Device::CPU ? "cpu" : "cuda"; }

// true when compiled with CUDA support and at least one device is usable
bool cuda_available();
// "cpu" or "cuda" (an Error when CUDA is not available); with allow_auto also "auto": CUDA when available, else CPU
Device parse_device(const std::string& name, bool allow_auto = false);
// Page-locked host buffers (raw windows, download staging) are kept for reuse when a reconstructor is destroyed
// (page-locking costs ~0.2 s/GB); this frees the ones not in use. No-op without CUDA.
void release_pinned_memory();

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace vc_tomography
