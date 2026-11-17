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
// The device scratch the CUDA kernels keep between calls on the calling thread (buffers and FFT plans that only grow)
// is freed after the device's work is done. A reconstructor on CUDA does this when it is destroyed, so its memory does
// not outlive it on a thread that goes on (e.g. Python's main thread). No-op without CUDA.
void release_device_scratch();

// What vc_tomography throws. In Python (the bindings translate them): Error a ValueError (a value, a parameter or
// data that cannot be used), KeyError a KeyError (an unknown or removed name), Unsupported a NotImplementedError
// (a case the code does not handle), IoError a RuntimeError (a file that cannot be read or written).
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct KeyError : Error {
    using Error::Error;
};
struct Unsupported : Error {
    using Error::Error;
};
struct IoError : Error {
    using Error::Error;
};

}  // namespace vc_tomography
