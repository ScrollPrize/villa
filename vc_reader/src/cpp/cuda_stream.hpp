#pragma once
// The CUDA stream vc_reader work uses on the calling thread. By default the
// thread's own default stream (cudaStreamPerThread): threads sampling at once do
// not serialize on the legacy default stream, which still orders against it.
#include "gpu_runtime.hpp"
#include <cstdint>
namespace vcr::device {
inline gpu::Stream& current_stream() { thread_local gpu::Stream stream = gpu::per_thread_stream(); return stream; }
inline gpu::Stream stream() { return current_stream(); }
struct StreamScope {
    gpu::Stream previous;
    explicit StreamScope(gpu::Stream s) : previous(current_stream()) { current_stream() = s; }
    ~StreamScope() { current_stream() = previous; }
    StreamScope(const StreamScope&) = delete;
};
// A stream from its DLPack/array-API handle: 1 is the legacy default stream,
// 2 the per-thread default stream.
inline gpu::Stream stream_handle(std::uintptr_t handle) {
    if (handle == 0 || handle == 1) return gpu::legacy_stream();
    if (handle == 2) return gpu::per_thread_stream();
    return reinterpret_cast<gpu::Stream>(handle);
}
// The DLPack consumer handle for a stream. The default streams map to DLPack's
// default stream (1 on CUDA, 0 on ROCm): PyTorch rejects 2, and the per-thread
// default stream is a blocking stream, so work on it already orders after the
// legacy default stream.
inline std::uintptr_t dlpack_handle(gpu::Stream s) {
    if (s == nullptr || s == gpu::legacy_stream() || s == gpu::per_thread_stream()) {
        // DLPack's default stream is 1 on CUDA but 0 on ROCm, which rejects 1.
#ifdef VC_READER_HIP
        return 0;
#else
        return 1;
#endif
    }
    return reinterpret_cast<std::uintptr_t>(s);
}
inline std::uintptr_t dlpack_handle() { return dlpack_handle(current_stream()); }
}
