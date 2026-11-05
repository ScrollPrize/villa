#pragma once
// The CUDA stream vc_reader work uses on the calling thread. By default the
// thread's own default stream (cudaStreamPerThread): threads sampling at once do
// not serialize on the legacy default stream, which still orders against it.
#include <cuda_runtime.h>
#include <cstdint>
namespace vcr::device {
inline cudaStream_t& current_stream() { thread_local cudaStream_t stream = cudaStreamPerThread; return stream; }
inline cudaStream_t stream() { return current_stream(); }
struct StreamScope {
    cudaStream_t previous;
    explicit StreamScope(cudaStream_t s) : previous(current_stream()) { current_stream() = s; }
    ~StreamScope() { current_stream() = previous; }
    StreamScope(const StreamScope&) = delete;
};
// A stream from its DLPack/array-API handle: 1 is the legacy default stream,
// 2 the per-thread default stream.
inline cudaStream_t stream_handle(std::uintptr_t handle) {
    if (handle == 0 || handle == 1) return cudaStreamLegacy;
    if (handle == 2) return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(handle);
}
// The DLPack consumer handle for a stream. The default streams map to 1 (legacy):
// PyTorch rejects 2, and the per-thread default stream is a blocking stream, so
// work on it already orders after the legacy default stream.
inline std::uintptr_t dlpack_handle(cudaStream_t s) {
    if (s == nullptr || s == cudaStreamLegacy || s == cudaStreamPerThread) return 1;
    return reinterpret_cast<std::uintptr_t>(s);
}
inline std::uintptr_t dlpack_handle() { return dlpack_handle(current_stream()); }
}
