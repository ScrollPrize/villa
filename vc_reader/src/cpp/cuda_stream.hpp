#pragma once
// The CUDA stream vc_reader work uses on the calling thread. Null (the
// default) is the legacy default stream: calls then behave as before streams.
#include <cuda_runtime.h>
namespace vcr::device {
inline cudaStream_t& current_stream() { thread_local cudaStream_t stream = nullptr; return stream; }
inline cudaStream_t stream() { return current_stream(); }
struct StreamScope {
    cudaStream_t previous;
    explicit StreamScope(cudaStream_t s) : previous(current_stream()) { current_stream() = s; }
    ~StreamScope() { current_stream() = previous; }
    StreamScope(const StreamScope&) = delete;
};
}
