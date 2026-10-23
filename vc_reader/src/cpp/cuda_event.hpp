#pragma once
// Completion of asynchronous work: recorded on the current stream when created.
// Consumers either wait on it from their own stream (no host blocking) or
// synchronize the host. Stream handles follow the DLPack/array-API convention:
// 1 is the legacy default stream, 2 the per-thread default stream.
#include "cuda_stream.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace vcr::device {
inline cudaStream_t stream_handle(std::uintptr_t handle) {
    if (handle == 0 || handle == 1) return cudaStreamLegacy;
    if (handle == 2) return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(handle);
}
class Event {
    cudaEvent_t event_ = nullptr;
    int device_ = 0;
    pid_t pid_ = getpid();
    static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e)); }
public:
    Event() {
        check(cudaGetDevice(&device_));
        check(cudaEventCreateWithFlags(&event_,cudaEventDisableTiming));
        check(cudaEventRecord(event_,stream()));
    }
    Event(const Event&) = delete;
    ~Event() { if (event_ && pid_ == getpid()) cudaEventDestroy(event_); }
    void synchronize() const { check(cudaEventSynchronize(event_)); }
    void wait(std::uintptr_t handle) const { check(cudaStreamWaitEvent(stream_handle(handle),event_,0)); }
    bool done() const {
        const auto status = cudaEventQuery(event_);
        if (status == cudaErrorNotReady) return false;
        check(status); return true;
    }
    int device() const { return device_; }
};
}
