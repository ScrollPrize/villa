#pragma once
#include "batch_queue.hpp"
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace vcr::augment::cuda {
struct KernelIndex { unsigned block, blocks; };

template<std::size_t I, class T> struct KernelArgument { T value; };
template<class Indices, class... T> struct KernelArguments;
template<std::size_t... I, class... T>
struct KernelArguments<std::index_sequence<I...>,T...> : KernelArgument<I,T>... {
    template<class F> __device__ void invoke(F function, KernelIndex index) const {
        function(index,static_cast<const KernelArgument<I,T>&>(*this).value...);
    }
};
template<class F, class... T> struct KernelCall {
    static constexpr bool inline_arguments = requires { F::inline_arguments; };
    F function;
    KernelArguments<std::index_sequence_for<T...>,T...> arguments;
    __device__ void operator()(KernelIndex index) const { arguments.invoke(function,index); }
};
template<class Call> struct KernelJob {
    unsigned blocks, threads;
    std::size_t shared;
    cudaStream_t stream;
    Call call;
    bool compatible(const KernelJob& other) const { return threads==other.threads && stream==other.stream; }
    static void run(const std::vector<const KernelJob*>& jobs);
};
#if CUDART_VERSION >= 11070
#define VCR_GRID_CONSTANT __grid_constant__
#else
#define VCR_GRID_CONSTANT
#endif
template<class Call> __global__ void single_kernel(Call call) {
    call(KernelIndex{blockIdx.x,gridDim.x});
}
// Opt in when arguments contain inline tables accessed by reference. Ordinary
// scalar arguments keep the compiler's normal parameter handling.
template<class Call> __global__ void single_inline_kernel(const VCR_GRID_CONSTANT Call call) {
    call(KernelIndex{blockIdx.x,gridDim.x});
}
template<class Call> void launch_single(const KernelJob<Call>& job) {
    if constexpr (Call::inline_arguments)
        single_inline_kernel<<<job.blocks,job.threads,job.shared,job.stream>>>(job.call);
    else single_kernel<<<job.blocks,job.threads,job.shared,job.stream>>>(job.call);
}
template<class Call> __global__ void batch_kernel(const KernelJob<Call>* jobs) {
    const auto& job = jobs[blockIdx.y];
    if (blockIdx.x < job.blocks) job.call(KernelIndex{blockIdx.x,job.blocks});
}
// Common training batches carry their descriptors in the launch parameters.
// This avoids allocating, uploading and freeing a descriptor buffer per stage.
// Keep the packet within the 4 KiB parameter limit of supported CUDA devices.
template<class Job> struct KernelPacket {
    static constexpr std::size_t capacity=std::min<std::size_t>(32,4096/sizeof(Job));
    alignas(Job) unsigned char bytes[sizeof(Job)*std::max<std::size_t>(1,capacity)];
};
template<class Call> __global__ void batch_inline_kernel(const VCR_GRID_CONSTANT KernelPacket<KernelJob<Call>> packet) {
    const auto& job=reinterpret_cast<const KernelJob<Call>*>(packet.bytes)[blockIdx.y];
    if(blockIdx.x<job.blocks)job.call(KernelIndex{blockIdx.x,job.blocks});
}
#undef VCR_GRID_CONSTANT
template<class Call> void KernelJob<Call>::run(const std::vector<const KernelJob*>& jobs) {
    const auto& first = *jobs.front();
    if (jobs.size() == 1) {
        launch_single(first);
    } else {
        unsigned blocks = 0;
        std::size_t shared = 0;
        for (const auto* job : jobs) {
            require(job->threads == first.threads && job->stream == first.stream,"incompatible batched launch");
            blocks = std::max(blocks,job->blocks);
            shared = std::max(shared,job->shared);
        }
        using Packet=KernelPacket<KernelJob>;
        static_assert(std::is_trivially_copyable_v<KernelJob>);
        if constexpr(Packet::capacity>=2) {
            if(jobs.size()<=Packet::capacity) {
                Packet packet{};
                for(std::size_t i=0;i<jobs.size();++i)std::memcpy(packet.bytes+i*sizeof(KernelJob),jobs[i],sizeof(KernelJob));
                batch_inline_kernel<Call><<<dim3(blocks,unsigned(jobs.size())),first.threads,shared,first.stream>>>(packet);
                check(cudaGetLastError());
                return;
            }
        }
        std::vector<KernelJob> packed;
        packed.reserve(jobs.size());
        for(const auto* job:jobs)packed.push_back(*job);
        Buffer<KernelJob> device(packed);
        batch_kernel<<<dim3(blocks,unsigned(jobs.size())),first.threads,shared,first.stream>>>(device.ptr);
    }
    check(cudaGetLastError());
}
template<class F, class... T>
void launch(unsigned blocks, unsigned threads, std::size_t shared, cudaStream_t stream, F function, T... arguments) {
    using Call = KernelCall<F,T...>;
    KernelJob<Call> job{blocks,threads,shared,stream,Call{function,{{arguments}...}}};
    if (!blocks) return;
    if (BatchQueue::current) BatchQueue::current->append(std::move(job));
    else launch_single(job);
}
}
