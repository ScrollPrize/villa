#pragma once
namespace vcr::transforms::cuda {
__device__ inline void copy_bytes(KernelIndex vcr_launch, unsigned char* to, const unsigned char* from, std::size_t size) {
    if ((reinterpret_cast<std::uintptr_t>(to) | reinterpret_cast<std::uintptr_t>(from)) % 16 == 0) {
        EACH_INDEX(size/16) reinterpret_cast<uint4*>(to)[i] = reinterpret_cast<const uint4*>(from)[i];
        if (!vcr_launch.block && !threadIdx.x)
            for (std::size_t i = size/16*16; i < size; ++i) to[i] = from[i];
    } else { EACH_INDEX(size) to[i] = from[i]; }
}
struct copy_bytes_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { copy_bytes(index,args...); }
};
inline cudaError_t copy_async(void* to, const void* from, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    if (BatchQueue::current && kind == cudaMemcpyDeviceToDevice) {
        launch(blocks((bytes+15)/16),256,0,stream,
            copy_bytes_launch{},
            static_cast<unsigned char*>(to),static_cast<const unsigned char*>(from),bytes);
        return cudaSuccess;
    }
    if (BatchQueue::current && kind == cudaMemcpyDeviceToHost) BatchQueue::current->flush();
    return cudaMemcpyAsync(to,from,bytes,kind,stream);
}
__device__ inline void fill_bytes(KernelIndex vcr_launch, unsigned char* to, unsigned char value, std::size_t size) {
    const unsigned word = unsigned(value)*0x01010101u;
    if (reinterpret_cast<std::uintptr_t>(to) % 16 == 0) {
        EACH_INDEX(size/16) reinterpret_cast<uint4*>(to)[i] = make_uint4(word,word,word,word);
        if (!vcr_launch.block && !threadIdx.x)
            for (std::size_t i = size/16*16; i < size; ++i) to[i] = value;
    } else { EACH_INDEX(size) to[i] = value; }
}
struct fill_bytes_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { fill_bytes(index,args...); }
};
inline cudaError_t memset_async(void* to, int value, std::size_t bytes, cudaStream_t stream) {
    if (!BatchQueue::current) return cudaMemsetAsync(to,value,bytes,stream);
    launch(blocks((bytes+15)/16),256,0,stream,
        fill_bytes_launch{},
        static_cast<unsigned char*>(to),static_cast<unsigned char>(value),bytes);
    return cudaSuccess;
}
}
