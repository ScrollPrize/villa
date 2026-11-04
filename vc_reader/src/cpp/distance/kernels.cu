#include "common.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::distance::cuda {
namespace {
__global__ void initialize(const std::uint8_t* input,std::uint8_t* output,std::size_t n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        output[i]=input[i]?255:0;
}
__global__ void lines(std::uint8_t* output,std::size_t count,std::size_t extent,std::size_t stride) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(gridDim.x)*blockDim.x)
        detail::line(output+(i/stride)*stride*extent+i%stride,extent,stride);
}
}
void manhattanDevice(const std::uint8_t* input,std::uint8_t* output,const std::vector<std::size_t>& shape) {
    const auto n=detail::validate(input,output,shape);
    if(!n) return;
    const auto stream=device::stream();
    initialize<<<unsigned(std::min<std::size_t>((n+255)/256,65535)),256,0,stream>>>(input,output,n);
    device::check(cudaGetLastError());
    std::size_t stride=1;
    for(std::size_t a=shape.size();a-->0;) {
        const auto count=n/shape[a];
        lines<<<unsigned(std::min<std::size_t>((count+255)/256,65535)),256,0,stream>>>(output,count,shape[a],stride);
        device::check(cudaGetLastError());
        stride*=shape[a];
    }
    device::complete();
}
void manhattan(const std::uint8_t* input,std::uint8_t* output,const std::vector<std::size_t>& shape) {
    const auto n=detail::validate(input,output,shape);
    if(!n) return;
    int gpu=0;device::check(cudaGetDevice(&gpu));device::Buffer data(n,gpu);
    auto* ptr=static_cast<std::uint8_t*>(data.ptr);
    device::check(cudaMemcpyAsync(ptr,input,n,cudaMemcpyHostToDevice,device::stream()));
    manhattanDevice(ptr,ptr,shape);
    device::check(cudaMemcpyAsync(output,ptr,n,cudaMemcpyDeviceToHost,device::stream()));
    device::complete();
}
}
