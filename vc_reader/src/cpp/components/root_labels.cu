#include "common.hpp"
#include "../transforms/cuda/components.cuh"
namespace vcr::components::cuda {
namespace tc=transforms::cuda;
namespace {
__global__ void exportLabels(const tc::ComponentIndex* parents,std::uint32_t* output,std::size_t n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        output[i]=parents[i]==n?0:std::uint32_t(parents[i]+1);
}
}
void labelDevice(const std::uint8_t* input,std::uint32_t* output,const std::vector<std::size_t>& shape,bool full) {
    const auto n=detail::validate(input,output,shape);if(!n)return;
    if(tc::BatchQueue::current)tc::BatchQueue::current->flush();
    tc::Buffer<tc::ComponentIndex> parents(n);
    tc::label_components<false>(input,transforms::Grid(shape),0,full,parents);
    if(tc::BatchQueue::current)tc::BatchQueue::current->flush();
    exportLabels<<<tc::blocks(n),256,0,device::stream()>>>(parents.ptr,output,n);
    tc::check(cudaGetLastError());tc::check(cudaStreamSynchronize(device::stream()));
}
void label(const std::uint8_t* input,std::uint32_t* output,const std::vector<std::size_t>& shape,bool full) {
    const auto n=detail::validate(input,output,shape);if(!n)return;
    tc::Buffer<std::uint8_t> data(n);tc::Buffer<std::uint32_t> labels(n);
    data.upload(input);labelDevice(data.ptr,labels.ptr,shape,full);labels.download(output);
}
}
