#include "common.hpp"
#include "rules.hpp"
#include "../cuda_buffer.hpp"
#include <vc_reader/distance_transform.hpp>
namespace vcr::thinning {
namespace {
unsigned blocks(std::size_t n) {return unsigned(std::min<std::size_t>((n+255)/256,65535));}
__global__ void binarize(const std::uint8_t* input,std::uint8_t* output,std::size_t n) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        output[i]=input[i]?255:0;
}
__global__ void subpass(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols,
                        std::size_t n,int phase,Algorithm algorithm,bool preserveBorder,int* changed) {
    bool any=false;
    const auto plane=rows*cols;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x) {
        const auto p=i%plane,y=p/cols,x=p%cols;
        auto value=input[i];
        if(value && !(preserveBorder && (!y||!x||y+1==rows||x+1==cols)) &&
           detail::remove(detail::neighbors(input+(i-p),p,rows,cols),phase,algorithm)) {value=0;any=true;}
        output[i]=value;
    }
    if(__syncthreads_or(any) && threadIdx.x==0) atomicExch(changed,1);
}
}
namespace cuda {
void skeletonDevice(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols,
                    std::size_t slices,Algorithm algorithm,bool preserveBorder) {
    const auto n=detail::validate(input,rows,cols,slices);detail::validateAlgorithm(algorithm);
    if(n && !output) throw std::invalid_argument("null skeleton output");
    if(!n) return;
    int gpu=0;device::check(cudaGetDevice(&gpu));
    device::Buffer scratch(n,gpu),flag(sizeof(int),gpu);
    auto* tmp=static_cast<std::uint8_t*>(scratch.ptr);auto* changed=static_cast<int*>(flag.ptr);
    const auto stream=device::stream();
    binarize<<<blocks(n),256,0,stream>>>(input,output,n);device::check(cudaGetLastError());
    for(int stage=0;stage<(algorithm==Algorithm::ZhangSuenGuoHall?2:1);++stage) {
        const auto method=algorithm==Algorithm::ZhangSuenGuoHall ? (stage?Algorithm::GuoHall:Algorithm::ZhangSuen) : algorithm;
        int any=1;
        while(any) {
            device::check(cudaMemsetAsync(changed,0,sizeof(int),stream));
            // Accumulate changes across four full iterations to amortize host
            // synchronization. An extra stable iteration cannot change output.
            for(int iteration=0;iteration<4;++iteration) {
                subpass<<<blocks(n),256,0,stream>>>(output,tmp,rows,cols,n,0,method,preserveBorder,changed);
                subpass<<<blocks(n),256,0,stream>>>(tmp,output,rows,cols,n,1,method,preserveBorder,changed);
            }
            device::check(cudaGetLastError());
            device::check(cudaMemcpyAsync(&any,changed,sizeof(int),cudaMemcpyDeviceToHost,stream));device::complete();
        }
    }
}
void skeleton(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols,
              std::size_t slices,Algorithm algorithm,bool preserveBorder) {
    const auto n=detail::validate(input,rows,cols,slices);detail::validateAlgorithm(algorithm);
    if(n && !output) throw std::invalid_argument("null skeleton output");
    if(!n) return;
    int gpu=0;device::check(cudaGetDevice(&gpu));device::Buffer data(n,gpu);
    auto* ptr=static_cast<std::uint8_t*>(data.ptr);const auto stream=device::stream();
    device::check(cudaMemcpyAsync(ptr,input,n,cudaMemcpyHostToDevice,stream));
    skeletonDevice(ptr,ptr,rows,cols,slices,algorithm,preserveBorder);
    device::check(cudaMemcpyAsync(output,ptr,n,cudaMemcpyDeviceToHost,stream));device::complete();
}
}
}
