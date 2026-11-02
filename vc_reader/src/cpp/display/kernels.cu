#include "pixels.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::display::cuda {
namespace {
__global__ void convert(const float* in,std::size_t n,uint8_t* out,uint8_t* coverage) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x) {
        out[i]=detail::byte(in[i]);if(coverage) coverage[i]=detail::finite(in[i]);
    }
}
__global__ void apply(const uint8_t* in,std::size_t rows,std::size_t cols,std::size_t stride,const uint32_t* lut,uint32_t* out,std::size_t outStride) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*cols;i+=std::size_t(gridDim.x)*blockDim.x)
        out[(i/cols)*outStride+i%cols]=lut[in[(i/cols)*stride+i%cols]];
}
__global__ void mixing(const uint32_t* base,const uint32_t* over,std::size_t n,float opacity,uint32_t* out) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        out[i]=detail::mix(base[i],over[i],opacity);
}
__global__ void makeLut(uint32_t* out,const uint32_t* in,float low,float high,int kind,bool flag,bool premultiplied) {
    const int i=threadIdx.x;
    out[i]=kind==0?detail::window(i,low,high):kind==1?detail::palette(i,in,low,high,flag):detail::overlay(i,in[i],low,high,flag,premultiplied);
}
unsigned blocks(std::size_t n) {return unsigned(std::min<std::size_t>((n+255)/256,65535));}
void finish() { device::check(cudaGetLastError()); device::complete(); }
void lut(std::array<uint32_t,256>& out,const uint32_t* in,float low,float high,int kind,bool flag,bool premultiplied) {
    int gpu;device::check(cudaGetDevice(&gpu));device::Buffer dest(1024,gpu),src(1024,gpu);
    if(in) device::check(cudaMemcpyAsync(src.ptr,in,1024,cudaMemcpyHostToDevice,device::stream()));
    makeLut<<<1,256,0,device::stream()>>>(static_cast<uint32_t*>(dest.ptr),static_cast<uint32_t*>(src.ptr),low,high,kind,flag,premultiplied);
    device::check(cudaGetLastError());
    device::check(cudaMemcpyAsync(out.data(),dest.ptr,1024,cudaMemcpyDeviceToHost,device::stream()));device::complete();
}
}
void toUint8(const float* in,std::size_t n,uint8_t* out,uint8_t* coverage) {
    if(!n)return;convert<<<blocks(n),256,0,device::stream()>>>(in,n,out,coverage);finish();
}
void applyLut(const uint8_t* in,std::size_t rows,std::size_t cols,std::size_t stride,const uint32_t* lut,uint32_t* out,std::size_t outStride) {
    if(!rows||!cols)return;apply<<<blocks(rows*cols),256,0,device::stream()>>>(in,rows,cols,stride,lut,out,outStride);finish();
}
void blend(const uint32_t* base,const uint32_t* over,std::size_t n,float opacity,uint32_t* out) {
    if(!n)return;mixing<<<blocks(n),256,0,device::stream()>>>(base,over,n,opacity,out);finish();
}
void windowLevelLut(std::array<uint32_t,256>& out,float low,float high) {lut(out,nullptr,low,high,0,false,false);}
void paletteLut(std::array<uint32_t,256>& out,const std::array<uint32_t,256>& in,float low,float high,bool blackZero) {lut(out,in.data(),low,high,1,blackZero,false);}
void overlayLut(std::array<uint32_t,256>& out,const std::array<uint32_t,256>& in,float low,float high,bool intensity,bool premultiplied) {lut(out,in.data(),low,high,2,intensity,premultiplied);}
}
