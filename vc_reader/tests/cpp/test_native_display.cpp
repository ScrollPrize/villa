#include <vc_reader/display.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
int main() {
    using namespace vcr::display;
    std::array<float,8> values{-1,0,1.9f,254.9f,256,NAN,INFINITY,-INFINITY};
    std::array<uint8_t,8> bytes{},coverage{};
    toUint8(values.data(),8,bytes.data(),coverage.data());
    CHECK((bytes==std::array<uint8_t,8>{0,0,1,254,255,0,0,0}));
    CHECK((coverage==std::array<uint8_t,8>{1,1,1,1,1,0,0,0}));
    std::array<uint32_t,256> ramp{},palette{},result{};
    windowLevelLut(ramp,10,110);
    CHECK(ramp[10]==0xff000000u && ramp[110]==0xffffffffu && ramp[60]==0xff7f7f7fu);
    for(int i=0;i<256;++i)palette[i]=0xff000000u|uint32_t(i);
    paletteLut(result,palette,10,110);
    CHECK(result[60]==palette[127] && result[0]==0xff000000u);
    CHECK(blendArgb(0xff204060u,0x80402010u,1)==0xff504040u);
#ifdef VC_READER_CUDA
    int count=0;
    if(cudaGetDeviceCount(&count)==cudaSuccess && count) {
        std::array<uint32_t,256> gpu{};
        for(float lo : {0.f,10.f,255.f}) for(float hi : {0.f,110.f,255.f}) {
            windowLevelLut(ramp,lo,hi); cuda::windowLevelLut(gpu,lo,hi); CHECK(ramp==gpu);
            paletteLut(result,palette,lo,hi);cuda::paletteLut(gpu,palette,lo,hi);CHECK(result==gpu);
            for(bool intensity : {false,true}) for(bool premul : {false,true}) {
                overlayLut(result,palette,lo,hi,intensity,premul);
                cuda::overlayLut(gpu,palette,lo,hi,intensity,premul);CHECK(result==gpu);
            }
        }
        float* input=nullptr;uint8_t *output=nullptr,*mask=nullptr;
        CHECK(cudaMalloc(&input,sizeof(values))==cudaSuccess);
        CHECK(cudaMalloc(&output,8)==cudaSuccess);CHECK(cudaMalloc(&mask,8)==cudaSuccess);
        cudaMemcpy(input,values.data(),sizeof(values),cudaMemcpyHostToDevice);
        cuda::toUint8(input,8,output,mask);
        std::array<uint8_t,8> actual{},valid{};
        cudaMemcpy(actual.data(),output,8,cudaMemcpyDeviceToHost);
        cudaMemcpy(valid.data(),mask,8,cudaMemcpyDeviceToHost);
        CHECK(actual==bytes && valid==coverage);
        cudaFree(input);cudaFree(output);cudaFree(mask);
        std::cout << "CPU/CUDA display parity passed\n";
    }
#endif
}
