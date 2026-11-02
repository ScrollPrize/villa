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
    static_assert(noexcept(windowLevelLut(ramp,0,0)));
    // Window edge cases: endpoints clamp into [0,255]; an empty or inverted
    // window becomes a step at low+1.
    windowLevelLut(ramp,-50,500);
    CHECK(ramp[0]==0xff000000u && ramp[128]==0xff808080u && ramp[255]==0xffffffffu);
    windowLevelLut(ramp,100,100);
    CHECK(ramp[100]==0xff000000u && ramp[101]==0xffffffffu);
    windowLevelLut(ramp,200,50);
    CHECK(ramp[200]==0xff000000u && ramp[201]==0xffffffffu);
    for(int i=0;i<256;++i)palette[i]=0xff000000u|uint32_t(i);
    paletteLut(result,palette,10,110);
    CHECK(result[60]==palette[127] && result[0]==0xff000000u);
    CHECK(blendArgb(0xff204060u,0x80402010u,1)==0xff504040u);
    // An opaque overlay mixes by opacity per channel, rounded; a transparent one keeps the base.
    CHECK(blendArgb(0xff204060u,0xffc08010u,0.3f)==0xff505348u);
    CHECK(blendArgb(0xff204060u,0u,0.3f)==0xff204060u);
    // Strided input; output padding past cols is left untouched.
    const std::array<uint8_t,6> pixels{1,2,99,3,4,99};
    std::array<uint32_t,8> mapped; mapped.fill(0xdeadbeefu);
    applyLut(pixels.data(),2,2,3,palette.data(),mapped.data(),4);
    CHECK((mapped==std::array<uint32_t,8>{palette[1],palette[2],0xdeadbeefu,0xdeadbeefu,
                                          palette[3],palette[4],0xdeadbeefu,0xdeadbeefu}));
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
