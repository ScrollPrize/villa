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
    CHECK((bytes==std::array<uint8_t,8>{0,0,2,255,255,0,0,0}));
    CHECK((coverage==std::array<uint8_t,8>{1,1,1,1,1,0,0,0}));
    // Halfway values round to even, without changing saturation or explicit policies.
    const std::array<float,10> ties{-1,.5f,1.5f,2.5f,20.6f,254.5f,255.5f,65535,NAN,INFINITY};
    std::array<uint8_t,10> rounded{};
    std::array<uint16_t,10> wide{};
    toUint8(ties.data(), ties.size(), rounded.data());
    CHECK((rounded==std::array<uint8_t,10>{0,0,2,2,21,254,255,255,0,0}));
    toUint16(ties.data(), ties.size(), wide.data());
    CHECK((wide==std::array<uint16_t,10>{0,0,2,2,21,254,256,65535,0,65535}));
    toUint8(ties.data(), ties.size(), rounded.data(), nullptr, {1,Rounding::Truncate,true});
    CHECK((rounded==std::array<uint8_t,10>{0,0,1,2,20,254,255,255,0,255}));
    toUint8(ties.data(), ties.size(), rounded.data(), nullptr, {1,Rounding::HalfUp,true});
    CHECK((rounded==std::array<uint8_t,10>{0,1,2,3,21,255,255,255,0,255}));
    const std::array<float,3> nearTie{std::nextafter(.5f,0.f), .5f, std::nextafter(.5f,1.f)};
    std::array<uint8_t,3> nearBytes{};
    toUint8(nearTie.data(), nearTie.size(), nearBytes.data());
    CHECK((nearBytes==std::array<uint8_t,3>{0,0,1}));
    std::array<uint32_t,256> ramp{},palette{},result{};
    windowLevelLut(ramp,10,110);
    CHECK(ramp[10]==0xff000000u && ramp[110]==0xffffffffu && ramp[60]==0xff808080u);
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
    CHECK(result[60]==palette[128] && result[0]==0xff000000u);
    CHECK(blendArgb(0xff204060u,0x80402010u,1)==0xff504040u);
    // An opaque overlay mixes by opacity per channel, rounded; a transparent one keeps the base.
    CHECK(blendArgb(0xff204060u,0xffc08010u,0.3f)==0xff505348u);
    CHECK(blendArgb(0xff204060u,0u,0.3f)==0xff204060u);
    CHECK(blendArgb(0xff000000u,0xff010305u,0.5f)==0xff000202u);
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
