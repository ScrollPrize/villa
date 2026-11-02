#include <vc_reader/thinning.hpp>
#include <vc_reader/distance_transform.hpp>
#include <vc_reader/components.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
using namespace vcr::thinning;
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}CHECK(caught);}
void verify(bool gpu) {
    std::mt19937 random(719);
    for(int dim:{2,3}) for(int trial=0;trial<20;++trial) {
        std::vector<std::size_t> shape{std::size_t(1+random()%8),std::size_t(1+random()%9)};
        if(dim==3)shape.insert(shape.begin(),1+random()%6);
        std::size_t n=1;for(auto d:shape)n*=d;
        std::vector<std::uint8_t> input(n),out(n),ref(n,255);
        for(auto& v:input)v=random()%4?255:0;
        if(trial==0)std::fill(input.begin(),input.end(),255);
        // Independent exhaustive nearest-background oracle.
        for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)if(!input[j]){
            auto p=i,q=j;unsigned dist=0;
            for(std::size_t a=shape.size();a-->0;){dist+=std::abs(int(p%shape[a])-int(q%shape[a]));p/=shape[a];q/=shape[a];}
            ref[i]=std::min<unsigned>(ref[i],dist);
        }
        vcr::distance::manhattan(input.data(),out.data(),shape);CHECK(out==ref);
#ifdef VC_READER_CUDA
        if(gpu){vcr::distance::cuda::manhattan(input.data(),out.data(),shape);CHECK(out==ref);}
#endif
    }
    for(bool full:{false,true})for(const auto& shape:std::vector<std::vector<std::size_t>>{{7,9},{5,7,9}}) {
        std::size_t n=1;for(auto d:shape)n*=d;
        std::vector<std::uint8_t> mask(n);for(auto& v:mask)v=random()%3?0:255;
        std::vector<std::uint32_t> cpu(n),other(n);
        vcr::components::label(mask.data(),cpu.data(),shape,full);
        for(std::size_t i=0;i<n;++i){CHECK(bool(cpu[i])==bool(mask[i]));if(cpu[i])CHECK(cpu[i]-1<=i && cpu[cpu[i]-1]==cpu[i]);}
#ifdef VC_READER_CUDA
        if(gpu){vcr::components::cuda::label(mask.data(),other.data(),shape,full);CHECK(cpu==other);}
#endif
    }
    std::vector<std::uint8_t> longLine(600,255),longOut(600);longLine[0]=0;
    vcr::distance::manhattan(longLine.data(),longOut.data(),{1,600});CHECK(longOut[254]==254 && longOut[255]==255 && longOut.back()==255);
    for(int trial=0;trial<45;++trial) {
        const auto h=std::size_t(1+random()%40),w=std::size_t(1+random()%41),n=h*w;
        std::vector<std::uint8_t> input(n*2),out(n*2),again(n*2),gpuOut(n*2);
        for(auto& v:input)v=random()%3?255:0;
        if(trial==0)std::fill(input.begin(),input.end(),0);
        if(trial==1)std::fill(input.begin(),input.end(),255);
        for(auto method:{Algorithm::ZhangSuen,Algorithm::GuoHall,Algorithm::ZhangSuenGuoHall})for(bool border:{false,true}){
            skeleton(input.data(),out.data(),h,w,2,method,border);
            skeleton(out.data(),again.data(),h,w,2,method,border);if(method!=Algorithm::ZhangSuenGuoHall)CHECK(out==again);
            for(std::size_t i=0;i<out.size();++i){CHECK(!out[i] || input[i]);if(border && (!(i%w)||i%w==w-1||((i/w)%h)==0||((i/w)%h)==h-1))CHECK(out[i]==input[i]);}
            for(std::size_t z=0;z<2;++z)skeleton(input.data()+z*n,again.data()+z*n,h,w,1,method,border);
            CHECK(out==again);
#ifdef VC_READER_CUDA
            if(gpu){cuda::skeleton(input.data(),gpuOut.data(),h,w,2,method,border);CHECK(out==gpuOut);}
#endif
        }
        Scratch scratch;Traces paths,repeat;Stats stats;
        trace(input.data(),h,w,out.data(),&paths,scratch,&stats);
        trace(input.data(),h,w,again.data(),&repeat,scratch);CHECK(paths==repeat);CHECK(std::equal(out.begin(),out.begin()+n,again.begin()));
        trace(input.data(),h,w,nullptr,&repeat,scratch);CHECK(paths==repeat);
        std::vector<std::uint8_t> padded(h*(w+5),0);
        for(std::size_t y=0;y<h;++y)std::copy_n(input.data()+y*w,w,padded.data()+y*(w+5));
        trace(padded.data(),h,w,again.data(),&repeat,scratch,nullptr,{},Backend::Cpu,w+5);CHECK(paths==repeat);
#ifdef VC_READER_CUDA
        if(gpu){trace(padded.data(),h,w,gpuOut.data(),&repeat,scratch,nullptr,{},Backend::Cuda,w+5);CHECK(paths==repeat);CHECK(std::equal(out.begin(),out.begin()+n,gpuOut.begin()));}
#endif
        trace(input.data(),h,w,out.data(),&paths,scratch,nullptr,{100000});CHECK(paths.empty());CHECK(std::all_of(out.begin(),out.begin()+n,[](auto v){return v==0;}));
    }
    Traces paths{{{0,0}},{{0,0},{3,4}},{{0,0},{1,0}}};Stats stats;
    prune(paths,{5},&stats);CHECK(paths.size()==1 && stats.tracesPruned==2 && stats.tracesKept==1);
    Stats sum;sum.accumulate(stats);sum.accumulate(stats);CHECK(sum.tracesPruned==4);
    Scratch scratch;trace(nullptr,0,10,nullptr,&paths,scratch);CHECK(paths.empty());
    rejects([&]{trace(nullptr,1,1,nullptr,&paths,scratch);});
    std::uint8_t pixel=1;
    rejects([&]{trace(&pixel,1,1,nullptr,nullptr,scratch,nullptr,{1});});
    rejects([&]{trace(&pixel,1,1,nullptr,&paths,scratch,nullptr,{-1});});
    rejects([&]{skeleton(&pixel,nullptr,1,1);});
    rejects([&]{skeleton(&pixel,&pixel,1,1,1,Algorithm(99));});
    rejects([&]{vcr::distance::manhattan(&pixel,&pixel,{1});});
    rejects([&]{prune(paths,{-1});});
}
int main(int argc,char**) {
    bool gpu=argc>1;
#ifdef VC_READER_CUDA
    if(gpu){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;}
#else
    if(gpu)return 77;
#endif
    verify(gpu);std::cout<<"thinning and Manhattan distance passed\n";
}
