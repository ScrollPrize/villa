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
    // Labels against a flood fill: same partition, numbered by first voxel in scan order.
    for(int connectivity:{4,8,6,18,26})for(int trial=0;trial<12;++trial) {
        const bool three=connectivity!=4&&connectivity!=8;
        std::vector<std::size_t> shape{std::size_t(1+random()%17),std::size_t(1+random()%70)};
        if(three)shape.insert(shape.begin(),1+random()%9);
        std::size_t n=1;for(auto d:shape)n*=d;
        std::vector<std::uint8_t> mask(n);const auto density=1+random()%4;for(auto& v:mask)v=random()%5<density;
        const std::size_t nz=three?shape[0]:1,ny=shape[three],nx=shape[three+1];
        std::vector<std::int32_t> expected(n,0);std::int32_t count=0;std::vector<std::size_t> stack;
        for(std::size_t start=0;start<n;++start)if(mask[start]&&!expected[start]) {
            expected[start]=++count;stack.push_back(start);
            while(!stack.empty()) {
                const auto i=stack.back();stack.pop_back();
                const long long x=i%nx,y=i/nx%ny,z=i/(nx*ny);
                for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                    const int axes=(dz!=0)+(dy!=0)+(dx!=0),reach=connectivity==4||connectivity==6?1:connectivity==18||connectivity==8?2:3;
                    if(!axes||axes>reach||(!three&&dz))continue;
                    const long long zz=z+dz,yy=y+dy,xx=x+dx;
                    if(zz<0||yy<0||xx<0||zz>=(long long)nz||yy>=(long long)ny||xx>=(long long)nx)continue;
                    const auto j=std::size_t((zz*(long long)ny+yy)*(long long)nx+xx);
                    if(mask[j]&&!expected[j]){expected[j]=count;stack.push_back(j);}
                }
            }
        }
        for(std::size_t threads:{1,3,0}) {
            std::vector<std::int32_t> labels(n,-1);
            CHECK(vcr::components::label(mask.data(),shape,connectivity,labels.data(),threads)==std::size_t(count));CHECK(labels==expected);
        }
#ifdef VC_READER_CUDA
        if(gpu) {
            std::uint8_t* deviceMask=nullptr;std::int32_t* deviceLabels=nullptr;std::vector<std::int32_t> labels(n);
            CHECK(cudaMalloc(&deviceMask,n)==cudaSuccess&&cudaMalloc(&deviceLabels,n*4)==cudaSuccess);
            CHECK(cudaMemcpy(deviceMask,mask.data(),n,cudaMemcpyHostToDevice)==cudaSuccess);
            CHECK(vcr::components::cuda::label(deviceMask,shape,connectivity,deviceLabels)==std::size_t(count));
            CHECK(cudaMemcpy(labels.data(),deviceLabels,n*4,cudaMemcpyDeviceToHost)==cudaSuccess);CHECK(labels==expected);
            cudaFree(deviceMask);cudaFree(deviceLabels);
        }
#endif
    }
    // 3D connectivity codes on a planar mask use their in-plane neighbours.
    // The two diagonal pixels distinguish 6 -> 4 from 18/26 -> 8.
    for(int connectivity:{6,18,26}) {
        const std::uint8_t mask[]{1,0,0,1};std::int32_t labels[4]{};
        CHECK(vcr::components::label(mask,std::vector<std::size_t>{2,2},connectivity,labels)==
              std::size_t(connectivity==6?2:1));
        CHECK(labels[0]==1 && labels[1]==0 && labels[2]==0 && labels[3]==(connectivity==6?2:1));
    }
    rejects([]{std::uint8_t m=1;std::int32_t l;vcr::components::label(&m,std::vector<std::size_t>{1,1},5,&l);});
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
