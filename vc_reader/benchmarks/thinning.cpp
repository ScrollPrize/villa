// Generated raw uint8 crops are supplied by bench_thinning.py (via ZarrReader).
#include <vc_reader/thinning.hpp>
#include <vc_reader/distance_transform.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <nlohmann/json.hpp>
#ifdef VCR_THINNING_REFERENCE_HEADER
#include VCR_THINNING_REFERENCE_HEADER
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#endif
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif
using Clock=std::chrono::steady_clock;
using namespace vcr::thinning;
using json=nlohmann::json;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
struct Measurements {
    std::map<std::string,std::vector<double>> times;
    template<class F> void run(const std::string& name,int repetitions,F f) {
        for(int i=-2;i<repetitions;++i) {
            const auto start=Clock::now();f();
            if(i>=0) times[name].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        }
    }
    json result() const {
        json result;
        for(const auto& [name,unsorted]:times) {
            auto v=unsorted;std::sort(v.begin(),v.end());
            result[name]={{"samples",v.size()},{"mean_ms",std::accumulate(v.begin(),v.end(),0.)/v.size()},
                          {"min_ms",v.front()},{"median_ms",v[v.size()/2]},
                          {"p95_ms",v[std::size_t(.95*(v.size()-1))]},{"max_ms",v.back()}};
        }
        return result;
    }
};
void checkReference(const std::uint8_t* input,std::size_t rows,std::size_t cols,bool gpu) {
    const auto n=rows*cols;std::vector<std::uint8_t> cpu(n),other(n);Scratch scratch;Traces paths,second;
    trace(input,rows,cols,cpu.data(),&paths,scratch);
#ifdef VCR_THINNING_REFERENCE_HEADER
    const cv::Mat image(int(rows),int(cols),CV_8UC1,const_cast<std::uint8_t*>(input));cv::Mat ref;
    std::vector<std::vector<cv::Point>> original;
    for(double length:{0.,3.,1000.}) {
        customThinning(image,ref,&original,nullptr,{length});
        trace(input,rows,cols,cpu.data(),&paths,scratch,nullptr,{length});
        CHECK(std::equal(cpu.begin(),cpu.end(),ref.data));CHECK(paths.size()==original.size());
        for(std::size_t j=0;j<paths.size();++j) {
            CHECK(paths[j].size()==original[j].size());
            for(std::size_t k=0;k<paths[j].size();++k)CHECK(paths[j][k].x==original[j][k].x && paths[j][k].y==original[j][k].y);
        }
#ifdef VC_READER_CUDA
        if(gpu){trace(input,rows,cols,other.data(),&second,scratch,nullptr,{length},Backend::Cuda);CHECK(cpu==other && paths==second);}
#endif
    }
    cv::distanceTransform(image,ref,1,cv::DIST_MASK_5,CV_8U);
    vcr::distance::manhattan(input,cpu.data(),{rows,cols});CHECK(std::equal(cpu.begin(),cpu.end(),ref.data));
#endif
    for(int m=0;m<3;++m) {
        const auto method=Algorithm(m);skeleton(input,cpu.data(),rows,cols,1,method);
#ifdef VCR_THINNING_REFERENCE_HEADER
        cv::ximgproc::thinning(image,ref,m==1?1:0);if(m==2)cv::ximgproc::thinning(ref,ref,1);
        CHECK(std::equal(cpu.begin(),cpu.end(),ref.data));
#endif
#ifdef VC_READER_CUDA
        if(gpu){cuda::skeleton(input,other.data(),rows,cols,1,method);CHECK(cpu==other);}
#endif
    }
}
int main(int argc,char** argv) try {
    if(argc<6) {std::cerr<<"Usage: bench_thinning crop.raw rows cols slices repetitions [cuda]\n";return 1;}
    const auto rows=std::stoull(argv[2]),cols=std::stoull(argv[3]),slices=std::stoull(argv[4]);
    const int repetitions=std::stoi(argv[5]);const bool gpu=argc>6;
    CHECK(rows && cols && slices && repetitions>0);
    CHECK(rows<=SIZE_MAX/cols && rows*cols<=SIZE_MAX/slices);
    const std::size_t n=rows*cols;
    std::vector<std::uint8_t> input(n*slices),out(n);
    std::ifstream file(argv[1],std::ios::binary);CHECK(file.read(reinterpret_cast<char*>(input.data()),input.size()));
#ifdef VCR_THINNING_REFERENCE_HEADER
    cv::setNumThreads(1);
#endif
#ifndef VC_READER_CUDA
    CHECK(!gpu);
#endif
    Measurements measurements;
    for(std::size_t z=0;z<slices;++z) {
        auto* data=input.data()+z*n;checkReference(data,rows,cols,gpu);
        measurements.run("custom_cpu",repetitions,[&]{Scratch scratch;trace(data,rows,cols,out.data(),nullptr,scratch);});
#ifdef VC_READER_CUDA
        if(gpu)measurements.run("custom_cuda_upload",repetitions,[&]{Scratch scratch;trace(data,rows,cols,out.data(),nullptr,scratch,nullptr,{},Backend::Cuda);});
#endif
#ifdef VCR_THINNING_REFERENCE_HEADER
        const cv::Mat image(int(rows),int(cols),CV_8UC1,data);cv::Mat ref;
        measurements.run("custom_original",repetitions,[&]{customThinning(image,ref);});
#endif
        Scratch traceScratch;Traces paths;
        measurements.run("polylines_cpu",repetitions,[&]{trace(data,rows,cols,nullptr,&paths,traceScratch);});
#ifdef VC_READER_CUDA
        if(gpu)measurements.run("polylines_cuda_upload",repetitions,[&]{trace(data,rows,cols,nullptr,&paths,traceScratch,nullptr,{},Backend::Cuda);});
        if(gpu) {
            std::uint8_t *deviceInput=nullptr,*deviceOutput=nullptr;
            CHECK(cudaMalloc(&deviceInput,n)==cudaSuccess);CHECK(cudaMalloc(&deviceOutput,n)==cudaSuccess);
            CHECK(cudaMemcpy(deviceInput,data,n,cudaMemcpyHostToDevice)==cudaSuccess);
            measurements.run("custom_cuda_resident",repetitions,[&]{cuda::traceDevice(deviceInput,deviceOutput,rows,cols);});
            CHECK(cudaMemcpy(out.data(),deviceOutput,n,cudaMemcpyDeviceToHost)==cudaSuccess);
            std::vector<std::uint8_t> expected(n);trace(data,rows,cols,expected.data(),nullptr,traceScratch);CHECK(out==expected);
            cudaFree(deviceInput);cudaFree(deviceOutput);
        }
#endif
#ifdef VCR_THINNING_REFERENCE_HEADER
        ThinningScratch originalScratch;std::vector<std::vector<cv::Point>> originalPaths;
        measurements.run("polylines_original",repetitions,[&]{customThinningTraceOnly(image,originalPaths,nullptr,originalScratch);});
#endif
        for(int m=0;m<3;++m) {
            const std::string name=m==0?"zhang_suen":m==1?"guo_hall":"combined";
            measurements.run(name+"_cpu",repetitions,[&]{skeleton(data,out.data(),rows,cols,1,Algorithm(m));});
#ifdef VC_READER_CUDA
            if(gpu) {
                measurements.run(name+"_cuda_upload",repetitions,[&]{cuda::skeleton(data,out.data(),rows,cols,1,Algorithm(m));});
                std::uint8_t *deviceInput=nullptr,*deviceOutput=nullptr;
                CHECK(cudaMalloc(&deviceInput,n)==cudaSuccess);CHECK(cudaMalloc(&deviceOutput,n)==cudaSuccess);
                CHECK(cudaMemcpy(deviceInput,data,n,cudaMemcpyHostToDevice)==cudaSuccess);
                measurements.run(name+"_cuda_resident",repetitions,[&]{cuda::skeletonDevice(deviceInput,deviceOutput,rows,cols,1,Algorithm(m));});
                CHECK(cudaMemcpy(out.data(),deviceOutput,n,cudaMemcpyDeviceToHost)==cudaSuccess);
                std::vector<std::uint8_t> expected(n);skeleton(data,expected.data(),rows,cols,1,Algorithm(m));CHECK(out==expected);
                cudaFree(deviceInput);cudaFree(deviceOutput);
            }
#endif
#ifdef VCR_THINNING_REFERENCE_HEADER
            measurements.run(name+"_opencv",repetitions,[&]{cv::ximgproc::thinning(image,ref,m==1?1:0);if(m==2)cv::ximgproc::thinning(ref,ref,1);});
#endif
        }
    }
    // Border/plateau/tie coverage against the same independent reference.
    std::mt19937 rng(91);
    for(int i=0;i<40;++i) {
        const std::size_t h=1+rng()%30,w=1+rng()%31;std::vector<std::uint8_t> mask(h*w);
        for(auto& v:mask)v=(i==0 || (i>1 && rng()%3))?255:0;
        checkReference(mask.data(),h,w,gpu);
    }
    json result={{"shape",{slices,rows,cols}},{"repetitions_per_slice",repetitions},{"warmups_per_slice",2},
                 {"exact_reference_match",true},{"timings",measurements.result()}};
#ifdef VCR_THINNING_REFERENCE_HEADER
    result["opencv_version"]=CV_VERSION;
#else
    result["exact_reference_match"]=nullptr;
#endif
    std::cout<<result.dump(2)<<'\n';
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
