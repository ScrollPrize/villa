// Sampling-profiler workload. File I/O and conversion happen once before the loop.
#include "fields.hpp"
#include <fstream>
#include <iostream>
using namespace vcr::elastic_experiment;
int main(int argc,char** argv) {
    if(argc!=5) {
        std::cerr << "usage: profile_elastic RAW_128_CHUNK dense|coarse|bspline SCALE ITERATIONS\n";
        return 2;
    }
    std::vector<unsigned char> raw(128*128*128);
    std::ifstream input(argv[1],std::ios::binary);
    if(!input.read(reinterpret_cast<char*>(raw.data()),raw.size())) return 2;
    Array image({1,128,128,128});
    for(std::size_t i=0;i<raw.size();++i) image.set(i,raw[i]/255.);
    const auto scale=std::stod(argv[3]); const auto reps=std::stoi(argv[4]);
    if(reps<1) return 2;
    double checksum=0;
    for(int i=0;i<reps;++i) {
        auto p=geometry({128,128,128});
        p.field=make_field(argv[2],128,scale,123);
        normalize(p.field,128,4,.8);
        auto result=resample(image,p,false,0);
        checksum+=result.get(64*128*128+64*128+64);
    }
    std::cout<<checksum<<'\n';
}
