// Standalone gprof driver: the same channel implementations used by Pipeline.
// Read a normalized float32 cubic crop exported by bench_augment_cpu.py.
#include "augment/defocus.hpp"
#include "augment/ringing_overshoot.hpp"
#include "augment/bias_field.hpp"
#include "augment/plasma_shadow.hpp"
#include "augment/gibbs_ringing.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    using namespace vcr::augment;
    if(argc!=5){std::cerr<<"profile_cpu_augment crop.f32 size transform repetitions\n";return 2;}
    const std::size_t n=std::stoul(argv[2]);Grid g({n,n,n});
    std::vector<float> input(g.size);std::ifstream file(argv[1],std::ios::binary);
    if(!file.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float)))return 2;
    const std::string name=argv[3];
    for(int repeat=0;repeat<std::stoi(argv[4]);++repeat){
        std::vector<double> v(input.begin(),input.end());ChannelPlan p;p.seed=123;
        if(name=="Defocus"){p.values={2};channel_apply(Defocus{},v,g,p);}
        else if(name=="RingingOvershoot"){p.values={.7,4};channel_apply(RingingOvershoot{},v,g,p);}
        else if(name=="BiasField"){p.values={.3,.025};channel_apply(BiasField{},v,g,p);}
        else if(name=="PlasmaShadow"){p.values={.5,.025,.5};channel_apply(PlasmaShadow{},v,g,p);}
        else if(name=="GibbsRinging"){p.values={.7};channel_apply(GibbsRinging{},v,g,p);}
        else return 2;
        std::cout<<v[g.size/2]<<'\n';
    }
}
