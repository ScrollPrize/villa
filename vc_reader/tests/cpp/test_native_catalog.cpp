#include "vc_reader/catalog.hpp"
#include <iostream>
#include <stdexcept>
using namespace vcr::catalog;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(int argc,char** argv) {
    try {
        require(argc==2,"fixture argument");
        auto catalog=loadOpenDataManifestFile(argv[1]);
        const auto* sample=catalog.findSample("PHerc0009B");require(sample,"real sample");
        RegistrationGraph graph;graph.addSample(*sample);
        const auto result=graph.resolve("20250820154339","20250521125136");
        require(result.status==TransformStatus::Resolved,"published registration");
        require(result.resolved().forward(0,0)==0.2776774047309435,"published coefficients retained exactly");
        require(!sample->segments.empty()&&!sample->volumes.empty(),"native browsing");
        for(const auto& v:sample->volumes)for(const auto& a:v.artifacts)
            if(a.hasResolvedUrl())require(a.resolvedUrl.starts_with("https://"),"public HTTPS resolution");
        RegistrationGraph custom;
        Affine a; a(0,3)=10;a(0,1)=0.25;
        custom.add({"a-b","1","a","b",a,"test"});
        custom.add({"b-c","1","b","c",Affine::scale(2),"test"});
        auto c=custom.resolve("a","c").resolved();
        auto point=std::array<double,3>{3,4,5};
        auto roundtrip=c.inverse.apply(c.forward.apply(point));
        require(roundtrip==point,"affine inverse/composition");
        custom.add({"alternate","1","a","c",Affine::scale(3),"test"});
        auto conflict=custom.resolve("a","c");require(conflict.status==TransformStatus::Ambiguous,"conflict detected");
        require(custom.resolve("a","c",c.path).status==TransformStatus::Resolved,"explicit choice");
        auto stale=c.path;stale[0]+="changed";
        require(custom.resolve("a","c",stale).status==TransformStatus::Unavailable,"stale choice");
        require(custom.resolve("a","missing").status==TransformStatus::Unavailable,"missing path");
        try { auto invalid=Affine{};invalid(0,0)=0;(void)invalid.inverse();throw std::runtime_error("accepted singular"); }
        catch(const std::invalid_argument&){}
        std::cout<<"native catalog and registration checks passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
