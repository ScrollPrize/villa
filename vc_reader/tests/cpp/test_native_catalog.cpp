#include "vc_reader/catalog.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
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
        // A registration file as the CLI tools read it.
        const auto file=std::filesystem::temp_directory_path()/"vcr_test_transform.json";
        std::ofstream(file)<<R"({"transformation_matrix":[[2,0,0,1],[0,2,0,2],[0,0,2,3],[0,0,0,1]]})";
        const auto loaded=Affine::loadTransformJson(file);
        std::filesystem::remove(file);
        require(loaded.apply({1,1,1})==std::array<double,3>{3,4,5},"transform.json 4x4 rows");
        require(Affine::fromTransformJson({{"transformation_matrix",{{1,0,0,0},{0,1,0,0},{0,0,1,0}}}})==Affine{},"3x4 rows");
        for(const auto* bad:{R"({"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]})",
                             R"({"transformation_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]]})",
                             R"({"transformation_matrix":[[1,0,0,0],[1,0,0,0],[0,0,1,0]]})"})
            try{(void)Affine::fromTransformJson(nlohmann::json::parse(bad));throw std::runtime_error(std::string("accepted ")+bad);}
            catch(const std::invalid_argument&){}

        // Attributes between frames. A uniform scale keeps directions and
        // normals exactly and scales lengths by its factor.
        const std::array<double,3> skew{0.3,-0.4,0.5};
        require(loaded.isUniformScale()&&loaded.lengthScale()==2,"uniform scale");
        require(loaded.mapDirection(skew)==skew&&loaded.mapNormal(skew)==skew,"uniform scale keeps attributes");
        // The published registration: a rotation and scale with column lengths
        // about 0.4% apart, a similarity only at a loose tolerance.
        const auto& published=result.resolved().forward;
        require(!published.isUniformScale()&&!published.similarityScale(1e-3),"published registration is not exactly similar");
        const auto similarity=published.similarityScale(0.04);
        require(similarity&&std::abs(*similarity-published.lengthScale())<0.01**similarity,"near-similarity scale");
        // A shear: a mapped normal stays perpendicular to mapped tangents.
        Affine shear;shear(0,1)=0.5;shear(2,0)=-0.25;shear(1,1)=3;
        require(!shear.similarityScale()&&std::abs(shear.lengthScale()-std::cbrt(3.))<1e-15,"shear has no similarity scale");
        const std::array<double,3> n{0,0,1},t1{1,0,0},t2{0,1,0};
        const auto mn=shear.mapNormal(n),m1=shear.mapDirection(t1),m2=shear.mapDirection(t2);
        const auto dot=[](const auto& a,const auto& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
        require(std::abs(dot(mn,m1))<1e-15&&std::abs(dot(mn,m2))<1e-15,"normals map by the inverse transpose");
        require(std::abs(dot(mn,mn)-1)<1e-15&&std::abs(dot(m1,m1)-1)<1e-15,"mapped attributes are unit length");
        const auto t=shear.normalTransform();
        for(int i=0;i<3;++i)require(std::abs(shear.mapNormal(skew)[i]-t.mapDirection(skew)[i])<1e-15,"normal transform");
        std::cout<<"native catalog and registration checks passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
