#include "vc_object_store/objects/FiberAxialPriors.hpp"
#include "vc_object_store/objects/UmbilicusFile.hpp"
#include "vc_object_store/geometry/FiberAxialAlignment.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace vc::objects {
std::unique_ptr<float[]> fiberAxialPriorColumns(std::span<const int64_t> offsets,
    std::span<const double> xyz,const std::filesystem::path& path,
    const std::string& volume,const std::optional<std::array<int64_t,3>>& requestedShape,
    double spacing,size_t workers)
{
    if(offsets.empty()||offsets.front()!=0||xyz.size()%3||offsets.back()!=int64_t(xyz.size()/3)||!workers)
        throw std::invalid_argument("invalid packed axial-prior geometry");
    const auto file=readUmbilicusFile(path);
    if(file.volume&&*file.volume!=volume&&!file.volume->starts_with(volume+"-"))
        throw std::invalid_argument("umbilicus/fiber volume coordinate frame mismatch");
    std::array<int64_t,3> shape;
    if(requestedShape)shape=*requestedShape;
    else{
        if(!file.volumeSlices||!file.volumeHeight||!file.volumeWidth)
            throw std::invalid_argument("axial priors require an explicit base shape or stamped umbilicus dimensions");
        shape={*file.volumeSlices,*file.volumeHeight,*file.volumeWidth};
    }
    for(const auto value:shape)if(value<=0||value>std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid axial-prior volume dimensions");
    const auto binding=umbilicusBinding(file,{{double(shape[2]),double(shape[1]),double(shape[0])},std::nullopt},std::nullopt);
    if(binding.refusal)throw std::invalid_argument("umbilicus refused: "+*binding.refusal);
    if(binding.confirm||!binding.storedUnitsPerVoxel)
        throw std::invalid_argument("umbilicus file does not state a verified coordinate frame");
    std::vector<cv::Vec3f> control;control.reserve(file.points.size());
    for(const auto& p:file.points)
        control.emplace_back(cv::Vec3f(float(p[0]),float(p[1]),float(p[2]))/float(*binding.storedUnitsPerVoxel));
    std::sort(control.begin(),control.end(),[](const auto& a,const auto& b){return a[2]<b[2];});
    std::vector<cv::Vec3f> centers(static_cast<size_t>(shape[0]));
    for(size_t z=0;z<centers.size();++z)centers[z]=vc::geometry::umbilicusCenter(control,double(z));
    const auto axisAt=[&](const cv::Vec3d& p){
        const int z=std::clamp(int(std::lround(p[2])),0,int(centers.size())-1);
        const int lo=std::max(0,z-8),hi=std::min(int(centers.size())-1,z+8);
        return cv::Vec3d(centers[hi]-centers[lo]);
    };
    struct Points {
        std::span<const double> values;
        size_t size()const{return values.size()/3;}
        cv::Vec3d operator[](size_t i)const{const auto* p=values.data()+3*i;return {p[0],p[1],p[2]};}
    };
    const size_t n=offsets.size()-1;
    auto output=std::make_unique_for_overwrite<float[]>(n);
    parallelFor((n+63)/64,workers,[&](size_t task){
        for(size_t i=task*64;i<std::min(n,(task+1)*64);++i){
            const auto begin=offsets[i],end=offsets[i+1];
            if(begin<0||end<begin||uint64_t(end)>xyz.size()/3||end-begin<2)
                throw std::invalid_argument("invalid axial-prior fiber interval");
            output[i]=float(vc::geometry::fiberAxialAlignment(Points{xyz.subspan(3*begin,3*(end-begin))},axisAt,
                [](const auto& p){return cv::norm(p);},spacing));
        }
    });return output;
}
} // namespace vc::objects
