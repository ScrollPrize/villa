#include "spatial/PatchGrid.hpp"
#include "vc_object_store/Tiff.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <limits>
namespace vc::object_links {
namespace {
template<class T> double scalar(const uint8_t* p){T v;std::memcpy(&v,p,sizeof v);return double(v);}
cv::Mat_<double> band(const std::filesystem::path& path){
    const auto samples = readTiffSamples(path, false);
    const auto bits = samples.bits, fmt = samples.sampleFormat;
    if (bits == 64 && fmt != SAMPLEFORMAT_IEEEFP)
        throw std::runtime_error("unsupported TIFF samples: " + path.string());
    auto value = [&](const uint8_t* p) -> double {
        if (fmt == SAMPLEFORMAT_IEEEFP) return bits == 32 ? scalar<float>(p) : scalar<double>(p);
        if (fmt == SAMPLEFORMAT_UINT) {
            if (bits == 8) return scalar<uint8_t>(p); if (bits == 16) return scalar<uint16_t>(p);
            return scalar<uint32_t>(p);
        }
        if (bits == 8) return scalar<int8_t>(p); if (bits == 16) return scalar<int16_t>(p);
        return scalar<int32_t>(p);
    };
    cv::Mat_<double> output(int(samples.height), int(samples.width), 0.0);
    const size_t stride = (bits/8)*samples.components;
    for (int y = 0; y < output.rows; ++y) for (int x = 0; x < output.cols; ++x)
        output(y,x) = value(samples.bytes.data() + (size_t(y)*samples.width+x)*stride);
    return output;
}
}
PatchGrid readPatchGrid(const std::filesystem::path& path){
    std::ifstream file(path/"meta.json");auto metadata=nlohmann::json::parse(file);
    auto x=band(path/"x.tif"),y=band(path/"y.tif"),z=band(path/"z.tif");
    if(x.size()!=y.size()||x.size()!=z.size())throw std::runtime_error("tifxyz dimensions differ");
    cv::Mat_<cv::Vec3f> points(x.size());
    for(int r=0;r<x.rows;++r)for(int c=0;c<x.cols;++c){
        points(r,c)={float(x(r,c)),float(y(r,c)),float(z(r,c))};
        if(points(r,c)[2]<=0)points(r,c)={-1,-1,-1};
    }
    if(std::filesystem::exists(path/"mask.tif")){
        auto mask=band(path/"mask.tif");
        if(mask.rows%x.rows==0&&mask.cols%x.cols==0){
            int sy=mask.rows/x.rows,sx=mask.cols/x.cols;
            for(int r=0;r<mask.rows;++r)for(int c=0;c<mask.cols;++c)if(!(mask(r,c)>=255))points(r/sy,c/sx)={-1,-1,-1};
        }
    }
    return PatchGrid(points,{metadata.at("scale").at(0).get<float>(),metadata.at("scale").at(1).get<float>()});
}
void writePatchGrid(const std::filesystem::path& path,const PatchGrid& grid){
    const auto& points=grid.rawPoints();
    for(int a=0;a<3;++a){cv::Mat_<float> plane(points.size());
        for(int r=0;r<points.rows;++r)for(int c=0;c<points.cols;++c)plane(r,c)=points(r,c)[a];
        writeTiff(path/(std::string(1,"xyz"[a])+".tif"),plane);
    }
}
}
