#include <vc_object_store/geometry/SurfaceGrid.hpp>
#include "vc_object_store/Tiff.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
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
SurfaceGrid readSurfaceGrid(const std::filesystem::path& path){
    std::ifstream file(path/"meta.json");auto metadata=nlohmann::json::parse(file);
    auto x=band(path/"x.tif"),y=band(path/"y.tif"),z=band(path/"z.tif");
    if(x.size()!=y.size()||x.size()!=z.size())throw std::runtime_error("tifxyz dimensions differ");
    cv::Mat_<cv::Vec3f> points(x.size());
    for(int r=0;r<x.rows;++r)for(int c=0;c<x.cols;++c){
        points(r,c)={float(x(r,c)),float(y(r,c)),float(z(r,c))};
        if(!vc::geometry::validSurfacePoint({points(r,c)[0],points(r,c)[1],points(r,c)[2]}))points(r,c)={-1,-1,-1};
    }
    return SurfaceGrid(points,{metadata.at("scale").at(0).get<float>(),metadata.at("scale").at(1).get<float>()});
}
std::optional<std::vector<uint8_t>> readSurfaceValidity(const std::filesystem::path& path,int rows,int cols){
    if(!std::filesystem::exists(path/"mask.tif"))return std::nullopt;
    // Page 0, first sample; a later render page is not validity.
    const auto mask=band(path/"mask.tif");
    try {
        return vc::geometry::validityFromMask({mask.ptr<double>(),mask.total()},mask.rows,mask.cols,rows,cols);
    } catch(const std::invalid_argument&) {
        throw std::runtime_error("mask.tif must be a whole multiple of the points grid: "+path.string());
    }
}
}
