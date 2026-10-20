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
    std::unique_ptr<TIFF,decltype(&TIFFClose)> tif(TIFFOpen(path.string().c_str(),"r"),TIFFClose);
    if(!tif)throw std::runtime_error("cannot read TIFF: "+path.string());
    uint32_t w=0,h=0;uint16_t bits=0,fmt=0,samples=1,planar=1;
    TIFFGetField(tif.get(),TIFFTAG_IMAGEWIDTH,&w);TIFFGetField(tif.get(),TIFFTAG_IMAGELENGTH,&h);
    TIFFGetFieldDefaulted(tif.get(),TIFFTAG_BITSPERSAMPLE,&bits);TIFFGetFieldDefaulted(tif.get(),TIFFTAG_SAMPLEFORMAT,&fmt);
    TIFFGetFieldDefaulted(tif.get(),TIFFTAG_SAMPLESPERPIXEL,&samples);TIFFGetFieldDefaulted(tif.get(),TIFFTAG_PLANARCONFIG,&planar);
    if(!w||!h||w>INT_MAX||h>INT_MAX||!samples||!((fmt==SAMPLEFORMAT_IEEEFP&&(bits==32||bits==64))||((fmt==SAMPLEFORMAT_UINT||fmt==SAMPLEFORMAT_INT)&&(bits==8||bits==16||bits==32))))
        throw std::runtime_error("unsupported TIFF samples: "+path.string());
    auto value=[&](const uint8_t* p){
        if(fmt==SAMPLEFORMAT_IEEEFP)return bits==32?scalar<float>(p):scalar<double>(p);
        if(fmt==SAMPLEFORMAT_UINT){if(bits==8)return scalar<uint8_t>(p);if(bits==16)return scalar<uint16_t>(p);return scalar<uint32_t>(p);}
        if(bits==8)return scalar<int8_t>(p);if(bits==16)return scalar<int16_t>(p);return scalar<int32_t>(p);
    };
    cv::Mat_<double> output(int(h),int(w),0.0);
    size_t stride=(bits/8)*(planar==PLANARCONFIG_SEPARATE?1:samples);
    if(TIFFIsTiled(tif.get())){
        uint32_t tw=0,th=0;TIFFGetField(tif.get(),TIFFTAG_TILEWIDTH,&tw);TIFFGetField(tif.get(),TIFFTAG_TILELENGTH,&th);
        auto bytes=TIFFTileSize(tif.get());if(!tw||!th||bytes<=0||uint64_t(tw)*th*stride>uint64_t(bytes))throw std::runtime_error("invalid TIFF tile");
        std::vector<uint8_t> buffer(size_t(bytes),0);
        for(uint32_t y=0;y<h;y+=th)for(uint32_t x=0;x<w;x+=tw){
            if(TIFFReadTile(tif.get(),buffer.data(),x,y,0,0)<bytes)throw std::runtime_error("truncated TIFF tile");
            for(uint32_t dy=0;dy<std::min(th,h-y);++dy)for(uint32_t dx=0;dx<std::min(tw,w-x);++dx)
                output(int(y+dy),int(x+dx))=value(buffer.data()+(size_t(dy)*tw+dx)*stride);
        }
    }else{
        auto bytes=TIFFScanlineSize(tif.get());if(bytes<=0||uint64_t(w)*stride>uint64_t(bytes))throw std::runtime_error("invalid TIFF row");
        std::vector<uint8_t> buffer(size_t(bytes),0);
        for(uint32_t y=0;y<h;++y){
            if(TIFFReadScanline(tif.get(),buffer.data(),y,0)!=1)throw std::runtime_error("truncated TIFF row");
            for(uint32_t x=0;x<w;++x)output(int(y),int(x))=value(buffer.data()+x*stride);
        }
    }
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
