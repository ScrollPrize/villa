#include "vc_reader/reader.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <stdexcept>
namespace fs=std::filesystem;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){
    const auto path=fs::temp_directory_path()/("vc-reader-native-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path path;~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{path};
    try{
        fs::create_directories(path/"0");
        std::ofstream(path/".zattrs")<<R"({"multiscales":[{"datasets":[{"path":"0"}]}]})";
        std::ofstream(path/"0/.zarray")<<R"({"zarr_format":2,"shape":[109,109,109],"chunks":[109,109,109],"dtype":"|u1","compressor":null,"filters":null,"fill_value":7,"order":"C"})";
        std::vector<uint8_t> data(109*109*109);
        for(size_t i=0;i<data.size();++i)data[i]=uint8_t(120+70*std::sin((int(i%109)-54)*.3));
        {std::ofstream out(path/"0/0.0.0",std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size());}
        vcr::CtReader reader(path.string());require(reader.shape()==std::array<int64_t,3>{109,109,109},"shape");
        std::array<uint8_t,8> crop{};reader.readRegion({-1,-1,-1},{2,2,2},crop);
        require(crop[0]==7&&crop[7]==data[0],"clipping/fill");
        bool rejected=false;try{reader.readRegion({0,0,0},{3,3,3},crop);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"output size validation");
        std::array<std::array<double,3>,3> points{{{54,54,54},{55.25,56.5,54.125},{53.99,54,54}}};
        auto result=reader.sheetNormals(points);
        for(int i=0;i<2;++i){require(std::abs(result[i][0])>.999f&&std::abs(result[i][1])<1e-5&&std::abs(result[i][2])<1e-5,"normal direction");require(result[i][3]>.99f,"eigengap");}
        require(result[2]==std::array<float,4>{},"support validity");
        require(reader.sheetNormals({}).empty(),"empty batch");
        std::cout<<"Native reader regions, pyramid resolution, support and normals passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
