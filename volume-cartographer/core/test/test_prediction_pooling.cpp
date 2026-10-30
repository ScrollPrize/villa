#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include "vc/lasagna/PredictionPooling.hpp"
#include <vc_reader/zarr.hpp>
#include <fstream>
#include <future>
#include <random>
#include <cstring>

namespace {
struct Fixture {
    std::filesystem::path root=std::filesystem::temp_directory_path()/
        ("vc_pool_"+std::to_string(std::mt19937_64(std::random_device{}())()));
    ~Fixture(){std::error_code e;std::filesystem::remove_all(root,e);}
    nlohmann::json channels;
    std::shared_ptr<vcr::MemoryCache> cache=std::make_shared<vcr::MemoryCache>(1<<20);
    template<class T> void channel(const std::string& name,std::vector<T> values,const std::string& dtype) {
        const auto path=root/"custom"/name/"3";
        auto writer=vcr::zarr::Writer::create(path,{.shape={2,2,2},.chunks={2,2,2},.dtype=dtype,.compressor=""});
        writer.writeChunk({0,0,0},reinterpret_cast<const std::uint8_t*>(values.data()),values.size()*sizeof(T));
        channels[name]=path.string();
    }
    void setup(){
        channel<uint8_t>("presence",{0,0,0,0,0,0,200,100},"|u1");
        for(const auto& name:{"nx","ny","sheet_nx","sheet_ny"})
            channel<uint8_t>(name,std::vector<uint8_t>(8,128),"|u1");
        channel<uint8_t>("sheet_grad_mag",std::vector<uint8_t>(8,255),"|u1");
        channel<float>("width",std::vector<float>(8,20),"<f4");
    }
    std::filesystem::path write(const std::string& name){
        auto path=root/("view_"+name);std::filesystem::create_directories(path);
        std::ofstream(path/".native_pool.json")<<nlohmann::json{
            {"revision",1},{"factor",2},{"mode","top2_spatial_comp"},{"source",root.string()},
            {"channels",channels},{"channel",name}};
        std::ofstream(path/".zarray")<<nlohmann::json{
            {"zarr_format",2},{"shape",{1,1,1}},{"chunks",{1,1,1}},{"dtype",name=="width"?"<f4":"|u1"},
            {"fill_value",0},{"order","C"},{"dimension_separator","/"},{"compressor",nullptr},{"filters",nullptr}};
        return path;
    }
    vcr::zarr::OpenedArray view(const std::string& name){
        return vc::lasagna::openPooledPredictionArray(write(name),cache);
    }
};
}
TEST_CASE("native pooling accepts explicit source levels and shares one source between channel views"){
    Fixture f;f.setup();auto presence=f.view("presence"),width=f.view("width"),nx=f.view("nx");
    auto a=std::async(std::launch::async,[&]{uint8_t v=0;presence.array->read_points(std::array<int64_t,3>{0,0,0}.data(),1,&v);return v;});
    auto b=std::async(std::launch::async,[&]{float v=0;width.array->read_points(std::array<int64_t,3>{0,0,0}.data(),1,reinterpret_cast<uint8_t*>(&v));return v;});
    CHECK(int(a.get())==150);
    CHECK(b.get()==20);
    uint8_t value=0;nx.array->read_points(std::array<int64_t,3>{0,0,0}.data(),1,&value);
    CHECK(int(value)==128);
    CHECK(presence.meta.shape==vcr::Index{1,1,1});
}
TEST_CASE("native pooled views reject metadata with a wrong channel dtype"){
    Fixture f;f.setup();
    const auto path=f.write("width");
    nlohmann::json meta;std::ifstream(path/".zarray")>>meta;
    meta["dtype"]="|u1";std::ofstream(path/".zarray")<<meta;
    CHECK_THROWS(vc::lasagna::openPooledPredictionArray(path,f.cache));
}
