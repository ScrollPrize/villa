#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include "vc/lasagna/PredictionPooling.hpp"
#include "vc/core/types/Volume.hpp"
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
    template<class T> void channel(const std::string& name,std::vector<T> values,utils::ZarrDtype dtype) {
        utils::ZarrMetadata m;m.version=utils::ZarrVersion::v2;m.shape={2,2,2};m.chunks={2,2,2};
        m.dtype=dtype;m.fill_value=0;m.compressor_id.clear();
        const auto path=root/"custom"/name/"3";
        auto a=utils::ZarrArray::create(path,m);
        a.write_chunk(std::array<size_t,3>{0,0,0},std::as_bytes(std::span(values)));
        channels[name]=path.string();
    }
    void setup(){
        channel<uint8_t>("presence",{0,0,0,0,0,0,200,100},utils::ZarrDtype::uint8);
        for(const auto& name:{"nx","ny","sheet_nx","sheet_ny"})
            channel<uint8_t>(name,std::vector<uint8_t>(8,128),utils::ZarrDtype::uint8);
        channel<uint8_t>("sheet_grad_mag",std::vector<uint8_t>(8,255),utils::ZarrDtype::uint8);
        channel<float>("width",std::vector<float>(8,20),utils::ZarrDtype::float32);
    }
    std::shared_ptr<utils::Store> view(const std::string& name){
        auto path=root/("view_"+name);std::filesystem::create_directories(path);
        std::ofstream(path/".native_pool.json")<<nlohmann::json{
            {"revision",1},{"factor",2},{"mode","top2_spatial_comp"},{"source",root.string()},
            {"channels",channels},{"channel",name}};
        std::ofstream(path/".zarray")<<nlohmann::json{
            {"zarr_format",2},{"shape",{1,1,1}},{"chunks",{1,1,1}},{"dtype",name=="width"?"<f4":"|u1"},
            {"fill_value",0},{"order","C"},{"dimension_separator","/"},{"compressor",nullptr},{"filters",nullptr}};
        return vc::lasagna::openPooledPredictionStore(path);
    }
};
}
TEST_CASE("native pooling accepts explicit source levels and shares channel bundles"){
    Fixture f;f.setup();auto presence=f.view("presence"),width=f.view("width"),nx=f.view("nx");
    auto a=std::async(std::launch::async,[&]{return presence->get_if_exists("0/0/0").value();});
    auto b=std::async(std::launch::async,[&]{return width->get_if_exists("0/0/0").value();});
    CHECK(std::to_integer<int>(a.get().at(0))==150);
    float value;auto bytes=b.get();std::memcpy(&value,bytes.data(),4);CHECK(value==20);
    CHECK(std::to_integer<int>(nx->get_if_exists("0/0/0")->at(0))==128);
    CHECK_THROWS(presence->set("0/0/0",{}));
    CHECK_THROWS(presence->get_if_exists("1/0/0"));
}
TEST_CASE("native pooled views reject metadata with a wrong channel dtype"){
    Fixture f;f.setup();f.view("width");
    const auto path=f.root/"view_width";
    nlohmann::json meta;std::ifstream(path/".zarray")>>meta;
    meta["dtype"]="|u1";std::ofstream(path/".zarray")<<meta;
    CHECK_THROWS(vc::lasagna::openPooledPredictionStore(path));
}
