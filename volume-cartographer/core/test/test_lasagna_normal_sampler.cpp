#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"
#include "vc/atlas/Atlas.hpp"

#include "utils/zarr.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <barrier>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto dir = fs::temp_directory_path() /
        ("vc_lasagna_normal_sampler_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

void createU8Zarr(
    const fs::path& path,
    std::vector<size_t> shape,
    std::vector<size_t> chunks,
    const std::vector<uint8_t>* payload,
    std::optional<double> fillValue = 0.0)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape = std::move(shape);
    meta.chunks = std::move(chunks);
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id.clear();
    meta.fill_value = fillValue;
    auto array = utils::ZarrArray::create(path, meta);
    if (payload == nullptr) {
        return;
    }
    std::vector<std::byte> bytes(payload->size());
    for (size_t i = 0; i < payload->size(); ++i) {
        bytes[i] = static_cast<std::byte>((*payload)[i]);
    }
    std::vector<size_t> zero(meta.shape.size(), 0);
    array.write_chunk(zero, bytes);
}

void createConstantChunkedU8Zarr(
    const fs::path& path,
    const std::array<size_t, 3>& shape,
    const std::array<size_t, 3>& chunks,
    uint8_t value)
{
    utils::ZarrMetadata meta;
    meta.version = utils::ZarrVersion::v2;
    meta.shape = {shape[0], shape[1], shape[2]};
    meta.chunks = {chunks[0], chunks[1], chunks[2]};
    meta.dtype = utils::ZarrDtype::uint8;
    meta.compressor_id.clear();
    meta.fill_value = 0.0;
    auto array = utils::ZarrArray::create(path, meta);
    std::vector<std::byte> bytes(chunks[0] * chunks[1] * chunks[2],
                                 static_cast<std::byte>(value));
    for (size_t z = 0; z < (shape[0] + chunks[0] - 1) / chunks[0]; ++z) {
        for (size_t y = 0; y < (shape[1] + chunks[1] - 1) / chunks[1]; ++y) {
            for (size_t x = 0; x < (shape[2] + chunks[2] - 1) / chunks[2]; ++x) {
                const std::array<size_t, 3> indices{z, y, x};
                array.write_chunk(indices, bytes);
            }
        }
    }
}

void writeText(const fs::path& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
}

} // namespace

TEST_CASE("LasagnaNormalSampler rejects 4D channel groups")
{
    const auto dir = tmpDir("4d");
    const auto zarrPath = dir / "pred.zarr";
    std::vector<uint8_t> payload(3 * 2 * 2 * 2, 128);
    createU8Zarr(zarrPath, {3, 2, 2, 2}, {3, 2, 2, 2}, &payload);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 1.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "pred": {
                "zarr": "pred.zarr",
                "scaledown": 0,
                "channels": ["grad_mag", "nx", "ny"]
            }
        }
    })");

    const vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_WITH_AS(
        vc::lasagna::LasagnaNormalSampler(dataset),
        doctest::Contains("must reference a 3D (Z,Y,X) zarr"),
        std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler supports 3D per-channel zarr groups and coordinate scaling")
{
    const auto dir = tmpDir("3d");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx(2 * 2 * 2, 255);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 2.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 1, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 1, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 1, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto sample = sampler.sampleNormal({4.0, 4.0, 4.0});

    REQUIRE(sample.valid);
    CHECK(sample.normal[0] == doctest::Approx(1.0));
    CHECK(sample.normal[1] == doctest::Approx(0.0));
    CHECK(sample.normal[2] == doctest::Approx(0.0));

    vc::lasagna::LasagnaDataset scaledDataset =
        vc::lasagna::LasagnaDataset::open(manifestPath, {4.0});
    vc::lasagna::LasagnaNormalSampler scaledSampler(scaledDataset);
    const auto scaledSample = scaledSampler.sampleNormal({1.0, 1.0, 1.0});
    REQUIRE(scaledSample.valid);
    CHECK(scaledSample.normal[0] == doctest::Approx(sample.normal[0]));
    CHECK(scaledSample.normal[1] == doctest::Approx(sample.normal[1]));
    CHECK(scaledSample.normal[2] == doctest::Approx(sample.normal[2]));
    REQUIRE(scaledSampler.predDtSpacing() == std::nullopt);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler samples pred_dt channel with accepted threshold")
{
    const auto dir = tmpDir("pred_dt");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx(2 * 2 * 2, 128);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    std::vector<uint8_t> predDtValues(2 * 2 * 2, 170);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    createU8Zarr(dir / "pred_dt.zarr", {2, 2, 2}, {2, 2, 2}, &predDtValues);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]},
            "pred_dt_group": {"zarr": "pred_dt.zarr", "scaledown": 0, "channels": ["pred_dt"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    REQUIRE(sampler.hasPredDtChannel());
    REQUIRE(sampler.predDtSpacing().has_value());
    CHECK(*sampler.predDtSpacing() == doctest::Approx(1.0));
    const auto predDt = sampler.samplePredDt({1.0, 1.0, 1.0});
    REQUIRE(predDt.has_value());
    CHECK(*predDt == doctest::Approx(170.0));
    CHECK(vc::atlas::atlasPredDtIsInside(*predDt));
    CHECK(vc::atlas::atlasPredDtIsInside(110.0));
    CHECK_FALSE(vc::atlas::atlasPredDtIsInside(109.0));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler integrates decoded grad_mag as winding distance")
{
    const auto dir = tmpDir("winding_distance");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 100);
    std::vector<uint8_t> nx(2 * 2 * 2, 255);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 100.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto density = sampler.sampleWindingDensity({0.5, 0.5, 0.5});
    REQUIRE(density.has_value());
    CHECK(*density == doctest::Approx(1.0));
    CHECK(sampler.windingDistance({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.25) ==
          doctest::Approx(1.0));
    CHECK(sampler.windingDistance(
              {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, 0.25) ==
          doctest::Approx(1.0));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler interpolates unoriented normal tensors")
{
    const auto dir = tmpDir("tensor_interp");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    std::vector<uint8_t> nx{
        255, 1,
        255, 1,
        255, 1,
        255, 1,
    };
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);

    const auto sample = sampler.sampleNormal({0.5, 0.5, 0.5});

    REQUIRE(sample.valid);
    CHECK(std::abs(sample.normal[0]) == doctest::Approx(1.0).epsilon(1.0e-9));
    CHECK(sample.normal[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(sample.normal[2] == doctest::Approx(0.0).epsilon(1.0e-9));
    fs::remove_all(dir);
}

TEST_CASE("pinned scalar stride gather matches masked general reads at chunk and volume edges")
{
    using namespace vc::lasagna;
    const auto dir=tmpDir("pinned_scalar_strides");
    std::mt19937 random(7123);
    nlohmann::json groups;
    for(bool mask:{false,true}) {
        const std::string name=mask?"mask":"width";
        utils::ZarrMetadata meta;
        meta.version=utils::ZarrVersion::v2;meta.shape={5,6,7};meta.chunks={3,3,3};
        meta.dtype=mask?utils::ZarrDtype::uint8:utils::ZarrDtype::float32;
        meta.compressor_id.clear();meta.fill_value=mask?1:3;
        auto array=utils::ZarrArray::create(dir/name,meta);
        for(size_t z=0;z<2;++z)for(size_t y=0;y<2;++y)for(size_t x=0;x<3;++x) {
            if(z==1 && x==2)continue; // nonzero sparse fill
            const std::array<size_t,3> key{z,y,x};
            if(mask) {
                std::vector<uint8_t> values(27);for(auto& v:values)v=random()%3?255:0;
                array.write_chunk(key,std::as_bytes(std::span(values)));
            } else {
                std::vector<float> values(27);for(auto& v:values)v=float(int(random()%1000)-200)/7;
                array.write_chunk(key,std::as_bytes(std::span(values)));
            }
        }
        groups[name]={{"zarr",(dir/name).string()},{"scaledown",0},{"channels",{name}}};
    }
    const auto manifest=LasagnaDatasetManifest::parseText(nlohmann::json{{"version",2},{"groups",groups}}.dump(),dir/"manifest.json");
    const auto width=bindLasagnaChannel(manifest,"width",true),mask=bindLasagnaChannel(manifest,"mask");
    std::vector<cv::Vec3f> points{{0,0,0},{6,5,4},{3,3,3},{2.999f,2.999f,2.999f},{-1,0,0},{7,5,4}};
    for(int i=0;i<2000;++i)points.emplace_back(float(random()%601)/100,float(random()%501)/100,float(random()%401)/100);
    vc::render::PinnedChunkRegion pinned;
    pinLasagnaRegion(pinned,width,{0,0,0},{6,5,4});pinLasagnaRegion(pinned,mask,{0,0,0},{6,5,4});pinned.seal();
    for(bool masked:{false,true}) {
        std::vector<std::vector<float>> expected,actual;
        sampleLasagnaVectorBatch({width},points,expected,false,masked?&mask:nullptr);
        sampleLasagnaVectorBatch({width},points,actual,false,masked?&mask:nullptr,&pinned);
        CHECK(actual==expected);
    }
    fs::remove_all(dir);
}

TEST_CASE("validity masks companion corners without diluting width or presence")
{
    using namespace vc::lasagna;
    const auto dir=tmpDir("valid_corners");
    std::vector<uint8_t> mask(8,0),nx(8,0),ny(8,0),width(8,0);
    mask[0]=255; nx[0]=255; ny[0]=128; width[0]=40;
    createU8Zarr(dir/"mask.zarr",{2,2,2},{2,2,2},&mask);
    createU8Zarr(dir/"nx.zarr",{2,2,2},{2,2,2},&nx);
    createU8Zarr(dir/"ny.zarr",{2,2,2},{2,2,2},&ny);
    createU8Zarr(dir/"width.zarr",{2,2,2},{2,2,2},&width);
    writeText(dir/"dataset.lasagna.json",R"({"version":2,"mask_invalid_normal_corners":true,
        "grad_mag_encode_scale":255,"grad_mag_factor":1,"groups":{
        "m":{"zarr":"mask.zarr","scaledown":0,"channels":["grad_mag"]},
        "x":{"zarr":"nx.zarr","scaledown":0,"channels":["nx"]},
        "y":{"zarr":"ny.zarr","scaledown":0,"channels":["ny"]},
        "w":{"zarr":"width.zarr","scaledown":0,"channels":["width"]}}})");
    auto dataset=LasagnaDataset::open(dir/"dataset.lasagna.json");
    const auto m=bindLasagnaChannel(dataset.manifest(),"grad_mag");
    const auto w=bindLasagnaChannel(dataset.manifest(),"width");
    std::vector<std::vector<float>> out;
    sampleLasagnaVectorBatch({w},{{.5f,.5f,.5f},{1,1,1},{0,0,0}},out,false,&m);
    REQUIRE(out[0].size()==1); CHECK(out[0][0]==doctest::Approx(40));
    CHECK(out[1].empty()); REQUIRE(out[2].size()==1); CHECK(out[2][0]==doctest::Approx(40));
    const auto expected=out;
    vc::render::PinnedChunkRegion pinned;
    pinLasagnaRegion(pinned,w,{0,0,0},{1,1,1});pinLasagnaRegion(pinned,m,{0,0,0},{1,1,1});pinned.seal();
    {
        sampleLasagnaVectorBatch({w},{{.5f,.5f,.5f},{1,1,1},{0,0,0}},out,false,&m,&pinned);
        CHECK(out==expected);
    }
    CHECK(*sampleLasagnaChannel(m,cv::Vec3d(.5,.5,.5))==doctest::Approx(255./8));
    LasagnaNormalSampler sampler(dataset);
    auto n=sampler.sampleNormal({.5,.5,.5});
    REQUIRE(n.valid); CHECK(std::abs(n.normal[0])==doctest::Approx(1));
    CHECK_FALSE(sampler.sampleNormal({1,1,1}).valid);
    std::array<uint8_t,8> a{},b{},v{};
    a[0]=255;b[0]=128;v[0]=255;
    auto weights=lasagnaCornerWeights({.5f,.5f,.5f});
    auto axis=interpolateLasagnaCompactAxisCorners(a,b,validLasagnaCornerWeights(weights,v));
    CHECK(std::abs(axis[0])==doctest::Approx(1)); CHECK(std::abs(axis[1])<1e-6);
    auto zero=interpolateLasagnaCompactAxisCorners(a,b,validLasagnaCornerWeights(lasagnaCornerWeights({1,1,1}),v));
    CHECK(zero.dot(zero)==doctest::Approx(0));
    v.fill(255); CHECK(validLasagnaCornerWeights(weights,v)==weights);
    CHECK(sampler.masksInvalidCorners());
    auto legacyManifest=dataset.manifest();
    legacyManifest.raw["mask_invalid_normal_corners"]=false;
    LasagnaDataset legacyDataset(std::move(legacyManifest));
    LasagnaNormalSampler legacy(legacyDataset);
    CHECK_FALSE(legacy.masksInvalidCorners());
    const auto legacyNormal=legacy.sampleNormal({.5,.5,.5});
    REQUIRE(legacyNormal.valid);
    const auto legacyAxis=interpolateLasagnaCompactAxisCorners(a,b,weights);
    CHECK(std::abs(cv::Vec3d(legacyAxis).dot(legacyNormal.normal))==doctest::Approx(1).epsilon(1e-5));
    CHECK(std::abs(legacyNormal.normal[1])>.1);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler fetches only interpolation source chunks across boundaries")
{
    const auto dir = tmpDir("chunk_boundary");
    createConstantChunkedU8Zarr(dir / "grad_mag.zarr", {4, 4, 4}, {2, 2, 2}, 255);
    createConstantChunkedU8Zarr(dir / "nx.zarr", {4, 4, 4}, {2, 2, 2}, 128);
    createConstantChunkedU8Zarr(dir / "ny.zarr", {4, 4, 4}, {2, 2, 2}, 128);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
    const auto report = sampler.sampleNormalBatch({{1.5, 1.5, 1.5}}, false, samples);

    REQUIRE(samples.size() == 1);
    REQUIRE(samples.front().sample.valid);
    CHECK(samples.front().sample.normal[2] == doctest::Approx(1.0));
    CHECK(report.prefetch.requestedChunks == 24);
    CHECK(report.prefetch.chunksRead == 24);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler requires grad_mag channel")
{
    const auto dir = tmpDir("missing_grad_mag_channel");
    std::vector<uint8_t> nx(2 * 2 * 2, 128);
    std::vector<uint8_t> ny(2 * 2 * 2, 128);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &nx);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &ny);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "groups": {
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_AS(vc::lasagna::LasagnaNormalSampler(dataset), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler requires explicit grad_mag scale metadata")
{
    const auto dir = tmpDir("missing_grad_mag_scale");
    std::vector<uint8_t> payload(2 * 2 * 2, 128);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, &payload);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    CHECK_THROWS_AS(vc::lasagna::LasagnaNormalSampler(dataset), std::runtime_error);
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler reports invalid samples for missing chunks and zero grad_mag")
{
    const auto dir = tmpDir("invalid");
    std::vector<uint8_t> zeroGradMag(2 * 2 * 2, 0);
    std::vector<uint8_t> normals(2 * 2 * 2, 128);
    createU8Zarr(dir / "zero_grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &zeroGradMag);
    createU8Zarr(dir / "zero_nx.zarr", {2, 2, 2}, {2, 2, 2}, &normals);
    createU8Zarr(dir / "zero_ny.zarr", {2, 2, 2}, {2, 2, 2}, &normals);
    createU8Zarr(dir / "missing_grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);
    createU8Zarr(dir / "missing_nx.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);
    createU8Zarr(dir / "missing_ny.zarr", {2, 2, 2}, {2, 2, 2}, nullptr);

    const auto zeroManifest = dir / "zero.lasagna.json";
    writeText(zeroManifest, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "zero_grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "zero_nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "zero_ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");
    vc::lasagna::LasagnaDataset zeroDataset = vc::lasagna::LasagnaDataset::open(zeroManifest);
    vc::lasagna::LasagnaNormalSampler zeroSampler(zeroDataset);
    CHECK_FALSE(zeroSampler.sampleNormal({1.0, 1.0, 1.0}).valid);

    const auto missingManifest = dir / "missing.lasagna.json";
    writeText(missingManifest, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "missing_grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "missing_nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "missing_ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");
    vc::lasagna::LasagnaDataset missingDataset = vc::lasagna::LasagnaDataset::open(missingManifest);
    vc::lasagna::LasagnaNormalSampler missingSampler(missingDataset);
    CHECK_FALSE(missingSampler.sampleNormal({1.0, 1.0, 1.0}).valid);

    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler reads absent chunks from the Zarr fill value")
{
    const auto dir = tmpDir("fill_value");
    std::vector<uint8_t> gradMag(2 * 2 * 2, 255);
    createU8Zarr(dir / "grad_mag.zarr", {2, 2, 2}, {2, 2, 2}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {2, 2, 2}, {2, 2, 2}, nullptr, 128.0);
    createU8Zarr(dir / "ny.zarr", {2, 2, 2}, {2, 2, 2}, nullptr, 128.0);

    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    const auto sample = sampler.sampleNormal({1.0, 1.0, 1.0});

    REQUIRE(sample.valid);
    CHECK(sample.normal[0] == doctest::Approx(0.0));
    CHECK(sample.normal[1] == doctest::Approx(0.0));
    CHECK(sample.normal[2] == doctest::Approx(1.0));
    fs::remove_all(dir);
}

TEST_CASE("LasagnaNormalSampler integrates with LineOptimizer")
{
    const auto dir = tmpDir("optimizer");
    std::vector<uint8_t> gradMag(4 * 4 * 4, 255);
    std::vector<uint8_t> normals(4 * 4 * 4, 128);
    createU8Zarr(dir / "grad_mag.zarr", {4, 4, 4}, {4, 4, 4}, &gradMag);
    createU8Zarr(dir / "nx.zarr", {4, 4, 4}, {4, 4, 4}, &normals);
    createU8Zarr(dir / "ny.zarr", {4, 4, 4}, {4, 4, 4}, &normals);
    const auto manifestPath = dir / "dataset.lasagna.json";
    writeText(manifestPath, R"({
        "version": 2,
        "source_to_base": 100.0,
        "grad_mag_encode_scale": 255.0,
        "grad_mag_factor": 1.0,
        "groups": {
            "grad_mag_group": {"zarr": "grad_mag.zarr", "scaledown": 0, "channels": ["grad_mag"]},
            "nx_group": {"zarr": "nx.zarr", "scaledown": 0, "channels": ["nx"]},
            "ny_group": {"zarr": "ny.zarr", "scaledown": 0, "channels": ["ny"]}
        }
    })");

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 50.0;
    config.maxIterations = 5;
    const auto result = optimizer.optimizeFromSeed({100.0, 100.0, 100.0}, config);

    CHECK(result.line.points.size() == 3);
    CHECK(result.report.validNormalSamples > 0);
    CHECK(result.line.points[1].sampledNormal.valid);
    CHECK(result.line.points[1].sampledNormal.normal[2] == doctest::Approx(1.0));
    fs::remove_all(dir);
}


TEST_CASE("Lasagna scalar and corner paths share regular decoded chunks in both warm orders")
{
    using namespace vc::lasagna;
    const auto dir = tmpDir("shared_cache");
    const auto manifest = dir / "dataset.lasagna.json";
    writeText(manifest, R"({"version":2,"groups":{
        "v":{"zarr":"values.zarr","scaledown":0,"channels":["nx"]}}})");
    createConstantChunkedU8Zarr(dir / "values.zarr", {8,8,8}, {4,4,4}, 42);
    const auto dataset = LasagnaDataset::open(manifest);
    const auto a = bindLasagnaChannel(dataset.manifest(), "nx");
    const auto b = bindLasagnaChannel(dataset.manifest(), "nx");
    CHECK(a.chunkSource->sourceId() == b.chunkSource->sourceId());
    LasagnaChannelCornerSampler corners(b);
    auto request = prepareLasagnaCubeRequest(a, cv::Vec3d{1,1,1});
    LasagnaLocalChunkResolver resolver(a);
    resolver.resolve(request);
    const auto pin = request.chunks[0].bytes;
    REQUIRE(pin);
    CHECK(pin == b.chunkSource->getChunkIfCached(0,0,0,0).bytes);
    const auto scalarFirstStats = a.chunkSource->stats();
    std::vector<LasagnaCornerSample> samples;
    (void)corners.sampleBatch({{1,1,1}}, samples);
    REQUIRE(samples.size() == 1);
    CHECK(samples[0].valid);
    CHECK(samples[0].values[0] == 42);
    CHECK(b.chunkSource->getChunkIfCached(0,0,0,0).bytes == pin);
    CHECK(a.chunkSource->stats().decodeAttempts == scalarFirstStats.decodeAttempts);

    (void)corners.sampleBatch({{4,4,4}}, samples);
    const auto cornerPin = b.chunkSource->getChunkIfCached(0,1,1,1).bytes;
    REQUIRE(cornerPin);
    const auto cornerFirstStats = b.chunkSource->stats();
    const auto value = sampleLasagnaChannel(a, cv::Vec3d{4,4,4});
    REQUIRE(value);
    CHECK(*value == 42);
    CHECK(a.chunkSource->getChunkIfCached(0,1,1,1).bytes == cornerPin);
    CHECK(a.chunkSource->stats().decodeAttempts == cornerFirstStats.decodeAttempts);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna regular cache coalesces concurrent cold readers and preserves pins on eviction")
{
    using namespace vc::lasagna;
    const auto dir = tmpDir("coalesced_cache");
    createConstantChunkedU8Zarr(dir / "values.zarr", {4,4,4}, {2,2,2}, 43);
    const auto manifest = dir / "dataset.lasagna.json";
    writeText(manifest, R"({"version":2,"groups":{
        "v":{"zarr":"values.zarr","scaledown":0,"channels":["nx"]}}})");
    const auto dataset = LasagnaDataset::open(manifest);
    const auto binding = bindLasagnaChannel(dataset.manifest(), "nx");
    std::barrier start(8);
    std::vector<std::future<vc::render::ChunkResult>> readers;
    for (int i=0; i<8; ++i)
        readers.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return binding.chunkSource->getChunkBlocking(0,0,0,0);
        }));
    auto pin = readers.front().get();
    REQUIRE(pin.bytes);
    for (size_t i=1; i<readers.size(); ++i)
        CHECK(readers[i].get().bytes == pin.bytes);
    const auto budget = vc::render::processChunkCacheService()->decodedByteBudget();
    struct RestoreBudget {
        std::shared_ptr<vc::render::DecodedChunkCacheBudget> budget;
        size_t bytes;
        ~RestoreBudget() { budget->setMaximumBytes(bytes); }
    } restore{budget, budget->maximumBytes()};
    budget->setMaximumBytes(1);
    (void)binding.chunkSource->getChunkBlocking(0,1,1,1);
    // Drop cache ownership under pressure, but keep the caller's pin readable.
    CHECK(binding.chunkSource->getChunkIfCached(0,0,0,0).status == vc::render::ChunkStatus::MissQueued);
    CHECK(pin.bytes->at(0) == std::byte{43});
    CHECK(budget->stats().decodedBytes <= 1);
    pin = {};
    // No new read or budget update is needed after releasing the pin.
    CHECK(budget->stats().decodedBytes <= 1);
    CHECK(binding.chunkSource->getChunkIfCached(0,0,0,0).status == vc::render::ChunkStatus::MissQueued);
    fs::remove_all(dir);
}

TEST_CASE("Lasagna shared cache distinguishes absent fill from stored zero and rejects corrupt chunks")
{
    using namespace vc::lasagna;
    for (const bool stored : {false, true}) {
        const auto dir = tmpDir(stored ? "stored_zero" : "no_fill");
        const std::vector<uint8_t> zero(8, 0);
        createU8Zarr(dir / "values.zarr", {2,2,2}, {2,2,2}, stored ? &zero : nullptr, std::nullopt);
        const auto manifest = dir / "dataset.lasagna.json";
        writeText(manifest, R"({"version":2,"groups":{
            "v":{"zarr":"values.zarr","scaledown":0,"channels":["nx"]}}})");
        const auto dataset = LasagnaDataset::open(manifest);
        const auto binding = bindLasagnaChannel(dataset.manifest(), "nx");
        const auto value = sampleLasagnaChannel(binding, cv::Vec3d{0,0,0});
        CHECK(value.has_value() == stored);
        if (value) CHECK(*value == 0);
        fs::remove_all(dir);
    }
    for (const bool corrupt : {false, true}) {
        const auto dir = tmpDir(corrupt ? "corrupt" : "bad_fill");
        createU8Zarr(dir / "values.zarr", {2,2,2}, {2,2,2}, nullptr, corrupt ? 0 : 300);
        if (corrupt) writeText(dir / "values.zarr" / "0.0.0", "x");
        const auto manifest = dir / "dataset.lasagna.json";
        writeText(manifest, R"({"version":2,"groups":{
            "v":{"zarr":"values.zarr","scaledown":0,"channels":["nx"]}}})");
        const auto dataset = LasagnaDataset::open(manifest);
        CHECK_THROWS([&] {
            const auto binding = bindLasagnaChannel(dataset.manifest(), "nx");
            (void)sampleLasagnaChannel(binding, cv::Vec3d{0,0,0});
        }());
        fs::remove_all(dir);
    }
}
