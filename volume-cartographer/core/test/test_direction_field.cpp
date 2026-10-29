#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Geometry.hpp"
#include "vc/tracer/CostFunctions.hpp"
#include "vc/tracer/DirectionField.hpp"
#include "vc_reader/slice.hpp"

#include <chrono>
#include <fstream>
#include <future>

namespace {
struct Fixture {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("vc-direction-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Fixture() { std::filesystem::remove_all(root); }

    void write(const std::string& codec, const std::string& separator) {
        vcr::zarr::CreateOptions options;
        options.shape = {5, 6, 7}; options.chunks = {2, 3, 4};
        options.compressor = codec; options.separator = separator;
        const std::array<const char*, 4> names{"x", "y", "z", "confidence"};
        for (int c = 0; c < 4; ++c) {
            options.fillValue = c == 3 ? 0 : 128;
            auto writer = vcr::zarr::Writer::create(root / names[c] / "1", options);
            std::vector<std::uint8_t> data(5*6*7);
            for (int z=0; z<5; ++z) for (int y=0; y<6; ++y) for (int x=0; x<7; ++x)
                data[(z*6+y)*7+x] = c == 3 ? 192 : 128 + (c+1)*10 + z*3 + y*2 + x;
            writer.writeRegion({0,0,0}, options.shape, data.data());
            writer.removeChunk({1,1,1});
        }
    }
};
void checkVector(cv::Vec3f actual, cv::Vec3f expected) {
    for (int a=0;a<3;++a) CHECK(actual[a] == doctest::Approx(expected[a]).epsilon(1e-6));
}
}

TEST_CASE("growth directions use native sources, scale, XYZ channels and sparse confidence") {
    for (const auto& codec : {"none", "gzip"}) for (const auto& separator : {".", "/"}) {
        Fixture fixture;
        fixture.write(codec, separator);
        DirectionField field("horizontal", fixture.root, 1, 0.75f, fixture.root/"confidence");
        CHECK(field.native->service == vcr::chunks::processService());
        CHECK(field.direction == "horizontal");
        CHECK(field.weight == 0.75f);
        // XYZ (6,2,1) at level 1, at the odd X edge and across chunk boundaries.
        const auto sample = sampleDirectionConstraint(field, {12,4,2});
        checkVector(sample.directionZYX, cv::Vec3f{43,33,23}/127.f);
        CHECK(sample.confidence == doctest::Approx(192.f/255));
        const auto fetched = field.native->service->stats().fetched;
        for (int i=0;i<5;++i) checkVector(sampleDirectionConstraint(field,{12,4,2}).directionZYX,sample.directionZYX);
        CHECK(field.native->service->stats().fetched == fetched);

        // Missing chunks and coordinates outside the array follow each channel's fill.
        for (const auto p : {cv::Vec3d{10,8,4}, cv::Vec3d{-10,0,0}, cv::Vec3d{100,0,0}}) {
            const auto fill = sampleDirectionConstraint(field,p);
            checkVector(fill.directionZYX, {0,0,0});
            CHECK(fill.confidence == 0);
        }
        // The native nearest sampler rounds half ties up.
        checkVector(sampleDirectionConstraint(field,{5,4,2}).directionZYX,
                    sampleDirectionConstraint(field,{6,4,2}).directionZYX);
        DirectionField unweighted("normal", fixture.root, 1);
        CHECK(sampleDirectionConstraint(unweighted,{12,4,2}).confidence == 1);

        std::vector<std::future<DirectionConstraintSample>> readers;
        for (int i=0;i<4;++i)
            readers.push_back(std::async(std::launch::async,[&] {return sampleDirectionConstraint(field,{8,2,8});}));
        for (auto& reader : readers) checkVector(reader.get().directionZYX, cv::Vec3f{48,38,28}/127.f);

#ifdef VC_READER_CUDA
        // The same registered field can be sampled on the GPU. Invalidate first
        // to exercise the CUDA consumer's cold service fetch as well as decoding.
        for (std::size_t c=0;c<4;++c) {
            auto& opened = *field.native->arrays[c];
            field.native->service->invalidate(field.native->sources[c]);
            vcr::slice::Cached source{*opened.array,*field.native->service,field.native->sources[c]};
            const float point[]{6,2,1};
            auto gpu = vcr::slice::cuda::points(source,opened.meta.type,point,1,vcr::slice::Order::Nearest);
            float value;
            vcr::device::check(cudaMemcpy(&value,gpu->ptr,sizeof(float),cudaMemcpyDeviceToHost));
            CHECK(value == (c==3 ? 192.f : 128.f+(c+1)*10+13));
        }
#endif
    }
}

TEST_CASE("fiber and normal residuals retain direction decoding and confidence weighting") {
    Fixture fixture;
    fixture.write("gzip", "/");
    DirectionField field("normal",fixture.root,1,0.75f,fixture.root/"confidence");
    const double p[]{12,4,2}, u[]{13,4.5,2.25}, v[]{12.25,5,2.5};
    const double* params[]{p,u,v};
    const auto sample = sampleDirectionConstraint(field,{p[0],p[1],p[2]});
    cv::Vec3d du{u[2]-p[2],u[1]-p[1],u[0]-p[0]}, dv{v[2]-p[2],v[1]-p[1],v[0]-p[0]};
    const auto normal = du.cross(dv);
    for (const bool normalLoss : {false,true}) {
        std::unique_ptr<ceres::CostFunction> cost(normalLoss ? NormalDirectionLoss::Create(field,field.weight)
                                                           : FiberDirectionLoss::Create(field,field.weight));
        double residual, j0[3],j1[3],j2[3]; double* jacobians[]{j0,j1,j2};
        REQUIRE(cost->Evaluate(params,&residual,jacobians));
        const auto vector = normalLoss ? normal : du;
        const double length = normalLoss ? cv::norm(normal) : std::sqrt(du.dot(du)+1e-12);
        const double expected = field.weight*(1-std::abs(vector.dot(cv::Vec3d(sample.directionZYX)))/length)*sample.confidence;
        CHECK(residual == doctest::Approx(expected).epsilon(1e-6));
        for (int block=0;block<(normalLoss ? 3 : 2);++block)
            for(int axis=0;axis<3;++axis) CHECK(std::isfinite(jacobians[block][axis]));
    }
}

TEST_CASE("direction sources survive moves and are unregistered on destruction") {
    Fixture fixture;
    fixture.write("none", ".");
    const auto service = vcr::chunks::processService();
    vcr::chunks::SourceId id;
    std::weak_ptr<vcr::zarr::OpenedArray> owner;
    {
        DirectionField original("vertical",fixture.root,1);
        id = original.native->sources[0]; owner = original.native->arrays[0];
        DirectionField moved(std::move(original));
        CHECK(sampleDirectionConstraint(moved,{0,0,0}).confidence == 1);
        DirectionField assigned("normal",fixture.root,1);
        const auto replaced = assigned.native->sources[0];
        assigned = std::move(moved);
        CHECK_THROWS(service->get({replaced,0,0,0,0}));
        CHECK(sampleDirectionConstraint(assigned,{0,0,0}).confidence == 1);
    }
    CHECK(owner.expired());
    CHECK_THROWS(service->get({id,0,0,0,0}));
}

TEST_CASE("direction fields reject incompatible channels before creating residuals") {
    Fixture fixture;
    fixture.write("none", ".");
    vcr::zarr::CreateOptions options;
    options.shape={5,6,7}; options.chunks={2,3,4}; options.dtype="<u2";
    (void)vcr::zarr::Writer::create(fixture.root/"y/1",options);
    CHECK_THROWS(DirectionField("normal",fixture.root,1));
    options.dtype="|u1"; options.shape={5,6,8};
    (void)vcr::zarr::Writer::create(fixture.root/"y/1",options);
    CHECK_THROWS(DirectionField("normal",fixture.root,1));
}
