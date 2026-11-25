// QuadSurface round trip through a vc_object_store surface object: points,
// validity (the "mask" channel), lazily loaded channels, sparse rectangle
// writes that keep masked coordinates, channel replacement and growth.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreSurfaces.hpp"
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {
struct Store {
    fs::path dir = fs::temp_directory_path() / ("vc-quad-store-" + util::newUuid());
    std::shared_ptr<ObjectStoreSession> session = std::make_shared<ObjectStoreSession>();
    AttachmentIdentity attachment;
    Store() {
        fs::create_directories(dir);
        OpenOptions options; options.author = "quad-store-test";
        { auto created = ObjectStore::create(dir / "objects.vcobj", options); }
        attachment = session->attach(dir / "objects.vcobj", options);
    }
    ~Store() { session.reset(); std::error_code ignored; fs::remove_all(dir, ignored); }
    std::string head(const std::string& uuid) {
        std::string result;
        session->readStore(attachment, [&](const StoreReader& reader, const auto&) { result = reader.currentGeometry(uuid)->uuid; });
        return result;
    }
    // A new object `record` (its uuid expected absent) holding `geometry`, in one checked request.
    void create(const ObjectRecord& record, const vc::objects::GeometryData& geometry) {
        auto request = session->newRequest(attachment);
        request.expectedObjects = {{record.uuid, std::nullopt}};
        request.operations.push_back(vc::objects::CreateObjectOp{vc::objects::objectRecordJson(record)});
        request.operations.push_back(vc::objects::PutGeometryOp{record.uuid, geometry, {}});
        session->submit(attachment, request);
    }
    // One checked request editing an existing surface at its current version:
    // `build` adds its operations, reading through the reader.
    template<class Build> void edit(const std::string& uuid, Build&& build) {
        auto request = session->newRequest(attachment);
        session->readStore(attachment, [&](const StoreReader& reader, const auto&) {
            request.expectedObjects = {{uuid, reader.currentObject(uuid)->reference()}};
            build(reader, request.operations);
        });
        session->submit(attachment, request);
    }
    std::unique_ptr<QuadSurface> load(const std::string& uuid) {
        std::unique_ptr<QuadSurface> surface;
        session->readStore(attachment, [&](const StoreReader& reader, const auto&) { surface = load_quad_from_store(reader, uuid); });
        surface->setChannelLoader(store_channel_loader(session, attachment, head(uuid)));
        return surface;
    }
};

cv::Vec3f stored(Store& store, const std::string& uuid, int row, int col)
{
    cv::Vec3f result;
    store.session->readStore(store.attachment, [&](const StoreReader& reader, const auto&) {
        const auto bytes = reader.readGeometryRectangle(reader.currentGeometry(uuid)->uuid, "points", row, col, 1, 1);
        std::memcpy(result.val, bytes.data(), 12);
    });
    return result;
}
uint8_t storedValid(Store& store, const std::string& uuid, int row, int col)
{
    uint8_t result = 0;
    store.session->readStore(store.attachment, [&](const StoreReader& reader, const auto&) {
        result = reader.readGeometryRectangle(reader.currentGeometry(uuid)->uuid, "valid", row, col, 1, 1).at(0);
    });
    return result;
}
}

TEST_CASE("QuadSurface round trip through the object store")
{
    Store store;
    cv::Mat_<cv::Vec3f> points(4, 5);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 5; ++c) points(r, c) = {float(c), float(r), 10.f};
    points(0, 4) = {-1.f, -1.f, -1.f};
    QuadSurface source(points, {0.5f, 0.5f});
    cv::Mat_<uint8_t> mask(8, 10, uint8_t(255));
    mask(5, 6) = 0;  // masks vertex (2,3)
    source.setChannel("mask", mask);
    cv::Mat_<uint16_t> generations(4, 5, uint16_t(7));
    source.setChannel("generations", generations);

    const auto uuid = util::newUuid();
    {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = "sheet"; record.tags = {"approved"};
        store.create(record, quad_surface_geometry(source));
    }

    auto loaded = store.load(uuid);
    CHECK(loaded->id == uuid);
    CHECK(loaded->tags == std::vector<std::string>{"approved"});
    CHECK(loaded->_scale == cv::Vec2f(0.5f, 0.5f));
    const auto& p = *loaded->rawPointsPtr();
    CHECK(p(1, 1) == cv::Vec3f(1, 1, 10));
    CHECK(p(0, 4) == cv::Vec3f(-1, -1, -1));
    CHECK(p(2, 3) == cv::Vec3f(-1, -1, -1));  // masked
    CHECK(loaded->channel("mask", SURF_CHANNEL_NORESIZE).at<uint8_t>(2, 3) == 0);
    const auto loadedGenerations = loaded->channel("generations", SURF_CHANNEL_NORESIZE);
    CHECK(loadedGenerations.type() == CV_16U);
    CHECK(cv::countNonZero(loadedGenerations != 7) == 0);

    // Edit one vertex, clear another, leave the masked one alone.
    (*loaded->rawPointsPtr())(1, 1) = {1.5f, 1.f, 11.f};
    (*loaded->rawPointsPtr())(3, 0) = {-1.f, -1.f, -1.f};
    store.edit(uuid, [&](const StoreReader& reader, auto& operations) {
        operations.push_back(*quad_rect_edit(GeometrySnapshot(reader).viewStored(uuid), *loaded, cv::Rect(0, 0, 5, 4)));
    });
    CHECK(stored(store, uuid, 1, 1) == cv::Vec3f(1.5f, 1.f, 11.f));
    CHECK(stored(store, uuid, 3, 0) == cv::Vec3f(-1, -1, -1));
    CHECK(stored(store, uuid, 2, 3) == cv::Vec3f(3, 2, 10));  // masked coordinates survive
    CHECK(storedValid(store, uuid, 2, 3) == 0);

    // Giving the masked vertex new coordinates makes it valid.
    (*loaded->rawPointsPtr())(2, 3) = {3.f, 2.f, 12.f};
    store.edit(uuid, [&](const StoreReader& reader, auto& operations) {
        operations.push_back(*quad_rect_edit(GeometrySnapshot(reader).viewStored(uuid), *loaded, cv::Rect(3, 2, 1, 1)));
    });
    // Nothing changed: no edit.
    store.session->readStore(store.attachment, [&](const StoreReader& reader, const auto&) {
        CHECK_FALSE(quad_rect_edit(GeometrySnapshot(reader).viewStored(uuid), *loaded, cv::Rect(0, 0, 5, 4)));
    });
    CHECK(storedValid(store, uuid, 2, 3) == 255);
    CHECK(stored(store, uuid, 2, 3) == cv::Vec3f(3, 2, 12));

    // Channels replace in place; the mask writes `valid`.
    loaded->setChannel("generations", cv::Mat_<uint16_t>(4, 5, uint16_t(9)));
    cv::Mat_<uint8_t> newMask(4, 5, uint8_t(255));
    newMask(0, 0) = 0;
    loaded->setChannel("mask", newMask);
    store.edit(uuid, [&](const StoreReader&, auto& operations) {
        operations.push_back(quad_channels_operation(uuid, *loaded, {"generations", "mask"}, {}, 4, 5));
    });
    auto reloaded = store.load(uuid);
    CHECK(cv::countNonZero(reloaded->channel("generations", SURF_CHANNEL_NORESIZE) != 9) == 0);
    CHECK((*reloaded->rawPointsPtr())(0, 0) == cv::Vec3f(-1, -1, -1));
    CHECK(storedValid(store, uuid, 0, 0) == 0);
}

TEST_CASE("A stored multi-component surface does not interpolate across its seams")
{
    // Two components side by side in the grid, far apart in the volume.
    cv::Mat_<cv::Vec3f> points(4, 16);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 16; ++c) points(r, c) = {float(c), float(r), c < 8 ? 10.f : 200.f};
    QuadSurface source(points, {1.f, 1.f});
    Store store;
    const auto put = [&](const char* meta) {
        const auto uuid = util::newUuid();
        ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.meta = meta;
        store.create(record, quad_surface_geometry(source));
        return uuid;
    };
    // Rendered depth along the middle row, across the seam between columns 7 and 8.
    const auto seamDepths = [&](const std::string& uuid) {
        const auto loaded = store.load(uuid);
        std::vector<std::uint8_t> valid;
        const vcr::grid::Surface surface(loaded->gridVertices(valid), loaded->gridOptions());
        std::vector<float> xyz(64 * 3);
        surface.resample(loaded->gridRaster(-2.0, 0.0, 4.0, cv::Size(64, 1)), xyz.data(), nullptr);
        std::vector<float> depths;
        for (int c = 0; c < 64; ++c)
            if (std::isfinite(xyz[3 * c + 2])) depths.push_back(xyz[3 * c + 2]);
        return depths;
    };
    const auto between = [](const std::vector<float>& depths) {
        return std::count_if(depths.begin(), depths.end(), [](float z) { return z > 10.5f && z < 199.5f; });
    };
    CHECK(between(seamDepths(put("{}"))) > 0);
    const auto split = seamDepths(put(R"({"components":[[0,8],[8,16]]})"));
    CHECK(split.size() > 32);
    CHECK(between(split) == 0);
}

TEST_CASE("A stored vertex without coordinates loads as missing")
{
    Store store;
    const auto uuid = util::newUuid();
    GeometryData data{"surface", 1, "source_xyz", R"({"scale":[1,1]})", {}};
    const std::vector<float> xyz{1, 2, 3, 5, -1, 7, std::nanf(""), 1, 2, -3, 1, 4, 1, -3, 4};
    GeometryArray points{"points", "<f4", {1, 5, 3}, std::vector<uint8_t>(xyz.size() * 4)};
    std::memcpy(points.bytes.data(), xyz.data(), points.bytes.size());
    data.arrays.push_back(points);
    {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        store.create(record, data);
    }
    const auto loaded = store.load(uuid);
    CHECK((*loaded->rawPointsPtr())(0, 0) == cv::Vec3f(1, 2, 3));
    CHECK((*loaded->rawPointsPtr())(0, 1) == cv::Vec3f(-1, -1, -1));
    CHECK((*loaded->rawPointsPtr())(0, 2) == cv::Vec3f(-1, -1, -1));
    // Native coordinates can be negative; only the missing sentinel is invalid.
    CHECK((*loaded->rawPointsPtr())(0, 3) == cv::Vec3f(-3, 1, 4));
    CHECK((*loaded->rawPointsPtr())(0, 4) == cv::Vec3f(1, -3, 4));
    CHECK(stored(store, uuid, 0, 3) == cv::Vec3f(-3, 1, 4));
    CHECK(stored(store, uuid, 0, 4) == cv::Vec3f(1, -3, 4));
}

TEST_CASE("A view hides stored z <= 0 vertices and keeps them on save")
{
    Store store;
    const auto uuid = util::newUuid();
    cv::Mat_<cv::Vec3f> grid(2, 2);
    grid(0, 0) = {1, 1, 4}; grid(0, 1) = {2, 1, 4}; grid(1, 0) = {1, 2, 4}; grid(1, 1) = {2, 2, -3};
    QuadSurface surface(grid, {1.f, 1.f});
    // Written as stored bytes: a tracer extrapolation below z = 0, marked valid.
    auto data = quad_surface_geometry(surface);
    std::memcpy(data.arrays.front().bytes.data() + 3 * 12, cv::Vec3f(2, 2, -3).val, 12);
    {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":0.5}})";
        store.create(record, data);
    }
    std::unique_ptr<QuadSurface> shown;
    store.session->readStore(store.attachment, [&](const StoreReader& reader, const auto&) {
        const auto view = GeometrySnapshot(reader).viewTransformedAt(uuid, "vol");
        shown = load_quad_from_store(reader, uuid, std::nullopt, true, &view);
    });
    CHECK((*shown->rawPointsPtr())(0, 0) == cv::Vec3f(2, 2, 8));
    CHECK((*shown->rawPointsPtr())(1, 1) == cv::Vec3f(-1, -1, -1));
    (*shown->rawPointsPtr())(0, 0) = {4, 4, 8};
    store.edit(uuid, [&](const StoreReader& reader, auto& operations) {
        const auto view = GeometrySnapshot(reader).viewTransformedAt(uuid, "vol");
        operations.push_back(*quad_rect_edit(view, *shown, cv::Rect(0, 0, 2, 2)));
    });
    CHECK(stored(store, uuid, 0, 0) == cv::Vec3f(2, 2, 4));
    CHECK(stored(store, uuid, 1, 1) == cv::Vec3f(2, 2, -3));
}

TEST_CASE("Tools read surfaces in a bound volume's voxels and write back only stored units")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-store-units-" + util::newUuid());
    fs::create_directories(dir);
    OpenOptions options; options.author = "units";
    cv::Mat_<cv::Vec3f> points(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) points(r, c) = {float(2 * c), float(2 * r), 8.f};
    QuadSurface stored(points, {1.f, 1.f});
    const auto uuid = util::newUuid();
    {
        auto store = ObjectStore::create(dir / "objects.vcobj", options);
        auto txn = store.write();
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        // Two stored units per voxel of "vol".
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":2}})";
        store.upsertObject(txn, record);
        store.putGeometry(txn, uuid, quad_surface_geometry(stored));
        txn.commit();
    }
    {
        StoreSurfaces surfaces(dir / "objects.vcobj", "vc_test_tool");
        auto inVolume = surfaces.load(uuid, true, VolumeVoxels{"vol"});
        CHECK((*inVolume->rawPointsPtr())(1, 1) == cv::Vec3f(1, 1, 4));
        CHECK_THROWS(surfaces.update(uuid, *inVolume));
        CHECK_THROWS(surfaces.load(uuid, true, VolumeVoxels{"other-volume"}));
        // A volume registered to "vol" (here shifted by 10 in x) reads through the store's mapping.
        {
            VolumeAffine shift; shift(0, 3) = 10;
            vc::objects::registerVolumeTransform(*surfaces.objects().session(), surfaces.objects().attachment(), "vol",
                                                 "registered", shift, "test registration");
        }
        auto registered = surfaces.load(uuid, true, VolumeVoxels{"registered"});
        CHECK((*registered->rawPointsPtr())(1, 1) == cv::Vec3f(11, 1, 4));
        CHECK_THROWS(surfaces.update(uuid, *registered));
        // A new surface made from it is in the registered volume's voxels.
        const auto derived = surfaces.create(*registered, "derived", nlohmann::json::object(), {uuid});
        auto reread = surfaces.load(derived, true, VolumeVoxels{"registered"});
        CHECK((*reread->rawPointsPtr())(1, 1) == cv::Vec3f(11, 1, 4));
        auto derivedInStore = surfaces.load(derived);
        // A native read in the registered frame is compatible with a source
        // shown in that same frame; physical read paths do not define units.
        const auto sameFrame = surfaces.create(*derivedInStore, "same-frame", nlohmann::json::object(), {uuid, derived});
        CHECK((*surfaces.load(sameFrame, true, VolumeVoxels{"registered"})->rawPointsPtr())(1, 1) == cv::Vec3f(11, 1, 4));
        auto inStore = surfaces.load(uuid);
        CHECK_THROWS(surfaces.create(*derivedInStore, "mixed", nlohmann::json::object(), {uuid, derived}));
        CHECK((*inStore->rawPointsPtr())(1, 1) == cv::Vec3f(2, 2, 8));
        CHECK(surfaces.update(uuid, *inStore) >= 1);  // unchanged: the version it loaded
        // At working level 1 (half the volume's voxels): read there, and what is
        // made from it is bound to the volume at that scale.
        auto coarse = surfaces.load(uuid, true, VolumeVoxels{"vol", 0.5});
        CHECK((*coarse->rawPointsPtr())(1, 1) == cv::Vec3f(0.5f, 0.5f, 2));
        const auto atLevel = surfaces.create(*coarse, "at-level", nlohmann::json::object(), {uuid});
        surfaces.objects().session()->readStore(surfaces.objects().attachment(), [&](const StoreReader& reader, const auto&) {
            CHECK(nlohmann::json::parse(reader.currentObject(atLevel)->record.meta).at("coordinates") ==
                  vc::objects::volumeCoordinates("vol", 0.5));
        });
        CHECK((*surfaces.load(atLevel, true, VolumeVoxels{"vol"})->rawPointsPtr())(1, 1) == cv::Vec3f(1, 1, 4));
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Tools create surfaces with provenance that other clients read")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-store-surfaces-" + util::newUuid());
    fs::create_directories(dir);
    const auto path = dir / "objects.vcobj";
    OpenOptions options; options.author = "curator";
    cv::Mat_<cv::Vec3f> points(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 4.f};
    QuadSurface seed(points, {1.f, 1.f});
    const auto source = util::newUuid();
    {
        auto store = ObjectStore::create(path, options);
        auto txn = store.write();
        ObjectRecord record; record.uuid = source; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
        store.upsertObject(txn, record);
        store.putGeometry(txn, source, quad_surface_geometry(seed));
        txn.commit();
    }

    std::string created;
    {
        StoreSurfaces surfaces(path, "vc_test_tool");
        auto input = surfaces.load(source);
        CHECK(input->id == source);
        (*input->rawPointsPtr())(1, 1)[2] = 5.f;
        created = surfaces.create(*input, "grown", nlohmann::json{{"steps", 3}}, {source}, {"paths"});
        CHECK(input->id == created);
        surfaces.objects().finish();
    }
    OpenOptions readOnly; readOnly.readOnly = true;
    auto viewer = ObjectStore::open(path, readOnly);
    auto txn = viewer.read();
    const auto object = viewer.objectByUuid(txn, created);
    REQUIRE(object.has_value());
    const auto meta = nlohmann::json::parse(object->meta);
    CHECK(meta.at("provenance").at("tool") == "vc_test_tool");
    CHECK(meta.at("provenance").at("sources") == nlohmann::json::array({source}));
    CHECK(meta.at("provenance").at("parameters").at("steps") == 3);
    CHECK(meta.at("coordinates").at("volume_uuid") == "vol");
    CHECK(object->tags == std::vector<std::string>{"paths"});
    const auto loaded = load_quad_from_store(LocalStoreReader(viewer, txn), created);
    CHECK((*loaded->rawPointsPtr())(1, 1) == cv::Vec3f(1, 1, 5));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}
