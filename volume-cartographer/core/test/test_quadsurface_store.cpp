// QuadSurface round trip through a vc_object_store surface object: points,
// validity (the "mask" channel), lazily loaded channels, sparse rectangle
// writes that keep masked coordinates, channel replacement and growth.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/StoreSurfaces.hpp"
#include <vc_object_store/objects/Hub.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/Uuid.hpp>

#include <opencv2/core.hpp>
#include <filesystem>
#include <cstring>
#include <cmath>

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
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) { result = db.currentGeometry(txn, uuid)->uuid; });
        return result;
    }
    RevisionSelection pin(const std::string& uuid) {
        RevisionSelection result;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) { result[uuid] = db.currentObjectRevision(txn, uuid)->uuid; });
        return result;
    }
    // One checked edit of an existing surface against its current geometry.
    template<class Edit> void edit(const std::string& uuid, Edit&& apply) {
        const auto geometry = head(uuid);
        session->act(attachment, pin(uuid), [&](auto& db, auto& txn) { apply(db, txn, geometry); });
    }
    std::unique_ptr<QuadSurface> load(const std::string& uuid) {
        std::unique_ptr<QuadSurface> surface;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) { surface = load_quad_from_store(db, txn, uuid); });
        surface->setChannelLoader(store_channel_loader(session, attachment, head(uuid)));
        return surface;
    }
};

cv::Vec3f stored(Store& store, const std::string& uuid, int row, int col)
{
    cv::Vec3f result;
    store.session->read(store.attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto bytes = db.readGeometryRectangle(txn, db.currentGeometry(txn, uuid)->uuid, "points", row, col, 1, 1);
        std::memcpy(result.val, bytes.data(), 12);
    });
    return result;
}
uint8_t storedValid(Store& store, const std::string& uuid, int row, int col)
{
    uint8_t result = 0;
    store.session->read(store.attachment, [&](const auto& db, const auto& txn, const auto&) {
        result = db.readGeometryRectangle(txn, db.currentGeometry(txn, uuid)->uuid, "valid", row, col, 1, 1).at(0);
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
    store.session->act(store.attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface"; record.name = "sheet"; record.tags = {"approved"};
        db.upsertObject(txn, record);
        db.putGeometry(txn, uuid, quad_surface_geometry(source));
    });

    auto loaded = store.load(uuid);
    CHECK(loaded->id == uuid);
    CHECK(loaded->meta["tags"].contains("approved"));
    CHECK(loaded->_scale == cv::Vec2f(0.5f, 0.5f));
    const auto& p = *loaded->rawPointsPtr();
    CHECK(p(1, 1) == cv::Vec3f(1, 1, 10));
    CHECK(p(0, 4) == cv::Vec3f(-1, -1, -1));
    CHECK(p(2, 3) == cv::Vec3f(-1, -1, -1));  // masked
    CHECK(loaded->channel("mask", SURF_CHANNEL_NORESIZE).at<uint8_t>(2, 3) == 0);
    const auto loadedGenerations = loaded->channel("generations", SURF_CHANNEL_NORESIZE);
    CHECK(loadedGenerations.type() == CV_16U);
    CHECK(cv::countNonZero(loadedGenerations != 7) == 0);

    // Loaded into a volume of twice the stored resolution: coordinates and grid step double.
    store.session->read(store.attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto scaled = load_quad_from_store(db, txn, uuid, std::nullopt, true, 2.0);
        CHECK(scaled->_scale == cv::Vec2f(0.25f, 0.25f));
        CHECK((*scaled->rawPointsPtr())(1, 1) == cv::Vec3f(2, 2, 20));
        CHECK((*scaled->rawPointsPtr())(0, 4) == cv::Vec3f(-1, -1, -1));
    });

    // Edit one vertex, clear another, leave the masked one alone.
    (*loaded->rawPointsPtr())(1, 1) = {1.5f, 1.f, 11.f};
    (*loaded->rawPointsPtr())(3, 0) = {-1.f, -1.f, -1.f};
    store.edit(uuid, [&](auto& db, auto& txn, const auto& head) {
        write_quad_rect_to_store(db, txn, uuid, *loaded, cv::Rect(0, 0, 5, 4), head);
    });
    CHECK(stored(store, uuid, 1, 1) == cv::Vec3f(1.5f, 1.f, 11.f));
    CHECK(stored(store, uuid, 3, 0) == cv::Vec3f(-1, -1, -1));
    CHECK(stored(store, uuid, 2, 3) == cv::Vec3f(3, 2, 10));  // masked coordinates survive
    CHECK(storedValid(store, uuid, 2, 3) == 0);

    // Giving the masked vertex new coordinates makes it valid.
    (*loaded->rawPointsPtr())(2, 3) = {3.f, 2.f, 12.f};
    store.edit(uuid, [&](auto& db, auto& txn, const auto& head) {
        write_quad_rect_to_store(db, txn, uuid, *loaded, cv::Rect(3, 2, 1, 1), head);
    });
    CHECK(storedValid(store, uuid, 2, 3) == 255);
    CHECK(stored(store, uuid, 2, 3) == cv::Vec3f(3, 2, 12));

    // Channels replace in place; the mask writes `valid`.
    loaded->setChannel("generations", cv::Mat_<uint16_t>(4, 5, uint16_t(9)));
    cv::Mat_<uint8_t> newMask(4, 5, uint8_t(255));
    newMask(0, 0) = 0;
    loaded->setChannel("mask", newMask);
    store.edit(uuid, [&](auto& db, auto& txn, const auto& head) {
        write_quad_channels_to_store(db, txn, uuid, *loaded, {"generations", "mask"}, {}, head);
    });
    auto reloaded = store.load(uuid);
    CHECK(cv::countNonZero(reloaded->channel("generations", SURF_CHANNEL_NORESIZE) != 9) == 0);
    CHECK((*reloaded->rawPointsPtr())(0, 0) == cv::Vec3f(-1, -1, -1));
    CHECK(storedValid(store, uuid, 0, 0) == 0);
}

TEST_CASE("A stored vertex without coordinates loads as missing")
{
    Store store;
    const auto uuid = util::newUuid();
    GeometryData data{"surface", 1, "source_xyz", R"({"scale":[1,1]})", {}};
    const std::vector<float> xyz{1, 2, 3, 5, -1, 7, std::nanf(""), 1, 2};
    GeometryArray points{"points", "<f4", {1, 3, 3}, std::vector<uint8_t>(xyz.size() * 4)};
    std::memcpy(points.bytes.data(), xyz.data(), points.bytes.size());
    data.arrays.push_back(points);
    store.session->act(store.attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        db.upsertObject(txn, record);
        db.putGeometry(txn, uuid, data);
    });
    const auto loaded = store.load(uuid);
    CHECK((*loaded->rawPointsPtr())(0, 0) == cv::Vec3f(1, 2, 3));
    CHECK((*loaded->rawPointsPtr())(0, 1) == cv::Vec3f(-1, -1, -1));
    CHECK((*loaded->rawPointsPtr())(0, 2) == cv::Vec3f(-1, -1, -1));
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
    store.session->act(store.attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        ObjectRecord record; record.uuid = uuid; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":0.5}})";
        db.upsertObject(txn, record);
        db.putGeometry(txn, uuid, data);
    });
    std::unique_ptr<QuadSurface> shown;
    store.session->read(store.attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto view = GeometrySnapshot(db, txn).viewTransformedAt(uuid, "vol");
        shown = load_quad_from_store(db, txn, uuid, std::nullopt, true, 1, &view);
    });
    CHECK((*shown->rawPointsPtr())(0, 0) == cv::Vec3f(2, 2, 8));
    CHECK((*shown->rawPointsPtr())(1, 1) == cv::Vec3f(-1, -1, -1));
    (*shown->rawPointsPtr())(0, 0) = {4, 4, 8};
    const auto geometry = store.head(uuid);
    store.session->act(store.attachment, store.pin(uuid), [&](auto& db, auto& txn) {
        const auto view = GeometrySnapshot(db, txn).viewTransformedAt(uuid, "vol");
        write_quad_rect_to_store(db, txn, uuid, *shown, cv::Rect(0, 0, 2, 2), geometry, {}, &view);
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
        auto inVolume = surfaces.load(uuid, true, std::string("vol"));
        CHECK((*inVolume->rawPointsPtr())(1, 1) == cv::Vec3f(1, 1, 4));
        CHECK_THROWS(surfaces.update(uuid, *inVolume));
        CHECK_THROWS(surfaces.load(uuid, true, std::string("other-volume")));
        // A volume registered to "vol" (here shifted by 10 in x) reads through the store's mapping.
        const auto registration = util::newUuid();
        surfaces.session()->act(surfaces.attachment(), {{registration, std::nullopt}}, [&](auto& db, auto& txn) {
            VolumeAffine shift; shift(0, 3) = 10;
            db.registerVolumeTransform(txn, "vol", "registered", shift, "test registration", registration);
        });
        auto registered = surfaces.load(uuid, true, std::string("registered"));
        CHECK((*registered->rawPointsPtr())(1, 1) == cv::Vec3f(11, 1, 4));
        CHECK_THROWS(surfaces.update(uuid, *registered));
        // A new surface made from it is in the registered volume's voxels.
        const auto derived = surfaces.create(*registered, "derived", nlohmann::json::object(), {uuid});
        auto reread = surfaces.load(derived, true, std::string("registered"));
        CHECK((*reread->rawPointsPtr())(1, 1) == cv::Vec3f(11, 1, 4));
        auto derivedInStore = surfaces.load(derived);
        CHECK_THROWS(surfaces.create(*derivedInStore, "mixed", nlohmann::json::object(), {uuid, derived}));
        auto inStore = surfaces.load(uuid);
        CHECK((*inStore->rawPointsPtr())(1, 1) == cv::Vec3f(2, 2, 8));
        CHECK_NOTHROW(surfaces.update(uuid, *inStore));
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

TEST_CASE("Tools create surfaces with provenance and push them to a hub")
{
    const fs::path dir = fs::temp_directory_path() / ("vc-store-surfaces-" + util::newUuid());
    fs::create_directories(dir / "share");
    const auto hub = dir / "share" / "hub.vcobj";
    OpenOptions options; options.author = "curator";
    cv::Mat_<cv::Vec3f> points(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 4.f};
    QuadSurface seed(points, {1.f, 1.f});
    const auto source = util::newUuid();
    {
        auto store = ObjectStore::create(hub, options);
        auto txn = store.write();
        ObjectRecord record; record.uuid = source; record.kind = "surface";
        record.meta = R"({"coordinates":{"volume_uuid":"vol","scale_factor":1}})";
        store.upsertObject(txn, record);
        store.putGeometry(txn, source, quad_surface_geometry(seed));
        txn.commit();
    }
    initHub(hub, "curator");
    cloneHub(hub, dir / "tool.vcobj", options);
    cloneHub(hub, dir / "viewer.vcobj", options);

    std::string created;
    {
        StoreSurfaces surfaces(dir / "tool.vcobj", "vc_test_tool");
        auto input = surfaces.load(source);
        CHECK(input->id == source);
        (*input->rawPointsPtr())(1, 1)[2] = 5.f;
        created = surfaces.create(*input, "grown", nlohmann::json{{"steps", 3}}, {source}, {"paths"});
        CHECK(input->id == created);
        CHECK(fs::is_directory(surfaces.session()->artifactDirectory(surfaces.attachment(), created)));
        surfaces.finish();
    }
    auto viewer = ObjectStore::open(dir / "viewer.vcobj", options);
    hubPull(viewer);
    auto txn = viewer.read();
    const auto object = viewer.objectByUuid(txn, created);
    REQUIRE(object.has_value());
    const auto meta = nlohmann::json::parse(object->meta);
    CHECK(meta.at("provenance").at("tool") == "vc_test_tool");
    CHECK(meta.at("provenance").at("sources") == nlohmann::json::array({source}));
    CHECK(meta.at("provenance").at("parameters").at("steps") == 3);
    CHECK(meta.at("coordinates").at("volume_uuid") == "vol");
    CHECK(object->tags == std::vector<std::string>{"paths"});
    const auto loaded = load_quad_from_store(viewer, txn, created);
    CHECK((*loaded->rawPointsPtr())(1, 1) == cv::Vec3f(1, 1, 5));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}
