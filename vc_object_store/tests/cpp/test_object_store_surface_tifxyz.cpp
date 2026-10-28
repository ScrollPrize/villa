#include "TestSupport.hpp"
#include "vc_object_store/Tiff.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/objects/GeometryExport.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <memory>

using namespace vc::objects;
namespace fs = std::filesystem;

namespace {
// VC writes mask.tif as a validity page followed by a render page of another size.
void writeVcMask(const fs::path& path, const cv::Mat_<uint8_t>& mask)
{
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tif(TIFFOpen(path.c_str(), "w"), TIFFClose);
    for (const auto& page : {mask, cv::Mat_<uint8_t>(5, 7, uint8_t(9))}) {
        TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, uint32_t(page.cols));
        TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, uint32_t(page.rows));
        TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, uint32_t(page.rows));
        for (int y = 0; y < page.rows; ++y)
            require(TIFFWriteScanline(tif.get(), const_cast<uint8_t*>(page.ptr(y)), uint32_t(y), 0) >= 0, "mask row");
        require(TIFFWriteDirectory(tif.get()), "mask page");
    }
}

std::vector<uint8_t> array(ObjectStore& db, const std::string& uuid, const std::string& name)
{
    auto read = db.read();
    return db.readGeometryArray(read, db.currentGeometry(read, uuid)->uuid, name);
}

vc::geometry::Surface plane(float offset)
{
    vc::geometry::Surface surface{33, 33, {1, 1}, {}};
    for (int r = 0; r < 33; ++r)
        for (int c = 0; c < 33; ++c) surface.points.push_back({float(c) + offset, float(r), 20});
    return surface;
}
}

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "tifxyz-test";
    auto db = ObjectStore::create(directory.path / "objects.vcobj", options);

    // A tifxyz as VC writes it: a two-page mask at twice the grid and an ink
    // channel rendered at 2x rows and 3x columns.
    const int rows = 8, cols = 10;
    const auto source = directory.path / "segment";
    fs::create_directory(source);
    cv::Mat_<float> x(rows, cols), y(rows, cols), z(rows, cols, 20.f);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) { x(r, c) = float(c); y(r, c) = float(r); }
    z(5, 6) = 0;  // missing coordinates
    writeTiff(source / "x.tif", x); writeTiff(source / "y.tif", y); writeTiff(source / "z.tif", z);
    cv::Mat_<uint8_t> mask(2 * rows, 2 * cols, uint8_t(255));
    mask(7, 9) = 128;  // one covered pixel below 255 invalidates vertex (3,4)
    writeVcMask(source / "mask.tif", mask);
    cv::Mat_<float> ink(2 * rows, 3 * cols);
    for (int i = 0; i < int(ink.total()); ++i) ink(i / ink.cols, i % ink.cols) = float(i) * 0.5f;
    writeTiff(source / "ink.tif", ink, -1, 0, 0);
    std::ofstream(source / "meta.json") << nlohmann::json{{"uuid", util::newUuid()}, {"scale", {1, 1}},
        {"format", "tifxyz"}, {"type", "seg"}, {"tags", {{"approved", {{"user", "reviewer"}}}}}}.dump();

    const auto report = ingestGeometrySource(db, "surface", source);
    require(report.ingested == 1 && report.problems.empty(), "tifxyz with a VC mask ingests");
    std::string uuid;
    {
        auto read = db.read();
        const auto object = db.selectObjects(read, {})[0];
        uuid = object.uuid;
        require(object.tags == std::vector<std::string>{"approved"}, "meta.json status tags become record tags");
    }

    std::vector<uint8_t> expectedValid(rows * cols, 255);
    expectedValid[3 * cols + 4] = 0;
    require(array(db, uuid, "valid") == expectedValid, "mask page 0 becomes per-vertex validity");
    const auto points = array(db, uuid, "points");
    float stored[3]; std::memcpy(stored, points.data() + (3 * cols + 4) * 12, 12);
    require(stored[0] == 4 && stored[1] == 3 && stored[2] == 20, "validity is not baked into points");
    std::memcpy(stored, points.data() + (5 * cols + 6) * 12, 12);
    require(stored[0] == -1 && stored[1] == -1 && stored[2] == -1, "z <= 0 is the missing-coordinate sentinel");
    require(array(db, uuid, "ink") == std::vector<uint8_t>(ink.datastart, ink.dataend), "finer channel keeps exact samples");

    {
        GeometrySnapshot snapshot(db);
        const auto view = snapshot.surface(uuid);
        require(!view.valid(3, 4) && !view.valid(5, 6) && view.valid(0, 0), "views apply validity and sentinels");
        const auto tile = view.readTile(2, 3, 3, 3);
        require(tile.validity.size() == 9 && !tile.valid(1, 1) && tile.valid(0, 0), "tiles carry validity");
        const auto owned = view.load();
        require(owned.validity == expectedValid && owned.channels.size() == 1 && owned.channels[0].name == "ink",
                "owned surfaces carry validity and channels");
        const auto data = vc::geometry::encode(owned);
        require(data.arrays.size() == 3, "encode keeps validity and channels");
    }

    // An ink prediction rendered at 2x attaches as a channel of the stored surface.
    cv::Mat_<uint8_t> prediction(2 * rows, 2 * cols, uint8_t(3));
    writeTiff(directory.path / "prediction.tif", prediction);
    {
        auto txn = db.write();
        importSurfaceChannel(db, txn, uuid, "ink_prediction", directory.path / "prediction.tif");
        txn.commit();
    }
    require(array(db, uuid, "ink_prediction") == std::vector<uint8_t>(prediction.datastart, prediction.dataend),
            "imported channels keep exact samples");
    {
        cv::Mat_<uint8_t> misaligned(2 * rows + 1, 2 * cols, uint8_t(3));
        writeTiff(directory.path / "misaligned.tif", misaligned);
        auto txn = db.write();
        mustFail([&] { importSurfaceChannel(db, txn, uuid, "bad", directory.path / "misaligned.tif"); });
    }

    // Export writes VC's mask (single page, grid resolution) and the channel;
    // ingesting that export reproduces the stored arrays.
    const auto exported = directory.path / "exported";
    exportGeometry(db, uuid, exported);
    const auto exportedMask = readTiffSamples(exported / "mask.tif");
    require(exportedMask.width == uint32_t(cols) && exportedMask.height == uint32_t(rows) &&
            exportedMask.bytes == expectedValid, "valid exports as mask.tif");
    auto other = ObjectStore::create(directory.path / "other.vcobj", options);
    require(ingestGeometrySource(other, "surface", exported).ingested == 1, "export ingests");
    for (const auto* name : {"points", "valid", "ink", "ink_prediction"})
        require(array(other, uuid, name) == array(db, uuid, name), "export round-trips every array");

    // A mask that is not a whole multiple of the grid is rejected, not ignored.
    writeVcMask(source / "mask.tif", cv::Mat_<uint8_t>(rows + 1, cols, uint8_t(255)));
    require(!ingestGeometrySource(other, "surface", source).problems.empty(), "misaligned masks are rejected");

    // Discovery treats invalid vertices as holes: masking the whole overlap
    // removes the link between two coincident planes.
    std::array<std::string, 2> planes;
    for (auto& id : planes) {
        auto txn = db.write();
        ObjectRecord record; record.kind = "surface";
        id = db.upsertObject(txn, record);
        db.putGeometry(txn, id, vc::geometry::encode(plane(0)));
        txn.commit();
    }
    vc::object_links::SurfaceOverlapParams params; params.surfaceUuids = {planes[0], planes[1]};
    require(vc::object_links::linkSurfaceOverlaps(db, params).edgesCreated == 1, "coincident planes link");
    {
        auto txn = db.write();
        for (const auto& edge : db.edgesOf(txn, planes[1])) db.removeEdge(txn, edge.uuid);
        const auto revision = db.currentGeometry(txn, planes[1])->uuid;
        GeometryArray invalid{"valid", "|u1", {33, 33}, std::vector<uint8_t>(33 * 33, 0)};
        db.updateSurfaceChannels(txn, planes[1], {invalid}, {}, revision);
        txn.commit();
    }
    require(vc::object_links::linkSurfaceOverlaps(db, params).edgesCreated == 0, "masked vertices do not link");

    // A folder of surfaces ingests into a session attachment one surface at a
    // time: a broken one is reported and the rest are stored.
    const auto folder = directory.path / "folder";
    for (const auto* name : {"b", "a", "broken"}) {
        const auto surfaceDir = folder / name;
        fs::create_directories(surfaceDir);
        writeTiff(surfaceDir / "x.tif", x); writeTiff(surfaceDir / "y.tif", y);
        if (std::string(name) != "broken") writeTiff(surfaceDir / "z.tif", z);
        std::ofstream(surfaceDir / "meta.json") << nlohmann::json{{"uuid", util::newUuid()}, {"scale", {1, 1}},
            {"format", "tifxyz"}, {"type", "seg"}, {"tags", {"reviewed"}}}.dump();
    }
    fs::create_directory(folder / "notes");
    const auto sessionPath = directory.path / "session.vcobj";
    ObjectStore::create(sessionPath, options);
    ObjectStoreSession session;
    auto attachment = session.attach(sessionPath, options);
    session.takeEvents();
    std::vector<std::string> progressed;
    const auto folderReport = ingestGeometrySource(session, attachment, "surface", folder,
        [&](const auto& update) { if (!update.detail.empty()) progressed.emplace_back(update.detail); },
        "{}", {"folder", "reviewed"});
    require(folderReport.ingested == 2 && folderReport.problems.size() == 1 &&
            folderReport.problems[0].find("broken") != std::string::npos, "a folder ingests each surface on its own");
    require(progressed == std::vector<std::string>{"a", "b", "broken"}, "surfaces ingest in name order");
    bool surfacesCommitted = false;
    for (const auto& event : session.takeEvents())
        surfacesCommitted |= event.kind == StoreSessionEvent::Kind::Committed && event.touches({"surface"});
    require(surfacesCommitted, "session observers see the ingested surfaces");
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        const auto surfaces = store.objectsByKind(txn, "surface");
        require(surfaces.size() == 2, "the session store holds both surfaces");
        for (auto object : surfaces) {
            std::sort(object.tags.begin(), object.tags.end());
            require(object.tags == std::vector<std::string>{"folder", "reviewed"}, "import tags join the source's own");
        }
    });
    require(!ingestGeometrySource(session, attachment, "surface", folder / "notes").problems.empty(),
            "a folder without surfaces is reported");
    auto readOnly = options; readOnly.readOnly = true;
    attachment = session.replace(attachment, sessionPath, readOnly);
    mustFail([&] { ingestGeometrySource(session, attachment, "surface", folder / "a"); });
    std::cout << "surface tifxyz workflow passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
