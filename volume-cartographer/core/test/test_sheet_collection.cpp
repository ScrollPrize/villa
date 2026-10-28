#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/types/SheetCollection.hpp"

#include <sqlite3.h>
#include <zlib.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

// Writes a minimal .asv in the on-disk format (see hercunet/meshes/asv.py) so the
// reader can be tested without Python: one zlib-coded sheet whose payload is the
// "ASV1" record (nu, nv, lo, hi, packed footprint, uint16 verts).
namespace
{
void putU16(std::vector<uint8_t>& b, uint16_t v)
{
    b.push_back(uint8_t(v & 0xff));
    b.push_back(uint8_t(v >> 8));
}
void putF32(std::vector<uint8_t>& b, float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    for (int i = 0; i < 4; ++i)
        b.push_back(uint8_t((u >> (8 * i)) & 0xff));
}

// codec byte (1 = zlib) + zlib.compress(raw), matching AsvWriter.pack_sheet.
std::vector<uint8_t> buildSheetBlob(
    int nu, int nv, const std::vector<std::array<float, 3>>& verts)  // full footprint
{
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (const auto& v : verts)
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::min(lo[c], v[c]);
            hi[c] = std::max(hi[c], v[c]);
        }
    std::vector<uint8_t> raw = {'A', 'S', 'V', '1'};
    putU16(raw, uint16_t(nu));
    putU16(raw, uint16_t(nv));
    for (int c = 0; c < 3; ++c) putF32(raw, lo[c]);
    for (int c = 0; c < 3; ++c) putF32(raw, hi[c]);
    const int cells = nu * nv;
    std::vector<uint8_t> packed(size_t((cells + 7) / 8), 0);  // full footprint: all bits set
    for (int k = 0; k < cells; ++k)
        packed[size_t(k / 8)] |= uint8_t(1u << (7 - (k & 7)));  // numpy big bit order
    raw.insert(raw.end(), packed.begin(), packed.end());
    for (const auto& v : verts)
        for (int c = 0; c < 3; ++c) {
            const double rng = std::max<double>(double(hi[c]) - double(lo[c]), 1e-6);
            long q = std::lround((double(v[c]) - double(lo[c])) / rng * 65535.0);
            q = std::max<long>(0, std::min<long>(65535, q));
            putU16(raw, uint16_t(q));
        }
    uLongf bound = compressBound(uLong(raw.size()));
    std::vector<uint8_t> comp(bound);
    REQUIRE(compress2(comp.data(), &bound, raw.data(), uLong(raw.size()), 9) == Z_OK);
    comp.resize(bound);
    std::vector<uint8_t> blob = {1};  // codec id: zlib
    blob.insert(blob.end(), comp.begin(), comp.end());
    return blob;
}

void exec(sqlite3* db, const char* sql) { REQUIRE(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK); }

void setMeta(sqlite3* db, const char* key, const char* value)
{
    sqlite3_stmt* s = nullptr;
    REQUIRE(sqlite3_prepare_v2(db, "INSERT INTO metadata(key,value) VALUES(?,?)", -1, &s, nullptr) == SQLITE_OK);
    sqlite3_bind_text(s, 1, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(s, 2, value, -1, SQLITE_TRANSIENT);
    REQUIRE(sqlite3_step(s) == SQLITE_DONE);
    sqlite3_finalize(s);
}

// A flat 4x4 sheet in the X-Z plane at y=40, x,z in [10,40]. An XY slice at z=25 cuts
// it along y=40, z=25, x in [10,40].
std::string writeFixture()
{
    const auto path = (std::filesystem::temp_directory_path() /
                       ("vc_sheet_test_" + std::to_string(::getpid()) + ".asv")).string();
    std::filesystem::remove(path);
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    exec(db,
         "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
         "CREATE TABLE sheets(id INTEGER PRIMARY KEY, name TEXT NOT NULL, window TEXT NOT NULL DEFAULT '',"
         " family TEXT NOT NULL DEFAULT '', nu INTEGER NOT NULL, nv INTEGER NOT NULL,"
         " vert_count INTEGER NOT NULL CHECK(vert_count >= 3), min_x REAL,max_x REAL,min_y REAL,max_y REAL,"
         " min_z REAL,max_z REAL, payload BLOB NOT NULL, annotation TEXT NOT NULL DEFAULT '{}');"
         "CREATE VIRTUAL TABLE sheet_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);");
    setMeta(db, "format_version", "1");
    setMeta(db, "complete", "true");
    setMeta(db, "codec", "zlib");
    setMeta(db, "grid_step_um", "30.0");

    const int nu = 4, nv = 4;
    std::vector<std::array<float, 3>> verts;
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j)
            verts.push_back({10.0f + i * 10.0f, 40.0f, 10.0f + j * 10.0f});
    const auto blob = buildSheetBlob(nu, nv, verts);

    sqlite3_stmt* s = nullptr;
    REQUIRE(sqlite3_prepare_v2(
                db,
                "INSERT INTO sheets(id,name,window,family,nu,nv,vert_count,min_x,max_x,min_y,max_y,min_z,max_z,"
                "payload,annotation) VALUES(1,'s','','',?,?,?,10,40,40,40,10,40,?,'{}')",
                -1, &s, nullptr) == SQLITE_OK);
    sqlite3_bind_int(s, 1, nu);
    sqlite3_bind_int(s, 2, nv);
    sqlite3_bind_int(s, 3, nu * nv);
    sqlite3_bind_blob(s, 4, blob.data(), int(blob.size()), SQLITE_TRANSIENT);
    REQUIRE(sqlite3_step(s) == SQLITE_DONE);
    sqlite3_finalize(s);
    exec(db, "INSERT INTO sheet_bounds VALUES(1,10,40,40,40,10,40)");
    sqlite3_close(db);
    return path;
}

// Like writeFixture but the sheets table carries the (nullable) quality columns, with
// three co-located sheets whose `confidence` is 0.9, 0.2 and NULL. Used to test the
// quality filter: thresholds drop low sheets, NULL always passes.
std::string writeQualityFixture()
{
    const auto path = (std::filesystem::temp_directory_path() /
                       ("asv_q_" + std::to_string(::getpid()) + ".asv")).string();
    std::filesystem::remove(path);
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    exec(db,
         "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
         "CREATE TABLE sheets(id INTEGER PRIMARY KEY, name TEXT NOT NULL, window TEXT NOT NULL DEFAULT '',"
         " family TEXT NOT NULL DEFAULT '', nu INTEGER NOT NULL, nv INTEGER NOT NULL,"
         " vert_count INTEGER NOT NULL CHECK(vert_count >= 3),"
         " area_mm2 REAL, n_voxels INTEGER, self_mass REAL, confidence REAL, mass_frac REAL,"
         " fit_rms_um REAL, coverage REAL,"
         " min_x REAL,max_x REAL,min_y REAL,max_y REAL, min_z REAL,max_z REAL,"
         " payload BLOB NOT NULL, annotation TEXT NOT NULL DEFAULT '{}');"
         "CREATE VIRTUAL TABLE sheet_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);");
    setMeta(db, "format_version", "1");
    setMeta(db, "complete", "true");
    setMeta(db, "codec", "zlib");
    setMeta(db, "grid_step_um", "30.0");

    const int nu = 4, nv = 4;
    std::vector<std::array<float, 3>> verts;
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j)
            verts.push_back({10.0f + i * 10.0f, 40.0f, 10.0f + j * 10.0f});
    const auto blob = buildSheetBlob(nu, nv, verts);

    // id, confidence (NaN sentinel -> bind NULL)
    const double kNull = std::nan("");
    const std::array<std::pair<int, double>, 3> rows = {{{1, 0.9}, {2, 0.2}, {3, kNull}}};
    for (const auto& [id, conf] : rows) {
        sqlite3_stmt* s = nullptr;
        REQUIRE(sqlite3_prepare_v2(
                    db,
                    "INSERT INTO sheets(id,name,window,family,nu,nv,vert_count,confidence,coverage,mass_frac,"
                    "fit_rms_um,min_x,max_x,min_y,max_y,min_z,max_z,payload,annotation) "
                    "VALUES(?,'s','','',?,?,?,?,1.0,1.0,10.0,10,40,40,40,10,40,?,'{}')",
                    -1, &s, nullptr) == SQLITE_OK);
        sqlite3_bind_int(s, 1, id);
        sqlite3_bind_int(s, 2, nu);
        sqlite3_bind_int(s, 3, nv);
        sqlite3_bind_int(s, 4, nu * nv);
        if (std::isnan(conf)) sqlite3_bind_null(s, 5);
        else sqlite3_bind_double(s, 5, conf);
        sqlite3_bind_blob(s, 6, blob.data(), int(blob.size()), SQLITE_TRANSIENT);
        REQUIRE(sqlite3_step(s) == SQLITE_DONE);
        sqlite3_finalize(s);
        const std::string ins =
            "INSERT INTO sheet_bounds VALUES(" + std::to_string(id) + ",10,40,40,40,10,40)";
        exec(db, ins.c_str());
    }
    sqlite3_close(db);
    return path;
}
}  // namespace

TEST_CASE("sheet collection: open, metadata, decode")
{
    using namespace vc::sheets;
    const auto path = writeFixture();
    SheetCollection sc(path);

    CHECK(sc.sheetCount() == 1);
    CHECK(sc.metadata("format_version") == "1");
    CHECK(sc.metadata("grid_step_um") == "30.0");

    const auto cat = sc.catalogBySize(INT64_MAX, 0, 100, 0, FamilyFilter::All);
    CHECK(cat.size() == 1);
    CHECK(cat[0].vertCount == 16);

    const Mesh m = sc.mesh(1);
    CHECK(m.nu == 4);
    CHECK(m.nv == 4);
    CHECK(m.verts.size() == 16);
    CHECK(m.tris.size() == 3 * 3 * 2);  // full 3x3 quads, 2 triangles each
    double miny = 1e9, maxy = -1e9, minx = 1e9, maxx = -1e9;
    for (const auto& v : m.verts) {
        miny = std::min(miny, v[1]); maxy = std::max(maxy, v[1]);
        minx = std::min(minx, v[0]); maxx = std::max(maxx, v[0]);
    }
    CHECK(miny == doctest::Approx(40).epsilon(0.001));
    CHECK(maxy == doctest::Approx(40).epsilon(0.001));
    CHECK(minx == doctest::Approx(10).epsilon(0.01));
    CHECK(maxx == doctest::Approx(40).epsilon(0.01));
    std::filesystem::remove(path);
}

TEST_CASE("sheet collection: region query and plane intersection")
{
    using namespace vc::sheets;
    const auto path = writeFixture();
    SheetCollection sc(path);

    Bounds box{{0, 0, 0}, {100, 100, 100}};
    const auto meshes = sc.region(box, 0, 0, FamilyFilter::All);
    REQUIRE(meshes.size() == 1);
    CHECK(meshes[0].id == 1);
    CHECK(meshes[0].bounds.lower[2] == doctest::Approx(10));
    CHECK(meshes[0].bounds.upper[2] == doctest::Approx(40));

    // XY slice at z=25 -> crossing curve at z=25, y=40.
    ViewRegion view;
    view.origin = {0, 0, 25};
    view.u = {1, 0, 0};
    view.v = {0, 1, 0};
    view.n = {0, 0, 1};
    view.box = {{0, 0, 24}, {100, 100, 26}};
    view.local = {{0, 0, -1}, {100, 100, 1}};
    const auto page = sc.viewRegion(view);
    REQUIRE(page.sheets.size() == 1);
    bool haveCurve = false;
    for (const auto& c : page.sheets[0].curves)
        for (const auto& p : c.points) {
            CHECK(p[2] == doctest::Approx(25).epsilon(0.01));
            CHECK(p[1] == doctest::Approx(40).epsilon(0.01));
            haveCurve = true;
        }
    CHECK(haveCurve);

    // A slab far from the sheet in z returns nothing.
    Bounds empty{{0, 0, 200}, {100, 100, 300}};
    CHECK(sc.region(empty, 0, 0, FamilyFilter::All).empty());
    std::filesystem::remove(path);
}

TEST_CASE("sheet collection: quality filter")
{
    using namespace vc::sheets;
    const Bounds box{{0, 0, 0}, {100, 100, 100}};

    // Quality columns present: confidence 0.9 (id 1), 0.2 (id 2), NULL (id 3).
    {
        const auto path = writeQualityFixture();
        SheetCollection sc(path);
        const auto all = sc.region(box, 0, 0, FamilyFilter::All);  // no filter -> all three
        REQUIRE(all.size() == 3);
        // Metric values are read onto SheetMesh (the colour-by-metric data path).
        std::map<int64_t, const SheetMesh*> byId;
        for (const auto& m : all) byId[m.id] = &m;
        CHECK(byId[1]->confidence == doctest::Approx(0.9));
        CHECK(byId[2]->confidence == doctest::Approx(0.2));
        CHECK(std::isnan(byId[3]->confidence));          // NULL -> NaN
        CHECK(byId[1]->coverage == doctest::Approx(1.0));
        CHECK(byId[1]->fitRmsUm == doctest::Approx(10.0));

        QualityFilter q;
        q.minConfidence = 0.5;
        const auto kept = sc.region(box, 0, 0, FamilyFilter::All, nullptr, 64u << 20, q);
        // Drops the 0.2 sheet; keeps 0.9 and the NULL one (NULL always passes).
        REQUIRE(kept.size() == 2);
        std::set<int64_t> ids;
        for (const auto& m : kept) ids.insert(m.id);
        CHECK(ids.count(1) == 1);
        CHECK(ids.count(2) == 0);
        CHECK(ids.count(3) == 1);

        // fit_rms_um is 10 for all; a 5 µm cap drops them all, a 20 µm cap keeps them.
        QualityFilter qr;
        qr.maxFitRmsUm = 5.0;
        CHECK(sc.region(box, 0, 0, FamilyFilter::All, nullptr, 64u << 20, qr).empty());
        qr.maxFitRmsUm = 20.0;
        CHECK(sc.region(box, 0, 0, FamilyFilter::All, nullptr, 64u << 20, qr).size() == 3);
        std::filesystem::remove(path);
    }

    // Columns absent (older schema): a quality filter must be ignored, not error.
    {
        const auto path = writeFixture();
        SheetCollection sc(path);
        QualityFilter q;
        q.minConfidence = 0.99;
        q.maxFitRmsUm = 1.0;
        CHECK(sc.region(box, 0, 0, FamilyFilter::All, nullptr, 64u << 20, q).size() == 1);
        std::filesystem::remove(path);
    }
}
