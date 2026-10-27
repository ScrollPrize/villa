#include "vc/core/types/SheetCollection.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include <sqlite3.h>
#include <zlib.h>

// zstd is compiled in only when the build links it (CMake defines VC_SHEET_HAVE_ZSTD),
// so the compiled code and the linked library can never disagree. Our files are zlib by
// default; zstd appears only when the writer had zstandard available.
#ifdef VC_SHEET_HAVE_ZSTD
#include <zstd.h>
#define VC_SHEET_ZSTD 1
#endif

namespace vc::sheets
{
namespace
{
constexpr uint8_t kCodecZlib = 1;
constexpr uint8_t kCodecZstd = 2;

void check(int rc, sqlite3* db)
{
    if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE)
        throw std::runtime_error(db ? sqlite3_errmsg(db) : "sqlite error");
}

// RAII prepared statement, mirroring FiberCollection's helper.
struct Statement {
    sqlite3* db_{};
    sqlite3_stmt* value{};
    Statement(sqlite3* db, const char* sql) : db_(db)
    {
        check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr), db);
    }
    ~Statement() { sqlite3_finalize(value); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    void bind(int i, int64_t n) { check(sqlite3_bind_int64(value, i, n), db_); }
    void bindText(int i, const std::string& s)
    {
        check(sqlite3_bind_text(value, i, s.c_str(), -1, SQLITE_TRANSIENT), db_);
    }
    bool row()
    {
        const int rc = sqlite3_step(value);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        throw std::runtime_error(sqlite3_errmsg(db_));
    }
    void reset() { check(sqlite3_reset(value), db_); }
};

std::string columnText(sqlite3_stmt* s, int col)
{
    const auto* p = sqlite3_column_text(s, col);
    return p ? std::string(reinterpret_cast<const char*>(p), sqlite3_column_bytes(s, col)) : "";
}

// ---- little-endian scalar reads from the decompressed record ----
uint16_t readU16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
float readF32(const uint8_t* p)
{
    uint32_t u = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<uint8_t> zlibInflate(const uint8_t* src, size_t srcLen)
{
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK)
        throw std::runtime_error("inflateInit failed");
    zs.next_in = const_cast<Bytef*>(src);
    zs.avail_in = uInt(srcLen);
    std::vector<uint8_t> out;
    out.resize(std::max<size_t>(srcLen * 4, 4096));
    size_t have = 0;
    int rc = Z_OK;
    do {
        if (have == out.size()) out.resize(out.size() * 2);
        zs.next_out = out.data() + have;
        zs.avail_out = uInt(out.size() - have);
        rc = inflate(&zs, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) {
            inflateEnd(&zs);
            throw std::runtime_error("zlib inflate failed");
        }
        have = out.size() - zs.avail_out;
    } while (rc != Z_STREAM_END);
    inflateEnd(&zs);
    out.resize(have);
    return out;
}

std::vector<uint8_t> zstdDecompress(const uint8_t* src, size_t srcLen)
{
#ifdef VC_SHEET_ZSTD
    const unsigned long long n = ZSTD_getFrameContentSize(src, srcLen);
    if (n == ZSTD_CONTENTSIZE_ERROR || n == ZSTD_CONTENTSIZE_UNKNOWN)
        throw std::runtime_error("bad zstd frame in sheet blob");
    const size_t outLen = size_t(n);
    std::vector<uint8_t> out(outLen);
    const size_t got = ZSTD_decompress(out.data(), out.size(), src, srcLen);
    if (ZSTD_isError(got) || got != out.size())
        throw std::runtime_error("zstd decompress failed");
    return out;
#else
    (void)src;
    (void)srcLen;
    throw std::runtime_error("sheet blob is zstd-coded but zstd is not compiled in");
#endif
}

// grid_triangles (hercunet.meshes.fit.grid_triangles): each full quad -> 2 triangles
// whose three corners are all in the footprint, indices into the flat nu*nv grid.
void appendGridTriangles(
    int64_t nu, int64_t nv, const std::vector<uint8_t>& foot,
    const std::vector<int64_t>& remap, std::vector<std::array<int64_t, 3>>& tris)
{
    auto in = [&](int64_t i, int64_t j) { return foot[i * nv + j] != 0; };
    auto id = [&](int64_t i, int64_t j) { return remap[i * nv + j]; };
    for (int64_t i = 0; i + 1 < nu; ++i) {
        for (int64_t j = 0; j + 1 < nv; ++j) {
            const bool fa = in(i, j), fb = in(i + 1, j), fc = in(i, j + 1), fd = in(i + 1, j + 1);
            if (fa && fb && fc)
                tris.push_back({id(i, j), id(i + 1, j), id(i, j + 1)});
            if (fb && fd && fc)
                tris.push_back({id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
        }
    }
}
}  // namespace

Mesh decodeSheetBlob(const unsigned char* data, size_t size)
{
    if (size < 2)
        throw std::runtime_error("sheet blob too small");
    const uint8_t codec = data[0];
    std::vector<uint8_t> raw;
    if (codec == kCodecZlib)
        raw = zlibInflate(data + 1, size - 1);
    else if (codec == kCodecZstd)
        raw = zstdDecompress(data + 1, size - 1);
    else
        throw std::runtime_error("unknown sheet blob codec");

    if (raw.size() < 32 || std::memcmp(raw.data(), "ASV1", 4) != 0)
        throw std::runtime_error("not an .asv sheet blob");
    Mesh m;
    m.nu = readU16(raw.data() + 4);
    m.nv = readU16(raw.data() + 6);
    size_t off = 8;
    float lo[3], hi[3];
    for (int i = 0; i < 3; ++i) { lo[i] = readF32(raw.data() + off); off += 4; }
    for (int i = 0; i < 3; ++i) { hi[i] = readF32(raw.data() + off); off += 4; }
    const int64_t cells = m.nu * m.nv;
    const size_t nbits = size_t((cells + 7) / 8);
    if (off + nbits > raw.size())
        throw std::runtime_error("sheet blob truncated (footprint)");
    // numpy.packbits default order is big-endian (MSB first).
    m.foot.resize(size_t(cells));
    for (int64_t k = 0; k < cells; ++k)
        m.foot[size_t(k)] = (raw[off + size_t(k / 8)] >> (7 - (k & 7))) & 1;
    off += nbits;

    int64_t nvert = 0;
    std::vector<int64_t> remap(size_t(cells), -1);
    for (int64_t k = 0; k < cells; ++k)
        if (m.foot[size_t(k)]) remap[size_t(k)] = nvert++;

    if (off + size_t(nvert) * 3 * 2 > raw.size())
        throw std::runtime_error("sheet blob truncated (verts)");
    double rng[3];
    for (int i = 0; i < 3; ++i) rng[i] = std::max<double>(double(hi[i]) - double(lo[i]), 1e-6);
    m.verts.resize(size_t(nvert));
    for (int64_t vi = 0; vi < nvert; ++vi) {
        Point p;
        for (int c = 0; c < 3; ++c) {
            const uint16_t q = readU16(raw.data() + off);
            off += 2;
            p[c] = double(q) / 65535.0 * rng[c] + double(lo[c]);
        }
        m.verts[size_t(vi)] = p;
    }
    appendGridTriangles(m.nu, m.nv, m.foot, remap, m.tris);
    return m;
}

bool intersectTrianglePlane(
    const Point& a, const Point& b, const Point& c,
    const Point& origin, const Point& n, std::array<Point, 2>& out)
{
    auto sd = [&](const Point& p) {
        return n[0] * (p[0] - origin[0]) + n[1] * (p[1] - origin[1]) + n[2] * (p[2] - origin[2]);
    };
    const double da = sd(a), db = sd(b), dc = sd(c);
    const Point* v[3] = {&a, &b, &c};
    const double d[3] = {da, db, dc};
    // Collect crossing points on the three edges (i -> i+1) whose endpoints straddle 0.
    Point hits[3];
    int nh = 0;
    for (int i = 0; i < 3 && nh < 3; ++i) {
        const int j = (i + 1) % 3;
        const double di = d[i], dj = d[j];
        if ((di < 0 && dj >= 0) || (di >= 0 && dj < 0)) {
            const double t = di / (di - dj);
            Point p;
            for (int k = 0; k < 3; ++k)
                p[k] = (*v[i])[k] + t * ((*v[j])[k] - (*v[i])[k]);
            hits[nh++] = p;
        }
    }
    if (nh < 2) return false;  // no crossing (or a single-vertex touch)
    out[0] = hits[0];
    out[1] = hits[1];
    return true;
}

Point ViewRegion::project(const Point& p) const
{
    const Point d = {p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]};
    return {d[0] * u[0] + d[1] * u[1] + d[2] * u[2],
            d[0] * v[0] + d[1] * v[1] + d[2] * v[2],
            d[0] * n[0] + d[1] * n[1] + d[2] * n[2]};
}

// ------------------------------------------------------------------ impl

struct SheetCollection::Impl {
    sqlite3* db{};
    ~Impl()
    {
        if (db) sqlite3_close(db);
    }
};

namespace
{
const char* kSummaryColumns =
    "id,name,window,family,nu,nv,vert_count,min_x,max_x,min_y,max_y,min_z,max_z";

Summary readSummary(sqlite3_stmt* s)
{
    Summary out;
    out.id = sqlite3_column_int64(s, 0);
    out.name = columnText(s, 1);
    out.window = columnText(s, 2);
    out.family = columnText(s, 3);
    out.nu = sqlite3_column_int64(s, 4);
    out.nv = sqlite3_column_int64(s, 5);
    out.vertCount = sqlite3_column_int64(s, 6);
    for (int i = 0; i < 3; ++i) {
        out.bounds.lower[i] = sqlite3_column_double(s, 7 + 2 * i);
        out.bounds.upper[i] = sqlite3_column_double(s, 8 + 2 * i);
    }
    return out;
}

std::string familyClause(FamilyFilter f)
{
    switch (f) {
        case FamilyFilter::Recto: return " AND family='R'";
        case FamilyFilter::Verso: return " AND family='V'";
        case FamilyFilter::None: return " AND family=''";
        case FamilyFilter::All: default: return "";
    }
}
}  // namespace

SheetCollection::SheetCollection(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>())
{
    const std::string name = path.string();
    const int rc = sqlite3_open_v2(
        name.c_str(), &impl_->db, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
    if (rc != SQLITE_OK) {
        const std::string msg = impl_->db ? sqlite3_errmsg(impl_->db) : "cannot open .asv";
        throw std::runtime_error(msg);
    }
    sqlite3_busy_timeout(impl_->db, 1000);
    check(sqlite3_exec(impl_->db,
                       "PRAGMA query_only=ON; PRAGMA cache_size=-2048; PRAGMA mmap_size=0; "
                       "PRAGMA temp_store=FILE;",
                       nullptr, nullptr, nullptr),
          impl_->db);
    // Validate it is a complete .asv: the tables exist and the file was finished.
    if (metadata("format_version") != "1")
        throw std::runtime_error("not an .asv file (format_version != 1)");
    if (metadata("complete") != "true")
        throw std::runtime_error(".asv file is incomplete");
    Statement probe(impl_->db, "SELECT id FROM sheet_bounds LIMIT 0");
    probe.row();
}

SheetCollection::~SheetCollection() = default;

std::string SheetCollection::metadata(const std::string& key) const
{
    Statement s(impl_->db, "SELECT value FROM metadata WHERE key=?");
    s.bindText(1, key);
    return s.row() ? columnText(s.value, 0) : std::string{};
}

int64_t SheetCollection::sheetCount() const
{
    Statement s(impl_->db, "SELECT COUNT(*) FROM sheets");
    return s.row() ? sqlite3_column_int64(s.value, 0) : 0;
}

Summary SheetCollection::summary(int64_t sheetId) const
{
    const std::string sql = std::string("SELECT ") + kSummaryColumns + " FROM sheets WHERE id=?";
    Statement s(impl_->db, sql.c_str());
    s.bind(1, sheetId);
    if (!s.row())
        throw std::runtime_error("no such sheet id");
    return readSummary(s.value);
}

std::vector<Summary> SheetCollection::catalogBySize(
    int64_t beforeVerts, int64_t afterId, int limit, int64_t minVerts, FamilyFilter family) const
{
    // Descending vert_count, then ascending id; paginate strictly after
    // (beforeVerts, afterId) so repeated calls stream the whole catalog.
    const std::string sql = std::string("SELECT ") + kSummaryColumns +
        " FROM sheets WHERE vert_count>=?" + familyClause(family) +
        " AND (vert_count<? OR (vert_count=? AND id>?)) ORDER BY vert_count DESC, id ASC LIMIT ?";
    Statement s(impl_->db, sql.c_str());
    s.bind(1, minVerts);
    s.bind(2, beforeVerts);
    s.bind(3, beforeVerts);
    s.bind(4, afterId);
    s.bind(5, int64_t(limit));
    std::vector<Summary> out;
    while (s.row())
        out.push_back(readSummary(s.value));
    return out;
}

Mesh SheetCollection::mesh(int64_t sheetId) const
{
    Statement s(impl_->db, "SELECT payload FROM sheets WHERE id=?");
    s.bind(1, sheetId);
    if (!s.row())
        throw std::runtime_error("no such sheet id");
    const auto* blob = static_cast<const unsigned char*>(sqlite3_column_blob(s.value, 0));
    const int n = sqlite3_column_bytes(s.value, 0);
    return decodeSheetBlob(blob, size_t(n));
}

std::string SheetCollection::annotation(int64_t sheetId) const
{
    Statement s(impl_->db, "SELECT annotation FROM sheets WHERE id=?");
    s.bind(1, sheetId);
    return s.row() ? columnText(s.value, 0) : std::string{};
}

std::vector<SheetMesh> SheetCollection::region(
    const Bounds& box, size_t maxSheets, int64_t minVerts, FamilyFilter family,
    const std::atomic_bool* cancelled, size_t byteBudget) const
{
    std::vector<SheetMesh> out;
    // Candidates: bbox intersects the query box, largest first so a cap keeps the most
    // visible sheets deterministically.
    const std::string sql =
        "SELECT s.id,s.family,s.payload,s.min_x,s.max_x,s.min_y,s.max_y,s.min_z,s.max_z "
        "FROM sheet_bounds b JOIN sheets s ON s.id=b.id WHERE "
        "b.max_x>=? AND b.min_x<=? AND b.max_y>=? AND b.min_y<=? AND b.max_z>=? AND b.min_z<=? "
        "AND s.vert_count>=?" + familyClause(family) + " ORDER BY s.vert_count DESC, s.id ASC";
    Statement s(impl_->db, sql.c_str());
    sqlite3_bind_double(s.value, 1, box.lower[0]);
    sqlite3_bind_double(s.value, 2, box.upper[0]);
    sqlite3_bind_double(s.value, 3, box.lower[1]);
    sqlite3_bind_double(s.value, 4, box.upper[1]);
    sqlite3_bind_double(s.value, 5, box.lower[2]);
    sqlite3_bind_double(s.value, 6, box.upper[2]);
    s.bind(7, minVerts);

    size_t decoded = 0;
    while (s.row()) {
        if (cancelled && cancelled->load())
            break;
        if (maxSheets != 0 && out.size() >= maxSheets)
            break;
        const auto* blob = static_cast<const unsigned char*>(sqlite3_column_blob(s.value, 2));
        const int n = sqlite3_column_bytes(s.value, 2);
        if (decoded + size_t(n) > byteBudget && !out.empty())
            break;
        SheetMesh sm;
        sm.id = sqlite3_column_int64(s.value, 0);
        sm.family = columnText(s.value, 1);
        for (int i = 0; i < 3; ++i) {
            sm.bounds.lower[i] = sqlite3_column_double(s.value, 3 + 2 * i);
            sm.bounds.upper[i] = sqlite3_column_double(s.value, 4 + 2 * i);
        }
        try {
            sm.mesh = decodeSheetBlob(blob, size_t(n));
        } catch (const std::exception&) {
            continue;  // skip a corrupt sheet rather than fail the whole region
        }
        decoded += size_t(n);
        out.push_back(std::move(sm));
    }
    return out;
}

RegionPage SheetCollection::viewRegion(
    const ViewRegion& view, size_t maximum, int64_t /*selected*/, size_t byteBudget,
    const std::atomic_bool* cancelled, int64_t minVerts, FamilyFilter family) const
{
    // Convenience: fetch the region, then intersect each mesh with the slice plane.
    // The interactive overlay instead caches region() and intersects per frame.
    RegionPage page;
    const auto meshes = region(view.box, maximum, minVerts, family, cancelled, byteBudget);
    std::array<Point, 2> seg;
    for (const auto& sm : meshes) {
        SheetRuns runs;
        runs.sheetId = sm.id;
        runs.family = sm.family;
        for (const auto& t : sm.mesh.tris) {
            if (intersectTrianglePlane(
                    sm.mesh.verts[size_t(t[0])], sm.mesh.verts[size_t(t[1])], sm.mesh.verts[size_t(t[2])],
                    view.origin, view.n, seg)) {
                Polyline pl;
                pl.points = {seg[0], seg[1]};
                runs.curves.push_back(std::move(pl));
            }
        }
        if (!runs.curves.empty())
            page.sheets.push_back(std::move(runs));
    }
    return page;
}

}  // namespace vc::sheets
