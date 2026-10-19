#pragma once
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vc/core/objects/GeometryIngest.hpp"
#include "vc/core/objects/GeometryExport.hpp"

namespace nb = nanobind;
namespace vs = vc::storage;
namespace vo = vc::objects;

namespace {
std::string jsonDumps(const nb::handle& obj)
{
    return nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(obj));
}

nb::object jsonLoads(const std::string& text)
{
    return nb::cast<nb::object>(nb::module_::import_("json").attr("loads")(text));
}

bool has(const nb::dict& d, const char* key) { return d.contains(key) && !d[key].is_none(); }

std::string getStr(const nb::dict& d, const char* key, const std::string& fallback = "")
{
    return has(d, key) ? nb::cast<std::string>(d[key]) : fallback;
}

int64_t getInt(const nb::dict& d, const char* key, int64_t fallback = 0)
{
    return has(d, key) ? nb::cast<int64_t>(d[key]) : fallback;
}

double getFloat(const nb::dict& d, const char* key, double fallback = 0.0)
{
    return has(d, key) ? nb::cast<double>(d[key]) : fallback;
}

bool getBool(const nb::dict& d, const char* key, bool fallback = false)
{
    return has(d, key) ? nb::cast<bool>(d[key]) : fallback;
}

std::vector<std::string> getTags(const nb::dict& d)
{
    std::vector<std::string> out;
    if (has(d, "tags"))
        for (nb::handle t : d["tags"])
            out.push_back(nb::cast<std::string>(t));
    return out;
}

std::optional<std::array<double, 6>> getBbox(const nb::dict& d)
{
    if (!has(d, "bbox")) return std::nullopt;
    nb::sequence seq = nb::cast<nb::sequence>(d["bbox"]);
    if (nb::len(seq) != 6)
        throw std::runtime_error("bbox must have 6 numbers");
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = nb::cast<double>(seq[i]);
    return box;
}

nb::object bboxToPy(const std::optional<std::array<double, 6>>& box)
{
    if (!box) return nb::none();
    nb::list out;
    for (double v : *box) out.append(v);
    return out;
}

vs::Anchor anchorFromPy(const nb::handle& obj)
{
    vs::Anchor a;
    if (obj.is_none()) return a;
    nb::dict d = nb::cast<nb::dict>(obj);
    if (has(d, "index")) a.index = nb::cast<int64_t>(d["index"]);
    for (const char* key : {"position", "direction"}) {
        if (!has(d, key)) continue;
        nb::sequence seq = nb::cast<nb::sequence>(d[key]);
        if (nb::len(seq) != 3)
            throw std::runtime_error(std::string("anchor ") + key + " must have 3 numbers");
        std::array<double, 3> v{nb::cast<double>(seq[0]), nb::cast<double>(seq[1]), nb::cast<double>(seq[2])};
        if (std::string(key) == "position") a.position = v; else a.direction = v;
    }
    return a;
}

nb::dict anchorToPy(const vs::Anchor& a)
{
    nb::dict d;
    if (a.index) d["index"] = *a.index;
    if (a.position) d["position"] = nb::make_tuple((*a.position)[0], (*a.position)[1], (*a.position)[2]);
    if (a.direction) d["direction"] = nb::make_tuple((*a.direction)[0], (*a.direction)[1], (*a.direction)[2]);
    return d;
}

nb::str pyStr(const std::string& s) { return nb::str(s.data(), s.size()); }

void commonToPy(nb::dict& d, const vs::RecordCommon& r)
{
    d["uuid"] = r.uuid;
    d["kind"] = r.kind;
    d["generation"] = r.generation;
    d["created_by"] = r.createdBy;
    d["created_at"] = r.createdAt;
    d["created_host"] = r.createdHost;
    d["changed_by"] = r.changedBy;
    d["changed_at"] = r.changedAt;
    d["changed_host"] = r.changedHost;
    d["approval_state"] = r.approvalState;
    d["approved_by"] = r.approvedBy ? nb::object(pyStr(*r.approvedBy)) : nb::none();
    d["approved_at"] = r.approvedAt ? nb::object(pyStr(*r.approvedAt)) : nb::none();
    d["approved_host"] = r.approvedHost ? nb::object(pyStr(*r.approvedHost)) : nb::none();
    d["approved_generation"] = r.approvedGeneration ? nb::object(nb::int_(*r.approvedGeneration)) : nb::none();
    nb::list tags;
    for (const auto& t : r.tags) tags.append(pyStr(t));
    d["tags"] = tags;
    d["meta"] = jsonLoads(r.meta);
}

nb::dict objectToPy(const vs::ObjectRecord& r)
{
    nb::dict d;
    commonToPy(d, r);
    d["name"] = r.name;
    d["format_version"] = r.formatVersion;
    d["storage_kind"] = r.storageKind;
    d["storage_ref"] = r.storageRef;
    d["data_generation"] = r.dataGeneration;
    d["data_checksum"] = r.dataChecksum ? nb::object(pyStr(*r.dataChecksum)) : nb::none();
    if (r.fiber) {
        nb::dict f;
        f["family"] = r.fiber->family;
        f["manual_family"] = r.fiber->manualFamily;
        f["control_point_count"] = r.fiber->controlPointCount;
        f["line_point_count"] = r.fiber->linePointCount;
        f["length"] = r.fiber->length;
        f["width"] = r.fiber->width;
        f["width_gap_fraction"] = r.fiber->widthGapFraction;
        f["optimization_mode"] = r.fiber->optimizationMode;
        f["bbox"] = bboxToPy(r.fiber->bbox);
        d["fiber"] = f;
    }
    if (r.patch) {
        nb::dict p;
        p["grid_width"] = r.patch->gridWidth;
        p["grid_height"] = r.patch->gridHeight;
        p["scale"] = nb::make_tuple(r.patch->scale[0], r.patch->scale[1]);
        p["bbox"] = bboxToPy(r.patch->bbox);
        d["patch"] = p;
    }
    if (r.track) {
        nb::dict t;
        t["dataset_ref"] = r.track->datasetRef;
        t["external_key"] = r.track->externalKey;
        t["family"] = r.track->family;
        t["point_count"] = r.track->pointCount;
        t["length"] = r.track->length;
        t["bbox"] = bboxToPy(r.track->bbox);
        d["track"] = t;
    }
    return d;
}

vs::ObjectRecord objectFromPy(const nb::dict& d)
{
    vs::ObjectRecord r;
    r.uuid = getStr(d, "uuid");
    r.kind = getStr(d, "kind");
    r.name = getStr(d, "name");
    r.formatVersion = getInt(d, "format_version");
    r.storageKind = getStr(d, "storage_kind");
    r.storageRef = getStr(d, "storage_ref");
    r.dataGeneration = getInt(d, "data_generation");
    if (has(d, "data_checksum")) r.dataChecksum = nb::cast<std::string>(d["data_checksum"]);
    r.tags = getTags(d);
    if (has(d, "meta")) r.meta = jsonDumps(d["meta"]);
    if (has(d, "fiber")) {
        const nb::dict f = nb::cast<nb::dict>(d["fiber"]);
        vs::FiberDetail det;
        det.family = getStr(f, "family");
        det.manualFamily = getBool(f, "manual_family");
        det.controlPointCount = getInt(f, "control_point_count");
        det.linePointCount = getInt(f, "line_point_count");
        det.length = getFloat(f, "length");
        det.width = getFloat(f, "width");
        det.widthGapFraction = getFloat(f, "width_gap_fraction", 0.2);
        det.optimizationMode = getStr(f, "optimization_mode");
        det.bbox = getBbox(f);
        r.fiber = det;
    }
    if (has(d, "patch")) {
        const nb::dict p = nb::cast<nb::dict>(d["patch"]);
        vs::PatchDetail det;
        det.gridWidth = getInt(p, "grid_width");
        det.gridHeight = getInt(p, "grid_height");
        if (has(p, "scale")) {
            const nb::object scale = p["scale"];
            if (nb::isinstance<nb::float_>(scale) || nb::isinstance<nb::int_>(scale))
                det.scale.fill(nb::cast<double>(scale));
            else det.scale = nb::cast<std::array<double, 2>>(scale);
        }
        det.bbox = getBbox(p);
        r.patch = det;
    }
    if (has(d, "track")) {
        const nb::dict t = nb::cast<nb::dict>(d["track"]);
        vs::TrackDetail det;
        det.datasetRef = getStr(t, "dataset_ref");
        det.externalKey = getStr(t, "external_key");
        det.family = getStr(t, "family");
        det.pointCount = getInt(t, "point_count");
        det.length = getFloat(t, "length");
        det.bbox = getBbox(t);
        r.track = det;
    }
    return r;
}

nb::dict edgeToPy(const vs::EdgeRecord& e)
{
    nb::dict d;
    commonToPy(d, e);
    d["a"] = e.aUuid;
    d["b"] = e.bUuid;
    d["a_anchor"] = anchorToPy(e.aAnchor);
    d["b_anchor"] = anchorToPy(e.bAnchor);
    return d;
}

vs::EdgeRecord edgeFromPy(const nb::dict& d)
{
    vs::EdgeRecord e;
    e.uuid = getStr(d, "uuid");
    e.kind = getStr(d, "kind");
    e.aUuid = getStr(d, "a");
    e.bUuid = getStr(d, "b");
    if (has(d, "a_anchor")) e.aAnchor = anchorFromPy(d["a_anchor"]);
    if (has(d, "b_anchor")) e.bAnchor = anchorFromPy(d["b_anchor"]);
    e.tags = getTags(d);
    if (has(d, "meta")) e.meta = jsonDumps(d["meta"]);
    return e;
}

vo::GeometryData geometryFromPy(const nb::dict& input)
{
    vo::GeometryData result;
    result.kind = nb::cast<std::string>(input["kind"]);
    if (input.contains("format_version")) result.formatVersion = nb::cast<int64_t>(input["format_version"]);
    if (input.contains("coordinate_system")) result.coordinateSystem = nb::cast<std::string>(input["coordinate_system"]);
    if (input.contains("attributes")) result.attributes = jsonDumps(input["attributes"]);
    for (const auto item : nb::cast<nb::list>(input["arrays"])) {
        const auto a = nb::cast<nb::dict>(item);
        vo::GeometryArray array;
        array.name = nb::cast<std::string>(a["name"]);
        array.dtype = nb::cast<std::string>(a["dtype"]);
        array.shape = nb::cast<std::vector<int64_t>>(a["shape"]);
        const auto bytes = nb::cast<nb::bytes>(a["data"]);
        const auto* begin = reinterpret_cast<const uint8_t*>(bytes.c_str());
        array.bytes.assign(begin, begin + bytes.size());
        result.arrays.push_back(std::move(array));
    }
    return result;
}
nb::dict geometryToPy(const vo::GeometryRevision& r)
{
    nb::dict result;
    result["uuid"] = r.uuid; result["object_uuid"] = r.objectUuid;
    result["parent_uuid"] = r.parentUuid; result["content_hash"] = r.contentHash;
    result["kind"] = r.kind; result["format_version"] = r.formatVersion;
    result["coordinate_system"] = r.coordinateSystem; result["attributes"] = jsonLoads(r.attributes);
    result["object_generation"] = r.objectGeneration;
    result["created_by"] = r.createdBy; result["created_at"] = r.createdAt; result["created_host"] = r.createdHost;
    nb::list arrays;
    for (const auto& a : r.arrays) {
        nb::dict item; item["name"] = a.name; item["dtype"] = a.dtype;
        item["shape"] = a.shape; item["byte_length"] = a.byteLength; arrays.append(item);
    }
    result["arrays"] = arrays;
    return result;
}
nb::dict geometryReportToPy(const vo::GeometryIngestReport& r)
{
    nb::dict result;
    result["ingested"] = r.ingested; result["unchanged"] = r.unchanged; result["problems"] = r.problems;
    return result;
}

}
namespace vc::python {
template <typename Database>
struct PyDatabase {
    std::shared_ptr<Database> db;

    Database& require()
    {
        if (!db) throw std::runtime_error("graph database is closed");
        return *db;
    }

    static PyDatabase create(const std::string& path, const std::string& author,
                                             const std::optional<std::string>& hostname, int readerPoolSize)
    {
        vs::OpenOptions o;
        o.author = author;
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        PyDatabase self;
        self.db = std::make_shared<Database>(Database::create(path, o));
        return self;
    }

    static PyDatabase open(const std::string& path, const std::optional<std::string>& author,
                                           bool readOnly, const std::optional<std::string>& hostname,
                                           int readerPoolSize)
    {
        vs::OpenOptions o;
        o.readOnly = readOnly;
        o.author = author.value_or("");
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        PyDatabase self;
        self.db = std::make_shared<Database>(Database::open(path, o));
        return self;
    }

    void close() {
        if (db && db.use_count() > 1) throw std::runtime_error("cannot close a database with an active transaction");
        db.reset();
    }

    template <typename Fn>
    auto withRead(Fn&& fn)
    {
        require();
        auto lease = db;
        auto& g = *lease;
        auto txn = g.read();
        return fn(g, txn);
    }
    template <typename Fn>
    auto withWrite(Fn&& fn)
    {
        require();
        auto lease = db;
        auto& g = *lease;
        auto txn = g.write();
        auto result = fn(g, txn);
        txn.commit();
        return result;
    }

    std::string upsertObjectIn(Database& g, vs::WriteTxn& txn, const nb::dict& rec)
    {
        return g.upsertObject(txn, objectFromPy(rec));
    }
    std::string addEdgeIn(Database& g, vs::WriteTxn& txn, const nb::dict& rec)
    {
        return g.addEdge(txn, edgeFromPy(rec));
    }
};

// Batch write transaction usable as a context manager:
//   with db.write() as w: w.upsert_object(...); w.add_edge(...)
// Commits on clean exit, rolls back on exception.
template <typename Database>
struct PyTransaction {
    PyDatabase<Database>* owner = nullptr;
    std::shared_ptr<Database> lease;
    std::unique_ptr<vs::WriteTxn> txn;

    vs::WriteTxn& require()
    {
        if (!txn) throw std::runtime_error("write transaction is finished; use it as a context manager");
        return *txn;
    }
    PyTransaction& enter()
    {
        if (txn) throw std::runtime_error("transaction is already active");
        owner->require();
        lease = owner->db;
        try { txn = std::make_unique<vs::WriteTxn>(lease->write()); }
        catch (...) { lease.reset(); throw; }
        return *this;
    }
    void exit(bool failed)
    {
        if (txn) {
            try { if (!failed) txn->commit(); }
            catch (...) { txn.reset(); lease.reset(); throw; }
            txn.reset();  // RAII rolls back when not committed
            lease.reset();
        }
    }
};


using PyObjectStore = PyDatabase<vo::ObjectStore>;
using PyObjectWriteTxn = PyTransaction<vo::ObjectStore>;
}
