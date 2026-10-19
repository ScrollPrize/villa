#include <set>
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

vo::Anchor anchorFromPy(const nb::handle& obj)
{
    vo::Anchor a;
    if (obj.is_none()) return a;
    nb::dict d = nb::cast<nb::dict>(obj);
    for (auto item : d) {
        const auto key = nb::cast<std::string>(item.first);
        if (key != "points" && key != "whole_geometry" && key != "position" && key != "direction")
            throw std::invalid_argument("unknown anchor field: " + key);
    }
    if (has(d, "whole_geometry")) a.wholeGeometry = nb::cast<bool>(d["whole_geometry"]);
    if (has(d, "points")) for (nb::handle item : nb::cast<nb::list>(d["points"])) {
        const auto point = nb::cast<nb::dict>(item);
        a.points.push_back({nb::cast<std::string>(point["array"]), nb::cast<std::vector<int64_t>>(point["index"])});
    }
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

nb::dict anchorToPy(const vo::Anchor& a)
{
    nb::dict d;
    if (a.wholeGeometry) d["whole_geometry"] = true;
    if (!a.points.empty()) {
        nb::list points;
        for (const auto& point : a.points) {
            nb::dict value; value["array"] = point.array; value["index"] = point.index; points.append(value);
        }
        d["points"] = points;
    }
    if (a.position) d["position"] = nb::make_tuple((*a.position)[0], (*a.position)[1], (*a.position)[2]);
    if (a.direction) d["direction"] = nb::make_tuple((*a.direction)[0], (*a.direction)[1], (*a.direction)[2]);
    return d;
}

nb::str pyStr(const std::string& s) { return nb::str(s.data(), s.size()); }

void commonToPy(nb::dict& d, const vo::RecordCommon& r, const std::set<std::string>* fields = nullptr)
{
    auto include = [&](const char* name) { return !fields || fields->contains(name); };
    d["uuid"] = r.uuid;
    d["kind"] = r.kind;
    if (include("generation")) d["generation"] = r.generation;
    if (include("created_by")) d["created_by"] = r.createdBy;
    if (include("created_at")) d["created_at"] = r.createdAt;
    if (include("created_host")) d["created_host"] = r.createdHost;
    if (include("changed_by")) d["changed_by"] = r.changedBy;
    if (include("changed_at")) d["changed_at"] = r.changedAt;
    if (include("changed_host")) d["changed_host"] = r.changedHost;
    if (include("approval_state")) d["approval_state"] = r.approvalState;
    if (include("approved_by")) d["approved_by"] = r.approvedBy ? nb::object(pyStr(*r.approvedBy)) : nb::none();
    if (include("approved_at")) d["approved_at"] = r.approvedAt ? nb::object(pyStr(*r.approvedAt)) : nb::none();
    if (include("approved_host")) d["approved_host"] = r.approvedHost ? nb::object(pyStr(*r.approvedHost)) : nb::none();
    if (include("approved_generation")) d["approved_generation"] = r.approvedGeneration ? nb::object(nb::int_(*r.approvedGeneration)) : nb::none();
    if (include("tags")) {
        nb::list tags;
        for (const auto& t : r.tags) tags.append(pyStr(t));
        d["tags"] = tags;
    }
    if (include("meta")) d["meta"] = jsonLoads(r.meta);
}

nb::dict objectToPy(const vo::ObjectRecord& r, const std::set<std::string>* fields = nullptr)
{
    auto include = [&](const char* name) { return !fields || fields->contains(name); };
    nb::dict d;
    commonToPy(d, r, fields);
    if (include("name")) d["name"] = r.name;
    if (include("format_version")) d["format_version"] = r.formatVersion;
    if (include("storage_kind")) d["storage_kind"] = r.storageKind;
    if (include("storage_ref")) d["storage_ref"] = r.storageRef;
    if (include("data_generation")) d["data_generation"] = r.dataGeneration;
    if (include("data_checksum")) d["data_checksum"] = r.dataChecksum ? nb::object(pyStr(*r.dataChecksum)) : nb::none();
    if (r.fiber && include("fiber")) {
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
    if (r.patch && include("patch")) {
        nb::dict p;
        p["grid_width"] = r.patch->gridWidth;
        p["grid_height"] = r.patch->gridHeight;
        p["scale"] = nb::make_tuple(r.patch->scale[0], r.patch->scale[1]);
        p["bbox"] = bboxToPy(r.patch->bbox);
        d["patch"] = p;
    }
    if (r.track && include("track")) {
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

vo::ObjectRecord objectFromPy(const nb::dict& d)
{
    vo::ObjectRecord r;
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
        vo::FiberDetail det;
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
        vo::PatchDetail det;
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
        vo::TrackDetail det;
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

nb::dict edgeToPy(const vo::EdgeRecord& e)
{
    nb::dict d;
    commonToPy(d, e);
    d["a"] = e.aUuid;
    d["b"] = e.bUuid;
    d["a_anchor"] = anchorToPy(e.aAnchor);
    d["b_anchor"] = anchorToPy(e.bAnchor);
    return d;
}

vo::EdgeRecord edgeFromPy(const nb::dict& d)
{
    vo::EdgeRecord e;
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
    result["edges_imported"] = r.edgesImported; result["ingested"] = r.ingested; result["unchanged"] = r.unchanged; result["problems"] = r.problems;
    return result;
}

}
namespace vc::python {
struct PyObjectStore {
    std::shared_ptr<vo::ObjectStore> db;

    vo::ObjectStore& require()
    {
        if (!db) throw std::runtime_error("object database is closed");
        return *db;
    }

    static PyObjectStore create(const std::string& path, const std::string& author,
                                             const std::optional<std::string>& hostname, int readerPoolSize,
                                             size_t geometryCacheBytes)
    {
        vo::OpenOptions o;
        o.author = author;
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        o.geometryCacheBytes = geometryCacheBytes;
        PyObjectStore self;
        self.db = std::make_shared<vo::ObjectStore>(vo::ObjectStore::create(path, o));
        return self;
    }

    static PyObjectStore open(const std::string& path, const std::optional<std::string>& author,
                                           bool readOnly, const std::optional<std::string>& hostname,
                                           int readerPoolSize, size_t geometryCacheBytes)
    {
        vo::OpenOptions o;
        o.readOnly = readOnly;
        o.author = author.value_or("");
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        o.geometryCacheBytes = geometryCacheBytes;
        PyObjectStore self;
        self.db = std::make_shared<vo::ObjectStore>(vo::ObjectStore::open(path, o));
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

    std::string upsertObjectIn(vo::ObjectStore& g, vo::WriteTxn& txn, const nb::dict& rec)
    {
        return g.upsertObject(txn, objectFromPy(rec));
    }
    std::string addEdgeIn(vo::ObjectStore& g, vo::WriteTxn& txn, const nb::dict& rec)
    {
        return g.addEdge(txn, edgeFromPy(rec));
    }
};

// Batch write transaction usable as a context manager:
//   with db.write() as w: w.upsert_object(...); w.add_edge(...)
// Commits on clean exit, rolls back on exception.
struct PyObjectWriteTxn {
    PyObjectStore* owner = nullptr;
    std::shared_ptr<vo::ObjectStore> lease;
    std::unique_ptr<vo::WriteTxn> txn;

    vo::WriteTxn& require()
    {
        if (!txn) throw std::runtime_error("write transaction is finished; use it as a context manager");
        return *txn;
    }
    PyObjectWriteTxn& enter()
    {
        if (txn) throw std::runtime_error("transaction is already active");
        owner->require();
        lease = owner->db;
        try { txn = std::make_unique<vo::WriteTxn>(lease->write()); }
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


}
