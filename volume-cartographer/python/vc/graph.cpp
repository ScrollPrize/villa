// Python bindings for the unified metadata graph database (vc_graph).
// Records cross the boundary as plain dicts using the same shape as the
// vc_graph_export JSON format: kind is a field in the dict, never part
// of a method name, and arbitrary extra metadata rides in "meta".

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

#include "vc/core/graph/GraphDb.hpp"
#include "vc/core/graph/GeometryIngest.hpp"
#include "vc/core/graph/GeometryExport.hpp"
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/CtSheetNormal.hpp"

namespace nb = nanobind;
namespace vg = vc::graph;

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

vg::Anchor anchorFromPy(const nb::handle& obj)
{
    vg::Anchor a;
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

nb::dict anchorToPy(const vg::Anchor& a)
{
    nb::dict d;
    if (a.index) d["index"] = *a.index;
    if (a.position) d["position"] = nb::make_tuple((*a.position)[0], (*a.position)[1], (*a.position)[2]);
    if (a.direction) d["direction"] = nb::make_tuple((*a.direction)[0], (*a.direction)[1], (*a.direction)[2]);
    return d;
}

nb::str pyStr(const std::string& s) { return nb::str(s.data(), s.size()); }

void commonToPy(nb::dict& d, const vg::RecordCommon& r)
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

nb::dict objectToPy(const vg::ObjectRecord& r)
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

vg::ObjectRecord objectFromPy(const nb::dict& d)
{
    vg::ObjectRecord r;
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
        vg::FiberDetail det;
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
        vg::PatchDetail det;
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
        vg::TrackDetail det;
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

nb::dict edgeToPy(const vg::EdgeRecord& e)
{
    nb::dict d;
    commonToPy(d, e);
    d["a"] = e.aUuid;
    d["b"] = e.bUuid;
    d["a_anchor"] = anchorToPy(e.aAnchor);
    d["b_anchor"] = anchorToPy(e.bAnchor);
    return d;
}

vg::EdgeRecord edgeFromPy(const nb::dict& d)
{
    vg::EdgeRecord e;
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

nb::dict detectorReportToPy(const vc::graph_links::DetectorReport& rep)
{
    nb::dict d;
    d["objects_considered"] = rep.objectsConsidered;
    d["candidates_found"] = rep.candidatesFound;
    d["edges_created"] = rep.edgesCreated;
    d["edges_updated"] = rep.edgesUpdated;
    d["edges_skipped_existing"] = rep.edgesSkippedExisting;
    d["stale_removed"] = rep.staleRemoved;
    nb::list problems;
    for (const auto& p : rep.problems) problems.append(pyStr(p));
    d["problems"] = problems;
    return d;
}

nb::dict reportToPy(const vg::ImportReport& rep)
{
    nb::dict d;
    d["objects_imported"] = rep.objectsImported;
    d["edges_imported"] = rep.edgesImported;
    nb::list problems;
    for (const auto& p : rep.problems) problems.append(pyStr(p));
    d["problems"] = problems;
    return d;
}

vg::GeometryData geometryFromPy(const nb::dict& input)
{
    vg::GeometryData result;
    result.kind = nb::cast<std::string>(input["kind"]);
    if (input.contains("format_version")) result.formatVersion = nb::cast<int64_t>(input["format_version"]);
    if (input.contains("coordinate_system")) result.coordinateSystem = nb::cast<std::string>(input["coordinate_system"]);
    if (input.contains("attributes")) result.attributes = jsonDumps(input["attributes"]);
    for (const auto item : nb::cast<nb::list>(input["arrays"])) {
        const auto a = nb::cast<nb::dict>(item);
        vg::GeometryArray array;
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
nb::dict geometryToPy(const vg::GeometryRevision& r)
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
nb::dict geometryReportToPy(const vg::GeometryIngestReport& r)
{
    nb::dict result;
    result["ingested"] = r.ingested; result["unchanged"] = r.unchanged; result["problems"] = r.problems;
    return result;
}

struct PyGraphDb {
    std::optional<vg::GraphDb> db;

    vg::GraphDb& require()
    {
        if (!db) throw std::runtime_error("graph database is closed");
        return *db;
    }

    static PyGraphDb create(const std::string& path, const std::string& author,
                                             const std::optional<std::string>& hostname, int readerPoolSize)
    {
        vg::OpenOptions o;
        o.author = author;
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        PyGraphDb self;
        self.db = vg::GraphDb::create(path, o);
        return self;
    }

    static PyGraphDb open(const std::string& path, const std::optional<std::string>& author,
                                           bool readOnly, const std::optional<std::string>& hostname,
                                           int readerPoolSize)
    {
        vg::OpenOptions o;
        o.readOnly = readOnly;
        o.author = author.value_or("");
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        PyGraphDb self;
        self.db = vg::GraphDb::open(path, o);
        return self;
    }

    void close() { db.reset(); }

    template <typename Fn>
    auto withRead(Fn&& fn)
    {
        auto& g = require();
        auto txn = g.read();
        return fn(g, txn);
    }
    template <typename Fn>
    auto withWrite(Fn&& fn)
    {
        auto& g = require();
        auto txn = g.write();
        auto result = fn(g, txn);
        txn.commit();
        return result;
    }

    std::string upsertObjectIn(vg::GraphDb& g, vg::WriteTxn& txn, const nb::dict& rec)
    {
        return g.upsertObject(txn, objectFromPy(rec));
    }
    std::string addEdgeIn(vg::GraphDb& g, vg::WriteTxn& txn, const nb::dict& rec)
    {
        return g.addEdge(txn, edgeFromPy(rec));
    }
};

// Batch write transaction usable as a context manager:
//   with db.write() as w: w.upsert_object(...); w.add_edge(...)
// Commits on clean exit, rolls back on exception.
struct PyWriteTxn {
    PyGraphDb* owner = nullptr;
    std::unique_ptr<vg::WriteTxn> txn;

    vg::WriteTxn& require()
    {
        if (!txn) throw std::runtime_error("write transaction is finished; use it as a context manager");
        return *txn;
    }
    PyWriteTxn& enter()
    {
        txn = std::make_unique<vg::WriteTxn>(owner->require().write());
        return *this;
    }
    void exit(bool failed)
    {
        if (txn) {
            if (!failed) txn->commit();
            txn.reset();  // RAII rolls back when not committed
        }
    }
};

bool setLinePatchParameter(vc::graph_links::LinePatchParams& p, const std::string& key, nb::handle val)
{
    if (key == "patch_scale") p.patchScale = nb::cast<double>(val);
    else if (key == "normals_manifest") p.normalsManifest = nb::cast<std::string>(val);
    else if (key == "ct_level") p.ctLevel = nb::cast<int>(val);
    else if (key == "ct_to_base_scale") p.ctToBaseScale = nb::cast<double>(val);
    else if (key == "working_to_base_scale") p.workingToBaseScale = nb::cast<double>(val);
    else if (key == "proximity") p.proximity = nb::cast<double>(val);
    else if (key == "min_contacts") p.minContacts = nb::cast<std::size_t>(val);
    else if (key == "strip_spacing") p.stripSpacing = nb::cast<double>(val);
    else if (key == "strip_rows_each_side") p.stripRowsEachSide = nb::cast<int>(val);
    else if (key == "strip_march_step") p.stripMarchStep = nb::cast<double>(val);
    else if (key == "contact_gap") p.contactGap = nb::cast<double>(val);
    else if (key == "anchor_contact") p.anchorContact = nb::cast<double>(val);
    else if (key == "same_sheet") p.sameSheet = nb::cast<double>(val);
    else if (key == "adjacent_low") p.adjacentLow = nb::cast<double>(val);
    else if (key == "adjacent_high") p.adjacentHigh = nb::cast<double>(val);
    else if (key == "same_normal") p.sameNormal = nb::cast<double>(val);
    else if (key == "adjacent_normal") p.adjacentNormal = nb::cast<double>(val);
    else if (key == "grow_same") p.growSame = nb::cast<double>(val);
    else if (key == "grow_ratio") p.growRatio = nb::cast<double>(val);
    else if (key == "strip_gate_ratio") p.stripGateRatio = nb::cast<double>(val);
    else if (key == "strip_gate_margin") p.stripGateMargin = nb::cast<double>(val);
    else if (key == "strip_margin") p.stripMargin = nb::cast<double>(val);
    else if (key == "max_rounds") p.maxRounds = nb::cast<int>(val);
    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(val);
    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(val);
    else if (key == "strip_weights") {
        const auto weights = nb::cast<std::vector<double>>(val);
        if (weights.size() != p.stripWeights.size())
            throw std::runtime_error("strip_weights requires four values");
        std::copy(weights.begin(), weights.end(), p.stripWeights.begin());
    } else return false;
    return true;
}

}  // namespace

NB_MODULE(graph, m)
{
    nb::class_<vc::CtSheetNormalSampler>(m, "CtSheetNormalSampler")
        .def(nb::init<const std::filesystem::path&, int, double>(),
             nb::arg("ct_volume"), nb::arg("level") = 0, nb::arg("working_to_ct_scale") = 1.0)
        .def("sample_normal", [](const vc::CtSheetNormalSampler& self, const std::array<double, 3>& xyz) {
            vc::CtSheetNormal sample;
            { nb::gil_scoped_release release; sample = self.sampleNormal({xyz[0], xyz[1], xyz[2]}); }
            nb::dict result;
            result["normal"] = std::array<double, 3>{sample.normal[0], sample.normal[1], sample.normal[2]};
            result["valid"] = sample.valid;
            result["gap"] = sample.gap;
            result["reason"] = sample.reason;
            return result;
        }, nb::arg("point_xyz"));

    m.doc() = "Unified metadata graph database (vc_graph). Records are dicts; "
              "'kind' is metadata in the dict, never part of a method name.";

    nb::class_<PyGraphDb>(m, "GraphDb")
        .def_static("create", &PyGraphDb::create, nb::arg("path"), nb::arg("author"),
                    nb::arg("hostname") = std::nullopt, nb::arg("reader_pool_size") = 4,
                    "Create a new graph database at path (refuses existing files).")
        .def_static("open", &PyGraphDb::open, nb::arg("path"), nb::arg("author") = std::nullopt,
                    nb::arg("read_only") = false, nb::arg("hostname") = std::nullopt,
                    nb::arg("reader_pool_size") = 4,
                    "Open an existing graph database. author is required unless read_only.")
        .def("close", &PyGraphDb::close)
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](PyGraphDb& self, nb::args) { self.close(); })
        .def_prop_ro("path", [](PyGraphDb& self) { return self.require().path().string(); })
        .def_prop_ro("hostname", [](PyGraphDb& self) { return self.require().hostname(); })
        .def_prop_ro("author", [](PyGraphDb& self) { return self.require().author(); })
        .def(
            "upsert_object",
            [](PyGraphDb& self, const nb::dict& rec) {
                return self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) { return self.upsertObjectIn(g, t, rec); });
            },
            nb::arg("record"), "Create or update an object from a dict; returns its uuid.")
        .def(
            "object",
            [](PyGraphDb& self, const std::string& uuid) -> nb::object {
                auto rec = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.objectByUuid(t, uuid); });
                return rec ? nb::object(objectToPy(*rec)) : nb::none();
            },
            nb::arg("uuid"))
        .def(
            "objects_by_kind",
            [](PyGraphDb& self, const std::string& kind, const std::optional<std::string>& approvalState) {
                auto recs = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.objectsByKind(t, kind, approvalState); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("kind"), nb::arg("approval_state") = std::nullopt)
        .def("object_count", [](PyGraphDb& self, const std::string& kind) {
            return self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.objectCount(txn, kind); });
        }, nb::arg("kind"))
        .def(
            "objects_by_tag",
            [](PyGraphDb& self, const std::string& tag) {
                auto recs = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.objectsByTag(t, tag); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("tag"))
        .def(
            "delete_object",
            [](PyGraphDb& self, const std::string& uuid) {
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.deleteObject(t, uuid);
                    return 0;
                });
            },
            nb::arg("uuid"))
        .def(
            "add_edge",
            [](PyGraphDb& self, const nb::dict& rec) {
                return self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) { return self.addEdgeIn(g, t, rec); });
            },
            nb::arg("record"), "Create an edge from a dict (keys: a, b, kind, anchors, tags, meta); returns uuid.")
        .def(
            "edge",
            [](PyGraphDb& self, const std::string& uuid) -> nb::object {
                auto rec = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.edgeByUuid(t, uuid); });
                return rec ? nb::object(edgeToPy(*rec)) : nb::none();
            },
            nb::arg("uuid"))
        .def(
            "remove_edge",
            [](PyGraphDb& self, const std::string& uuid) {
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.removeEdge(t, uuid);
                    return 0;
                });
            },
            nb::arg("uuid"))
        .def(
            "edges_of",
            [](PyGraphDb& self, const std::string& uuid) {
                auto recs = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.edgesOf(t, uuid); });
                nb::list out;
                for (const auto& e : recs) out.append(edgeToPy(e));
                return out;
            },
            nb::arg("uuid"))
        .def(
            "edges_by_kind",
            [](PyGraphDb& self, const std::string& kind) {
                auto recs = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.edgesByKind(t, kind); });
                nb::list out;
                for (const auto& e : recs) out.append(edgeToPy(e));
                return out;
            },
            nb::arg("edge_kind"))
        .def("neighbours",
             [](PyGraphDb& self, const std::string& uuid, const std::optional<std::string>& edgeKind,
                const std::optional<std::string>& neighbourKind) {
                 return self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.neighbours(t, uuid, edgeKind, neighbourKind); });
             },
             nb::arg("uuid"), nb::arg("edge_kind") = std::nullopt, nb::arg("neighbour_kind") = std::nullopt)
        .def("component",
             [](PyGraphDb& self, const std::string& uuid, const std::vector<std::string>& edgeKinds) {
                 return self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.componentOf(t, uuid, edgeKinds); });
             },
             nb::arg("uuid"), nb::arg("edge_kinds") = std::vector<std::string>{})
        .def("reachable_within",
             [](PyGraphDb& self, const std::string& uuid, int maxDepth, const std::vector<std::string>& edgeKinds) {
                 return self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.reachableWithin(t, uuid, maxDepth, edgeKinds); });
             },
             nb::arg("uuid"), nb::arg("max_depth"), nb::arg("edge_kinds") = std::vector<std::string>{})
        .def(
            "set_approval",
            [](PyGraphDb& self, const std::string& uuid, const std::string& state) {
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.setApproval(t, uuid, state);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("state"), "state: 'pending', 'approved' or 'rejected' (objects and edges).")
        .def(
            "set_meta",
            [](PyGraphDb& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                const std::string json = jsonDumps(value);
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.setMeta(t, uuid, key, json);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"), nb::arg("value"), "Set one open-ended metadata key (any JSON value).")
        .def(
            "remove_meta",
            [](PyGraphDb& self, const std::string& uuid, const std::string& key) {
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.removeMeta(t, uuid, key);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"))
        .def(
            "journal_since",
            [](PyGraphDb& self, int64_t afterSeq) {
                auto entries = self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.journalSince(t, afterSeq); });
                nb::list out;
                for (const auto& e : entries) {
                    nb::dict d;
                    d["seq"] = e.seq;
                    d["record_uuid"] = e.recordUuid;
                    d["generation"] = e.generation;
                    d["author"] = e.author;
                    d["hostname"] = e.hostname;
                    d["timestamp"] = e.timestamp;
                    d["change_kind"] = e.changeKind;
                    d["approval_state"] = e.approvalState ? nb::object(pyStr(*e.approvalState)) : nb::none();
                    d["approved_by"] = e.approvedBy ? nb::object(pyStr(*e.approvedBy)) : nb::none();
                    d["approved_at"] = e.approvedAt ? nb::object(pyStr(*e.approvedAt)) : nb::none();
                    d["approved_host"] = e.approvedHost ? nb::object(pyStr(*e.approvedHost)) : nb::none();
                    d["approved_generation"] = e.approvedGeneration ? nb::object(nb::int_(*e.approvedGeneration)) : nb::none();
                    out.append(d);
                }
                return out;
            },
            nb::arg("after_seq") = 0)
        .def("put_geometry", [](PyGraphDb& self, const std::string& uuid, const nb::dict& input,
                               const std::optional<std::string>& expected) {
            const auto data = geometryFromPy(input);
            nb::gil_scoped_release release;
            return self.withWrite([&](vg::GraphDb& db, vg::WriteTxn& txn) { return db.putGeometry(txn, uuid, data, expected); });
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt)
        .def("current_geometry", [](PyGraphDb& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.currentGeometry(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("object_uuid"))
        .def("geometry_revision", [](PyGraphDb& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.geometryRevision(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("revision_uuid"))
        .def("geometry_history", [](PyGraphDb& self, const std::string& uuid) {
            const auto revisions = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.geometryHistory(txn, uuid); });
            nb::list result; for (const auto& r : revisions) result.append(geometryToPy(r)); return result;
        }, nb::arg("object_uuid"))
        .def("read_geometry_array", [](PyGraphDb& self, const std::string& uuid, const std::string& name,
                                      int64_t offset, const std::optional<int64_t>& count) {
            std::vector<uint8_t> bytes;
            { nb::gil_scoped_release release;
              bytes = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.readGeometryArray(txn, uuid, name, offset, count); }); }
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("byte_offset") = 0, nb::arg("byte_count") = std::nullopt)
        .def("edge_geometry_state", [](PyGraphDb& self, const std::string& uuid) {
            const auto state = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.edgeGeometryState(txn, uuid); });
            nb::dict result; result["a_revision"] = state.aRevision; result["b_revision"] = state.bRevision; result["stale"] = state.stale; return result;
        }, nb::arg("edge_uuid"))
        .def("revalidate_edge_geometry", [](PyGraphDb& self, const std::string& uuid) {
            self.withWrite([&](vg::GraphDb& db, vg::WriteTxn& txn) { db.revalidateEdgeGeometry(txn, uuid); return 0; });
        }, nb::arg("edge_uuid"))
        .def("ingest_registered_geometry", [](PyGraphDb& self, const std::optional<std::filesystem::path>& root,
                                             const std::vector<std::string>& uuids) {
            vg::GeometryIngestReport result;
            { nb::gil_scoped_release release; result = vg::ingestRegisteredGeometry(self.require(), root, uuids); }
            return geometryReportToPy(result);
        }, nb::arg("dataset_root") = std::nullopt, nb::arg("object_uuids") = std::vector<std::string>{})
        .def("ingest_geometry_source", [](PyGraphDb& self, const std::string& kind, const std::filesystem::path& source) {
            vg::GeometryIngestReport result;
            { nb::gil_scoped_release release; result = vg::ingestGeometrySource(self.require(), kind, source); }
            return geometryReportToPy(result);
        }, nb::arg("kind"), nb::arg("source"))
        .def("export_geometry", [](PyGraphDb& self, const std::string& object,
                                   const std::filesystem::path& destination,
                                   const std::optional<std::string>& revision, bool overwrite) {
            nb::gil_scoped_release release;
            return vg::exportGeometry(self.require(), object, destination, revision, overwrite);
        }, nb::arg("object_uuid"), nb::arg("destination"), nb::arg("revision_uuid") = std::nullopt,
           nb::arg("overwrite") = false,
           "Export one immutable revision to native files; return its UUID. Does not modify the graph.")
        .def("resolve_storage_path",
             [](PyGraphDb& self, const std::string& uuid,
                const std::optional<std::filesystem::path>& datasetRoot) {
                 return self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) {
                     return g.resolveStoragePath(t, uuid, datasetRoot);
                 });
             }, nb::arg("uuid"), nb::arg("dataset_root") = std::nullopt,
             "Find a patch/fiber by source UUID, verifying the saved path before searching the dataset root.")
        .def("sweep_external_data",
             [](PyGraphDb& self) { return self.withRead([](vg::GraphDb& g, vg::ReadTxn& t) { return g.sweepExternalData(t); }); })
        .def("checkpoint", [](PyGraphDb& self) { self.require().checkpoint(); })
        .def(
            "export_json",
            [](PyGraphDb& self, const std::optional<std::string>& objectKind, const std::optional<std::string>& tag) {
                return self.withRead([&](vg::GraphDb& g, vg::ReadTxn& t) { return g.exportJson(t, objectKind, tag); });
            },
            nb::arg("object_kind") = std::nullopt, nb::arg("tag") = std::nullopt)
        .def(
            "import_json",
            [](PyGraphDb& self, const std::string& jsonText) {
                self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) {
                    g.importJson(t, jsonText);
                    return 0;
                });
            },
            nb::arg("json_text"))
        .def(
            "import_fibers",
            [](PyGraphDb& self, const std::string& dir) {
                auto rep = self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) { return vg::importFibers(g, t, dir); });
                return reportToPy(rep);
            },
            nb::arg("dir"))
        .def(
            "import_patches",
            [](PyGraphDb& self, const std::string& dir) {
                auto rep = self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) { return vg::importPatches(g, t, dir); });
                return reportToPy(rep);
            },
            nb::arg("dir"))
        .def(
            "import_track_list",
            [](PyGraphDb& self, const std::string& file) {
                auto rep = self.withWrite([&](vg::GraphDb& g, vg::WriteTxn& t) { return vg::importTrackList(g, t, file); });
                return reportToPy(rep);
            },
            nb::arg("file"))
        .def(
            "link_point_collections_to_patches",
            [](PyGraphDb& self, const std::string& collectionsFile, const std::string& patchesDir,
               const nb::kwargs& kwargs) {
                vc::graph_links::PointCollectionPatchParams p;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle value = item.second;
                    if (key == "tolerance") p.tolerance = nb::cast<double>(value);
                    else if (key == "point_scale") p.pointScale = nb::cast<double>(value);
                    else if (key == "patch_scale") p.patchScale = nb::cast<double>(value);
                    else if (key == "window_points") p.windowPoints = nb::cast<int>(value);
                    else if (key == "window_min_points") p.windowMinPoints = nb::cast<int>(value);
                    else if (key == "absolute_winding") p.absoluteWinding = nb::cast<std::optional<bool>>(value);
                    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(value);
                    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(value);
                    else throw std::runtime_error("link_point_collections_to_patches: unknown parameter '" + key + "'");
                }
                vc::graph_links::DetectorReport report;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::graph_links::linkPointCollectionsToPatches(db, collectionsFile, patchesDir, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("collections_file"), nb::arg("patches_dir"), nb::arg("kwargs"),
            "Link regular PCL points to patches using Spiral's largest-area/nearest-named-pair rules. "
            "Point IDs anchor the edges; winding annotations and patch grid coordinates are retained. "
            "Defaults: tolerance=2.5, window_points=1, window_min_points=1; no CT required.")
        .def(
            "link_patch_overlaps",
            [](PyGraphDb& self, const std::string& dir, const nb::kwargs& kwargs) {
                vc::graph_links::PatchOverlapParams p;
                for (auto item : kwargs) {
                    const std::string key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "workers") p.workers = nb::cast<int>(val);
                    else if (key == "tolerance") p.tolerance = nb::cast<double>(val);
                    else if (key == "sample_spacing") p.sampleSpacing = nb::cast<double>(val);
                    else if (key == "thinning_spacing") p.thinningSpacing = nb::cast<double>(val);
                    else if (key == "max_correspondences") p.maxCorrespondences = nb::cast<std::size_t>(val);
                    else if (key == "uv_inlier_tolerance") p.uvInlierTolerance = nb::cast<double>(val);
                    else if (key == "min_inliers") p.minInliers = nb::cast<std::size_t>(val);
                    else if (key == "min_major_spread") p.minMajorSpread = nb::cast<double>(val);
                    else if (key == "min_minor_spread") p.minMinorSpread = nb::cast<double>(val);
                    else if (key == "max_refit_rms") p.maxRefitRms = nb::cast<double>(val);
                    else if (key == "min_agreement_ratio") p.minAgreementRatio = nb::cast<double>(val);
                    else if (key == "min_overlap_fraction") p.minOverlapFraction = nb::cast<double>(val);
                    else if (key == "ransac_confidence") p.ransacConfidence = nb::cast<double>(val);
                    else if (key == "ransac_max_hypotheses") p.ransacMaxHypotheses = nb::cast<std::size_t>(val);
                    else if (key == "allow_reflection") p.allowReflection = nb::cast<bool>(val);
                    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(val);
                    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(val);
                    else throw std::runtime_error("link_patch_overlaps: unknown parameter '" + key + "'");
                }
                auto rep = vc::graph_links::linkPatchOverlaps(self.require(), dir, p);
                return detectorReportToPy(rep);
            },
            "Run the patch-overlap detector over the tifxyz segment dirs in dir; "
            "kwargs mirror PatchOverlapParams (tolerance, min_minor_spread, ...).")
        .def(
            "link_tracks_to_tracks",
            [](PyGraphDb& self, const std::string& tracksPath, const nb::kwargs& kwargs) {
                vc::graph_links::TrackTrackParams p;
                p.tracksPath = tracksPath;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle value = item.second;
                    if (key == "min_angle_degrees") p.minAngleDegrees = nb::cast<double>(value);
                    else if (key == "tangent_radius_voxels") p.tangentRadiusVoxels = nb::cast<double>(value);
                    else if (key == "cluster_index_radius") p.clusterIndexRadius = nb::cast<int>(value);
                    else if (key == "workers") p.workers = nb::cast<int>(value);
                    else if (key == "source_ids") p.sourceIds = nb::cast<std::vector<uint64_t>>(value);
                    else if (key == "dataset_ref") p.datasetRef = nb::cast<std::string>(value);
                    else if (key == "track_uuids") p.trackUuids = nb::cast<std::vector<std::string>>(value);
                    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(value);
                    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(value);
                    else throw std::runtime_error("link_tracks_to_tracks: unknown parameter '" + key + "'");
                }
                vc::graph_links::DetectorReport report;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::graph_links::linkTracksToTracks(db, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("tracks_dbm") = "", nb::arg("kwargs"),
            "Compute track crossings directly into the graph. A DBM path discovers adjacent "
            "packed geometry and registers tracks automatically; no crossings.npz or JSON list is used. "
            "source_ids optionally selects a subset. Omit the path to link registered tracks. "
            "Anchors are raw point indices and XYZ positions in dataset voxels.")
        .def(
            "link_tracks_to_patches",
            [](PyGraphDb& self, const std::string& tracksPath, const std::string& patchesDir,
               const std::string& ctVolume, const nb::kwargs& kwargs) {
                vc::graph_links::TrackPatchParams p;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "track_scale") p.trackScale = nb::cast<double>(val);
                    else if (key == "source_ids") p.sourceIds = nb::cast<std::vector<uint64_t>>(val);
                    else if (key == "track_uuids") p.trackUuids = nb::cast<std::vector<std::string>>(val);
                    else if (!setLinePatchParameter(p, key, val))
                        throw std::runtime_error("link_tracks_to_patches: unknown parameter '" + key + "'");
                }
                vc::graph_links::DetectorReport report;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::graph_links::linkTracksToPatches(db, tracksPath, patchesDir, ctVolume, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("tracks_dbm"), nb::arg("patches_dir"), nb::arg("ct_volume"), nb::arg("kwargs"),
            "Link DBM/packed tracks to tifxyz patches using the shared fiber/track strip rules. "
            "CT supplies normals by default; source_ids optionally limits the tracks. "
            "Anchors use raw point indices and source XYZ coordinates.")
        .def(
            "link_fibers_to_patches",
            [](PyGraphDb& self, const std::string& fibersDir, const std::string& patchesDir,
               const std::string& ctVolume, const nb::kwargs& kwargs) {
                vc::graph_links::FiberPatchParams p;
                for (auto item : kwargs) {
                    const std::string key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "fiber_scale") p.fiberScale = nb::cast<double>(val);
                    else if (key == "rebuild_strips") p.rebuildStrips = nb::cast<bool>(val);
                    else if (!setLinePatchParameter(p, key, val))
                        throw std::runtime_error("link_fibers_to_patches: unknown parameter '" + key + "'");
                }
                return detectorReportToPy(vc::graph_links::linkFibersToPatches(
                    self.require(), fibersDir, patchesDir, ctVolume, p));
            },
            nb::arg("fibers_dir"), nb::arg("patches_dir"), nb::arg("ct_volume"), nb::arg("kwargs"),
            "Link fiber JSONs to tifxyz patches using CT sheet normals; creates pending edges. "
            "kwargs mirror FiberPatchParams; normals_manifest selects a Lasagna reference source. "
            "The default strip has one row at 20 voxels on each side. "
            "Strips are cached in fibers_dir/.strip-cache; rebuild_strips=True refreshes them.")
        .def(
            "write",
            [](PyGraphDb& self) {
                PyWriteTxn w;
                w.owner = &self;
                return w;
            },
            "Batch write transaction context manager: 'with db.write() as w: w.upsert_object(...)'");

    nb::class_<PyWriteTxn>(m, "WriteTransaction")
        .def("__enter__", [](nb::object self) {
            nb::cast<PyWriteTxn&>(self).enter();
            return self;
        })
        .def("__exit__", [](PyWriteTxn& self, nb::args args) {
            const bool failed = nb::len(args) > 0 && !args[0].is_none();
            self.exit(failed);
        })
        .def("put_geometry", [](PyWriteTxn& self, const std::string& uuid, const nb::dict& input,
                               const std::optional<std::string>& expected) {
            const auto data = geometryFromPy(input);
            return self.owner->require().putGeometry(self.require(), uuid, data, expected);
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt)
        .def("revalidate_edge_geometry", [](PyWriteTxn& self, const std::string& uuid) {
            self.owner->require().revalidateEdgeGeometry(self.require(), uuid);
        }, nb::arg("edge_uuid"))
        .def("upsert_object",
             [](PyWriteTxn& self, const nb::dict& rec) { return self.owner->upsertObjectIn(self.owner->require(), self.require(), rec); })
        .def("add_edge",
             [](PyWriteTxn& self, const nb::dict& rec) { return self.owner->addEdgeIn(self.owner->require(), self.require(), rec); })
        .def("set_meta",
             [](PyWriteTxn& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                 self.owner->require().setMeta(self.require(), uuid, key, jsonDumps(value));
             })
        .def("set_approval", [](PyWriteTxn& self, const std::string& uuid, const std::string& state) {
            self.owner->require().setApproval(self.require(), uuid, state);
        });
}
