#pragma once
#include <set>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/GeometryExport.hpp"
#include "internal/Anchors.hpp"
#include "internal/JsonFields.hpp"
#include "internal/RecordFields.hpp"

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

std::vector<std::string> getTags(const nb::dict& d)
{
    std::vector<std::string> result;
    if (has(d, "tags"))
        for (nb::handle tag : d["tags"]) result.push_back(nb::cast<std::string>(tag));
    return result;
}

template<class T> void readDetailField(const nb::handle& value, T& field) { field = nb::cast<T>(value); }
void readDetailField(const nb::handle& value, std::array<double, 2>& field) {
    field = vo::detail::scaleFromJson(nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value)
        ? nlohmann::json(nb::cast<double>(value)) : nlohmann::json(nb::cast<std::array<double, 2>>(value)));
}
template<class T> const T& detailFieldToPy(const T& value) { return value; }
nb::tuple detailFieldToPy(const std::array<double, 2>& value) { return nb::make_tuple(value[0], value[1]); }

vo::Anchor anchorFromPy(const nb::handle& obj)
{
    vo::Anchor a;
    if (obj.is_none()) return a;
    nb::dict d = nb::cast<nb::dict>(obj);
    for (auto item : d) {
        const auto key = nb::cast<std::string>(item.first);
        if (key != "points" && key != "whole_geometry" && key != "position" && key != "direction" && key != "metadata_keys" && key != "channels")
            throw std::invalid_argument("unknown anchor field: " + key);
    }
    if (has(d, "whole_geometry")) a.wholeGeometry = nb::cast<bool>(d["whole_geometry"]);
    if (has(d, "channels")) a.channels = nb::cast<std::vector<std::string>>(d["channels"]);
    if (has(d, "metadata_keys")) a.metadataKeys = nb::cast<std::vector<std::string>>(d["metadata_keys"]);
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
    vo::detail::validateAnchor(a);
    return a;
}

nb::dict anchorToPy(const vo::Anchor& a)
{
    nb::dict d;
    if (a.wholeGeometry) d["whole_geometry"] = true;
    if (!a.channels.empty()) d["channels"] = a.channels;
    if (!a.metadataKeys.empty()) d["metadata_keys"] = a.metadataKeys;
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
    vo::detail::commonFields(r, [&](const char* key, const auto& value) {
        if (include(key)) d[key] = value;
    });
    if (include("meta")) d["meta"] = jsonLoads(r.meta);
}

nb::dict objectToPy(const vo::ObjectRecord& r, const std::set<std::string>* fields = nullptr)
{
    auto include = [&](const char* name) { return !fields || fields->contains(name); };
    nb::dict d;
    commonToPy(d, r, fields);
    if (include("retired")) d["retired"] = r.retired;
    if (include("lifecycle_epoch")) d["lifecycle_epoch"] = r.lifecycleEpoch;
    vo::detail::objectFields(r, [&](const char* key, const auto& value) {
        if (include(key)) d[key] = value;
    });
    if (include("data_checksum")) d["data_checksum"] = r.dataChecksum;
    vo::detail::objectDetails(r, [&](const char* key, const auto& detail) {
        if (!detail || !include(key)) return;
        nb::dict value;
        vo::detail::detailFields(*detail, [&](const char* name, const auto& field) { value[name] = detailFieldToPy(field); });
        d[key] = value;
    });
    return d;
}

vo::ObjectRecord objectFromPy(const nb::dict& d)
{
    vo::ObjectRecord r;
    r.uuid = getStr(d, "uuid");
    r.kind = getStr(d, "kind");
    vo::detail::objectFields(r, [&](const char* key, auto& field) {
        if (has(d, key)) field = nb::cast<std::remove_cvref_t<decltype(field)>>(d[key]);
    });
    if (has(d, "data_checksum")) r.dataChecksum = nb::cast<std::string>(d["data_checksum"]);
    r.tags = getTags(d);
    if (has(d, "meta")) r.meta = jsonDumps(d["meta"]);
    vo::detail::objectDetails(r, [&](const char* key, auto& detail) {
        if (!has(d, key)) return;
        detail.emplace();
        const auto value = nb::cast<nb::dict>(d[key]);
        vo::detail::detailFields(*detail, [&](const char* name, auto& field) {
            if (has(value, name)) readDetailField(value[name], field);
        });
    });
    return r;
}

nb::dict edgeToPy(const vo::EdgeRecord& e, const std::set<std::string>* fields = nullptr)
{
    nb::dict d;
    commonToPy(d, e, fields);
    d["a"] = e.aUuid;
    d["b"] = e.bUuid;
    if (!fields || fields->contains("a_anchor")) d["a_anchor"] = anchorToPy(e.aAnchor);
    if (!fields || fields->contains("b_anchor")) d["b_anchor"] = anchorToPy(e.bAnchor);
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

std::vector<vo::GeometryArrayEdit> geometryEditsFromPy(const nb::list& input)
{
    std::vector<vo::GeometryArrayEdit> result;
    for (const auto item : input) {
        const auto entry = nb::cast<nb::dict>(item);
        const auto bytes = nb::cast<nb::bytes>(entry["data"]);
        vo::GeometryArrayEdit edit;
        edit.array = nb::cast<std::string>(entry["array"]);
        edit.elementOffset = nb::cast<int64_t>(entry["element_offset"]);
        const auto* p = reinterpret_cast<const uint8_t*>(bytes.c_str());
        edit.bytes.assign(p, p + bytes.size());
        result.push_back(std::move(edit));
    }
    return result;
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
    result["grid_origin"] = r.gridOrigin;
    result["object_generation"] = r.objectGeneration;
    result["created_by"] = r.createdBy; result["created_at"] = r.createdAt; result["created_host"] = r.createdHost;
    nb::list arrays;
    for (const auto& a : r.arrays) {
        nb::dict item; item["name"] = a.name; item["dtype"] = a.dtype;
        if (r.kind == "patch") item["components"] = a.shape.size() == 3 ? a.shape[2] : 1;
        item["shape"] = a.shape; item["byte_length"] = a.byteLength; arrays.append(item);
    }
    result["arrays"] = arrays;
    return result;
}
nb::dict geometryReportToPy(const vo::GeometryIngestReport& r)
{
    nb::dict result;
    result["edges_imported"] = r.edgesImported; result["ingested"] = r.ingested; result["problems"] = r.problems;
    return result;
}

std::vector<vo::GeometryEdgeEdit> edgeEditsFromPy(const nb::list& edits)
{
    std::vector<vo::GeometryEdgeEdit> result;
    for (nb::handle item : edits) {
        const auto d = nb::cast<nb::dict>(item);
        vo::GeometryEdgeEdit edit;
        edit.edgeUuid = nb::cast<std::string>(d["edge_uuid"]);
        edit.expectedGeneration = nb::cast<int64_t>(d["expected_generation"]);
        edit.action = nb::cast<std::string>(d["action"]);
        edit.aRevision = nb::cast<std::optional<std::string>>(d["a_revision"]);
        edit.bRevision = nb::cast<std::optional<std::string>>(d["b_revision"]);
        if (edit.action == "update") {
            edit.aAnchor = anchorFromPy(d["a_anchor"]); edit.bAnchor = anchorFromPy(d["b_anchor"]);
        }
        result.push_back(std::move(edit));
    }
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
                                             size_t geometryCacheBytes, size_t spatialCacheBytes)
    {
        vo::OpenOptions o;
        o.author = author;
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        o.geometryCacheBytes = geometryCacheBytes;
        o.spatialCacheBytes = spatialCacheBytes;
        PyObjectStore self;
        self.db = std::make_shared<vo::ObjectStore>(vo::ObjectStore::create(path, o));
        return self;
    }

    static PyObjectStore open(const std::string& path, const std::optional<std::string>& author,
                                           bool readOnly, const std::optional<std::string>& hostname,
                                           int readerPoolSize, size_t geometryCacheBytes, size_t spatialCacheBytes)
    {
        vo::OpenOptions o;
        o.readOnly = readOnly;
        o.author = author.value_or("");
        o.hostnameOverride = hostname;
        o.readerPoolSize = readerPoolSize;
        o.geometryCacheBytes = geometryCacheBytes;
        o.spatialCacheBytes = spatialCacheBytes;
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
    template <bool ReleaseGil = false, typename Fn>
    auto withWrite(Fn&& fn)
    {
        require();
        auto lease = db;
        auto write = [&] {
            auto txn = lease->write();
            if constexpr (std::is_void_v<std::invoke_result_t<Fn, vo::ObjectStore&, vo::WriteTxn&>>) {
                fn(*lease, txn);
                txn.commit();
            } else {
                auto result = fn(*lease, txn);
                txn.commit();
                return result;
            }
        };
        if constexpr (ReleaseGil) {
            nb::gil_scoped_release release;
            return write();
        } else return write();
    }

};

// Batch write transaction usable as a context manager:
//   with db.write() as w: w.upsert_object(...); w.add_edge(...)
// Commits on clean exit, rolls back on exception.
struct PyObjectWriteTxn {
    PyObjectStore* owner = nullptr;
    std::shared_ptr<vo::ObjectStore> lease;
    std::unique_ptr<vo::WriteTxn> txn;
    std::optional<vo::RevisionSelection> expected;

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
        try { txn = std::make_unique<vo::WriteTxn>(expected ? lease->write(*expected) : lease->write()); }
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
    template <typename Fn>
    auto withRead(Fn&& fn) { return fn(owner->require(), require()); }

    template <bool ReleaseGil = false, typename Fn>
    auto withWrite(Fn&& fn) { return withRead(std::forward<Fn>(fn)); }

};


}

vo::GeometrySequenceEdit sequenceEditFromPy(const nb::dict& input) {
    vo::GeometrySequenceEdit edit;
    edit.array=nb::cast<std::string>(input["array"]);
    edit.start=nb::cast<int64_t>(input["start"]); edit.eraseCount=nb::cast<int64_t>(input["erase_count"]);
    if (has(input,"transformed_annotations")) edit.transformedAnnotations=jsonDumps(input["transformed_annotations"]);
    if (has(input,"parts")) for (const auto item:nb::cast<nb::list>(input["parts"])) {
        const auto p=nb::cast<nb::dict>(item); vo::GeometrySequencePart part;
        if (has(p,"revision")) part.revision=nb::cast<std::string>(p["revision"]);
        part.start=getInt(p,"start"); part.count=getInt(p,"count");
        if (has(p,"reverse")) part.reverse=nb::cast<bool>(p["reverse"]);
        if (has(p,"data")) { const auto bytes=nb::cast<nb::bytes>(p["data"]); const auto* b=reinterpret_cast<const uint8_t*>(bytes.c_str()); part.bytes.assign(b,b+bytes.size()); }
        if (has(p,"ids")) part.ids=nb::cast<std::vector<int64_t>>(p["ids"]);
        if (has(p,"annotations")) part.annotations=jsonDumps(p["annotations"]);
        edit.parts.push_back(std::move(part));
    }
    return edit;
}
vo::GeometrySurfaceResize surfaceResizeFromPy(const nb::dict& input) {
    vo::GeometrySurfaceResize resize;
    resize.row=nb::cast<int64_t>(input["row"]); resize.column=nb::cast<int64_t>(input["column"]);
    resize.rows=nb::cast<int64_t>(input["rows"]); resize.columns=nb::cast<int64_t>(input["columns"]);
    nb::dict data; data["kind"]="patch"; data["arrays"]=input["fill"];
    resize.fill=geometryFromPy(data).arrays; return resize;
}
