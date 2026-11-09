#include "StorageBindings.hpp"
#include "AfvBindings.hpp"
#include "ObjectReadBindings.hpp"
#include "TypedGeometryBindings.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "vc_object_store/objects/SurfaceGrid.hpp"
#include "GeometryRegionBindings.hpp"
#include "SessionBindings.hpp"
#include <nanobind/stl/map.h>
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/Selections.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include "vc_object_store/objects/Project.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/object_links/LinkObjects.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/CtSheetNormal.hpp"
using namespace vc::python;
namespace {
nb::list collectionsToPy(const std::vector<vo::AttachedCollection>& collections)
{
    nb::list out;
    for (const auto& collection : collections) {
        nb::dict item;
        item["uuid"] = collection.uuid; item["name"] = collection.name;
        item["ids"] = collection.points.ids; item["points"] = collection.points.points;
        item["annotations"] = jsonLoads(collection.annotations.dump());
        item["source"] = jsonLoads(collection.source.dump());
        out.append(item);
    }
    return out;
}

nb::dict objectRevisionToPy(const vo::ObjectRevision& revision)
{
    nb::dict d;
    d["uuid"] = revision.uuid; d["object_uuid"] = revision.objectUuid;
    d["parent_uuid"] = revision.parentUuid; d["geometry_revision"] = revision.geometryRevision;
    d["sequence"] = revision.sequence; d["metadata_complete"] = revision.metadataComplete;
    d["published_at_us"] = revision.publishedAtUs; d["deleted"] = revision.deleted;
    d["version"] = revision.version; d["generation"] = revision.generation;
    d["commit_sequence"] = revision.commitSequence;

    d["record"] = objectToPy(revision.record);
    return d;
}

nb::dict relationshipRevisionToPy(const vo::RelationshipRevision& revision)
{
    nb::dict d;
    d["uuid"] = revision.uuid; d["record"] = edgeToPy(revision.record);
    d["a_revision"] = revision.aRevision; d["b_revision"] = revision.bRevision;
    d["a_geometry_revision"] = revision.aGeometryRevision; d["b_geometry_revision"] = revision.bGeometryRevision;
    d["version"] = revision.version; d["generation"] = revision.generation;
    d["a_version"] = revision.aVersion; d["b_version"] = revision.bVersion;
    return d;
}

// Convert native results after the transaction (and any GIL release) ends.
// Numbered references as dicts {store, generation, object, version}, through
// the shared wire codec.
std::vector<vo::ObjectVersion> objectVersionsFromPy(const nb::handle& refs)
{
    nb::dict request; request["versions"] = refs;
    return vo::objectVersionsRequestFromJson(jsonDumps(request));
}

nb::list objectVersionsToPy(const std::vector<vo::ObjectVersion>& refs)
{
    return nb::cast<nb::list>(jsonLoads(vo::objectVersionsRequestToJson(refs))["versions"]);
}

nb::dict selectionRecordToPy(const vo::SelectionRecord& r)
{
    nb::dict d;
    d["store"] = r.storeUuid; d["generation"] = r.generation; d["commit_sequence"] = r.commitSequence;
    d["query"] = jsonLoads(r.queryJson);
    d["transforms"] = objectVersionsToPy(r.transforms);
    d["objects"] = objectVersionsToPy(r.objects);
    return d;
}

vo::SelectionRecord selectionRecordFromPy(const nb::dict& d)
{
    vo::SelectionRecord r;
    r.storeUuid = nb::cast<std::string>(d["store"]); r.generation = nb::cast<std::string>(d["generation"]);
    r.commitSequence = nb::cast<int64_t>(d["commit_sequence"]);
    r.queryJson = nb::isinstance<nb::str>(d["query"]) ? nb::cast<std::string>(d["query"]) : jsonDumps(d["query"]);
    r.transforms = objectVersionsFromPy(d["transforms"]);
    r.objects = objectVersionsFromPy(d["objects"]);
    return r;
}

nb::list proximityHitsToPy(const std::vector<vo::SurfaceProximityHit>& hits)
{
    nb::list result;
    for (const auto& hit : hits) {
        nb::dict row; row["object_uuid"] = hit.objectUuid; row["revision_uuid"] = hit.revisionUuid;
        row["grid"] = hit.grid; row["point"] = hit.point; row["distance"] = hit.distance;
        result.append(row);
    }
    return result;
}

nb::bytes bytesToPy(const std::vector<uint8_t>& value)
{
    return nb::bytes(reinterpret_cast<const char*>(value.data()), value.size());
}

nb::object toPy(const vo::ObjectRecord& value) { return objectToPy(value); }
nb::object toPy(const vo::EdgeRecord& value) { return edgeToPy(value); }
nb::object toPy(const vo::GeometryRevision& value) { return geometryToPy(value); }
nb::object toPy(const vo::ObjectRevision& value) { return objectRevisionToPy(value); }
nb::object toPy(const vo::RelationshipRevision& value) { return relationshipRevisionToPy(value); }
nb::bytes toPy(const std::vector<uint8_t>& value) {
    return nb::bytes(reinterpret_cast<const char*>(value.data()), value.size());
}
template<class T> T toPy(T value) { return value; }
std::vector<std::string> toPy(std::vector<std::string> value) { return value; }
std::optional<std::filesystem::path> toPy(std::optional<std::filesystem::path> value) { return value; }
nb::object toPy(const std::optional<int64_t>& value) { return value ? nb::cast(*value) : nb::none(); }
template<class T> nb::object toPy(const std::optional<T>& value) {
    return value ? toPy(*value) : nb::none();
}
template<class T> nb::list toPy(const std::vector<T>& values) {
    nb::list result;
    for (const auto& value : values) result.append(toPy(value));
    return result;
}

template<auto Method, bool ReleaseGil = false>
auto readMethod()
{
    return []<class R, class... Args>(R (vo::ObjectStore::*)(const vo::Txn&, Args...) const) {
        return [](PyObjectStore& self, Args... args) {
            return toPy(self.withRead<ReleaseGil>([&](auto& db, auto& txn) {
                return (db.*Method)(txn, args...);
            }));
        };
    }(Method);
}

template<auto Method, class T = PyObjectStore>
auto writeMethod()
{
    return []<class R, class... Args>(R (vo::ObjectStore::*)(vo::WriteTxn&, Args...)) {
        return [](T& self, Args... args) {
            return self.withWrite([&](auto& db, auto& txn) { return (db.*Method)(txn, args...); });
        };
    }(Method);
}

// The geometry payload a mutation expects, from the object version a script
// read (a conflict when the object moved on); geometry revisions stay internal.
std::string expectedGeometry(const vo::ObjectStore& db, const vo::Txn& txn, const std::string& uuid, int64_t version)
{
    const auto geometry = db.geometryAtVersion(txn, uuid, version);
    if (!geometry) throw std::invalid_argument("geometry: object " + uuid + " has no geometry at version " + std::to_string(version));
    return *geometry;
}

vo::EdgeGeometryRepair repairFromPy(const nb::dict& d)
{
    vo::EdgeGeometryRepair repair;
    repair.edgeUuid = nb::cast<std::string>(d["edge_uuid"]);
    repair.expectedVersion = nb::cast<int64_t>(d["expected_version"]);
    repair.aRevision = nb::cast<std::optional<std::string>>(d["a_revision"]);
    repair.bRevision = nb::cast<std::optional<std::string>>(d["b_revision"]);
    repair.aAnchor = anchorFromPy(d["a_anchor"]);
    repair.bAnchor = anchorFromPy(d["b_anchor"]);
    return repair;
}

nb::dict detectorReportToPy(const vc::object_links::DetectorReport& rep)
{
    nb::dict d;
    d["objects_considered"] = rep.objectsConsidered;
    d["candidates_found"] = rep.candidatesFound;
    d["edges_created"] = rep.edgesCreated;
    d["edges_updated"] = rep.edgesUpdated;
    d["edges_skipped_existing"] = rep.edgesSkippedExisting;
    d["stale_removed"] = rep.staleRemoved;
    d["problems"] = rep.problems;
    return d;
}

// Detector keyword arguments as the JSON the detectors' parameter parsers take
// (object_links/LinkObjects.hpp), so Python and link_objects plans agree.
nlohmann::json kwargsJson(const nb::kwargs& kwargs)
{
    nb::dict d;
    for (auto item : kwargs) d[item.first] = item.second;
    return nlohmann::json::parse(jsonDumps(d));
}

template<class T>
void bindMutations(nb::class_<T>& cls)
{
    cls
        .def(
            "upsert_object",
            [](T& self, const nb::dict& rec) {
                return self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) { return g.upsertObject(t, objectFromPy(rec)); });
            },
            nb::arg("record"), "Create or update an object from a dict; returns its uuid.")
        .def(
            "add_edge",
            [](T& self, const nb::dict& rec) {
                return self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) { return g.addEdge(t, edgeFromPy(rec)); });
            },
            nb::arg("record"), "Create an edge from a dict (keys: a, b, kind, anchors, tags, meta); returns uuid.")
        .def(
            "attach",
            [](T& self, const std::string& data, const std::string& owner, const std::string& role) {
                return self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) { return g.attach(t, data, owner, role); });
            },
            nb::arg("data_uuid"), nb::arg("owner_uuid"), nb::arg("role"),
            "Attach object data to owner in role (an attachment relationship); returns the edge uuid.")
        .def(
            "create_surface",
            [](T& self, const std::string& uuid, const std::string& name, const std::vector<std::string>& tags,
               const nb::object& coordinates, const std::string& tool, const nb::object& parameters,
               const std::vector<std::string>& sources, const nb::object& meta, const nb::object& geometry) {
                const auto data = geometryInputFromPy(geometry);
                const vo::SurfaceProvenance provenance{tool, nlohmann::json::parse(jsonDumps(parameters)), sources};
                const auto frame = nlohmann::json::parse(jsonDumps(coordinates));
                const auto rest = nlohmann::json::parse(jsonDumps(meta));
                self.template withWrite<true>([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    vo::createSurface(g, t, uuid, name, tags, frame, provenance, rest, data);
                });
            },
            nb::arg("uuid"), nb::arg("name"), nb::arg("tags"), nb::arg("coordinates"), nb::arg("tool"),
            nb::arg("parameters"), nb::arg("sources"), nb::arg("meta"), nb::arg("geometry"),
            "Create surface uuid holding geometry (objects/SurfaceCreate.hpp): bound by coordinates, with "
            "meta.provenance {tool, parameters, sources} and the rest of meta without tifxyz file keys.")
        .def("remove_edge", writeMethod<&vo::ObjectStore::removeEdge, T>(),
             nb::arg("uuid"))
        .def("set_approval", writeMethod<&vo::ObjectStore::setApproval, T>(),
             nb::arg("uuid"), nb::arg("state"), "state: 'pending', 'approved' or 'rejected' (objects and edges).")
        .def(
            "set_meta",
            [](T& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                const std::string json = jsonDumps(value);
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.setMeta(t, uuid, key, json);
                });
            },
            nb::arg("uuid"), nb::arg("key"), nb::arg("value").none(), "Set one open-ended metadata key (any JSON value).")
        .def("remove_meta", writeMethod<&vo::ObjectStore::removeMeta, T>(),
             nb::arg("uuid"), nb::arg("key"))
        .def("put_geometry", [](T& self, const std::string& uuid, const nb::object& input,
                               const std::optional<int64_t>& version, const nb::list& choices, int64_t surfaceTileSize) {
            const auto data = geometryInputFromPy(input);
            const auto edits = edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](vo::ObjectStore& db, vo::WriteTxn& txn) {
                const auto expected = version ? db.geometryAtVersion(txn, uuid, *version) : std::nullopt;
                return db.putGeometry(txn, uuid, data, expected, edits, vo::GeometryStorageOptions{surfaceTileSize});
            });
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_version") = std::nullopt,
           nb::arg("edge_edits") = nb::list(), nb::arg("surface_tile_size") = vo::GeometryStorageOptions{}.surfaceTileSize,
           "Create geometry (expected_version None: the object has none yet) or replace it, the object expected at "
           "expected_version. Returns the new geometry revision.")
        .def("edit_transformed_coordinates", [](T& self,const vo::TransformedGeometryView& view,
            const std::string& array,int64_t start,const std::vector<vc::geometry::Point>& points,const nb::list& choices) {
            const auto edges=edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](auto& db,auto& txn){return view.writeCoordinates(db,txn,array,start,points,edges);});
        },nb::arg("view"),nb::arg("array"),nb::arg("start"),nb::arg("points"),nb::arg("edge_edits")=nb::list())
        .def("edit_transformed_surface", [](T& self,const vo::TransformedGeometryView& view,
            int64_t row,int64_t column,int64_t rows,int64_t columns,const std::vector<vc::geometry::SurfacePoint>& points,const nb::list& choices) {
            const auto edges=edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](auto& db,auto& txn){return view.writeSurface(db,txn,row,column,rows,columns,points,edges);});
        },nb::arg("view"),nb::arg("row"),nb::arg("column"),nb::arg("rows"),nb::arg("columns"),nb::arg("points"),nb::arg("edge_edits")=nb::list())
        .def("edit_geometry", [](T& self, const std::string& uuid, const nb::list& input,
                                int64_t version, const nb::list& choices) {
            const auto edits = geometryEditsFromPy(input);
            const auto edges = edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](vo::ObjectStore& db, vo::WriteTxn& txn) {
                return db.editGeometry(txn, uuid, edits, expectedGeometry(db, txn, uuid, version), edges);
            });
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_version"), nb::arg("edge_edits") = nb::list())
        .def("update_surface_channels", [](T& self, const std::string& uuid, const nb::list& channels,
                                           const std::vector<std::string>& remove, int64_t version, const nb::list& choices) {
            nb::dict input; input["kind"] = "surface"; input["arrays"] = channels;
            const auto data = geometryFromPy(input); const auto edges = edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](auto& db, auto& txn) {
                return db.updateSurfaceChannels(txn, uuid, data.arrays, remove, expectedGeometry(db, txn, uuid, version), edges);
            });
        }, nb::arg("object_uuid"), nb::arg("channels"), nb::arg("remove"), nb::arg("expected_version"), nb::arg("edge_edits") = nb::list())
        .def("edit_geometry_rectangle", [](T& self, const std::string& uuid, const std::string& name,
                                          int64_t row, int64_t column, int64_t rows, int64_t columns,
                                          const nb::bytes& data, int64_t version, const nb::list& choices) {
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::vector<uint8_t> bytes(p, p + data.size()); const auto edges = edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](auto& db, auto& txn) {
                return db.editGeometryRectangle(txn, uuid, name, row, column, rows, columns, bytes,
                                                expectedGeometry(db, txn, uuid, version), edges);
            });
        }, nb::arg("object_uuid"), nb::arg("array_name"), nb::arg("row"), nb::arg("column"), nb::arg("rows"), nb::arg("columns"),
           nb::arg("data"), nb::arg("expected_version"), nb::arg("edge_edits") = nb::list())
        .def("affected_geometry_edit_edges", [](T& self, const std::string& uuid,
                                                const nb::list& input, int64_t version) {
            const auto edits = geometryEditsFromPy(input);
            const auto edges = self.withRead([&](vo::ObjectStore& db, const vo::Txn& txn) {
                return db.affectedGeometryEditEdges(txn, uuid, edits, expectedGeometry(db, txn, uuid, version));
            });
            return toPy(edges);
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_version"))
        .def("affected_geometry_edges", [](T& self, const std::string& uuid, const nb::object& input) {
            const auto data = geometryInputFromPy(input);
            auto edges = self.withRead([&](vo::ObjectStore& db, const vo::Txn& txn) { return db.affectedGeometryEdges(txn, uuid, data); });
            return toPy(edges);
        }, nb::arg("object_uuid"), nb::arg("geometry"))
        .def("repair_edge_geometry", [](T& self, const nb::dict& input) {
            const auto repair = repairFromPy(input);
            self.withWrite([&](vo::ObjectStore& db, vo::WriteTxn& txn) { db.repairEdgeGeometry(txn, repair);  });
        }, nb::arg("repair"), "Atomically publish anchors and checked revisions; reject changed endpoints or edge version.");
}

}  // namespace

// Client of a store served by vc_object_server: the same request, query and
// numbered-reference forms as the local ObjectStore methods. with_status=True
// also returns the status of the server snapshot an answer came from.
void bindHostedStore(nb::module_& m)
{
    using Hosted = vo::HostedStore;
    auto statusPy = [](const vo::StoreStatus& status) { return jsonLoads(vo::storeStatusToJson(status)); };
    auto answer = [statusPy](nb::object result, const vo::StoreStatus& status, bool withStatus) -> nb::object {
        return withStatus ? nb::make_tuple(result, statusPy(status)) : result;
    };
    auto objectQuery = [](const nb::object& query, std::optional<std::vector<std::string>>& fields) {
        return vo::objectQueryRequestFromJson(nb::isinstance<nb::str>(query) ? nb::cast<std::string>(query) : jsonDumps(query), &fields);
    };
    auto edgeQuery = [](const nb::object& query, std::optional<std::vector<std::string>>& fields) {
        return vo::edgeQueryRequestFromJson(nb::isinstance<nb::str>(query) ? nb::cast<std::string>(query) : jsonDumps(query), &fields);
    };
    auto objectsPy = [](const std::vector<vo::ObjectRecord>& records, const std::optional<std::vector<std::string>>& fields) {
        const std::set<std::string> selected = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
        nb::list result;
        for (const auto& record : records) result.append(objectToPy(record, fields ? &selected : nullptr));
        return result;
    };
    auto edgesPy = [](const std::vector<vo::EdgeRecord>& records, const std::optional<std::vector<std::string>>& fields) {
        const std::set<std::string> selected = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
        nb::list result;
        for (const auto& record : records) result.append(edgeToPy(record, fields ? &selected : nullptr));
        return result;
    };
    nb::class_<Hosted>(m, "HostedStore")
        .def("__init__", [](Hosted* self, const std::string& endpoint, int connections, long connectTimeoutMs,
                            long stallTimeoutS, int retries, long requestTimeoutMs, long linkTimeoutS) {
            vo::HostedStoreOptions o;
            o.connections = connections; o.connectTimeoutMs = connectTimeoutMs; o.stallTimeoutS = stallTimeoutS;
            o.retries = retries; o.requestTimeoutMs = requestTimeoutMs; o.linkTimeoutS = linkTimeoutS;
            new (self) Hosted(endpoint, o);
        }, nb::arg("endpoint"), nb::arg("connections") = 4, nb::arg("connect_timeout_ms") = 10000,
           nb::arg("stall_timeout_s") = 60, nb::arg("retries") = 5, nb::arg("request_timeout_ms") = 0,
           nb::arg("link_timeout_s") = 300,
           "link_timeout_s: how long a link_objects request may take to be answered (its detection runs on the "
           "service); set it with the service's --request-deadline.")
        .def_prop_ro("endpoint", &Hosted::endpoint)
        .def("status", [statusPy](Hosted& self) {
            vo::StoreStatus status;
            { nb::gil_scoped_release release; status = self.status(); }
            return statusPy(status);
        })
        .def("service_report", [](Hosted& self) {
            std::string text;
            { nb::gil_scoped_release release; text = self.serviceReport(); }
            return jsonLoads(text);
        })
        .def("submit", [](Hosted& self, const nb::object& request, const nb::dict& payloads) {
            vo::TransactionPayloads bytes;
            for (const auto& [hash, value] : payloads) {
                const auto data = nb::cast<nb::bytes>(value);
                const auto* begin = reinterpret_cast<const uint8_t*>(data.c_str());
                bytes.emplace(nb::cast<std::string>(hash), std::vector<uint8_t>(begin, begin + data.size()));
            }
            const auto parsed = vo::transactionRequestFromJson(
                nb::isinstance<nb::str>(request) ? nb::cast<std::string>(request) : jsonDumps(request), bytes);
            vo::TransactionReceipt receipt;
            { nb::gil_scoped_release release; receipt = self.submit(parsed); }
            return jsonLoads(vo::transactionReceiptToJson(receipt));
        }, nb::arg("request"), nb::arg("payloads") = nb::dict(),
           "Stage payloads and submit; an unknown outcome is recovered by receipt() or the identical request.")
        .def("receipt", [](Hosted& self, const std::string& request) -> nb::object {
            std::optional<vo::TransactionReceipt> receipt;
            { nb::gil_scoped_release release; receipt = self.receipt(request); }
            return receipt ? jsonLoads(vo::transactionReceiptToJson(*receipt)) : nb::none();
        }, nb::arg("request_uuid"))
        .def("changes_since", [](Hosted& self, const nb::dict& cursor, int64_t maxChanges) {
            const vo::FeedCursor c{nb::cast<std::string>(cursor["store"]), nb::cast<std::string>(cursor["generation"]),
                                   nb::cast<int64_t>(cursor["sequence"]),
                                   cursor.contains("offset") ? nb::cast<int64_t>(cursor["offset"]) : 0};
            vo::ChangePage page;
            { nb::gil_scoped_release release; page = self.changesSince(c, maxChanges); }
            return jsonLoads(vo::changePageToJson(page));
        }, nb::arg("cursor"), nb::arg("max_changes") = 4096)
        .def("current_object_revision", [answer](Hosted& self, const std::string& uuid, bool withStatus) {
            vo::StoreStatus status; std::optional<vo::ObjectRevision> revision;
            { nb::gil_scoped_release release; revision = self.currentObjectRevision(uuid, &status); }
            return answer(toPy(revision), status, withStatus);
        }, nb::arg("object_uuid"), nb::arg("with_status") = false)
        .def("current_version", [](Hosted& self, const std::string& uuid) {
            std::optional<int64_t> version;
            { nb::gil_scoped_release release; version = self.currentVersion(uuid); }
            return toPy(version);
        }, nb::arg("object_uuid"))
        .def("objects_at_versions", [](Hosted& self, const nb::list& refs, int64_t batch) {
            const auto versions = objectVersionsFromPy(refs);
            std::vector<std::optional<vo::ObjectRevision>> revisions;
            { nb::gil_scoped_release release; revisions = self.objectsAtVersions(versions, batch); }
            return toPy(revisions);
        }, nb::arg("refs"), nb::arg("batch") = 1000)
        .def("geometry_revision", [](Hosted& self, const std::string& revision) {
            std::optional<vo::GeometryRevision> geometry;
            { nb::gil_scoped_release release; geometry = self.geometryRevision(revision); }
            return toPy(geometry);
        }, nb::arg("revision_uuid"))
        .def("read_geometry_array", [](Hosted& self, const std::string& revision, const std::string& array,
                                       int64_t offset, const std::optional<int64_t>& count) {
            std::vector<uint8_t> bytes;
            { nb::gil_scoped_release release; bytes = self.readGeometryArray(revision, array, offset, count); }
            return bytesToPy(bytes);
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("byte_offset") = 0, nb::arg("byte_count") = std::nullopt)
        .def("select_objects", [=](Hosted& self, const nb::object& query, bool withStatus) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = objectQuery(query, fields);
            vo::StoreStatus status; std::vector<vo::ObjectRecord> records;
            { nb::gil_scoped_release release; records = self.selectObjects(selection, fields, &status); }
            return answer(objectsPy(records, fields), status, withStatus);
        }, nb::arg("query"), nb::arg("with_status") = false, "One page; the query (wire form) needs a limit.")
        .def("select_all_objects", [=](Hosted& self, const nb::object& query, int64_t pageSize, int maxRestarts, bool withStatus) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = objectQuery(query, fields);
            vo::StoreStatus status; std::vector<vo::ObjectRecord> records;
            { nb::gil_scoped_release release; records = self.selectAllObjects(selection, fields, pageSize, maxRestarts, &status); }
            return answer(objectsPy(records, fields), status, withStatus);
        }, nb::arg("query"), nb::arg("page_size") = 1000, nb::arg("max_restarts") = 3, nb::arg("with_status") = false,
           "Every match, exact as of one commit; raises StoreBusy when commits never settle.")
        .def("select_edges", [=](Hosted& self, const nb::object& query, bool withStatus) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = edgeQuery(query, fields);
            vo::StoreStatus status; std::vector<vo::EdgeRecord> records;
            { nb::gil_scoped_release release; records = self.selectEdges(selection, fields, &status); }
            return answer(edgesPy(records, fields), status, withStatus);
        }, nb::arg("query"), nb::arg("with_status") = false)
        .def("select_all_edges", [=](Hosted& self, const nb::object& query, int64_t pageSize, int maxRestarts, bool withStatus) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = edgeQuery(query, fields);
            vo::StoreStatus status; std::vector<vo::EdgeRecord> records;
            { nb::gil_scoped_release release; records = self.selectAllEdges(selection, fields, pageSize, maxRestarts, &status); }
            return answer(edgesPy(records, fields), status, withStatus);
        }, nb::arg("query"), nb::arg("page_size") = 1000, nb::arg("max_restarts") = 3, nb::arg("with_status") = false)
        .def("record_selection", [](Hosted& self, const nb::object& query, int64_t pageSize, int maxRestarts) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = vo::objectQueryRequestFromJson(
                nb::isinstance<nb::str>(query) ? nb::cast<std::string>(query) : jsonDumps(query), &fields);
            vo::SelectionRecord record;
            { nb::gil_scoped_release release; record = self.recordSelection(selection, pageSize, maxRestarts); }
            return selectionRecordToPy(record);
        }, nb::arg("query"), nb::arg("page_size") = 1000, nb::arg("max_restarts") = 3)
        .def("object_history", [](Hosted& self, const std::string& uuid) {
            std::vector<vo::ObjectRevision> history;
            { nb::gil_scoped_release release; history = self.objectHistory(uuid); }
            return toPy(history);
        }, nb::arg("object_uuid"))
        .def("relationships_at_versions", [](Hosted& self, const nb::list& refs) {
            const auto versions = objectVersionsFromPy(refs);
            std::vector<vo::SelectedRelationship> revisions;
            { nb::gil_scoped_release release; revisions = self.relationshipsAt(versions); }
            nb::list result;
            for (const auto& revision : revisions) {
                auto d = relationshipRevisionToPy(revision.relationship);
                d["applicable"] = revision.applicable;
                result.append(d);
            }
            return result;
        }, nb::arg("refs"))
        .def("geometry_region", [answer](Hosted& self, const vo::RegionSlice& slice, const vo::RegionOptions& options, bool withStatus) {
            const auto view = slice; const auto request = options;
            vo::StoreStatus status; vo::RegionResult result;
            { nb::gil_scoped_release release; result = self.geometryRegion(view, request, &status); }
            return answer(regionResultToPy(result), status, withStatus);
        }, nb::arg("slice"), nb::arg("options") = vo::RegionOptions{}, nb::arg("with_status") = false)
        .def("nearest_surfaces", [answer](Hosted& self, const std::array<double, 3>& point, double tolerance,
            const std::optional<std::vector<std::string>>& uuids, const std::optional<std::string>& coordinates,
            const std::optional<std::string>& volume, double targetScale, bool withStatus) {
            const vo::SurfaceProximityQuery query{point, tolerance, uuids, coordinates, volume, targetScale};
            vo::StoreStatus status; std::vector<vo::SurfaceProximityHit> hits;
            { nb::gil_scoped_release release; hits = self.nearestSurfaces(query, &status); }
            return answer(proximityHitsToPy(hits), status, withStatus);
        }, nb::arg("point"), nb::arg("tolerance"), nb::arg("uuids") = std::nullopt,
           nb::arg("coordinates") = std::nullopt, nb::arg("volume") = std::nullopt, nb::arg("target_scale") = 1.0,
           nb::arg("with_status") = false)
        .def("put_artifact", [](Hosted& self, const std::filesystem::path& file) {
            nb::gil_scoped_release release;
            return self.putArtifact(file);
        }, nb::arg("file"))
        .def("artifact_length", [](Hosted& self, const std::string& sha256) -> nb::object {
            std::optional<uint64_t> length;
            { nb::gil_scoped_release release; length = self.artifactLength(sha256); }
            return length ? nb::cast(*length) : nb::none();
        }, nb::arg("sha256"))
        .def("read_artifact", [](Hosted& self, const std::string& sha256, uint64_t offset, const std::optional<uint64_t>& count) {
            std::vector<uint8_t> bytes;
            { nb::gil_scoped_release release; bytes = self.readArtifact(sha256, offset, count); }
            return bytesToPy(bytes);
        }, nb::arg("sha256"), nb::arg("offset") = 0, nb::arg("count") = std::nullopt)
        .def("fetch_artifact", [](Hosted& self, const nb::dict& file, const std::filesystem::path& destination) {
            const vo::ArtifactFile f{nb::cast<std::string>(file["name"]), nb::cast<std::string>(file["sha256"]),
                                     nb::cast<uint64_t>(file["length"])};
            nb::gil_scoped_release release;
            self.fetchArtifact(f, destination);
        }, nb::arg("file"), nb::arg("destination"), "Write one manifest entry {name, sha256, length} atomically, verified.");
    nb::exception<vo::StoreBusy>(m, "StoreBusy", PyExc_RuntimeError);
    nb::exception<vo::PreparationFailed>(m, "PreparationFailed", PyExc_RuntimeError);
}

// ObjectStoreSession operations that take requests, numbered references and
// artifacts in the same forms as ObjectStore and HostedStore.
void bindSessionOperations(nb::class_<vo::ObjectStoreSession> session)
{
    using Session = vo::ObjectStoreSession;
    auto statusPy = [](const vo::StoreStatus& status) { return jsonLoads(vo::storeStatusToJson(status)); };
    auto requestFromPy = [](const nb::object& request, const nb::dict& payloads) {
        vo::TransactionPayloads bytes;
        for (const auto& [hash, value] : payloads) {
            const auto data = nb::cast<nb::bytes>(value);
            const auto* begin = reinterpret_cast<const uint8_t*>(data.c_str());
            bytes.emplace(nb::cast<std::string>(hash), std::vector<uint8_t>(begin, begin + data.size()));
        }
        return vo::transactionRequestFromJson(nb::isinstance<nb::str>(request) ? nb::cast<std::string>(request) : jsonDumps(request), bytes);
    };
    auto receiptPy = [](const vo::TransactionReceipt& receipt) { return jsonLoads(vo::transactionReceiptToJson(receipt)); };
    auto filesPy = [](const std::vector<vo::ArtifactFile>& files) {
        nb::list result;
        for (const auto& f : files) { nb::dict d; d["name"] = f.name; d["sha256"] = f.sha256; d["length"] = f.length; result.append(d); }
        return result;
    };
    session
        .def("attach_location", [](Session& self, const std::string& location, const std::string& author, bool readOnly) {
            vo::OpenOptions options; options.author = author; options.readOnly = readOnly;
            nb::gil_scoped_release release;
            return self.attachLocation(location, options);
        }, nb::arg("location"), nb::arg("author") = std::string(), nb::arg("read_only") = false,
           "Attach a store location: a vc_object_server endpoint (http:// or https://; always writable) or a store file.")
        .def("attach_hosted", [](Session& self, const std::string& endpoint, int64_t pollIntervalMs, size_t cacheBytes,
                                 int retries, long connectTimeoutMs, const std::string& author, long linkTimeoutS) {
            vo::HostedSessionOptions options;
            options.author = author;
            options.pollInterval = std::chrono::milliseconds(pollIntervalMs);
            options.cacheBytes = cacheBytes;
            options.client.retries = retries;
            options.client.connectTimeoutMs = connectTimeoutMs;
            options.client.linkTimeoutS = linkTimeoutS;
            nb::gil_scoped_release release;
            return self.attachHosted(endpoint, options);
        }, nb::arg("endpoint"), nb::arg("poll_interval_ms") = 2000, nb::arg("cache_bytes") = vo::HostedSessionOptions{}.cacheBytes,
           nb::arg("retries") = 5, nb::arg("connect_timeout_ms") = 10000, nb::arg("author") = std::string(),
           nb::arg("link_timeout_s") = 300,
           "Attach a store served by vc_object_server; use the typed operations (no local callbacks).")
        .def("read_surface_grid", [](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid,
                                     const std::string& volume, double targetScale, std::optional<int64_t> version,
                                     const std::optional<std::string>& generation) {
            vo::SurfaceGrid grid;
            {
                nb::gil_scoped_release release;
                self.readStore(attachment, [&](const vo::StoreReader& reader, const auto&) {
                    grid = vo::readSurfaceGrid(reader, uuid, volume, targetScale, version, generation);
                });
            }
            nb::dict d;
            const auto rows = size_t(grid.rows), columns = size_t(grid.columns);
            d["points"] = ownedArray(std::move(grid.points), {rows, columns, 3});
            d["valid"] = ownedArray(std::move(grid.valid), {rows, columns});
            d["scale"] = nb::make_tuple(grid.scale[0], grid.scale[1]);
            d["record"] = objectToPy(grid.record);
            d["version"] = grid.version; d["generation"] = grid.generation;
            d["geometry_revision"] = grid.geometryRevision; d["channels"] = grid.channels;
            return d;
        }, nb::arg("attachment"), nb::arg("surface_uuid"), nb::arg("volume"), nb::arg("target_scale") = 1.0,
           nb::arg("version") = nb::none(), nb::arg("generation") = nb::none(),
           "A surface (current, or numbered version) as a whole grid in volume's voxels (objects/SurfaceGrid.hpp): "
           "points (rows, columns, 3) float32 with -1 at invalid vertices, valid (uint8), scale, record, version, "
           "generation, geometry_revision, channels.")
        .def("select_objects", [](Session& self, const vo::AttachmentIdentity& attachment, const nb::object& query) {
            std::optional<std::vector<std::string>> fields;
            const auto selection = vo::objectQueryRequestFromJson(
                nb::isinstance<nb::str>(query) ? nb::cast<std::string>(query) : jsonDumps(query), &fields);
            std::vector<vo::ObjectRecord> records;
            {
                nb::gil_scoped_release release;
                self.readStore(attachment, [&](const vo::StoreReader& reader, const auto&) { records = reader.selectObjects(selection, fields); });
            }
            const std::set<std::string> selected = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list result;
            for (const auto& record : records) result.append(objectToPy(record, fields ? &selected : nullptr));
            return result;
        }, nb::arg("attachment"), nb::arg("query"),
           "Every record the query (wire form, as HostedStore.select_objects; complete without a limit) selects in one read.")
        .def("read_geometry_arrays", [](Session& self, const vo::AttachmentIdentity& attachment,
                                        const std::vector<std::string>& revisions, const std::string& name) {
            std::shared_ptr<vo::PackedGeometryArray> packed;
            {
                nb::gil_scoped_release release;
                self.readStore(attachment, [&](const vo::StoreReader& reader, const auto&) {
                    packed = std::make_shared<vo::PackedGeometryArray>(reader.readGeometryArrays(revisions, name));
                });
            }
            return PyPackedGeometryArray{std::move(packed)};
        }, nb::arg("attachment"), nb::arg("revision_uuids"), nb::arg("array_name"),
           "Read homogeneous arrays into one verified packed buffer with row offsets (as ObjectStore.read_geometry_arrays).")
        .def("create_surface", [](Session& self, const vo::AttachmentIdentity& attachment, const std::string& name,
                                  const std::vector<std::string>& tags, const nb::object& coordinates, const std::string& tool,
                                  const nb::object& parameters, const std::vector<std::string>& sources, const nb::object& meta,
                                  const nb::object& geometry) {
            auto data = geometryInputFromPy(geometry);
            const vo::SurfaceProvenance provenance{tool, nlohmann::json::parse(jsonDumps(parameters)), sources};
            const auto frame = nlohmann::json::parse(jsonDumps(coordinates));
            const auto rest = nlohmann::json::parse(jsonDumps(meta));
            nb::gil_scoped_release release;
            return vo::createSurface(self, attachment, name, tags, frame, provenance, rest, std::move(data));
        }, nb::arg("attachment"), nb::arg("name"), nb::arg("tags"), nb::arg("coordinates"), nb::arg("tool"),
           nb::arg("parameters"), nb::arg("sources"), nb::arg("meta"), nb::arg("geometry"),
           "Create a surface under a new UUID in one checked request (objects/SurfaceCreate.hpp); returns the UUID.")
        .def("attach_document", [](Session& self, const vo::AttachmentIdentity& attachment, const std::string& owner,
                                   const nb::object& document, const std::string& role, const std::string& listKey,
                                   const std::string& positionKey, const std::string& name, const nb::object& coordinates) {
            const auto parsed = nlohmann::json::parse(jsonDumps(document));
            const auto frame = nlohmann::json::parse(jsonDumps(coordinates));
            nb::gil_scoped_release release;
            return vo::attachDocument(self, attachment, owner, parsed, {role, listKey, positionKey}, name, frame);
        }, nb::arg("attachment"), nb::arg("owner_uuid"), nb::arg("document"), nb::arg("role"), nb::arg("list_key"),
           nb::arg("position_key"), nb::arg("name"), nb::arg("coordinates"),
           "Attach a document to the owner's current version as a new point collection (objects/AttachedDocument.hpp): "
           "one point per record of document[list_key], at record[position_key]; returns its UUID.")
        .def("attached_document", [](Session& self, const vo::AttachmentIdentity& attachment, const std::string& owner,
                                     const std::string& role) -> nb::object {
            std::optional<nlohmann::json> document;
            { nb::gil_scoped_release release; document = vo::attachedDocument(self, attachment, owner, role); }
            return document ? jsonLoads(document->dump()) : nb::none();
        }, nb::arg("attachment"), nb::arg("owner_uuid"), nb::arg("role"), "The document attached to owner in role, or None.")
        .def("point_collections", [](Session& self, const vo::AttachmentIdentity& attachment, const std::vector<std::string>& uuids,
                                     const std::string& volume, double targetScale) {
            std::vector<vo::AttachedCollection> collections;
            {
                nb::gil_scoped_release release;
                self.readStore(attachment, [&](const vo::StoreReader& reader, const auto&) {
                    collections = vo::pointCollections(reader, uuids, vo::VolumeVoxels{volume, targetScale});
                });
            }
            return collectionsToPy(collections);
        }, nb::arg("attachment"), nb::arg("uuids"), nb::arg("volume"), nb::arg("target_scale") = 1.0,
           "Point collections by UUID (any role), in the volume's voxels at target_scale: uuid, name, ids, points, "
           "annotations, source.")
        .def("hosted", &Session::hosted, nb::arg("attachment"))
        .def("freshness", [statusPy](const Session& self, const vo::AttachmentIdentity& attachment) -> nb::object {
            const auto fresh = self.freshness(attachment);
            if (!fresh) return nb::none();
            nb::dict d;
            d["status"] = fresh->status ? statusPy(*fresh->status) : nb::none();
            d["checked_at_us"] = fresh->checkedAtUs ? nb::cast(*fresh->checkedAtUs) : nb::none();
            d["error"] = fresh->error;
            return d;
        }, nb::arg("attachment"))
        .def("refresh", &Session::refresh, nb::arg("attachment"), nb::call_guard<nb::gil_scoped_release>())
        .def("status", [statusPy](Session& self, const vo::AttachmentIdentity& attachment) {
            vo::StoreStatus status;
            { nb::gil_scoped_release release; status = self.status(attachment); }
            return statusPy(status);
        }, nb::arg("attachment"))
        .def("submit", [=](Session& self, const vo::AttachmentIdentity& attachment, const nb::object& request, const nb::dict& payloads) {
            const auto parsed = requestFromPy(request, payloads);
            vo::TransactionReceipt receipt;
            { nb::gil_scoped_release release; receipt = self.submit(attachment, parsed); }
            return receiptPy(receipt);
        }, nb::arg("attachment"), nb::arg("request"), nb::arg("payloads") = nb::dict())
        .def("receipt", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& requestUuid) -> nb::object {
            std::optional<vo::TransactionReceipt> receipt;
            { nb::gil_scoped_release release; receipt = self.receipt(attachment, requestUuid); }
            return receipt ? receiptPy(*receipt) : nb::none();
        }, nb::arg("attachment"), nb::arg("request_uuid"))
        .def("current_object", [](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid) {
            std::optional<vo::ObjectRevision> revision;
            { nb::gil_scoped_release release; revision = self.currentObject(attachment, uuid); }
            return toPy(revision);
        }, nb::arg("attachment"), nb::arg("object_uuid"))
        .def("objects_at_versions", [](Session& self, const vo::AttachmentIdentity& attachment, const nb::list& refs) {
            const auto versions = objectVersionsFromPy(refs);
            std::vector<std::optional<vo::ObjectRevision>> revisions;
            { nb::gil_scoped_release release; revisions = self.objectsAtVersions(attachment, versions); }
            return toPy(revisions);
        }, nb::arg("attachment"), nb::arg("refs"))
        .def("artifacts", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid,
                              const std::optional<int64_t>& version, const std::optional<std::string>& generation) {
            std::vector<vo::ArtifactFile> files;
            { nb::gil_scoped_release release; files = self.artifacts(attachment, uuid, version, generation); }
            return filesPy(files);
        }, nb::arg("attachment"), nb::arg("object_uuid"), nb::arg("version") = std::nullopt, nb::arg("generation") = std::nullopt)
        .def("materialize_artifacts", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid,
                                          const std::filesystem::path& destination, const std::optional<int64_t>& version, const std::optional<std::string>& generation) {
            std::vector<vo::ArtifactFile> files;
            { nb::gil_scoped_release release; files = self.materializeArtifacts(attachment, uuid, destination, version, generation); }
            return filesPy(files);
        }, nb::arg("attachment"), nb::arg("object_uuid"), nb::arg("destination"), nb::arg("version") = std::nullopt, nb::arg("generation") = std::nullopt)
        .def("artifact_file", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid,
                                  const std::string& name, const std::filesystem::path& cacheDirectory,
                                  const std::optional<int64_t>& version, const std::optional<std::string>& generation) -> nb::object {
            std::optional<std::filesystem::path> path;
            { nb::gil_scoped_release release; path = self.artifactFile(attachment, uuid, name, cacheDirectory, version, generation); }
            return path ? nb::cast(*path) : nb::none();
        }, nb::arg("attachment"), nb::arg("object_uuid"), nb::arg("name"), nb::arg("cache_directory"),
           nb::arg("version") = std::nullopt, nb::arg("generation") = std::nullopt,
           "One artifact as a local file under cache_directory/<sha256>/ (written unless there); None without it.")
        .def("publish_artifacts", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& uuid,
                                      int64_t expectedVersion, const std::map<std::string, std::filesystem::path>& files,
                                      const std::string& author) {
            const std::vector<std::pair<std::string, std::filesystem::path>> list(files.begin(), files.end());
            vo::TransactionReceipt receipt;
            { nb::gil_scoped_release release; receipt = self.publishArtifacts(attachment, uuid, expectedVersion, list, author); }
            return receiptPy(receipt);
        }, nb::arg("attachment"), nb::arg("object_uuid"), nb::arg("expected_version"), nb::arg("files"), nb::arg("author") = "",
           "Store {name: local path} durably and name them in the object's manifest in one checked request.")
        .def("use_draft_directory", [](Session& self, const std::filesystem::path& directory) {
            vo::DraftRecovery recovery;
            { nb::gil_scoped_release release; recovery = self.useDraftDirectory(directory); }
            nb::dict d; d["checkpoint"] = recovery.checkpoint; d["drafts"] = recovery.drafts;
            nb::list pending;
            for (const auto& p : recovery.pending) {
                nb::dict item; item["request_uuid"] = p.requestUuid; item["store_uuid"] = p.storeUuid; item["drafts"] = p.drafts;
                pending.append(item);
            }
            d["pending"] = pending;
            return d;
        }, nb::arg("directory"), "Durable drafts: install the directory's last checkpoint; call once, before making drafts.")
        .def("checkpoint_drafts", &Session::checkpointDrafts, nb::call_guard<nb::gil_scoped_release>())
        .def("draft_checkpoint_error", &Session::draftCheckpointError)
        .def("pending_submissions", [](const Session& self) {
            nb::list result;
            for (const auto& p : self.pendingSubmissions()) {
                nb::dict item; item["request_uuid"] = p.requestUuid; item["store_uuid"] = p.storeUuid; item["drafts"] = p.drafts;
                result.append(item);
            }
            return result;
        })
        .def("submit_durably", [=](Session& self, const vo::AttachmentIdentity& attachment, const nb::object& request,
                                   const nb::dict& payloads, const std::vector<std::string>& drafts, bool keepDrafts) {
            const auto parsed = requestFromPy(request, payloads);
            vo::DurableSave save;
            for (const auto& uuid : drafts) save.drafts.push_back({uuid, 0});
            save.keepDrafts = keepDrafts;
            vo::TransactionReceipt receipt;
            { nb::gil_scoped_release release; receipt = self.submitDurably(attachment, parsed, save); }
            return receiptPy(receipt);
        }, nb::arg("attachment"), nb::arg("request"), nb::arg("payloads") = nb::dict(), nb::arg("drafts") = std::vector<std::string>{},
           nb::arg("keep_drafts") = false,
           "A checked save that survives a crash (request written before it is sent). keep_drafts: the saved "
           "drafts stay (on the committed version, marked saved) for an editor still open on them.")
        .def("resolve_pending", [=](Session& self, const vo::AttachmentIdentity& attachment, const std::string& requestUuid) {
            vo::TransactionReceipt receipt;
            { nb::gil_scoped_release release; receipt = self.resolvePending(attachment, requestUuid); }
            return receiptPy(receipt);
        }, nb::arg("attachment"), nb::arg("request_uuid"));
}

NB_MODULE(_native, m)
{
    bindAfvSource(m);
    m.def("artifact_manifest_json", [](const nb::list& files) {
        std::vector<vo::ArtifactFile> parsed;
        for (const auto& f : files) {
            const auto d = nb::cast<nb::dict>(f);
            parsed.push_back({nb::cast<std::string>(d["name"]), nb::cast<std::string>(d["sha256"]), nb::cast<uint64_t>(d["length"])});
        }
        return vo::artifactManifestJson(std::move(parsed));
    }, nb::arg("files"), "Canonical manifest text for the 'artifacts' metadata key; files are {name, sha256, length}.");
    m.attr("ARTIFACTS_META_KEY") = vo::kArtifactsMetaKey;
    // link_objects requests (object_links/LinkObjects.hpp) on local stores
    // run their detectors in this process.
    vc::object_links::enableLinkObjects();
    m.attr("LINK_OBJECTS_OPERATION") = vo::kLinkObjectsOperation;
    m.attr("LINK_MARKER_KIND") = vc::object_links::kLinkMarkerKind;
    m.def("link_marker_uuid", [] { return vc::object_links::linkMarkerUuid(); },
          "UUID of the store's link marker object (the same in every store).");
    m.def("encode_selection_record", [](const nb::dict& record) { return bytesToPy(vo::encodeSelectionRecord(selectionRecordFromPy(record))); },
          nb::arg("record"), "Compact binary selection record (about 24 bytes per object).");
    m.def("decode_selection_record", [](const nb::bytes& data) {
        return selectionRecordToPy(vo::decodeSelectionRecord({reinterpret_cast<const uint8_t*>(data.c_str()), data.size()}));
    }, nb::arg("data"));

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

    m.doc() = "Revisioned objects and relationships, with owned geometry and typed snapshot views.";

    m.def("enable_history", [](const std::filesystem::path& path) {
        nb::gil_scoped_release release; vo::changeHistoryMode(path, vo::HistoryMode::On);
    }, nb::arg("path"), "Offline history enablement; close all handles and reload geometry tokens afterward.");
    m.def("remove_history_and_disable", [](const std::filesystem::path& path) {
        nb::gil_scoped_release release; vo::changeHistoryMode(path, vo::HistoryMode::Off);
    }, nb::arg("path"), "Explicitly remove all retained history and disable future history; close all handles first.");

    bindGraphReadView(m);
    bindGeometryRegion(m);
    bindHostedStore(m);
    bindSessionOperations(bindObjectStoreSession(m));
    bindTypedGeometry(m); // Shared C++ geometry and revision-pinned views.

    m.def("geometry_scale_to_volume", [](const nb::handle& meta, const std::string& volume) {
        return vo::geometryScaleToVolume(nlohmann::json::parse(jsonDumps(meta)), volume);
    }, nb::arg("meta"), nb::arg("volume"),
       "Volume voxels per stored unit for an object whose meta binds it to volume; None when it is not.");
    m.def("create_project", [](const std::filesystem::path& path, const std::string& name, const nb::object& meta,
                               const std::string& author) {
        vo::OpenOptions options;
        options.author = author;
        vo::createProject(path, name, meta.is_none() ? nlohmann::json::object() : nlohmann::json::parse(jsonDumps(meta)), options);
    }, nb::arg("path"), nb::arg("name"), nb::arg("meta") = nb::none(), nb::arg("author"),
       "Create a store at path holding a project named name.");
    m.def("project_resource_uuid", &vo::projectResourceUuid, nb::arg("kind"), nb::arg("location"), nb::arg("owner") = "");


    auto store = nb::class_<PyObjectStore>(m, "ObjectStore")
        .def_static("create", &PyObjectStore::create, nb::arg("path"), nb::arg("author"),
                    nb::arg("hostname") = std::nullopt, nb::arg("reader_pool_size") = 4,
                    nb::arg("geometry_cache_bytes") = 16 * 1024 * 1024,
                    nb::arg("spatial_cache_bytes") = 16 * 1024 * 1024, nb::arg("catalog") = nb::none(),
                    nb::kw_only(), nb::arg("history_enabled") = true, nb::arg("spatial_index") = true,
                    "Create a new object database at path (refuses existing files).")
        .def_static("open", &PyObjectStore::open, nb::arg("path"), nb::arg("author") = std::nullopt,
                    nb::arg("read_only") = false, nb::arg("hostname") = std::nullopt,
                    nb::arg("reader_pool_size") = 4,
                    nb::arg("geometry_cache_bytes") = 16 * 1024 * 1024,
                    nb::arg("spatial_cache_bytes") = 16 * 1024 * 1024, nb::arg("catalog") = nb::none(),
                    "Open an existing object database. author is required unless read_only.")
        .def("geometry_snapshot", [](PyObjectStore& self) {
            self.require();
            auto lease = self.db;
            nb::gil_scoped_release release;
            return vo::GeometrySnapshot(std::move(lease));
        })
        .def("close", &PyObjectStore::close)
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](PyObjectStore& self, nb::args) { self.close(); })
        .def_prop_ro("path", [](PyObjectStore& self) { return self.require().path().string(); })
        .def_prop_ro("hostname", [](PyObjectStore& self) { return self.require().hostname(); })
        .def_prop_ro("author", [](PyObjectStore& self) { return self.require().author(); })
        .def("provisioning", [](PyObjectStore& self) {
            const auto p = self.withRead([](auto& db, auto& txn) { return db.provisioning(txn); });
            nb::dict result; result["schema_version"] = p.schemaVersion; result["generation"] = p.generation;
            result["commit_sequence"] = p.commitSequence;
            result["history_enabled"] = p.history == vo::HistoryMode::On;
            result["spatial_index"] = p.spatialIndex; result["sealed"] = p.sealed;
            return result;
        })
        .def("database_identity", [](PyObjectStore& self) {
            const auto id = self.withRead([](auto& db, auto& txn) { return db.databaseIdentity(txn); });
            nb::dict result; result["uuid"] = id.uuid; result["schema_version"] = id.schemaVersion;
            nb::list tables;
            for (const auto& table : id.tables) {
                nb::dict row; row["uuid"] = table.uuid; row["name"] = table.name; row["primary_key"] = table.primaryKey;
                tables.append(row);
            }
            result["tables"] = tables; return result;
        })
        .def("seal", [](PyObjectStore& self) { self.withWrite([](auto& db, auto& txn) { db.seal(txn); }); },
             "Permanently seal this database against writes; read-only opens remain available.")
        .def("object", readMethod<&vo::ObjectStore::objectByUuid>(),
             nb::arg("uuid"))
        .def("objects_by_kind", readMethod<&vo::ObjectStore::objectsByKind>(),
             nb::arg("kind"), nb::arg("approval_state") = std::nullopt, nb::arg("include_retired") = false)
        .def("object_count", readMethod<&vo::ObjectStore::objectCount>(), nb::arg("kind"),
             nb::arg("include_retired") = false)
        .def("unbound_object_count", readMethod<&vo::ObjectStore::unboundObjectCount>(), nb::arg("kind"), nb::arg("volume"),
             "Live objects of kind not bound to volume (no coordinates in its voxels).")
        .def("objects_by_tag", readMethod<&vo::ObjectStore::objectsByTag>(),
             nb::arg("tag"), nb::arg("include_retired") = false)
        .def("object_tags", readMethod<&vo::ObjectStore::objectTags>(),
             nb::arg("kind") = std::nullopt)
        .def("delete_object", writeMethod<&vo::ObjectStore::deleteObject>(),
             nb::arg("uuid"))
        .def("edge", readMethod<&vo::ObjectStore::edgeByUuid>(),
             nb::arg("uuid"))
        .def("edges_of", readMethod<&vo::ObjectStore::edgesOf>(),
             nb::arg("uuid"), nb::arg("include_retired") = false)
        .def("edges_by_kind", readMethod<&vo::ObjectStore::edgesByKind>(),
             nb::arg("edge_kind"), nb::arg("include_retired") = false)
        .def("neighbours", readMethod<&vo::ObjectStore::neighbours>(),
             nb::arg("uuid"), nb::arg("edge_kind") = std::nullopt, nb::arg("neighbour_kind") = std::nullopt, nb::arg("include_retired") = false,
             nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false)
        .def("component",
             [](PyObjectStore& self, const std::string& uuid, const std::vector<std::string>& edgeKinds, bool includeRetired, std::optional<bool> adjacent, bool applicableOnly) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.componentOf(t, uuid, edgeKinds, nullptr, includeRetired, adjacent, applicableOnly); });
             },
             nb::arg("uuid"), nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("include_retired") = false,
             nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false)
        .def("reachable_within",
             [](PyObjectStore& self, const std::string& uuid, int maxDepth, const std::vector<std::string>& edgeKinds, bool includeRetired, std::optional<bool> adjacent, bool applicableOnly) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.reachableWithin(t, uuid, maxDepth, edgeKinds, nullptr, includeRetired, adjacent, applicableOnly); });
             },
             nb::arg("uuid"), nb::arg("max_depth"), nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("include_retired") = false,
             nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false)
        .def("set_meta_many", [](PyObjectStore& self, const nb::list& updates) {
            struct Update { std::string uuid, key, value; };
            std::vector<Update> batch;
            for (nb::handle item : updates) {
                auto row = nb::cast<nb::sequence>(item);
                if (nb::len(row) != 3) throw std::invalid_argument("metadata updates must be (uuid,key,value) triples");
                batch.push_back({nb::cast<std::string>(row[0]), nb::cast<std::string>(row[1]), jsonDumps(row[2])});
            }
            self.withWrite<true>([&](auto& db, auto& txn) {
                for (const auto& update : batch) db.setMeta(txn, update.uuid, update.key, update.value);
                return 0;
            });
        }, nb::arg("updates"), "Apply metadata triples atomically in one durable transaction.")
        .def(
            "journal_since",
            [](PyObjectStore& self, int64_t afterSeq) {
                auto entries = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.journalSince(t, afterSeq); });
                nb::list out;
                for (const auto& e : entries) {
                    nb::dict d;
                    d["seq"] = e.seq;
                    d["record_uuid"] = e.recordUuid;
                    d["kind"] = e.kind;
                    d["commit_sequence"] = e.commitSequence;
                    d["author"] = e.author;
                    d["hostname"] = e.hostname;
                    d["timestamp"] = e.timestamp;
                    d["change_kind"] = e.changeKind;
                    d["approval_state"] = e.approvalState;
                    d["approved_by"] = e.approvedBy;
                    d["approved_at"] = e.approvedAt;
                    d["approved_host"] = e.approvedHost;
                    d["approved_version"] = e.approvedVersion;
                    out.append(d);
                }
                return out;
            },
            nb::arg("after_seq") = 0)
        .def("geometry_point_annotations", [](PyObjectStore& self,const std::string& revision,const std::string& array) {
            return jsonLoads(self.withRead([&](vo::ObjectStore& db,vo::ReadTxn& txn) { return db.geometryPointAnnotations(txn,revision,array).dump(); }));
        },nb::arg("revision_uuid"),nb::arg("array_name"))
        .def("current_geometry", readMethod<&vo::ObjectStore::currentGeometry>(),
             nb::arg("object_uuid"))
        .def("geometry_revision", readMethod<&vo::ObjectStore::geometryRevision>(),
             nb::arg("revision_uuid"))
        .def("geometry_history", readMethod<&vo::ObjectStore::geometryHistory>(),
             nb::arg("object_uuid"))
        .def("read_geometry_array", readMethod<&vo::ObjectStore::readGeometryArray, true>(), nb::arg("revision_uuid"), nb::arg("array_name"),
             nb::arg("byte_offset") = 0, nb::arg("byte_count") = std::nullopt)
        .def("read_geometry_arrays", [](PyObjectStore& self, const std::vector<std::string>& revisions,
                                       const std::string& name) {
            return PyPackedGeometryArray{self.withRead<true>([&](auto& db, auto& txn) {
                return std::make_shared<vo::PackedGeometryArray>(db.readGeometryArrays(txn, revisions, name));
            })};
        }, nb::arg("revision_uuids"), nb::arg("array_name"),
           "Read homogeneous arrays into one verified packed buffer with row offsets.")
        .def("read_geometry_rectangle", readMethod<&vo::ObjectStore::readGeometryRectangle, true>(), nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("row"), nb::arg("column"),
             nb::arg("rows"), nb::arg("columns"), "Read a dense row-major surface rectangle, preserving native dtype and component bytes.")
        .def("read_geometry_rectangles", readMethod<&vo::ObjectStore::readGeometryRectangles, true>(), nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("rectangles"),
             "Read [row,column,rows,columns] rectangles in order, sharing verified chunks.")
        .def("read_geometry_ranges", readMethod<&vo::ObjectStore::readGeometryRanges, true>(), nb::arg("revision_uuid"), nb::arg("array_name"),
             nb::arg("ranges"))
        .def("ingest_registered_geometry", [](PyObjectStore& self, const std::optional<std::filesystem::path>& root,
                                             const std::vector<std::string>& uuids) {
            return geometryReportToPy(self.withStore<true>([&](auto& db) {
                return vo::ingestRegisteredGeometry(db, root, uuids, {});
            }));
        }, nb::arg("dataset_root") = std::nullopt, nb::arg("object_uuids") = std::vector<std::string>{})
        .def("ingest_geometry_source", [](PyObjectStore& self, const std::string& kind, const std::filesystem::path& source,
                                         const nb::dict& context, const std::optional<std::string>& uuidNamespace) {
            const auto declarations = jsonDumps(context);
            const vo::SourceIngestOptions options{uuidNamespace};
            return geometryReportToPy(self.withStore<true>([&](auto& db) {
                return vo::ingestGeometrySource(db, kind, source, {}, declarations, {}, options);
            }));
        }, nb::arg("kind"), nb::arg("source"), nb::arg("coordinate_context") = nb::dict(),
           nb::arg("uuid_namespace") = std::nullopt)
        .def("export_geometry", [](PyObjectStore& self, const std::string& object,
                                   const std::filesystem::path& destination,
                                   const std::optional<int64_t>& version, const std::optional<std::string>& generation,
                                   bool overwrite, bool validOnly) {
            if (generation && !version) throw std::invalid_argument("export_geometry: generation needs version");
            const auto exported = self.withStore<true>([&](auto& db) {
                std::optional<vo::ObjectVersion> selected;
                if (version) {
                    const auto status = db.status(db.read());
                    selected = vo::ObjectVersion{status.storeUuid, generation.value_or(status.generation), object, *version};
                }
                return vo::exportGeometry(db, object, destination, selected, overwrite, validOnly);
            });
            return nb::cast<nb::dict>(objectVersionsToPy({exported})[0]);
        }, nb::arg("object_uuid"), nb::arg("destination"), nb::arg("version") = std::nullopt,
           nb::arg("generation") = std::nullopt, nb::arg("overwrite") = false, nb::arg("valid_only") = false,
           "Export an object version (default: current; generation defaults to the active one) to native files; "
           "return the exported {store, generation, object, version}. Does not modify the store.")
        .def("edge_geometry_state", [](PyObjectStore& self, const std::string& uuid) {
            const auto state = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.edgeGeometryState(txn, uuid); });
            nb::dict result; result["a_revision"] = state.aRevision; result["b_revision"] = state.bRevision; result["stale"] = state.stale; return result;
        }, nb::arg("edge_uuid"))
        .def("resolve_storage_path", readMethod<&vo::ObjectStore::resolveStoragePath>(), nb::arg("uuid"),
             nb::arg("dataset_root") = std::nullopt,
             "Find a surface/fiber by source UUID, verifying the saved path before searching the dataset root.")
        .def("checkpoint", [](PyObjectStore& self) { self.require().checkpoint(); })
        .def("attached_document", [](PyObjectStore& self, const std::string& owner, const std::string& role) -> nb::object {
            const auto document = self.withRead<true>([&](auto& db, auto& txn) { return vo::attachedDocument(vo::LocalStoreReader(db, txn), owner, role); });
            return document ? jsonLoads(document->dump()) : nb::none();
        }, nb::arg("owner_uuid"), nb::arg("role"), "The document attached to owner in role, or None.")
        .def("point_collections", [](PyObjectStore& self, const std::vector<std::string>& uuids, const std::string& volume,
                                     double targetScale) {
            const auto collections = self.withRead<true>([&](auto& db, auto& txn) {
                return vo::pointCollections(vo::LocalStoreReader(db, txn), uuids, vo::VolumeVoxels{volume, targetScale});
            });
            return collectionsToPy(collections);
        }, nb::arg("uuids"), nb::arg("volume"), nb::arg("target_scale") = 1.0,
           "Point collections by UUID (any role), in the volume's voxels at target_scale: uuid, name, ids, points, "
           "annotations, source.")
        .def("attachments", [](PyObjectStore& self, const std::string& owner, const std::optional<std::string>& role) {
            const auto records = self.withRead<true>([&](auto& db, auto& txn) { return db.attachments(txn, owner, role); });
            nb::list out; for (const auto& record : records) out.append(edgeToPy(record));
            return out;
        }, nb::arg("owner_uuid"), nb::arg("role") = std::nullopt,
           "Attachment relationships of owner (in role when given); the attached object is the other endpoint.")
        .def("select_edges", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::vector<std::string>& states, const std::optional<std::vector<std::string>>& endpoints,
            bool both, const std::optional<std::string>& tag, const std::optional<int64_t>& limit,
            const std::optional<std::string>& after, const std::optional<std::vector<std::string>>& fields, bool includeRetired, std::optional<bool> adjacent, bool applicableOnly) {
            vo::EdgeSelection selection; selection.kind = kind; selection.approvalStates = states;
            selection.endpointUuids = endpoints; selection.bothEndpoints = both;
            selection.tag = tag; selection.limit = limit; selection.afterUuid = after; selection.includeRetired = includeRetired;
            selection.adjacent = adjacent; selection.applicableOnly = applicableOnly;
            const auto records = self.withRead<true>([&](auto& db, auto& txn) { return db.selectEdges(txn, selection, fields); });
            const std::set<std::string> projection = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list out; for (const auto& record : records) out.append(edgeToPy(record, fields ? &projection : nullptr));
            return out;
        }, nb::arg("edge_kind") = std::nullopt, nb::arg("approval_states") = std::vector<std::string>{},
           nb::arg("endpoint_uuids") = std::nullopt, nb::arg("both_endpoints") = false,
           nb::arg("tag") = std::nullopt, nb::arg("limit") = std::nullopt,
           nb::arg("after_uuid") = std::nullopt, nb::arg("fields") = std::nullopt, nb::arg("include_retired") = false,
           nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false)
        .def("graph_read", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::vector<std::string>& ids, const std::optional<std::string>& edgeKind,
            const std::vector<std::string>& states, const std::optional<std::array<double, 6>>& bbox,
            const std::optional<std::string>& tag, bool includeRetired, std::optional<bool> adjacent,
            bool applicableOnly, const std::optional<nb::list>& versionRefs) {
            self.require();
            if (versionRefs && (kind || !ids.empty() || bbox || includeRetired))
                throw std::invalid_argument("versions specifies the complete graph object selection; cannot combine with object filters or include_retired");
            const auto versions = versionRefs ? std::optional(objectVersionsFromPy(*versionRefs)) : std::nullopt;
            auto view = std::make_unique<PyGraphReadView>(); view->store = self.db;
            vo::ObjectSelection objects; objects.kind = kind; objects.uuids = ids; objects.bbox = bbox; objects.includeRetired = includeRetired;
            vo::EdgeSelection edges; edges.kind = edgeKind; edges.approvalStates = states; edges.tag = tag;
            edges.adjacent = adjacent; edges.applicableOnly = applicableOnly;
            { nb::gil_scoped_release release;
              view->txn = std::make_unique<vo::ReadTxn>(view->store->read());
              view->graph = std::make_shared<vo::GraphReadView>(versions ? view->store->graphReadViewAt(*view->txn, *versions, edges)
                        : view->store->graphReadView(*view->txn, objects, edges)); }
            return view.release();
        }, nb::rv_policy::take_ownership, nb::arg("object_kind") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("edge_kind") = std::nullopt,
           nb::arg("approval_states") = std::vector<std::string>{"approved"},
           nb::arg("bbox") = std::nullopt, nb::arg("edge_tag") = std::nullopt, nb::arg("include_retired") = false,
           nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false, nb::arg("versions") = std::nullopt,
           "Pin a store graph read view; use as a context manager and close before checkpointing. versions: "
           "numbered references {store, generation, object, version}, the complete closed object selection.")
        .def("select_objects", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::optional<std::string>& tag, const std::vector<std::string>& ids,
            const std::optional<std::string>& connected, const std::vector<std::string>& edgeKinds,
            const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::array<double, 6>>& bbox, const std::optional<std::vector<std::string>>& fields,
            const std::optional<std::string>& family, const std::optional<double>& minLength,
            const std::string& orderBy, const std::optional<std::string>& afterUuid,
            const std::optional<double>& afterLength, bool includeRetired,
            const std::optional<std::string>& volume, bool includeUnbound, double targetScale) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.orderBy = orderBy;
            selection.afterUuid = afterUuid; selection.afterLength = afterLength; selection.includeRetired = includeRetired;
            selection.volume = volume; selection.includeUnbound = includeUnbound; selection.targetScale = targetScale;
            const auto records = self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.selectObjects(txn, selection, fields); });
            const std::set<std::string> selected = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list result;
            for (const auto& record : records) result.append(objectToPy(record, fields ? &selected : nullptr));
            return result;
        }, nb::arg("object_kind") = std::nullopt, nb::arg("tag") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("connected_to") = std::nullopt,
           nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("limit") = std::nullopt, nb::arg("offset") = 0,
           nb::arg("bbox") = std::nullopt, nb::arg("fields") = std::nullopt,
           nb::arg("family") = std::nullopt, nb::arg("min_length") = std::nullopt,
           nb::arg("order_by") = "uuid", nb::arg("after_uuid") = std::nullopt, nb::arg("after_length") = std::nullopt, nb::arg("include_retired") = false,
           nb::arg("volume") = std::nullopt, nb::arg("include_unbound") = false, nb::arg("target_scale") = 1.0)
        .def("geometry_region", &geometryRegionFromPy, nb::arg("slice"),
             nb::arg("options") = vo::RegionOptions{}, nb::arg("cancel") = nb::none())
        .def("nearest_surfaces", [](PyObjectStore& self, const std::array<double, 3>& point, double tolerance,
            const std::optional<std::vector<std::string>>& uuids, const std::optional<std::string>& coordinates,
            const std::optional<std::string>& volume, double targetScale) {
            vo::SurfaceProximityQuery query{point, tolerance, uuids, coordinates, volume, targetScale};
            return proximityHitsToPy(self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.nearestSurfaces(txn, query); }));
        }, nb::arg("point"), nb::arg("tolerance"), nb::arg("uuids") = std::nullopt,
           nb::arg("coordinates") = std::nullopt, nb::arg("volume") = std::nullopt, nb::arg("target_scale") = 1.0,
           "Nearest point of each surface within tolerance, ordered by distance.")
        .def("spatial_regions", [](PyObjectStore& self, const std::array<double, 6>& bbox,
            const std::optional<std::string>& kind, const std::optional<std::string>& tag,
            const std::vector<std::string>& ids, const std::optional<std::string>& connected,
            const std::vector<std::string>& edgeKinds, const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::string>& family, const std::optional<double>& minLength, bool includeRetired,
            const std::optional<std::string>& volume, double targetScale) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.includeRetired = includeRetired;
            selection.volume = volume; selection.targetScale = targetScale;
            const auto regions = self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.spatialRegions(txn, selection); });
            nb::list result;
            for (const auto& r : regions) {
                nb::dict row; row["object_uuid"] = r.objectUuid; row["kind"] = r.kind;
                row["revision_uuid"] = r.revisionUuid; row["array_name"] = r.arrayName;
                row["start"] = r.start; row["shape"] = r.shape; row["bbox"] = r.bbox;
                result.append(row);
            }
            return result;
        }, nb::arg("bbox"), nb::arg("object_kind") = std::nullopt, nb::arg("tag") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("connected_to") = std::nullopt,
           nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("limit") = std::nullopt, nb::arg("offset") = 0,
           nb::arg("family") = std::nullopt, nb::arg("min_length") = std::nullopt, nb::arg("include_retired") = false,
           nb::arg("volume") = std::nullopt, nb::arg("target_scale") = 1.0,
           "Return revision-bound surface tiles, polyline/point blocks, or external object boxes. Bounds are conservative candidates.")
        .def("export_json", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::optional<std::string>& tag, const std::vector<std::string>& ids,
            const std::optional<std::string>& connected, const std::vector<std::string>& edgeKinds,
            const std::vector<std::string>& fields, bool includeRetired) {
            vo::JsonExportOptions options{{kind, tag, connected, ids, edgeKinds}, fields}; options.selection.includeRetired = includeRetired;
            return self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.exportJson(txn, options); });
        }, nb::arg("object_kind") = std::nullopt, nb::arg("tag") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("connected_to") = std::nullopt,
           nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("fields") = std::vector<std::string>{}, nb::arg("include_retired") = false)
        .def("import_json", writeMethod<&vo::ObjectStore::importJson>(),
             nb::arg("json_text"))
        .def(
            "link_point_collections_to_surfaces",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                const auto p = vc::object_links::pointCollectionSurfaceParamsFromJson(kwargsJson(kwargs));
                return detectorReportToPy(self.withStore<true>([&](auto& db) {
                    return vc::object_links::linkPointCollectionsToSurfaces(db, p);
                }));
            },
            nb::arg("kwargs"),
            "Link regular PCL points to surfaces using Spiral's largest-area/nearest-named-pair rules. "
            "Point IDs anchor the edges; winding annotations and surface grid coordinates are retained. "
            "Defaults: tolerance=2.5, window_points=1, window_min_points=1; no CT required.")
        .def(
            "link_surface_overlaps",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                const auto p = vc::object_links::surfaceOverlapParamsFromJson(kwargsJson(kwargs));
                return detectorReportToPy(self.withStore<true>([&](auto& db) {
                    return vc::object_links::linkSurfaceOverlaps(db, p);
                }));
            },
            "Run surface-overlap discovery using stored geometry; surface_uuids optionally selects objects. "
            "kwargs mirror SurfaceOverlapParams (tolerance, min_minor_spread, ...).")
        .def(
            "link_tracks_to_tracks",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                const auto p = vc::object_links::trackTrackParamsFromJson(kwargsJson(kwargs));
                return detectorReportToPy(self.withStore<true>([&](auto& db) {
                    return vc::object_links::linkTracksToTracks(db, p);
                }));
            },
            nb::arg("kwargs"),
            "Discover crossings from stored tracks, optionally selected by track_uuids and dataset_ref. "
            "Anchors are raw point indices and source XYZ positions.")
        .def(
            "link_tracks_to_surfaces",
            [](PyObjectStore& self, const std::string& ctVolume, const nb::kwargs& kwargs) {
                const auto p = vc::object_links::trackSurfaceParamsFromJson(kwargsJson(kwargs));
                return detectorReportToPy(self.withStore<true>([&](auto& db) {
                    return vc::object_links::linkTracksToSurfaces(db, ctVolume, p);
                }));
            },
            nb::arg("ct_volume") = "", nb::arg("kwargs"),
            "Link stored tracks to stored surfaces; select with track_uuids and surface_uuids. "
            "Strip normals come from ct_volume, lasagna normal_x_volume/normal_y_volume, or a "
            "normal_pool pack; with none, the centerline alone gates anchors. "
            "Anchors retain raw indices and source XYZ.")
        .def(
            "link_fibers_to_surfaces",
            [](PyObjectStore& self, const std::string& ctVolume, const nb::kwargs& kwargs) {
                const auto p = vc::object_links::fiberSurfaceParamsFromJson(kwargsJson(kwargs));
                return detectorReportToPy(self.withStore<true>([&](auto& db) {
                    return vc::object_links::linkFibersToSurfaces(db, ctVolume, p);
                }));
            },
            nb::arg("ct_volume") = "", nb::arg("kwargs"),
            "Link stored fibers to stored surfaces; select with fiber_uuids and surface_uuids. "
            "Strip normals come from ct_volume, lasagna normal_x_volume/normal_y_volume, or a "
            "normal_pool pack; with none, the centerline alone gates anchors. "
            "Optional cache_directory stores disposable strips.")
        .def_prop_ro("uuid", readMethod<&vo::ObjectStore::uuid>())
        .def("current_object_revision", readMethod<&vo::ObjectStore::currentObjectRevision>(),
             nb::arg("object_uuid"))
        .def("object_history", readMethod<&vo::ObjectStore::objectHistory>(),
             nb::arg("object_uuid"))
        .def("status", [](PyObjectStore& self) {
            return jsonLoads(vo::storeStatusToJson(self.withRead([](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.status(txn); })));
        }, "Store UUID, lineage generation, commit sequence, feed start and history_enabled policy.")
        .def("current_version", readMethod<&vo::ObjectStore::currentVersion>(), nb::arg("object_uuid"))
        .def("current_relationship_version", readMethod<&vo::ObjectStore::currentRelationshipVersion>(),
             nb::arg("relationship_uuid"))
        .def("object_at_version", [](PyObjectStore& self, const std::string& store, const std::string& generation,
                                     const std::string& object, int64_t version) {
            return toPy(self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return db.objectAtVersion(txn, {store, generation, object, version});
            }));
        }, nb::arg("store_uuid"), nb::arg("generation"), nb::arg("object_uuid"), nb::arg("version"),
           "Exact numbered reference; None when not published in that generation.")
        .def("changes_since", [](PyObjectStore& self, const nb::dict& cursor, int64_t maxChanges) {
            const vo::FeedCursor c{nb::cast<std::string>(cursor["store"]), nb::cast<std::string>(cursor["generation"]),
                                   nb::cast<int64_t>(cursor["sequence"]),
                                   cursor.contains("offset") ? nb::cast<int64_t>(cursor["offset"]) : 0};
            return jsonLoads(vo::changePageToJson(self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return db.changesSince(txn, c, maxChanges);
            })));
        }, nb::arg("cursor"), nb::arg("max_changes") = 4096)
        .def("receipt", [](PyObjectStore& self, const std::string& request) -> nb::object {
            const auto receipt = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.receipt(txn, request); });
            return receipt ? jsonLoads(vo::transactionReceiptToJson(*receipt)) : nb::none();
        }, nb::arg("request_uuid"))
        .def("submit", [](PyObjectStore& self, const nb::object& request, const nb::dict& payloads, const nb::object& progress) {
            // The canonical JSON form (transaction_request_json); binary
            // payloads keyed by SHA-256, as on the wire.
            vo::TransactionPayloads bytes;
            for (const auto& [hash, value] : payloads) {
                const auto data = nb::cast<nb::bytes>(value);
                const auto* begin = reinterpret_cast<const uint8_t*>(data.c_str());
                bytes.emplace(nb::cast<std::string>(hash), std::vector<uint8_t>(begin, begin + data.size()));
            }
            const auto text = nb::isinstance<nb::str>(request) ? nb::cast<std::string>(request) : jsonDumps(request);
            const auto parsed = vo::transactionRequestFromJson(text, bytes);
            vo::PrepareOptions options;
            if (!progress.is_none()) {
                options.progress = [progress](const vo::util::ProgressUpdate& update) {
                    nb::gil_scoped_acquire acquire;
                    progress(nb::str(update.stage.data(), update.stage.size()), update.completed, update.total,
                             nb::str(update.detail.data(), update.detail.size()));
                };
            }
            vo::TransactionReceipt receipt;
            {
                nb::gil_scoped_release release;
                auto& db = self.require();
                const auto prepared = db.prepare(parsed, options);
                receipt = db.submit(parsed, prepared);
            }
            return jsonLoads(vo::transactionReceiptToJson(receipt));
        }, nb::arg("request"), nb::arg("payloads") = nb::dict(), nb::arg("progress") = nb::none(),
           "Atomic checked request; identical retries return the saved receipt. Conflicts raise RuntimeError. "
           "A link_objects request runs its detectors here first (progress(stage, completed, total, detail) "
           "follows them) and then commits their links with the link marker in one checked write.")
        .def("start_history_generation", [](PyObjectStore& self, const std::string& reason) {
            nb::gil_scoped_release release;
            return self.require().startHistoryGeneration(reason);
        }, nb::arg("reason"), "Restore tooling: a fresh generation for new writes.")
        .def("objects_at_versions", [](PyObjectStore& self, const nb::list& refs) {
            const auto versions = objectVersionsFromPy(refs);
            return toPy(self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.objectsAtVersions(txn, versions); }));
        }, nb::arg("refs"), "Numbered references {store, generation, object, version}, answered in order (None where not recorded).")
        .def("relationships_at_versions", [](PyObjectStore& self, const nb::list& refs) {
            const auto versions = objectVersionsFromPy(refs);
            const auto revisions = self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.relationshipsAt(txn, versions); });
            nb::list result;
            for (const auto& revision : revisions) {
                auto d = relationshipRevisionToPy(revision.relationship);
                d["applicable"] = revision.applicable;
                result.append(d);
            }
            return result;
        }, nb::arg("refs"), "Relationships recorded between these numbered object versions.")
        .def("record_selection", [](PyObjectStore& self, const nb::object& query) {
            const auto text = nb::isinstance<nb::str>(query) ? nb::cast<std::string>(query) : jsonDumps(query);
            std::optional<std::vector<std::string>> fields;
            const auto selection = vo::objectQueryRequestFromJson(text, &fields);
            return selectionRecordToPy(self.withRead<true>([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return vo::recordSelection(db, txn, selection);
            }));
        }, nb::arg("query"), "Every match of an object query (wire form) at its version, exact as of one commit.")
        .def("put_artifact", [](PyObjectStore& self, const std::filesystem::path& file, bool move) {
            return self.withStore<true>([&](vo::ObjectStore& db) { return db.putArtifact(file, move); });
        }, nb::arg("file"), nb::arg("move") = false,
           "Store a file's bytes (durably) and return their SHA-256; name it in a manifest afterwards.")
        .def("artifact_length", [](PyObjectStore& self, const std::string& sha256) {
            const auto length = self.withStore([&](vo::ObjectStore& db) { return db.artifactLength(sha256); });
            return length ? nb::cast(*length) : nb::none();
        }, nb::arg("sha256"))
        .def("read_artifact", [](PyObjectStore& self, const std::string& sha256, uint64_t offset,
                                 const std::optional<uint64_t>& count) {
            return bytesToPy(self.withStore<true>([&](vo::ObjectStore& db) { return db.readArtifact(sha256, offset, count); }));
        }, nb::arg("sha256"), nb::arg("offset") = 0, nb::arg("count") = std::nullopt)
        .def("versions_at", [](PyObjectStore& self, int64_t time, bool includeRetired) {
            return objectVersionsToPy(self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return db.versionsAt(txn, time, includeRetired);
            }));
        }, nb::arg("published_at_us"), nb::arg("include_retired") = false,
           "The version {store, generation, object, version} of every object live at a publication time (UTC Unix "
           "microseconds), by object UUID; read them with objects_at_versions and relationships_at_versions.")
        .def(
            "write",
            [](PyObjectStore& self, const std::optional<vo::VersionSelection>& objects,
               const std::optional<vo::VersionSelection>& relationships, const std::optional<vo::VersionSelection>& reads,
               const std::optional<int64_t>& commitSequence, const std::optional<std::string>& requestUuid,
               const std::optional<std::string>& author) {
                PyObjectWriteTxn w;
                w.owner = &self;
                if (objects || relationships || reads || commitSequence || requestUuid || author) {
                    vo::TransactionContext context;
                    if (objects) context.expectedObjects = *objects;
                    if (relationships) context.expectedRelationships = *relationships;
                    if (reads) context.readDependencies = *reads;
                    context.expectedCommitSequence = commitSequence;
                    context.requestUuid = requestUuid;
                    context.author = author;
                    w.context = std::move(context);
                }
                return w;
            },
            nb::arg("expected_versions") = std::nullopt, nb::arg("expected_relationships") = std::nullopt,
            nb::arg("read_dependencies") = std::nullopt, nb::arg("expected_commit_sequence") = std::nullopt,
            nb::arg("request_uuid") = std::nullopt, nb::arg("author") = std::nullopt, nb::keep_alive<0, 1>(),
            "Batch write transaction context manager: 'with db.write() as w: w.upsert_object(...)'. Checked when "
            "given expectations: {uuid: version, or None for absent} of every object (and relationship) the write "
            "changes, versions of records it only read, and/or the store commit sequence; a conflict raises "
            "RuntimeError and commits nothing. After the block, w.receipt is the commit's numbered receipt.");

    store.def("available_transforms",[](PyObjectStore& self,const std::string& uuid){
        return volumeTransformsToPy(self.withRead([&](auto& db,auto& txn){return db.availableTransforms(txn,uuid);}));
    },nb::arg("object_uuid"));
    store.def("geometry_length_in_volume",readMethod<&vo::ObjectStore::geometryLengthInVolume>(),
        nb::arg("object_uuid"),nb::arg("target_volume"),nb::arg("target_scale")=1.0);
    // A store as a project (objects/Project.hpp).
    store.def("project_record",[](PyObjectStore& self)->nb::object{
        const auto record=self.withRead([&](auto& db,auto& txn){return vo::projectRecord(vo::LocalStoreReader(db,txn));});
        return record?nb::object(objectToPy(*record)):nb::none();
    },"The store's project record, or None.");
    store.def("project_resources",[](PyObjectStore& self,const std::string& kind,const std::optional<std::string>& viewer){
        nb::list resources;
        for(const auto& resource:self.withRead([&](auto& db,auto& txn){return vo::projectResources(vo::LocalStoreReader(db,txn),kind,viewer);})){
            nb::dict d;
            d["uuid"]=resource.uuid;d["kind"]=resource.kind;d["location"]=resource.location;d["owner"]=resource.owner;
            d["tags"]=resource.tags;d["meta"]=jsonLoads(resource.meta.dump());
            resources.append(d);
        }
        return resources;
    },nb::arg("kind"),nb::arg("viewer")=std::nullopt,
      "The live resources of kind (volume, normal_grid, lasagna_dataset) viewer sees (all when None), in the order added.");
    store.def("add_project_resource",[](PyObjectStore& self,const std::string& kind,const std::string& location,
                                        const std::vector<std::string>& tags,const nb::object& meta,const std::string& owner){
        const auto record=vo::projectResourceRecord(kind,location,tags,
            meta.is_none()?nlohmann::json::object():nlohmann::json::parse(jsonDumps(meta)),owner);
        return self.withWrite([&](auto& db,auto& txn){return db.upsertObject(txn,record);});
    },nb::arg("kind"),nb::arg("location"),nb::arg("tags")=std::vector<std::string>{},nb::arg("meta")=nb::none(),
      nb::arg("owner")="","Add (or update) a project resource, shared or owner's alone; returns its uuid.");
    store.def("register_volume_transform",[](PyObjectStore& self,const std::string& source,const std::string& target,const nb::object& rows,const std::string& name){
        const auto matrix=vo::VolumeAffine::fromJson(nlohmann::json::parse(jsonDumps(rows)));
        return self.withWrite([&](auto& db,auto& txn){return db.registerVolumeTransform(txn,source,target,matrix,name);});
    },nb::arg("source_volume"),nb::arg("target_volume"),nb::arg("matrix"),nb::arg("name")="volume registration");
    store.def("choose_volume_transform",[](PyObjectStore& self,const std::string& source,const std::string& target,const std::vector<std::string>& path){
        return self.withWrite([&](auto& db,auto& txn){return db.chooseVolumeTransform(txn,source,target,path);});
    },nb::arg("source_volume"),nb::arg("target_volume"),nb::arg("path"));
    auto writer = nb::class_<PyObjectWriteTxn>(m, "WriteTransaction")
        .def("seal", [](PyObjectWriteTxn& self) { self.withWrite([](auto& db, auto& txn) { db.seal(txn); }); })
        .def("__enter__", [](nb::object self) {
            nb::cast<PyObjectWriteTxn&>(self).enter();
            return self;
        })
        .def("__exit__", [](PyObjectWriteTxn& self, nb::args args) {
            const bool failed = nb::len(args) > 0 && !args[0].is_none();
            self.exit(failed);
        })
        .def_prop_ro("receipt", [](const PyObjectWriteTxn& self) -> nb::object {
            return self.receipt ? jsonLoads(vo::transactionReceiptToJson(*self.receipt)) : nb::none();
        }, "The committed write's numbered receipt (commit sequence, versions); None before it commits.")
        .def("retire_object", writeMethod<&vo::ObjectStore::retireObject, PyObjectWriteTxn>(), nb::arg("uuid"))
        .def("restore_object", [](PyObjectWriteTxn& self, const std::string& uuid, const std::optional<int64_t>& version,
                                  const std::optional<std::string>& generation) {
            if (generation && !version) throw std::invalid_argument("restore_object: source_generation needs source_version");
            auto& db = self.owner->require(); auto& txn = self.require();
            std::optional<vo::ObjectVersion> source;
            if (version) {
                const auto status = db.status(txn);
                source = vo::ObjectVersion{status.storeUuid, generation.value_or(status.generation), uuid, *version};
            }
            db.restoreObject(txn, uuid, source);
        }, nb::arg("uuid"), nb::arg("source_version") = std::nullopt, nb::arg("source_generation") = std::nullopt,
           "Restore from a recorded version of the object (default: its last active one).")
        .def("retain_relationship", writeMethod<&vo::ObjectStore::retainRelationship, PyObjectWriteTxn>(), nb::arg("edge_uuid"))
        .def("keep_recorded_relationship", writeMethod<&vo::ObjectStore::keepRecordedRelationship, PyObjectWriteTxn>(), nb::arg("edge_uuid"))
        .def("keep_recorded_relationships_of", writeMethod<&vo::ObjectStore::keepRecordedRelationshipsOf, PyObjectWriteTxn>(), nb::arg("object_uuid"))
        .def("update_edge", [](PyObjectWriteTxn& self, const nb::dict& record) {
            self.owner->require().updateEdge(self.require(), edgeFromPy(record));
        })
        .def("edit_geometry_sequence", [](PyObjectWriteTxn& self,const std::string& uuid,const nb::dict& input,int64_t version,const nb::list& choices) {
            auto& db=self.owner->require(); auto& txn=self.require();
            return db.editGeometrySequence(txn,uuid,sequenceEditFromPy(input),expectedGeometry(db,txn,uuid,version),edgeEditsFromPy(choices));
        },nb::arg("object_uuid"),nb::arg("edit"),nb::arg("expected_version"),nb::arg("edge_edits")=nb::list())
        .def("resize_surface", [](PyObjectWriteTxn& self,const std::string& uuid,const nb::dict& input,int64_t version,const nb::list& choices) {
            auto& db=self.owner->require(); auto& txn=self.require();
            return db.resizeSurface(txn,uuid,surfaceResizeFromPy(input),expectedGeometry(db,txn,uuid,version),edgeEditsFromPy(choices));
        },nb::arg("object_uuid"),nb::arg("resize"),nb::arg("expected_version"),nb::arg("edge_edits")=nb::list());
    bindMutations(store);
    bindMutations(writer);
}
