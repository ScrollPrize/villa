#include "StorageBindings.hpp"
#include "ObjectReadBindings.hpp"
#include "TypedGeometryBindings.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/SurfaceCreate.hpp"
#include "HubBindings.hpp"
#include <nanobind/stl/map.h>
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Hub.hpp"
using namespace vc::python;
namespace {
nb::dict objectRevisionToPy(const vo::ObjectRevision& revision)
{
    nb::dict d;
    d["uuid"] = revision.uuid; d["object_uuid"] = revision.objectUuid;
    d["parent_uuid"] = revision.parentUuid; d["geometry_revision"] = revision.geometryRevision;
    d["sequence"] = revision.sequence; d["metadata_complete"] = revision.metadataComplete;
    d["published_at_us"] = revision.publishedAtUs; d["deleted"] = revision.deleted;
    d["record"] = objectToPy(revision.record);
    return d;
}

// Convert native results after the transaction (and any GIL release) ends.
nb::object toPy(const vo::ObjectRecord& value) { return objectToPy(value); }
nb::object toPy(const vo::GeometryRevision& value) { return geometryToPy(value); }
nb::object toPy(const vo::ObjectRevision& value) { return objectRevisionToPy(value); }
template<class T> T toPy(T value) { return value; }
std::vector<std::string> toPy(std::vector<std::string> value) { return value; }
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
            "attach_document",
            [](T& self, const std::string& owner, const std::string& collection, const nb::object& document,
               const std::string& role, const std::string& listKey, const std::string& positionKey,
               const std::string& name, const nb::object& coordinates) {
                const auto parsed = nlohmann::json::parse(jsonDumps(document));
                const auto frame = nlohmann::json::parse(jsonDumps(coordinates));
                self.template withWrite<true>([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    vo::attachDocument(g, t, owner, collection, parsed, {role, listKey, positionKey}, name, frame);
                });
            },
            nb::arg("owner_uuid"), nb::arg("collection_uuid"), nb::arg("document"), nb::arg("role"),
            nb::arg("list_key"), nb::arg("position_key"), nb::arg("name"), nb::arg("coordinates"),
            "Attach a document to owner as point collection collection_uuid (objects/AttachedDocument.hpp): one point "
            "per record of document[list_key], at record[position_key].")
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
        .def(
            "set_meta",
            [](T& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                const std::string json = jsonDumps(value);
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.setMeta(t, uuid, key, json);
                });
            },
            nb::arg("uuid"), nb::arg("key"), nb::arg("value").none(), "Set one open-ended metadata key (any JSON value).")
        .def("put_geometry", [](T& self, const std::string& uuid, const nb::object& input,
                               const std::optional<std::string>& expected, const nb::list& choices, int64_t surfaceTileSize) {
            const auto data = geometryInputFromPy(input);
            const auto edits = edgeEditsFromPy(choices);
            return self.template withWrite<true>([&](vo::ObjectStore& db, vo::WriteTxn& txn) { return db.putGeometry(txn, uuid, data, expected, edits, vo::GeometryStorageOptions{surfaceTileSize}); });
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt,
           nb::arg("edge_edits") = nb::list(), nb::arg("surface_tile_size") = vo::GeometryStorageOptions{}.surfaceTileSize);
}

}  // namespace

NB_MODULE(_native, m)
{
    m.doc() = "Revisioned objects and relationships, with owned geometry and typed snapshot views.";

    bindGraphReadView(m);
    bindObjectStoreSession(m);
    bindTypedGeometry(m); // Shared C++ geometry and revision-pinned views.

    m.def("geometry_scale_to_volume", [](const nb::handle& meta, const std::string& volume) {
        return vo::geometryScaleToVolume(nlohmann::json::parse(jsonDumps(meta)), volume);
    }, nb::arg("meta"), nb::arg("volume"),
       "Volume voxels per stored unit for an object whose meta binds it to volume; None when it is not.");
    m.def("hub_init", &vo::initHub, nb::arg("path"), nb::arg("author"),
          "Turn a local store with no other connections into a hub for a network share.");
    m.def("hub_clone", [](const std::filesystem::path& hub, const std::filesystem::path& replica, const std::string& author) {
        vo::OpenOptions options; options.author = author;
        nb::gil_scoped_release release;
        vo::cloneHub(hub, replica, options);
    }, nb::arg("hub"), nb::arg("replica"), nb::arg("author"), "Copy a hub into a new local replica.");

    auto store = nb::class_<PyObjectStore>(m, "ObjectStore")
        .def_static("create", &PyObjectStore::create, nb::arg("path"), nb::arg("author"),
                    nb::arg("hostname") = std::nullopt, nb::arg("reader_pool_size") = 4,
                    nb::arg("geometry_cache_bytes") = 16 * 1024 * 1024,
                    nb::arg("spatial_cache_bytes") = 16 * 1024 * 1024, nb::arg("catalog") = nb::none(),
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
        .def("object", readMethod<&vo::ObjectStore::objectByUuid>(),
             nb::arg("uuid"))
        .def("objects_by_kind", readMethod<&vo::ObjectStore::objectsByKind>(),
             nb::arg("kind"), nb::arg("approval_state") = std::nullopt, nb::arg("include_retired") = false)
        .def("unbound_object_count", readMethod<&vo::ObjectStore::unboundObjectCount>(), nb::arg("kind"), nb::arg("volume"),
             "Live objects of kind not bound to volume (no coordinates in its voxels).")
        .def("neighbours", readMethod<&vo::ObjectStore::neighbours>(),
             nb::arg("uuid"), nb::arg("edge_kind") = std::nullopt, nb::arg("neighbour_kind") = std::nullopt, nb::arg("include_retired") = false,
             nb::arg("adjacent") = std::nullopt, nb::arg("applicable_only") = false)
        .def("current_geometry", readMethod<&vo::ObjectStore::currentGeometry>(),
             nb::arg("object_uuid"))
        .def("geometry_revision", readMethod<&vo::ObjectStore::geometryRevision>(),
             nb::arg("revision_uuid"))
        .def("read_geometry_arrays", [](PyObjectStore& self, const std::vector<std::string>& revisions,
                                       const std::string& name) {
            return PyPackedGeometryArray{self.withRead<true>([&](auto& db, auto& txn) {
                return std::make_shared<vo::PackedGeometryArray>(db.readGeometryArrays(txn, revisions, name));
            })};
        }, nb::arg("revision_uuids"), nb::arg("array_name"),
           "Read homogeneous arrays into one verified packed buffer with row offsets.")
        .def("ingest_geometry_source", [](PyObjectStore& self, const std::string& kind, const std::filesystem::path& source,
                                         const nb::dict& context) {
            const auto declarations = jsonDumps(context);
            return geometryReportToPy(self.withStore<true>([&](auto& db) {
                return vo::ingestGeometrySource(db, kind, source, {}, declarations);
            }));
        }, nb::arg("kind"), nb::arg("source"), nb::arg("coordinate_context") = nb::dict())
        .def("checkpoint", [](PyObjectStore& self) { self.require().checkpoint(); })
        .def("attached_document", [](PyObjectStore& self, const std::string& owner, const std::string& role) -> nb::object {
            const auto document = self.withRead<true>([&](auto& db, auto& txn) { return vo::attachedDocument(db, txn, owner, role); });
            return document ? jsonLoads(document->dump()) : nb::none();
        }, nb::arg("owner_uuid"), nb::arg("role"), "The document attached to owner in role, or None.")
        .def("point_collections", [](PyObjectStore& self, const std::vector<std::string>& uuids, const std::string& volume) {
            const auto collections = self.withRead<true>([&](auto& db, auto& txn) { return vo::pointCollections(db, txn, uuids, volume); });
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
        }, nb::arg("uuids"), nb::arg("volume"),
           "Point collections by UUID (any role), in the volume's voxels: uuid, name, ids, points, annotations, source.")
        .def("graph_read", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::vector<std::string>& ids) {
            self.require();
            auto view = std::make_unique<PyGraphReadView>(); view->store = self.db;
            vo::ObjectSelection objects; objects.kind = kind; objects.uuids = ids;
            vo::EdgeSelection edges; edges.approvalStates = {"approved"};
            { nb::gil_scoped_release release;
              view->txn = std::make_unique<vo::ReadTxn>(view->store->read());
              view->graph = std::make_shared<vo::GraphReadView>(view->store->graphReadView(*view->txn, objects, edges)); }
            return view.release();
        }, nb::rv_policy::take_ownership, nb::arg("object_kind") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{},
           "Pin the selected objects in one read snapshot; use as a context manager and close before checkpointing.")
        .def("select_objects", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::optional<std::string>& tag, const std::vector<std::string>& ids,
            const std::optional<std::string>& connected, const std::vector<std::string>& edgeKinds,
            const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::array<double, 6>>& bbox, const std::optional<std::vector<std::string>>& fields,
            const std::optional<std::string>& family, const std::optional<double>& minLength,
            const std::string& orderBy, const std::optional<std::string>& afterUuid,
            const std::optional<double>& afterLength, bool includeRetired,
            const std::optional<std::string>& volume, bool includeUnbound) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.orderBy = orderBy;
            selection.afterUuid = afterUuid; selection.afterLength = afterLength; selection.includeRetired = includeRetired;
            selection.volume = volume; selection.includeUnbound = includeUnbound;
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
           nb::arg("volume") = std::nullopt, nb::arg("include_unbound") = false)
        .def("spatial_regions", [](PyObjectStore& self, const std::array<double, 6>& bbox,
            const std::optional<std::string>& kind, const std::optional<std::string>& tag,
            const std::vector<std::string>& ids, const std::optional<std::string>& connected,
            const std::vector<std::string>& edgeKinds, const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::string>& family, const std::optional<double>& minLength, bool includeRetired,
            const std::optional<std::string>& volume) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.includeRetired = includeRetired;
            selection.volume = volume;
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
           nb::arg("volume") = std::nullopt,
           "Return revision-bound surface tiles, polyline/point blocks, or external object boxes. Bounds are conservative candidates.")
        .def("hub_sync", [](PyObjectStore& self, bool pull, bool push) {
            auto& db = self.require();
            if (self.db.use_count() != 1) throw std::runtime_error("hub_sync requires an idle store");
            vo::HubSyncReport report;
            { nb::gil_scoped_release release;
              report = pull && push ? vo::hubSync(db) : pull ? vo::hubPull(db) : vo::hubPush(db); }
            return hubReport(report);
        }, nb::arg("pull") = true, nb::arg("push") = true,
           "Pull hub changes and/or push local ones. Pushing changed hub objects requires their leases.")
        .def("acquire_leases", [](PyObjectStore& self, const std::vector<std::string>& uuids, int64_t seconds) {
            auto& db = self.require();
            std::vector<vo::HubLease> leases;
            { nb::gil_scoped_release release; leases = vo::acquireLeases(db, uuids, seconds); }
            return hubLeases(leases);
        }, nb::arg("uuids"), nb::arg("seconds") = vo::kDefaultLeaseSeconds)
        .def_prop_ro("uuid", readMethod<&vo::ObjectStore::uuid>())
        .def("current_object_revision", readMethod<&vo::ObjectStore::currentObjectRevision>(),
             nb::arg("object_uuid"))
        .def(
            "write",
            [](PyObjectStore& self, const std::optional<vo::RevisionSelection>& expected) {
                PyObjectWriteTxn w;
                w.owner = &self;
                w.expected = expected;
                return w;
            },
            nb::arg("expected_revisions") = std::nullopt, nb::keep_alive<0, 1>(),
            "Batch write transaction context manager: 'with db.write() as w: w.upsert_object(...)'");

    store.def("register_volume_transform",[](PyObjectStore& self,const std::string& source,const std::string& target,const nb::object& rows,const std::string& name){
        const auto matrix=vo::VolumeAffine::fromJson(nlohmann::json::parse(jsonDumps(rows)));
        return self.withWrite([&](auto& db,auto& txn){return db.registerVolumeTransform(txn,source,target,matrix,name);});
    },nb::arg("source_volume"),nb::arg("target_volume"),nb::arg("matrix"),nb::arg("name")="volume registration");
    auto writer = nb::class_<PyObjectWriteTxn>(m, "WriteTransaction")
        .def("__enter__", [](nb::object self) {
            nb::cast<PyObjectWriteTxn&>(self).enter();
            return self;
        })
        .def("__exit__", [](PyObjectWriteTxn& self, nb::args args) {
            const bool failed = nb::len(args) > 0 && !args[0].is_none();
            self.exit(failed);
        })
        .def("retire_object", writeMethod<&vo::ObjectStore::retireObject, PyObjectWriteTxn>(), nb::arg("uuid"));
    bindMutations(store);
    bindMutations(writer);
}
