#include "StorageBindings.hpp"
#include "ObjectReadBindings.hpp"
#include "TypedGeometryBindings.hpp"
#include <nanobind/stl/map.h>
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/CtSheetNormal.hpp"
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

nb::dict relationshipRevisionToPy(const vo::RelationshipRevision& revision)
{
    nb::dict d;
    d["uuid"] = revision.uuid; d["record"] = edgeToPy(revision.record);
    d["a_revision"] = revision.aRevision; d["b_revision"] = revision.bRevision;
    d["a_geometry_revision"] = revision.aGeometryRevision; d["b_geometry_revision"] = revision.bGeometryRevision;
    return d;
}

nb::dict revisionViewToPy(vo::ObjectStore& db, const vo::Txn& txn, const vo::RevisionSelection& selection)
{
    const auto relationships = db.relationshipsAt(txn, selection);
    nb::dict result, resolved;
    nb::list objects, edges;
    for (const auto& [uuid, selected] : selection) {
        const auto revision = selected ? db.objectRevision(txn, *selected) : db.currentObjectRevision(txn, uuid);
        objects.append(objectRevisionToPy(*revision)); // selection was validated above
        resolved[pyStr(revision->objectUuid)] = revision->uuid;
    }
    for (const auto& relationship : relationships) {
        auto edge = relationshipRevisionToPy(relationship.relationship);
        edge["applicable"] = relationship.applicable; edges.append(edge);
    }
    result["objects"] = objects; result["relationships"] = edges; result["selection"] = resolved;
    return result;
}


vo::EdgeGeometryRepair repairFromPy(const nb::dict& d)
{
    vo::EdgeGeometryRepair repair;
    repair.edgeUuid = nb::cast<std::string>(d["edge_uuid"]);
    repair.expectedGeneration = nb::cast<int64_t>(d["expected_generation"]);
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
    nb::list problems;
    for (const auto& p : rep.problems) problems.append(pyStr(p));
    d["problems"] = problems;
    return d;
}

bool setLinePatchParameter(vc::object_links::LinePatchParams& p, const std::string& key, nb::handle val)
{
    if (key == "patch_uuids") p.patchUuids = nb::cast<std::vector<std::string>>(val);
    else if (key == "patch_scale") p.patchScale = nb::cast<double>(val);
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

NB_MODULE(_native, m)
{
    m.def("read_geometry_source_coordinates",
        [](const std::string& kind, const std::filesystem::path& source,
           const std::string& uuid, const nb::dict& context) {
            const auto declarations = jsonDumps(context);
            std::string result;
            { nb::gil_scoped_release release;
              result = vo::readGeometrySourceCoordinates(kind, source, uuid, declarations); }
            return jsonLoads(result);
        }, nb::arg("kind"), nb::arg("source"), nb::arg("object_uuid"),
           nb::arg("coordinate_context") = nb::dict());
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

    bindGraphReadView(m);
    bindTypedGeometry(m); // Shared C++ geometry and revision-pinned views.

    nb::class_<PyObjectStore>(m, "ObjectStore")
        .def_static("create", &PyObjectStore::create, nb::arg("path"), nb::arg("author"),
                    nb::arg("hostname") = std::nullopt, nb::arg("reader_pool_size") = 4,
                    nb::arg("geometry_cache_bytes") = 16 * 1024 * 1024,
                    nb::arg("spatial_cache_bytes") = 16 * 1024 * 1024,
                    "Create a new object database at path (refuses existing files).")
        .def_static("open", &PyObjectStore::open, nb::arg("path"), nb::arg("author") = std::nullopt,
                    nb::arg("read_only") = false, nb::arg("hostname") = std::nullopt,
                    nb::arg("reader_pool_size") = 4,
                    nb::arg("geometry_cache_bytes") = 16 * 1024 * 1024,
                    nb::arg("spatial_cache_bytes") = 16 * 1024 * 1024,
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
        .def(
            "upsert_object",
            [](PyObjectStore& self, const nb::dict& rec) {
                return self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) { return self.upsertObjectIn(g, t, rec); });
            },
            nb::arg("record"), "Create or update an object from a dict; returns its uuid.")
        .def(
            "object",
            [](PyObjectStore& self, const std::string& uuid) -> nb::object {
                auto rec = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.objectByUuid(t, uuid); });
                return rec ? nb::object(objectToPy(*rec)) : nb::none();
            },
            nb::arg("uuid"))
        .def(
            "objects_by_kind",
            [](PyObjectStore& self, const std::string& kind, const std::optional<std::string>& approvalState, bool includeRetired) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.objectsByKind(t, kind, approvalState, includeRetired); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("kind"), nb::arg("approval_state") = std::nullopt, nb::arg("include_retired") = false)
        .def("object_count", [](PyObjectStore& self, const std::string& kind, bool includeRetired) {
            return self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.objectCount(txn, kind, includeRetired); });
        }, nb::arg("kind"), nb::arg("include_retired") = false)
        .def(
            "objects_by_tag",
            [](PyObjectStore& self, const std::string& tag, bool includeRetired) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.objectsByTag(t, tag, includeRetired); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("tag"), nb::arg("include_retired") = false)
        .def(
            "delete_object",
            [](PyObjectStore& self, const std::string& uuid) {
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.deleteObject(t, uuid);
                    return 0;
                });
            },
            nb::arg("uuid"))
        .def(
            "add_edge",
            [](PyObjectStore& self, const nb::dict& rec) {
                return self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) { return self.addEdgeIn(g, t, rec); });
            },
            nb::arg("record"), "Create an edge from a dict (keys: a, b, kind, anchors, tags, meta); returns uuid.")
        .def(
            "edge",
            [](PyObjectStore& self, const std::string& uuid) -> nb::object {
                auto rec = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.edgeByUuid(t, uuid); });
                return rec ? nb::object(edgeToPy(*rec)) : nb::none();
            },
            nb::arg("uuid"))
        .def(
            "remove_edge",
            [](PyObjectStore& self, const std::string& uuid) {
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.removeEdge(t, uuid);
                    return 0;
                });
            },
            nb::arg("uuid"))
        .def(
            "edges_of",
            [](PyObjectStore& self, const std::string& uuid, bool includeRetired) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.edgesOf(t, uuid, includeRetired); });
                nb::list out;
                for (const auto& e : recs) out.append(edgeToPy(e));
                return out;
            },
            nb::arg("uuid"), nb::arg("include_retired") = false)
        .def(
            "edges_by_kind",
            [](PyObjectStore& self, const std::string& kind, bool includeRetired) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.edgesByKind(t, kind, includeRetired); });
                nb::list out;
                for (const auto& e : recs) out.append(edgeToPy(e));
                return out;
            },
            nb::arg("edge_kind"), nb::arg("include_retired") = false)
        .def("neighbours",
             [](PyObjectStore& self, const std::string& uuid, const std::optional<std::string>& edgeKind,
                const std::optional<std::string>& neighbourKind, bool includeRetired) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.neighbours(t, uuid, edgeKind, neighbourKind, includeRetired); });
             },
             nb::arg("uuid"), nb::arg("edge_kind") = std::nullopt, nb::arg("neighbour_kind") = std::nullopt, nb::arg("include_retired") = false)
        .def("component",
             [](PyObjectStore& self, const std::string& uuid, const std::vector<std::string>& edgeKinds, bool includeRetired) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.componentOf(t, uuid, edgeKinds, nullptr, includeRetired); });
             },
             nb::arg("uuid"), nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("include_retired") = false)
        .def("reachable_within",
             [](PyObjectStore& self, const std::string& uuid, int maxDepth, const std::vector<std::string>& edgeKinds, bool includeRetired) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.reachableWithin(t, uuid, maxDepth, edgeKinds, nullptr, includeRetired); });
             },
             nb::arg("uuid"), nb::arg("max_depth"), nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("include_retired") = false)
        .def(
            "set_approval",
            [](PyObjectStore& self, const std::string& uuid, const std::string& state) {
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.setApproval(t, uuid, state);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("state"), "state: 'pending', 'approved' or 'rejected' (objects and edges).")
        .def(
            "set_meta",
            [](PyObjectStore& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                const std::string json = jsonDumps(value);
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.setMeta(t, uuid, key, json);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"), nb::arg("value").none(), "Set one open-ended metadata key (any JSON value).")
        .def("set_meta_many", [](PyObjectStore& self, const nb::list& updates) {
            struct Update { std::string uuid, key, value; };
            std::vector<Update> batch;
            for (nb::handle item : updates) {
                auto row = nb::cast<nb::sequence>(item);
                if (nb::len(row) != 3) throw std::invalid_argument("metadata updates must be (uuid,key,value) triples");
                batch.push_back({nb::cast<std::string>(row[0]), nb::cast<std::string>(row[1]), jsonDumps(row[2])});
            }
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            self.withWrite([&](auto& db, auto& txn) {
                for (const auto& update : batch) db.setMeta(txn, update.uuid, update.key, update.value);
                return 0;
            });
        }, nb::arg("updates"), "Apply metadata triples atomically in one durable transaction.")
        .def(
            "remove_meta",
            [](PyObjectStore& self, const std::string& uuid, const std::string& key) {
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.removeMeta(t, uuid, key);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"))
        .def(
            "journal_since",
            [](PyObjectStore& self, int64_t afterSeq) {
                auto entries = self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) { return g.journalSince(t, afterSeq); });
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
        .def("put_geometry", [](PyObjectStore& self, const std::string& uuid, const nb::object& input,
                               const std::optional<std::string>& expected, const nb::list& choices, int64_t patchTileSize) {
            const auto data = geometryInputFromPy(input);
            const auto edits = edgeEditsFromPy(choices);
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return self.withWrite([&](vo::ObjectStore& db, vo::WriteTxn& txn) { return db.putGeometry(txn, uuid, data, expected, edits, vo::GeometryStorageOptions{patchTileSize}); });
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt,
           nb::arg("edge_edits") = nb::list(), nb::arg("patch_tile_size") = vo::GeometryStorageOptions{}.patchTileSize)
        .def("edit_geometry", [](PyObjectStore& self, const std::string& uuid, const nb::list& input,
                                const std::string& expected, const nb::list& choices) {
            const auto edits = geometryEditsFromPy(input);
            const auto edges = edgeEditsFromPy(choices);
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return self.withWrite([&](vo::ObjectStore& db, vo::WriteTxn& txn) {
                return db.editGeometry(txn, uuid, edits, expected, edges);
            });
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("update_surface_channels", [](PyObjectStore& self, const std::string& uuid, const nb::list& channels,
                                           const std::vector<std::string>& remove, const std::string& expected, const nb::list& choices) {
            nb::dict input; input["kind"] = "patch"; input["arrays"] = channels;
            const auto data = geometryFromPy(input); const auto edges = edgeEditsFromPy(choices);
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return self.withWrite([&](auto& db, auto& txn) { return db.updateSurfaceChannels(txn, uuid, data.arrays, remove, expected, edges); });
        }, nb::arg("object_uuid"), nb::arg("channels"), nb::arg("remove"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("edit_geometry_rectangle", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                          int64_t row, int64_t column, int64_t rows, int64_t columns,
                                          const nb::bytes& data, const std::string& expected, const nb::list& choices) {
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::vector<uint8_t> bytes(p, p + data.size()); const auto edges = edgeEditsFromPy(choices);
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return self.withWrite([&](auto& db, auto& txn) {
                return db.editGeometryRectangle(txn, uuid, name, row, column, rows, columns, bytes, expected, edges);
            });
        }, nb::arg("object_uuid"), nb::arg("array_name"), nb::arg("row"), nb::arg("column"), nb::arg("rows"), nb::arg("columns"),
           nb::arg("data"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("affected_geometry_edit_edges", [](PyObjectStore& self, const std::string& uuid,
                                                const nb::list& input, const std::string& expected) {
            const auto edits = geometryEditsFromPy(input);
            const auto edges = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return db.affectedGeometryEditEdges(txn, uuid, edits, expected);
            });
            nb::list result; for (const auto& edge : edges) result.append(edgeToPy(edge)); return result;
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_revision"))
        .def("affected_geometry_edges", [](PyObjectStore& self, const std::string& uuid, const nb::object& input) {
            const auto data = geometryInputFromPy(input);
            auto edges = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.affectedGeometryEdges(txn, uuid, data); });
            nb::list result; for (const auto& edge : edges) result.append(edgeToPy(edge)); return result;
        }, nb::arg("object_uuid"), nb::arg("geometry"))
        .def("current_geometry", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.currentGeometry(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("object_uuid"))
        .def("geometry_revision", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.geometryRevision(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("revision_uuid"))
        .def("geometry_history", [](PyObjectStore& self, const std::string& uuid) {
            const auto revisions = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.geometryHistory(txn, uuid); });
            nb::list result; for (const auto& r : revisions) result.append(geometryToPy(r)); return result;
        }, nb::arg("object_uuid"))
        .def("read_geometry_array", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                      int64_t offset, const std::optional<int64_t>& count) {
            std::vector<uint8_t> bytes;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              bytes = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.readGeometryArray(txn, uuid, name, offset, count); }); }
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("byte_offset") = 0, nb::arg("byte_count") = std::nullopt)
        .def("read_geometry_arrays", [](PyObjectStore& self, const std::vector<std::string>& revisions,
                                       const std::string& name) {
            std::shared_ptr<vo::PackedGeometryArray> packed;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              packed = std::make_shared<vo::PackedGeometryArray>(self.withRead([&](auto& db, auto& txn) {
                  return db.readGeometryArrays(txn, revisions, name); })); }
            return PyPackedGeometryArray{std::move(packed)};
        }, nb::arg("revision_uuids"), nb::arg("array_name"),
           "Read homogeneous arrays into one verified packed buffer with row offsets.")
        .def("read_geometry_rectangle", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                           int64_t row, int64_t column, int64_t rows, int64_t columns) {
            std::vector<uint8_t> bytes;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              bytes = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                  return db.readGeometryRectangle(txn, uuid, name, row, column, rows, columns); }); }
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("row"), nb::arg("column"),
           nb::arg("rows"), nb::arg("columns"), "Read a dense row-major patch rectangle, preserving native dtype and component bytes.")
        .def("read_geometry_rectangles", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                            const std::vector<std::array<int64_t, 4>>& rectangles) {
            std::vector<std::vector<uint8_t>> values;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              values = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                  return db.readGeometryRectangles(txn, uuid, name, rectangles); }); }
            nb::list out;
            for (const auto& bytes : values) out.append(nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
            return out;
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("rectangles"),
           "Read [row,column,rows,columns] rectangles in order, sharing verified chunks.")
        .def("read_geometry_ranges", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                        const std::vector<std::array<int64_t, 2>>& ranges) {
            std::vector<std::vector<uint8_t>> values;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              values = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.readGeometryRanges(txn, uuid, name, ranges); }); }
            nb::list out;
            for (const auto& bytes : values) out.append(nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
            return out;
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("ranges"))
        .def("ingest_registered_geometry", [](PyObjectStore& self, const std::optional<std::filesystem::path>& root,
                                             const std::vector<std::string>& uuids) {
            vo::GeometryIngestReport result;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release; result = vo::ingestRegisteredGeometry(self.require(), root, uuids, {}); }
            return geometryReportToPy(result);
        }, nb::arg("dataset_root") = std::nullopt, nb::arg("object_uuids") = std::vector<std::string>{})
        .def("ingest_geometry_source", [](PyObjectStore& self, const std::string& kind, const std::filesystem::path& source,
                                         const nb::dict& context) {
            vo::GeometryIngestReport result;
            const auto declarations = jsonDumps(context);
            { auto lease = self.db; self.require(); nb::gil_scoped_release release; result = vo::ingestGeometrySource(self.require(), kind, source, {}, declarations); }
            return geometryReportToPy(result);
        }, nb::arg("kind"), nb::arg("source"), nb::arg("coordinate_context") = nb::dict())
        .def("export_geometry", [](PyObjectStore& self, const std::string& object,
                                   const std::filesystem::path& destination,
                                   const std::optional<std::string>& revision, bool overwrite) {
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return vo::exportGeometry(self.require(), object, destination, revision, overwrite);
        }, nb::arg("object_uuid"), nb::arg("destination"), nb::arg("revision_uuid") = std::nullopt,
           nb::arg("overwrite") = false,
           "Export one immutable revision to native files; return its UUID. Does not modify the store.")
        .def("edge_geometry_state", [](PyObjectStore& self, const std::string& uuid) {
            const auto state = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.edgeGeometryState(txn, uuid); });
            nb::dict result; result["a_revision"] = state.aRevision; result["b_revision"] = state.bRevision; result["stale"] = state.stale; return result;
        }, nb::arg("edge_uuid"))
        .def("repair_edge_geometry", [](PyObjectStore& self, const nb::dict& input) {
            const auto repair = repairFromPy(input);
            self.withWrite([&](vo::ObjectStore& db, vo::WriteTxn& txn) { db.repairEdgeGeometry(txn, repair); return 0; });
        }, nb::arg("repair"), "Atomically publish anchors and checked revisions; reject changed endpoints or edge generation.")
        .def("resolve_storage_path",
             [](PyObjectStore& self, const std::string& uuid,
                const std::optional<std::filesystem::path>& datasetRoot) {
                 return self.withRead([&](vo::ObjectStore& g, vo::ReadTxn& t) {
                     return g.resolveStoragePath(t, uuid, datasetRoot);
                 });
             }, nb::arg("uuid"), nb::arg("dataset_root") = std::nullopt,
             "Find a patch/fiber by source UUID, verifying the saved path before searching the dataset root.")
        .def("checkpoint", [](PyObjectStore& self) { self.require().checkpoint(); })
        .def("select_edges", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::vector<std::string>& states, const std::optional<std::vector<std::string>>& endpoints,
            bool both, const std::optional<std::string>& tag, const std::optional<int64_t>& limit,
            const std::optional<std::string>& after, const std::optional<std::vector<std::string>>& fields, bool includeRetired) {
            vo::EdgeSelection selection; selection.kind = kind; selection.approvalStates = states;
            selection.endpointUuids = endpoints; selection.bothEndpoints = both;
            selection.tag = tag; selection.limit = limit; selection.afterUuid = after; selection.includeRetired = includeRetired;
            std::vector<vo::EdgeRecord> records;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              records = self.withRead([&](auto& db, auto& txn) { return db.selectEdges(txn, selection, fields); }); }
            const std::set<std::string> projection = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list out; for (const auto& record : records) out.append(edgeToPy(record, fields ? &projection : nullptr));
            return out;
        }, nb::arg("edge_kind") = std::nullopt, nb::arg("approval_states") = std::vector<std::string>{},
           nb::arg("endpoint_uuids") = std::nullopt, nb::arg("both_endpoints") = false,
           nb::arg("tag") = std::nullopt, nb::arg("limit") = std::nullopt,
           nb::arg("after_uuid") = std::nullopt, nb::arg("fields") = std::nullopt, nb::arg("include_retired") = false)
        .def("graph_read", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::vector<std::string>& ids, const std::optional<std::string>& edgeKind,
            const std::vector<std::string>& states, const std::optional<std::array<double, 6>>& bbox,
            const std::optional<std::string>& tag, bool includeRetired) {
            self.require();
            auto view = std::make_unique<PyGraphReadView>(); view->store = self.db;
            vo::ObjectSelection objects; objects.kind = kind; objects.uuids = ids; objects.bbox = bbox; objects.includeRetired = includeRetired;
            vo::EdgeSelection edges; edges.kind = edgeKind; edges.approvalStates = states; edges.tag = tag;
            { nb::gil_scoped_release release;
              view->txn = std::make_unique<vo::ReadTxn>(view->store->read());
              view->graph = std::make_shared<vo::GraphReadView>(view->store->graphReadView(*view->txn, objects, edges)); }
            return view.release();
        }, nb::rv_policy::take_ownership, nb::arg("object_kind") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("edge_kind") = std::nullopt,
           nb::arg("approval_states") = std::vector<std::string>{"approved"},
           nb::arg("bbox") = std::nullopt, nb::arg("edge_tag") = std::nullopt, nb::arg("include_retired") = false,
           "Pin a store graph read view; use as a context manager and close before checkpointing.")
        .def("select_objects", [](PyObjectStore& self, const std::optional<std::string>& kind,
            const std::optional<std::string>& tag, const std::vector<std::string>& ids,
            const std::optional<std::string>& connected, const std::vector<std::string>& edgeKinds,
            const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::array<double, 6>>& bbox, const std::optional<std::vector<std::string>>& fields,
            const std::optional<std::string>& family, const std::optional<double>& minLength,
            const std::string& orderBy, const std::optional<std::string>& afterUuid,
            const std::optional<double>& afterLength, bool includeRetired) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.orderBy = orderBy;
            selection.afterUuid = afterUuid; selection.afterLength = afterLength; selection.includeRetired = includeRetired;
            std::vector<vo::ObjectRecord> records;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              records = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.selectObjects(txn, selection, fields); }); }
            const std::set<std::string> selected = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list result;
            for (const auto& record : records) result.append(objectToPy(record, fields ? &selected : nullptr));
            return result;
        }, nb::arg("object_kind") = std::nullopt, nb::arg("tag") = std::nullopt,
           nb::arg("object_uuids") = std::vector<std::string>{}, nb::arg("connected_to") = std::nullopt,
           nb::arg("edge_kinds") = std::vector<std::string>{}, nb::arg("limit") = std::nullopt, nb::arg("offset") = 0,
           nb::arg("bbox") = std::nullopt, nb::arg("fields") = std::nullopt,
           nb::arg("family") = std::nullopt, nb::arg("min_length") = std::nullopt,
           nb::arg("order_by") = "uuid", nb::arg("after_uuid") = std::nullopt, nb::arg("after_length") = std::nullopt, nb::arg("include_retired") = false)
        .def("spatial_regions", [](PyObjectStore& self, const std::array<double, 6>& bbox,
            const std::optional<std::string>& kind, const std::optional<std::string>& tag,
            const std::vector<std::string>& ids, const std::optional<std::string>& connected,
            const std::vector<std::string>& edgeKinds, const std::optional<int64_t>& limit, int64_t offset,
            const std::optional<std::string>& family, const std::optional<double>& minLength, bool includeRetired) {
            vo::ObjectSelection selection{kind, tag, connected, ids, edgeKinds, limit, offset, bbox};
            selection.family = family; selection.minLength = minLength; selection.includeRetired = includeRetired;
            std::vector<vo::SpatialRegion> regions;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              regions = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.spatialRegions(txn, selection); }); }
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
        .def_prop_ro("replica_uuid", [](PyObjectStore& self) {
            return self.withRead([](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.replicaUuid(txn); });
        })
        .def("fork_replica", [](PyObjectStore& self) {
            self.require();
            if (self.db.use_count() != 1) throw std::runtime_error("fork_replica requires an idle store");
            return self.require().forkReplica();
        })
        .def("synchronize", [](PyObjectStore& self, PyObjectStore& peer, bool dryRun,
            const std::map<std::string, std::string>& resolutions, const std::optional<std::string>& expected) {
            self.require(); peer.require();
            if (self.db.use_count() != 1 || peer.db.use_count() != 1)
                throw std::runtime_error("synchronize requires idle stores with no active transactions");
            vo::SyncReport report;
            { auto localLease = self.db, remoteLease = peer.db; nb::gil_scoped_release release;
              report = localLease->synchronize(*remoteLease, {dryRun, resolutions, expected}); }
            nb::dict result; result["applied"] = report.applied; result["plan_token"] = report.planToken;
            result["local_replica"] = report.localReplica; result["remote_replica"] = report.remoteReplica;
            result["local_changes"] = report.localChanges; result["remote_changes"] = report.remoteChanges;
            nb::list conflicts;
            for (const auto& c : report.conflicts) {
                nb::dict item; item["uuid"] = c.uuid; item["reason"] = c.reason;
                item["base"] = jsonLoads(c.base); item["local"] = jsonLoads(c.local); item["remote"] = jsonLoads(c.remote);
                conflicts.append(item);
            }
            result["conflicts"] = conflicts; return result;
        }, nb::arg("peer"), nb::arg("dry_run") = false,
           nb::arg("resolutions") = std::map<std::string, std::string>{}, nb::arg("expected_plan") = std::nullopt)
        .def(
            "import_json",
            [](PyObjectStore& self, const std::string& jsonText) {
                self.withWrite([&](vo::ObjectStore& g, vo::WriteTxn& t) {
                    g.importJson(t, jsonText);
                    return 0;
                });
            },
            nb::arg("json_text"))
        .def(
            "link_point_collections_to_patches",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                vc::object_links::PointCollectionPatchParams p;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle value = item.second;
                    if (key == "collection_uuids") p.collectionUuids = nb::cast<std::vector<std::string>>(value);
                    else if (key == "patch_uuids") p.patchUuids = nb::cast<std::vector<std::string>>(value);
                    else if (key == "tolerance") p.tolerance = nb::cast<double>(value);
                    else if (key == "point_scale") p.pointScale = nb::cast<double>(value);
                    else if (key == "patch_scale") p.patchScale = nb::cast<double>(value);
                    else if (key == "window_points") p.windowPoints = nb::cast<int>(value);
                    else if (key == "window_min_points") p.windowMinPoints = nb::cast<int>(value);
                    else if (key == "absolute_winding") p.absoluteWinding = nb::cast<std::optional<bool>>(value);
                    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(value);
                    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(value);
                    else throw std::runtime_error("link_point_collections_to_patches: unknown parameter '" + key + "'");
                }
                vc::object_links::DetectorReport report;
                auto lease = self.db;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::object_links::linkPointCollectionsToPatches(db, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("kwargs"),
            "Link regular PCL points to patches using Spiral's largest-area/nearest-named-pair rules. "
            "Point IDs anchor the edges; winding annotations and patch grid coordinates are retained. "
            "Defaults: tolerance=2.5, window_points=1, window_min_points=1; no CT required.")
        .def(
            "link_patch_overlaps",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                vc::object_links::PatchOverlapParams p;
                for (auto item : kwargs) {
                    const std::string key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "patch_uuids") p.patchUuids = nb::cast<std::vector<std::string>>(val);
                    else if (key == "workers") p.workers = nb::cast<int>(val);
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
                vc::object_links::DetectorReport rep;
                { auto lease = self.db; self.require(); nb::gil_scoped_release release;
                  rep = vc::object_links::linkPatchOverlaps(*lease, p); }
                return detectorReportToPy(rep);
            },
            "Run patch-overlap discovery using stored geometry; patch_uuids optionally selects objects. "
            "kwargs mirror PatchOverlapParams (tolerance, min_minor_spread, ...).")
        .def(
            "link_tracks_to_tracks",
            [](PyObjectStore& self, const nb::kwargs& kwargs) {
                vc::object_links::TrackTrackParams p;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle value = item.second;
                    if (key == "min_angle_degrees") p.minAngleDegrees = nb::cast<double>(value);
                    else if (key == "tangent_radius_voxels") p.tangentRadiusVoxels = nb::cast<double>(value);
                    else if (key == "cluster_index_radius") p.clusterIndexRadius = nb::cast<int>(value);
                    else if (key == "workers") p.workers = nb::cast<int>(value);
                    else if (key == "dataset_ref") p.datasetRef = nb::cast<std::string>(value);
                    else if (key == "track_uuids") p.trackUuids = nb::cast<std::vector<std::string>>(value);
                    else if (key == "edge_kind") p.edgeKind = nb::cast<std::string>(value);
                    else if (key == "replace_stale_pending") p.replaceStalePending = nb::cast<bool>(value);
                    else throw std::runtime_error("link_tracks_to_tracks: unknown parameter '" + key + "'");
                }
                vc::object_links::DetectorReport report;
                auto lease = self.db;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::object_links::linkTracksToTracks(db, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("kwargs"),
            "Discover crossings from stored tracks, optionally selected by track_uuids and dataset_ref. "
            "Anchors are raw point indices and source XYZ positions.")
        .def(
            "link_tracks_to_patches",
            [](PyObjectStore& self, const std::string& ctVolume, const nb::kwargs& kwargs) {
                vc::object_links::TrackPatchParams p;
                for (auto item : kwargs) {
                    const auto key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "track_scale") p.trackScale = nb::cast<double>(val);
                    else if (key == "track_uuids") p.trackUuids = nb::cast<std::vector<std::string>>(val);
                    else if (!setLinePatchParameter(p, key, val))
                        throw std::runtime_error("link_tracks_to_patches: unknown parameter '" + key + "'");
                }
                vc::object_links::DetectorReport report;
                auto lease = self.db;
                auto& db = self.require();
                {
                    nb::gil_scoped_release release;
                    report = vc::object_links::linkTracksToPatches(db, ctVolume, p);
                }
                return detectorReportToPy(report);
            },
            nb::arg("ct_volume") = "", nb::arg("kwargs"),
            "Link stored tracks to stored patches; select with track_uuids and patch_uuids. "
            "vc_reader supplies CT normals. Anchors retain raw indices and source XYZ.")
        .def(
            "link_fibers_to_patches",
            [](PyObjectStore& self, const std::string& ctVolume, const nb::kwargs& kwargs) {
                vc::object_links::FiberPatchParams p;
                for (auto item : kwargs) {
                    const std::string key = nb::cast<std::string>(item.first);
                    const nb::handle val = item.second;
                    if (key == "fiber_uuids") p.fiberUuids = nb::cast<std::vector<std::string>>(val);
                    else if (key == "cache_directory") p.cacheDirectory = nb::cast<std::string>(val);
                    else if (key == "fiber_scale") p.fiberScale = nb::cast<double>(val);
                    else if (key == "rebuild_strips") p.rebuildStrips = nb::cast<bool>(val);
                    else if (!setLinePatchParameter(p, key, val))
                        throw std::runtime_error("link_fibers_to_patches: unknown parameter '" + key + "'");
                }
                vc::object_links::DetectorReport report;
                { auto lease = self.db; self.require(); nb::gil_scoped_release release;
                  report = vc::object_links::linkFibersToPatches(*lease, ctVolume, p); }
                return detectorReportToPy(report);
            },
            nb::arg("ct_volume") = "", nb::arg("kwargs"),
            "Link stored fibers to stored patches; select with fiber_uuids and patch_uuids. "
            "vc_reader supplies CT normals. Optional cache_directory stores disposable strips.")
        .def_prop_ro("uuid", [](PyObjectStore& self) {
            return self.withRead([](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.uuid(txn); });
        })
        .def("current_object_revision", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.currentObjectRevision(txn, uuid); });
            return revision ? nb::object(objectRevisionToPy(*revision)) : nb::none();
        }, nb::arg("object_uuid"))
        .def("object_revision", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.objectRevision(txn, uuid); });
            return revision ? nb::object(objectRevisionToPy(*revision)) : nb::none();
        }, nb::arg("revision_uuid"))
        .def("object_history", [](PyObjectStore& self, const std::string& uuid) {
            const auto revisions = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.objectHistory(txn, uuid); });
            nb::list result;
            for (const auto& revision : revisions) result.append(objectRevisionToPy(revision));
            return result;
        }, nb::arg("object_uuid"))
        .def("recorded_relationships", [](PyObjectStore& self, const std::string& uuid) {
            const auto revisions = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.recordedRelationships(txn, uuid); });
            nb::list result;
            for (const auto& revision : revisions) result.append(relationshipRevisionToPy(revision));
            return result;
        }, nb::arg("object_revision_uuid"))
        .def("relationships_at", [](PyObjectStore& self, const vo::RevisionSelection& selection) {
            const auto revisions = self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.relationshipsAt(txn, selection); });
            nb::list result;
            for (const auto& revision : revisions) {
                auto d = relationshipRevisionToPy(revision.relationship);
                d["applicable"] = revision.applicable;
                result.append(d);
            }
            return result;
        }, nb::arg("selection"))
        .def("revision_read_view", [](PyObjectStore& self, const vo::RevisionSelection& selection) {
            return self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return revisionViewToPy(db, txn, selection);
            });
        }, nb::arg("selection"),
           "Resolve selected objects, metadata, geometry references and relationship applicability in one read snapshot.")
        .def("revision_selection_at", [](PyObjectStore& self, int64_t time, bool includeRetired) {
            return self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) { return db.revisionSelectionAt(txn, time, includeRetired); });
        }, nb::arg("published_at_us"), nb::arg("include_retired") = false)
        .def("revision_read_view_at", [](PyObjectStore& self, int64_t time, bool includeRetired) {
            return self.withRead([&](vo::ObjectStore& db, vo::ReadTxn& txn) {
                return revisionViewToPy(db, txn, db.revisionSelectionAt(txn, time, includeRetired));
            });
        }, nb::arg("published_at_us"), nb::arg("include_retired") = false,
           "Recover all objects, geometry references and recorded relationships at a publication time (UTC Unix microseconds).")
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

    nb::class_<PyObjectWriteTxn>(m, "WriteTransaction")
        .def("__enter__", [](nb::object self) {
            nb::cast<PyObjectWriteTxn&>(self).enter();
            return self;
        })
        .def("__exit__", [](PyObjectWriteTxn& self, nb::args args) {
            const bool failed = nb::len(args) > 0 && !args[0].is_none();
            self.exit(failed);
        })
        .def("retire_object", [](PyObjectWriteTxn& self, const std::string& uuid) {
            self.owner->require().retireObject(self.require(), uuid);
        }, nb::arg("uuid"))
        .def("restore_object", [](PyObjectWriteTxn& self, const std::string& uuid,
                                  const std::optional<std::string>& source) {
            self.owner->require().restoreObject(self.require(), uuid, source);
        }, nb::arg("uuid"), nb::arg("source_revision") = std::nullopt)
        .def("repair_edge_geometry", [](PyObjectWriteTxn& self, const nb::dict& input) {
            self.owner->require().repairEdgeGeometry(self.require(), repairFromPy(input));
        }, nb::arg("repair"))
        .def("retain_relationship", [](PyObjectWriteTxn& self, const std::string& uuid) {
            self.owner->require().retainRelationship(self.require(), uuid);
        }, nb::arg("edge_uuid"))
        .def("keep_recorded_relationship", [](PyObjectWriteTxn& self, const std::string& uuid) {
            self.owner->require().keepRecordedRelationship(self.require(), uuid);
        }, nb::arg("edge_uuid"))
        .def("update_edge", [](PyObjectWriteTxn& self, const nb::dict& record) {
            self.owner->require().updateEdge(self.require(), edgeFromPy(record));
        })
        .def("remove_edge", [](PyObjectWriteTxn& self, const std::string& uuid) {
            self.owner->require().removeEdge(self.require(), uuid);
        })
        .def("remove_meta", [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& key) {
            self.owner->require().removeMeta(self.require(), uuid, key);
        })
        .def("put_geometry", [](PyObjectWriteTxn& self, const std::string& uuid, const nb::object& input,
                               const std::optional<std::string>& expected, const nb::list& choices, int64_t patchTileSize) {
            const auto data = geometryInputFromPy(input);
            return self.owner->require().putGeometry(self.require(), uuid, data, expected, edgeEditsFromPy(choices), vo::GeometryStorageOptions{patchTileSize});
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt,
           nb::arg("edge_edits") = nb::list(), nb::arg("patch_tile_size") = vo::GeometryStorageOptions{}.patchTileSize)
        .def("edit_geometry", [](PyObjectWriteTxn& self, const std::string& uuid, const nb::list& input,
                                const std::string& expected, const nb::list& choices) {
            return self.owner->require().editGeometry(self.require(), uuid, geometryEditsFromPy(input), expected, edgeEditsFromPy(choices));
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("update_surface_channels", [](PyObjectWriteTxn& self, const std::string& uuid, const nb::list& channels,
                                           const std::vector<std::string>& remove, const std::string& expected, const nb::list& choices) {
            nb::dict input; input["kind"] = "patch"; input["arrays"] = channels;
            const auto data = geometryFromPy(input); const auto edges = edgeEditsFromPy(choices);
            return self.owner->require().updateSurfaceChannels(self.require(), uuid, data.arrays, remove, expected, edges);
        }, nb::arg("object_uuid"), nb::arg("channels"), nb::arg("remove"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("edit_geometry_rectangle", [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& name,
                                          int64_t row, int64_t column, int64_t rows, int64_t columns,
                                          const nb::bytes& data, const std::string& expected, const nb::list& choices) {
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::vector<uint8_t> bytes(p, p + data.size()); const auto edges = edgeEditsFromPy(choices);
            return self.owner->require().editGeometryRectangle(self.require(), uuid, name, row, column, rows, columns, bytes, expected, edges);
        }, nb::arg("object_uuid"), nb::arg("array_name"), nb::arg("row"), nb::arg("column"), nb::arg("rows"), nb::arg("columns"),
           nb::arg("data"), nb::arg("expected_revision"), nb::arg("edge_edits") = nb::list())
        .def("affected_geometry_edit_edges", [](PyObjectWriteTxn& self, const std::string& uuid,
                                                const nb::list& input, const std::string& expected) {
            const auto edges = self.owner->require().affectedGeometryEditEdges(self.require(), uuid, geometryEditsFromPy(input), expected);
            nb::list result; for (const auto& edge : edges) result.append(edgeToPy(edge)); return result;
        }, nb::arg("object_uuid"), nb::arg("edits"), nb::arg("expected_revision"))
        .def("affected_geometry_edges", [](PyObjectWriteTxn& self, const std::string& uuid, const nb::object& input) {
            const auto edges = self.owner->require().affectedGeometryEdges(self.require(), uuid, geometryInputFromPy(input));
            nb::list result; for (const auto& edge : edges) result.append(edgeToPy(edge)); return result;
        }, nb::arg("object_uuid"), nb::arg("geometry"))
        .def("upsert_object",
             [](PyObjectWriteTxn& self, const nb::dict& rec) { return self.owner->upsertObjectIn(self.owner->require(), self.require(), rec); })
        .def("add_edge",
             [](PyObjectWriteTxn& self, const nb::dict& rec) { return self.owner->addEdgeIn(self.owner->require(), self.require(), rec); })
        .def("set_meta",
             [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                 self.owner->require().setMeta(self.require(), uuid, key, jsonDumps(value));
             }, nb::arg("uuid"), nb::arg("key"), nb::arg("value").none())
        .def("backfill_geometry_coordinates",
             [](PyObjectWriteTxn& self, const std::string& uuid, const nb::dict& declarations) {
                 self.owner->require().backfillGeometryCoordinates(self.require(), uuid, jsonDumps(declarations));
             }, nb::arg("uuid"), nb::arg("declarations") = nb::dict())
        .def("set_approval", [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& state) {
            self.owner->require().setApproval(self.require(), uuid, state);
        });
}
