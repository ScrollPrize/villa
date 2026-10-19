#include "StorageBindings.hpp"
#include "vc/core/graph/GraphDb.hpp"
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/CtSheetNormal.hpp"
namespace vg = vc::graph;
using namespace vc::python;
using PyGraphDb = PyDatabase<vg::GraphDb>;
using PyWriteTxn = PyTransaction<vg::GraphDb>;
namespace {
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
    nb::module_::import_("vc.objects");
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
        .def("selected_object", [](PyGraphDb& self, const std::string& uuid) -> nb::object {
            const auto selection = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.selectedObject(txn, uuid); });
            if (!selection) return nb::none();
            nb::dict result;
            result["object_uuid"] = selection->objectUuid;
            result["store_uuid"] = selection->storeUuid;
            result["revision_uuid"] = selection->revisionUuid;
            return result;
        }, nb::arg("object_uuid"))
        .def("available_updates", [](PyGraphDb& self, PyObjectStore& store) {
            const auto updates = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.availableUpdates(txn, store.require()); });
            nb::list result;
            for (const auto& update : updates) {
                nb::dict item;
                item["object_uuid"] = update.objectUuid;
                item["selected_revision"] = update.selectedRevision;
                item["current_revision"] = update.currentRevision;
                result.append(item);
            }
            return result;
        }, nb::arg("store"))
        .def("include_object", [](PyGraphDb& self, PyObjectStore& store, const std::string& object, const std::string& revision) {
            self.withWrite([&](vg::GraphDb& db, vg::WriteTxn& txn) { db.includeObject(txn, store.require(), object, revision); return 0; });
        }, nb::arg("store"), nb::arg("object_uuid"), nb::arg("revision"))
        .def("advance_object", [](PyGraphDb& self, PyObjectStore& store, const std::string& object, const std::string& revision, const std::string& expected) {
            self.withWrite([&](vg::GraphDb& db, vg::WriteTxn& txn) { db.advanceObject(txn, store.require(), object, revision, expected); return 0; });
        }, nb::arg("store"), nb::arg("object_uuid"), nb::arg("revision"), nb::arg("expected_revision"))
        .def("edge_geometry_state", [](PyGraphDb& self, const std::string& uuid) {
            const auto state = self.withRead([&](vg::GraphDb& db, vg::ReadTxn& txn) { return db.edgeGeometryState(txn, uuid); });
            nb::dict result; result["a_revision"] = state.aRevision; result["b_revision"] = state.bRevision; result["stale"] = state.stale; return result;
        }, nb::arg("edge_uuid"))
        .def("revalidate_edge_geometry", [](PyGraphDb& self, const std::string& uuid) {
            self.withWrite([&](vg::GraphDb& db, vg::WriteTxn& txn) { db.revalidateEdgeGeometry(txn, uuid); return 0; });
        }, nb::arg("edge_uuid"))
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
            nb::keep_alive<0, 1>(),
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
        .def("include_object", [](PyWriteTxn& self, PyObjectStore& store, const std::string& object, const std::string& revision) {
            self.owner->require().includeObject(self.require(), store.require(), object, revision);
        }, nb::arg("store"), nb::arg("object_uuid"), nb::arg("revision"))
        .def("advance_object", [](PyWriteTxn& self, PyObjectStore& store, const std::string& object, const std::string& revision, const std::string& expected) {
            self.owner->require().advanceObject(self.require(), store.require(), object, revision, expected);
        }, nb::arg("store"), nb::arg("object_uuid"), nb::arg("revision"), nb::arg("expected_revision"))
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
