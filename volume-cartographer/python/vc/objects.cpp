#include "StorageBindings.hpp"
using namespace vc::python;
NB_MODULE(objects, m) {
    m.doc() = "Independent versioned object storage.";
    nb::class_<PyObjectStore>(m, "ObjectStore")
        .def_static("create", &PyObjectStore::create, nb::arg("path"), nb::arg("author"),
                    nb::arg("hostname") = std::nullopt, nb::arg("reader_pool_size") = 4,
                    "Create a new object store at path (refuses existing files).")
        .def_static("open", &PyObjectStore::open, nb::arg("path"), nb::arg("author") = std::nullopt,
                    nb::arg("read_only") = false, nb::arg("hostname") = std::nullopt,
                    nb::arg("reader_pool_size") = 4,
                    "Open an existing object store. author is required unless read_only.")
        .def("close", &PyObjectStore::close)
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](PyObjectStore& self, nb::args) { self.close(); })
        .def_prop_ro("path", [](PyObjectStore& self) { return self.require().path().string(); })
        .def_prop_ro("hostname", [](PyObjectStore& self) { return self.require().hostname(); })
        .def_prop_ro("author", [](PyObjectStore& self) { return self.require().author(); })
        .def(
            "upsert_object",
            [](PyObjectStore& self, const nb::dict& rec) {
                return self.withWrite([&](vo::ObjectStore& g, vs::WriteTxn& t) { return self.upsertObjectIn(g, t, rec); });
            },
            nb::arg("record"), "Create or update an object from a dict; returns its uuid.")
        .def(
            "object",
            [](PyObjectStore& self, const std::string& uuid) -> nb::object {
                auto rec = self.withRead([&](vo::ObjectStore& g, vs::ReadTxn& t) { return g.objectByUuid(t, uuid); });
                return rec ? nb::object(objectToPy(*rec)) : nb::none();
            },
            nb::arg("uuid"))
        .def(
            "objects_by_kind",
            [](PyObjectStore& self, const std::string& kind, const std::optional<std::string>& approvalState) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vs::ReadTxn& t) { return g.objectsByKind(t, kind, approvalState); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("kind"), nb::arg("approval_state") = std::nullopt)
        .def("object_count", [](PyObjectStore& self, const std::string& kind) {
            return self.withRead([&](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.objectCount(txn, kind); });
        }, nb::arg("kind"))
        .def(
            "objects_by_tag",
            [](PyObjectStore& self, const std::string& tag) {
                auto recs = self.withRead([&](vo::ObjectStore& g, vs::ReadTxn& t) { return g.objectsByTag(t, tag); });
                nb::list out;
                for (const auto& r : recs) out.append(objectToPy(r));
                return out;
            },
            nb::arg("tag"))
        .def(
            "delete_object",
            [](PyObjectStore& self, const std::string& uuid) {
                self.withWrite([&](vo::ObjectStore& g, vs::WriteTxn& t) {
                    g.deleteObject(t, uuid);
                    return 0;
                });
            },
            nb::arg("uuid"))
        .def(
            "set_approval",
            [](PyObjectStore& self, const std::string& uuid, const std::string& state) {
                self.withWrite([&](vo::ObjectStore& g, vs::WriteTxn& t) {
                    g.setApproval(t, uuid, state);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("state"), "state: 'pending', 'approved' or 'rejected' (objects and edges).")
        .def(
            "set_meta",
            [](PyObjectStore& self, const std::string& uuid, const std::string& key, const nb::object& value) {
                const std::string json = jsonDumps(value);
                self.withWrite([&](vo::ObjectStore& g, vs::WriteTxn& t) {
                    g.setMeta(t, uuid, key, json);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"), nb::arg("value"), "Set one open-ended metadata key (any JSON value).")
        .def(
            "remove_meta",
            [](PyObjectStore& self, const std::string& uuid, const std::string& key) {
                self.withWrite([&](vo::ObjectStore& g, vs::WriteTxn& t) {
                    g.removeMeta(t, uuid, key);
                    return 0;
                });
            },
            nb::arg("uuid"), nb::arg("key"))
        .def(
            "journal_since",
            [](PyObjectStore& self, int64_t afterSeq) {
                auto entries = self.withRead([&](vo::ObjectStore& g, vs::ReadTxn& t) { return g.journalSince(t, afterSeq); });
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
        .def("put_geometry", [](PyObjectStore& self, const std::string& uuid, const nb::dict& input,
                               const std::optional<std::string>& expected) {
            const auto data = geometryFromPy(input);
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return self.withWrite([&](vo::ObjectStore& db, vs::WriteTxn& txn) { return db.putGeometry(txn, uuid, data, expected); });
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt)
        .def("current_geometry", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.currentGeometry(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("object_uuid"))
        .def("geometry_revision", [](PyObjectStore& self, const std::string& uuid) -> nb::object {
            const auto revision = self.withRead([&](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.geometryRevision(txn, uuid); });
            return revision ? nb::object(geometryToPy(*revision)) : nb::none();
        }, nb::arg("revision_uuid"))
        .def("geometry_history", [](PyObjectStore& self, const std::string& uuid) {
            const auto revisions = self.withRead([&](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.geometryHistory(txn, uuid); });
            nb::list result; for (const auto& r : revisions) result.append(geometryToPy(r)); return result;
        }, nb::arg("object_uuid"))
        .def("read_geometry_array", [](PyObjectStore& self, const std::string& uuid, const std::string& name,
                                      int64_t offset, const std::optional<int64_t>& count) {
            std::vector<uint8_t> bytes;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release;
              bytes = self.withRead([&](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.readGeometryArray(txn, uuid, name, offset, count); }); }
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }, nb::arg("revision_uuid"), nb::arg("array_name"), nb::arg("byte_offset") = 0, nb::arg("byte_count") = std::nullopt)
        .def("ingest_registered_geometry", [](PyObjectStore& self, const std::optional<std::filesystem::path>& root,
                                             const std::vector<std::string>& uuids) {
            vo::GeometryIngestReport result;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release; result = vo::ingestRegisteredGeometry(self.require(), root, uuids); }
            return geometryReportToPy(result);
        }, nb::arg("dataset_root") = std::nullopt, nb::arg("object_uuids") = std::vector<std::string>{})
        .def("ingest_geometry_source", [](PyObjectStore& self, const std::string& kind, const std::filesystem::path& source) {
            vo::GeometryIngestReport result;
            { auto lease = self.db; self.require(); nb::gil_scoped_release release; result = vo::ingestGeometrySource(self.require(), kind, source); }
            return geometryReportToPy(result);
        }, nb::arg("kind"), nb::arg("source"))
        .def("export_geometry", [](PyObjectStore& self, const std::string& object,
                                   const std::filesystem::path& destination,
                                   const std::optional<std::string>& revision, bool overwrite) {
            auto lease = self.db; self.require(); nb::gil_scoped_release release;
            return vo::exportGeometry(self.require(), object, destination, revision, overwrite);
        }, nb::arg("object_uuid"), nb::arg("destination"), nb::arg("revision_uuid") = std::nullopt,
           nb::arg("overwrite") = false,
           "Export one immutable revision to native files; return its UUID. Does not modify the store.")
        .def("resolve_storage_path",
             [](PyObjectStore& self, const std::string& uuid,
                const std::optional<std::filesystem::path>& datasetRoot) {
                 return self.withRead([&](vo::ObjectStore& g, vs::ReadTxn& t) {
                     return g.resolveStoragePath(t, uuid, datasetRoot);
                 });
             }, nb::arg("uuid"), nb::arg("dataset_root") = std::nullopt,
             "Find a patch/fiber by source UUID, verifying the saved path before searching the dataset root.")
        .def("sweep_external_data",
             [](PyObjectStore& self) { return self.withRead([](vo::ObjectStore& g, vs::ReadTxn& t) { return g.sweepExternalData(t); }); })
        .def("checkpoint", [](PyObjectStore& self) { self.require().checkpoint(); })
        .def_prop_ro("uuid", [](PyObjectStore& self) {
            return self.withRead([](vo::ObjectStore& db, vs::ReadTxn& txn) { return db.uuid(txn); });
        })
        .def(
            "write",
            [](PyObjectStore& self) {
                PyObjectWriteTxn w;
                w.owner = &self;
                return w;
            },
            nb::keep_alive<0, 1>(),
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
        .def("put_geometry", [](PyObjectWriteTxn& self, const std::string& uuid, const nb::dict& input,
                               const std::optional<std::string>& expected) {
            const auto data = geometryFromPy(input);
            return self.owner->require().putGeometry(self.require(), uuid, data, expected);
        }, nb::arg("object_uuid"), nb::arg("geometry"), nb::arg("expected_revision") = std::nullopt)
        .def("upsert_object", [](PyObjectWriteTxn& self, const nb::dict& rec) {
            return self.owner->upsertObjectIn(self.owner->require(), self.require(), rec);
        })
        .def("set_meta", [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& key, const nb::object& value) {
            self.owner->require().setMeta(self.require(), uuid, key, jsonDumps(value));
        })
        .def("set_approval", [](PyObjectWriteTxn& self, const std::string& uuid, const std::string& state) {
            self.owner->require().setApproval(self.require(), uuid, state);
        });
}
