#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

namespace vc::python {
// Lifecycle, artifacts and copies for the stores a process works with.
inline nb::class_<vo::ObjectStoreSession> bindObjectStoreSession(nb::module_& m)
{
    nb::class_<vo::AttachmentIdentity>(m, "AttachmentIdentity")
        .def_ro("store_uuid", &vo::AttachmentIdentity::storeUuid)
        .def_ro("generation", &vo::AttachmentIdentity::generation)
        .def("__eq__", [](const vo::AttachmentIdentity& a, const vo::AttachmentIdentity& b) { return a == b; })
        .def("__repr__", [](const vo::AttachmentIdentity& a) {
            return "AttachmentIdentity(" + a.storeUuid + ", " + std::to_string(a.generation) + ")";
        });
    auto session = nb::class_<vo::ObjectStoreSession>(m, "ObjectStoreSession");
    session
        .def(nb::init<>())
        .def("attach", [](vo::ObjectStoreSession& self, const std::filesystem::path& path, const std::string& author,
                          bool readOnly) {
            vo::OpenOptions options; options.author = author; options.readOnly = readOnly;
            nb::gil_scoped_release release;
            return self.attach(path, options);
        }, nb::arg("path"), nb::arg("author"), nb::arg("read_only") = false, "Attach an existing store.")
        .def("detach", &vo::ObjectStoreSession::detach, nb::arg("attachment"), nb::call_guard<nb::gil_scoped_release>())
        .def("attachments", &vo::ObjectStoreSession::attachments)
        .def("writable", &vo::ObjectStoreSession::writable, nb::arg("attachment"))
        .def("current", &vo::ObjectStoreSession::current, nb::arg("attachment"))
        .def("path", &vo::ObjectStoreSession::path, nb::arg("attachment"), "The store file of a local attachment.")
        .def("copy_object", &vo::ObjectStoreSession::copyObject, nb::arg("source"), nb::arg("destination"),
             nb::arg("object_uuid"), nb::arg("name") = std::string(), nb::call_guard<nb::gil_scoped_release>())
        .def("import_object", &vo::ObjectStoreSession::importObject, nb::arg("source"), nb::arg("destination"),
             nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("finalize", &vo::ObjectStoreSession::finalize, nb::call_guard<nb::gil_scoped_release>())
        .def("take_events", [](vo::ObjectStoreSession& self) {
            std::vector<vo::StoreSessionEvent> events;
            { nb::gil_scoped_release release; events = self.takeEvents(); }
            nb::list result;
            for (const auto& event : events) {
                static constexpr const char* kinds[] = {"attached", "detached", "committed", "draft"};
                nb::dict item; item["kind"] = kinds[int(event.kind)]; item["attachment"] = event.attachment;
                item["journal_position"] = event.journalPosition; item["revalidate"] = event.revalidate;
                nb::list changes;
                for (const auto& change : event.changes) {
                    nb::dict row; row["seq"] = change.seq; row["record_uuid"] = change.recordUuid;
                    row["kind"] = change.kind; row["change_kind"] = change.changeKind;
                    changes.append(row);
                }
                item["changes"] = changes;
                result.append(item);
            }
            return result;
        });
    return session;
}
}
