#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

namespace {
nb::list hubLeases(const std::vector<vo::HubLease>& leases)
{
    nb::list result;
    for (const auto& lease : leases) {
        nb::dict item; item["uuid"] = lease.objectUuid; item["holder"] = lease.holder;
        item["author"] = lease.author; item["host"] = lease.host;
        item["acquired_us"] = lease.acquiredUs; item["expires_us"] = lease.expiresUs;
        result.append(item);
    }
    return result;
}

nb::list hubConflicts(const std::vector<vo::HubConflict>& conflicts)
{
    nb::list result;
    for (const auto& conflict : conflicts) {
        nb::dict item; item["uuid"] = conflict.uuid; item["hub_version"] = conflict.hubVersion;
        item["hub"] = jsonLoads(conflict.hubRecord); item["local"] = jsonLoads(conflict.localRecord);
        result.append(item);
    }
    return result;
}

nb::dict hubReport(const vo::HubSyncReport& report)
{
    nb::dict result; result["pulled"] = report.pulled; result["pushed"] = report.pushed;
    result["conflicts"] = hubConflicts(report.conflicts);
    return result;
}
}

namespace vc::python {
// Lifecycle, leases, hub sync, artifacts and copies for the stores a process
// works with. Data access goes through ObjectStore connections to the same
// files; a session's hosted replica pushes their commits in the background.
inline void bindObjectStoreSession(nb::module_& m)
{
    nb::class_<vo::AttachmentIdentity>(m, "AttachmentIdentity")
        .def_ro("store_uuid", &vo::AttachmentIdentity::storeUuid)
        .def_ro("generation", &vo::AttachmentIdentity::generation)
        .def("__eq__", [](const vo::AttachmentIdentity& a, const vo::AttachmentIdentity& b) { return a == b; })
        .def("__repr__", [](const vo::AttachmentIdentity& a) {
            return "AttachmentIdentity(" + a.storeUuid + ", " + std::to_string(a.generation) + ")";
        });
    nb::class_<vo::LeaseHold>(m, "LeaseHold")
        .def("release", &vo::LeaseHold::reset, nb::call_guard<nb::gil_scoped_release>())
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](vo::LeaseHold& self, nb::args) { nb::gil_scoped_release release; self.reset(); });
    nb::class_<vo::ObjectStoreSession>(m, "ObjectStoreSession")
        .def(nb::init<>())
        .def("attach", [](vo::ObjectStoreSession& self, const std::filesystem::path& path, const std::string& author,
                          bool readOnly, int64_t syncIntervalMs, int64_t pushDelayMs) {
            vo::OpenOptions options; options.author = author; options.readOnly = readOnly;
            vo::HubSessionOptions hub;
            hub.syncInterval = std::chrono::milliseconds(syncIntervalMs);
            hub.pushDelay = std::chrono::milliseconds(pushDelayMs);
            nb::gil_scoped_release release;
            return self.attach(path, options, hub);
        }, nb::arg("path"), nb::arg("author"), nb::arg("read_only") = false,
           nb::arg("sync_interval_ms") = vo::HubSessionOptions{}.syncInterval.count(),
           nb::arg("push_delay_ms") = vo::HubSessionOptions{}.pushDelay.count(),
           "Attach an existing store; a writable hosted replica starts background hub sync.")
        .def("detach", &vo::ObjectStoreSession::detach, nb::arg("attachment"), nb::call_guard<nb::gil_scoped_release>())
        .def("attachments", &vo::ObjectStoreSession::attachments)
        .def("writable", &vo::ObjectStoreSession::writable, nb::arg("attachment"))
        .def("current", &vo::ObjectStoreSession::current, nb::arg("attachment"))
        .def("hold_leases", &vo::ObjectStoreSession::holdLeases, nb::arg("attachment"), nb::arg("uuids"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Lease objects before editing them; renewed in the background until released.")
        .def("hub_status", [](const vo::ObjectStoreSession& self, const vo::AttachmentIdentity& attachment) -> nb::object {
            const auto status = self.hubStatus(attachment);
            if (!status) return nb::none();
            nb::dict result; result["pending"] = status->pending; result["conflicts"] = hubConflicts(status->conflicts);
            result["error"] = status->error; result["synced_at_us"] = status->syncedAtUs;
            result["leases"] = hubLeases(status->leases);
            return result;
        }, nb::arg("attachment"))
        .def("sync_hub", &vo::ObjectStoreSession::syncHub, nb::arg("attachment"))
        .def("resolve_hub_conflicts", &vo::ObjectStoreSession::resolveHubConflicts, nb::arg("attachment"), nb::arg("keep"))
        .def("artifact_directory", &vo::ObjectStoreSession::artifactDirectory, nb::arg("attachment"), nb::arg("object_uuid"))
        .def("copy_object", &vo::ObjectStoreSession::copyObject, nb::arg("source"), nb::arg("destination"),
             nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("finalize", &vo::ObjectStoreSession::finalize, nb::call_guard<nb::gil_scoped_release>())
        .def("take_events", [](vo::ObjectStoreSession& self) {
            std::vector<vo::StoreSessionEvent> events;
            { nb::gil_scoped_release release; events = self.takeEvents(); }
            nb::list result;
            for (const auto& event : events) {
                static constexpr const char* kinds[] = {"attached", "detached", "committed", "hub"};
                nb::dict item; item["kind"] = kinds[int(event.kind)]; item["attachment"] = event.attachment;
                item["journal_position"] = event.journalPosition;
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
}
}
