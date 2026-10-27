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

nb::object sessionHubStatus(const std::optional<vo::HubSessionStatus>& status)
{
    if (!status) return nb::none();
    nb::dict result; result["pending"] = status->pending; result["conflicts"] = hubConflicts(status->conflicts);
    result["error"] = status->error; result["synced_at_us"] = status->syncedAtUs;
    result["leases"] = hubLeases(status->leases);
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
        .def("__repr__", [](const vo::AttachmentIdentity& a) {
            return "AttachmentIdentity(" + a.storeUuid + ", " + std::to_string(a.generation) + ")";
        });
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
        .def("sync_hub_and_wait", [](vo::ObjectStoreSession& self, const vo::AttachmentIdentity& attachment, double timeoutS) {
            std::optional<vo::HubSessionStatus> status;
            {
                nb::gil_scoped_release release;
                status = self.syncHubAndWait(attachment, std::chrono::milliseconds(int64_t(timeoutS * 1000)));
            }
            return sessionHubStatus(status);
        }, nb::arg("attachment"), nb::arg("timeout_s") = 120.0,
           "Sync now and return that sync's hub_status() once it finished (None without a hub); check error and conflicts.")
        .def("artifact_directory", &vo::ObjectStoreSession::artifactDirectory, nb::arg("attachment"), nb::arg("object_uuid"),
             nb::arg("create") = true)
        .def("finalize", &vo::ObjectStoreSession::finalize, nb::call_guard<nb::gil_scoped_release>());
}
}
