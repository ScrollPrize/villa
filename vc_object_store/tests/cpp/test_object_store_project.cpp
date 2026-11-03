#include "TestSupport.hpp"
#include "HubTestSupport.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/objects/Project.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <thread>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
template <class Condition>
void eventually(const Condition& condition, const char* message)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!condition()) {
        require(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::vector<ProjectResource> resources(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                       const std::string& kind, const std::optional<std::string>& viewer = std::nullopt)
{
    std::vector<ProjectResource> result;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        result = projectResources(store, txn, kind, viewer);
    });
    return result;
}

Json view(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& author)
{
    Json result;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) { result = projectView(store, txn, author); });
    return result;
}
} // namespace

int main() try
{
    TemporaryDirectory directory;
    const auto path = directory.path / "project.vcobj";

    // A new project names itself; any other store is an unnamed project.
    createProject(path, "PHerc", {{"open_data", {{"sample", "PHerc0139"}}}}, as("alice"));
    mustFail([&] { createProject(path, "again", Json::object(), as("alice")); });
    ObjectStoreSession session;
    const auto project = session.attach(path, as("alice"));
    session.read(project, [&](const auto& store, const auto& txn, const auto&) {
        const auto record = projectRecord(store, txn);
        require(record && record->name == "PHerc" &&
                Json::parse(record->meta).at("open_data").at("sample") == "PHerc0139", "project record");
    });
    setProjectName(session, project, "renamed");
    {
        auto plain = ObjectStore::create(directory.path / "plain.vcobj", as("alice"));
        auto txn = plain.read();
        require(!projectRecord(plain, txn), "a plain store has no project record");
    }
    const auto plain = session.attach(directory.path / "plain.vcobj", as("alice"));
    setProjectName(session, plain, "named later");
    session.read(plain, [&](const auto& store, const auto& txn, const auto&) {
        require(projectRecord(store, txn)->name == "named later", "naming creates the record");
    });

    // Resources: one per (kind, location), in the order added; removal retires.
    const std::string zarr = "s3://bucket/scan.zarr/", local = "/data/predictions.zarr";
    require(addProjectResource(session, project, kVolumeResource, zarr, {"vc-remote-auth:anonymous"}), "added");
    require(!addProjectResource(session, project, kVolumeResource, zarr, {"vc-remote-auth:anonymous"}), "unchanged");
    require(addProjectResource(session, project, kVolumeResource, local), "second volume");
    require(addProjectResource(session, project, kNormalGridResource, "/data/normals"), "normal grid");
    mustFail([&] { addProjectResource(session, project, "surface", zarr); });
    mustFail([&] { addProjectResource(session, project, kVolumeResource, ""); });
    auto volumes = resources(session, project, kVolumeResource);
    require(volumes.size() == 2 && volumes[0].location == zarr && volumes[1].location == local, "volumes in order");
    require(volumes[0].tags == std::vector<std::string>{"vc-remote-auth:anonymous"}, "tags kept");
    require(volumes[0].uuid == projectResourceUuid(kVolumeResource, zarr), "named UUID");
    require(resources(session, project, kNormalGridResource).size() == 1, "kinds are separate");
    updateProjectResourceMeta(session, project, volumes[0].uuid, {{"volume_id", "scan-L0"}});
    require(resources(session, project, kVolumeResource)[0].meta.at("volume_id") == "scan-L0", "meta merged");
    require(removeProjectResource(session, project, volumes[1].uuid), "removed");
    require(!removeProjectResource(session, project, volumes[1].uuid), "already removed");
    require(resources(session, project, kVolumeResource).size() == 1, "removed resources are not listed");
    require(addProjectResource(session, project, kVolumeResource, local), "re-added");
    require(resources(session, project, kVolumeResource).size() == 2, "re-added resource listed");

    // A resource of one owner (a path on their disk) is theirs alone, even at a path another uses.
    const std::string onDisk = "/home/user/predictions.zarr";
    require(addProjectResource(session, project, kVolumeResource, onDisk, {}, Json::object(), "alice"), "alice's own");
    require(addProjectResource(session, project, kVolumeResource, onDisk, {}, Json::object(), "bob"), "bob's own");
    require(projectResourceUuid(kVolumeResource, onDisk, "alice") != projectResourceUuid(kVolumeResource, onDisk, "bob"),
            "owners name their resources apart");
    const auto seenBy = [&](const std::string& viewer) {
        std::vector<std::string> owners;
        for (const auto& resource : resources(session, project, kVolumeResource, viewer))
            if (resource.location == onDisk) owners.push_back(resource.owner);
        return owners;
    };
    require(seenBy("alice") == std::vector<std::string>{"alice"} && seenBy("bob") == std::vector<std::string>{"bob"},
            "each owner sees only their own");
    require(seenBy("carol").empty() && resources(session, project, kVolumeResource, "carol").size() == 2,
            "others see the shared resources");
    require(removeProjectResource(session, project, projectResourceUuid(kVolumeResource, onDisk, "bob")), "bob removes his");
    require(seenBy("alice") == std::vector<std::string>{"alice"}, "removing one's own leaves the others'");

    // Each author has a view of their own.
    setProjectView(session, project, "alice", {{"current_volume", "scan-L0"}});
    setProjectView(session, project, "bob", {{"current_volume", "predictions"}});
    require(view(session, project, "alice").at("current_volume") == "scan-L0", "alice's view");
    require(view(session, project, "bob").at("current_volume") == "predictions", "bob's view");
    require(view(session, project, "carol").empty(), "no view yet");
    // A copy is a project of its own: same records, another store identity.
    copyProject(session, project, directory.path / "copy.vcobj", "copy", as("alice"));
    mustFail([&] { copyProject(session, project, directory.path / "copy.vcobj", "again", as("alice")); });
    const auto copy = session.attach(directory.path / "copy.vcobj", as("alice"));
    require(copy.storeUuid != project.storeUuid, "the copy has its own identity");
    require(resources(session, copy, kVolumeResource, "carol").size() == 2 && view(session, copy, "bob").at("current_volume") == "predictions",
            "the copy keeps the resources and views");
    session.read(copy, [&](const auto& store, const auto& txn, const auto&) {
        require(projectRecord(store, txn)->name == "copy", "the copy is renamed");
    });
    session.detach(copy);
    session.detach(plain);
    session.detach(project);

    // A hosted project shares its resources; each replica keeps its author's view.
    initHub(path, "curator");
    const auto alicePath = directory.path / "alice.vcobj", bobPath = directory.path / "bob.vcobj";
    cloneHub(path, alicePath, as("alice"));
    cloneHub(path, bobPath, as("bob"));
    HubSessionOptions fast;
    fast.syncInterval = std::chrono::milliseconds(100);
    fast.pushDelay = std::chrono::milliseconds(20);
    ObjectStoreSession aliceSession, bobSession;
    const auto alice = aliceSession.attach(alicePath, as("alice"), fast);
    const auto bob = bobSession.attach(bobPath, as("bob"), fast);
    const std::string shared = "https://example.org/shared.zarr/";
    require(addProjectResource(aliceSession, alice, kVolumeResource, shared), "alice adds a volume");
    setProjectView(aliceSession, alice, "alice", {{"current_volume", "shared"}});
    require(addProjectResource(aliceSession, alice, kVolumeResource, "/home/alice/scan.zarr", {}, Json::object(), "alice"),
            "alice adds a volume on her disk");
    aliceSession.syncHubAndWait(alice, std::chrono::seconds(20));
    bobSession.syncHubAndWait(bob, std::chrono::seconds(20));
    require(resources(bobSession, bob, kVolumeResource, "bob").size() == 3, "bob sees alice's shared volume, not her disk's");
    // Editing a pushed resource leases it.
    updateProjectResourceMeta(bobSession, bob, projectResourceUuid(kVolumeResource, shared), {{"volume_id", "shared"}});
    setProjectView(bobSession, bob, "bob", {{"current_volume", "shared"}});
    const auto pushed = bobSession.syncHubAndWait(bob, std::chrono::seconds(20));
    require(pushed && pushed->conflicts.empty() && pushed->error.empty(), "bob pushed");
    eventually([&] {
        aliceSession.syncHubAndWait(alice, std::chrono::seconds(20));
        const auto listed = resources(aliceSession, alice, kVolumeResource);
        return std::any_of(listed.begin(), listed.end(), [](const auto& r) { return r.meta.contains("volume_id") && r.meta.at("volume_id") == "shared"; });
    }, "alice sees bob's resource edit");
    require(view(aliceSession, alice, "alice").at("current_volume") == "shared" &&
            view(aliceSession, alice, "bob").at("current_volume") == "shared", "views synced without contention");
    // A copy of a hosted replica is standalone.
    copyProject(aliceSession, alice, directory.path / "standalone.vcobj", "standalone", as("alice"));
    {
        ObjectStoreSession standalone;
        const auto copied = standalone.attach(directory.path / "standalone.vcobj", as("alice"));
        require(!standalone.hubStatus(copied), "not a replica");
        require(resources(standalone, copied, kVolumeResource, "alice").size() == 5, "replica records copied");
    }
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
