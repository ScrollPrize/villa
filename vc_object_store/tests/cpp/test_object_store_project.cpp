#include "TestSupport.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/Project.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
OpenOptions as(const std::string& author)
{
    OpenOptions options;
    options.author = author;
    return options;
}

std::vector<ProjectResource> resources(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                       const std::string& kind, const std::optional<std::string>& viewer = std::nullopt)
{
    std::vector<ProjectResource> result;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
        result = projectResources(reader, kind, viewer);
    });
    return result;
}

Json view(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& author)
{
    Json result;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) { result = projectView(reader, author); });
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
    session.readStore(project, [&](const vc::objects::StoreReader& reader, const auto&) {
        const auto record = projectRecord(reader);
        require(record && record->name == "PHerc" &&
                Json::parse(record->meta).at("open_data").at("sample") == "PHerc0139", "project record");
    });
    setProjectName(session, project, "renamed");
    {
        auto plain = ObjectStore::create(directory.path / "plain.vcobj", as("alice"));
        auto txn = plain.read();
        require(!projectRecord(LocalStoreReader(plain, txn)), "a plain store has no project record");
    }
    const auto plain = session.attach(directory.path / "plain.vcobj", as("alice"));
    setProjectName(session, plain, "named later");
    session.readStore(plain, [&](const vc::objects::StoreReader& reader, const auto&) {
        require(projectRecord(reader)->name == "named later", "naming creates the record");
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
    session.readStore(copy, [&](const vc::objects::StoreReader& reader, const auto&) {
        require(projectRecord(reader)->name == "copy", "the copy is renamed");
    });
    session.detach(copy);
    session.detach(plain);
    session.detach(project);

    // Several local writers share one project; each keeps their own view.
    {
        ObjectStoreSession aliceSession, bobSession;
        const auto alice = aliceSession.attach(path, as("alice"));
        const auto bob = bobSession.attach(path, as("bob"));
        const std::string shared = "https://example.org/shared.zarr/";
        require(addProjectResource(aliceSession, alice, kVolumeResource, shared), "alice adds a volume");
        setProjectView(aliceSession, alice, "alice", {{"current_volume", "shared"}});
        require(addProjectResource(aliceSession, alice, kVolumeResource, "/home/alice/scan.zarr", {}, Json::object(), "alice"),
                "alice adds a volume on her disk");
        require(resources(bobSession, bob, kVolumeResource, "bob").size() == 3, "bob sees alice's shared volume, not her disk's");
        updateProjectResourceMeta(bobSession, bob, projectResourceUuid(kVolumeResource, shared), {{"volume_id", "shared"}});
        setProjectView(bobSession, bob, "bob", {{"current_volume", "shared"}});
        const auto listed = resources(aliceSession, alice, kVolumeResource);
        require(std::any_of(listed.begin(), listed.end(), [](const auto& r) {
                    return r.meta.contains("volume_id") && r.meta.at("volume_id") == "shared"; }),
                "alice sees bob's resource edit");
        require(view(aliceSession, alice, "alice").at("current_volume") == "shared" &&
                view(aliceSession, alice, "bob").at("current_volume") == "shared", "views kept per author");
        // A copy carries the project's artifacts.
        const auto file = directory.path / "notes.txt";
        std::ofstream(file) << "project notes";
        std::string recordUuid;
        int64_t version = 0;
        aliceSession.readStore(alice, [&](const vc::objects::StoreReader& reader, const auto&) {
            recordUuid = projectRecord(reader)->uuid;
            version = reader.currentObject(recordUuid)->version;
        });
        aliceSession.publishArtifacts(alice, recordUuid, version, {{"notes.txt", file}});
        const auto copyPath = directory.path / "with-artifacts.vcobj";
        copyProject(aliceSession, alice, copyPath, "with artifacts", as("alice"));
        ObjectStoreSession copied;
        const auto copy = copied.attach(copyPath, as("alice"));
        require(resources(copied, copy, kVolumeResource, "alice").size() == 5, "records copied");
        const auto artifacts = copied.artifacts(copy, recordUuid);
        require(artifacts.size() == 1 && std::filesystem::is_regular_file(artifactBlobPath(copyPath, artifacts.front().sha256)),
                "artifact bytes copied");
    }
    std::cout << "ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
}
