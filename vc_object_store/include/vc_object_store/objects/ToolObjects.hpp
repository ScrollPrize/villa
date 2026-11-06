#pragma once
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
// A command-line tool's objects in one store: the revisions it read (and the
// frame it read them in), the coordinates of what it creates and the pins of
// what it writes back. Several tools may use the same store at once; reads
// may be noted from several threads.
class ToolObjects {
public:
    ToolObjects(const std::filesystem::path& store, const std::string& tool);

    const std::string& tool() const { return tool_; }
    const std::shared_ptr<ObjectStoreSession>& session() const { return session_; }
    const AttachmentIdentity& attachment() const { return attachment_; }

    // The tool read `pin`: in stored coordinates (`volume` unset), or through
    // the view of it in `volume`, where `identity` says whether that view maps
    // stored coordinates unchanged.
    void noteRead(const PinnedObject& pin, const std::optional<VolumeVoxels>& volume, bool identity);
    // The coordinates of a new object made from `sources`: bound to `volume`
    // when given, else to the frame the sources were read in (the voxels they
    // were read through, else the first source's binding). Sources read in
    // different frames throw.
    CoordinateJson coordinatesFor(const std::vector<std::string>& sources, const std::optional<VolumeVoxels>& volume);
    // The pin (version and geometry) of writing back `uuid`: one this tool
    // read in stored units (writing a mapped grid back would rewrite every
    // stored coordinate). Throws otherwise.
    PinnedObject writable(const std::string& uuid);
    // Publish local files (name, path) as artifacts of `uuid`
    // (ObjectStoreSession::publishArtifacts), expecting the version this tool
    // read it at, else its current version (an object the tool created).
    TransactionReceipt publishArtifacts(const std::string& uuid,
                                        const std::vector<std::pair<std::string, std::filesystem::path>>& files);
    // Drain accepted writes.
    void finish();

private:
    struct Read {
        PinnedObject pin;
        std::optional<VolumeVoxels> volume;
        bool identity = true;
    };
    std::string tool_;
    std::shared_ptr<ObjectStoreSession> session_;
    AttachmentIdentity attachment_;
    std::mutex mutex_;
    std::map<std::string, Read> read_;
};
}
