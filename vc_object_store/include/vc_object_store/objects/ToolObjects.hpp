#pragma once
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
// A command-line tool's objects in one store: the revisions it read (and the
// frame it read them in), the coordinates of what it creates, the pins of
// what it writes back, and the wait for its writes to reach a hosted store's
// hub. Several tools may use the same replica at once; reads may be noted
// from several threads.
class ToolObjects {
public:
    ToolObjects(const std::filesystem::path& store, const std::string& tool);

    const std::string& tool() const { return tool_; }
    const std::shared_ptr<ObjectStoreSession>& session() const { return session_; }
    const AttachmentIdentity& attachment() const { return attachment_; }

    // The tool read `uuid` at these revisions: in stored coordinates
    // (`volume` unset), or through `volume`'s view of it, where `identity`
    // says whether that view maps stored coordinates unchanged.
    void noteRead(const std::string& uuid, const std::string& objectRevision, const std::string& geometryRevision,
                  const std::optional<std::string>& volume, bool identity);
    // The coordinates of a new object made from `sources`: bound to `volume`'s
    // voxels when given, else to the frame the sources were read in (the
    // volume they were read through, else the first source's binding).
    // Sources read in different frames throw.
    CoordinateJson coordinatesFor(const std::vector<std::string>& sources, const std::optional<std::string>& volume);
    // The pins (object and geometry revision) of writing back `uuid`: one this
    // tool read in stored units (writing a mapped grid back would rewrite every
    // stored coordinate). Throws otherwise.
    struct Pins {
        std::string objectRevision, geometryRevision;
    };
    Pins writable(const std::string& uuid);
    // The tool wrote to the store (finish waits for the hub then).
    void wrote() { wrote_ = true; }
    // Drain writes and, after any, wait until they reached the hub (no-op for
    // local stores); throws when they did not.
    void finish(std::chrono::seconds timeout = std::chrono::seconds(120));

private:
    struct Read {
        std::string objectRevision, geometryRevision;
        std::optional<std::string> volume;
        bool identity = true;
    };
    std::string tool_;
    std::shared_ptr<ObjectStoreSession> session_;
    AttachmentIdentity attachment_;
    bool wrote_ = false;
    std::mutex mutex_;
    std::map<std::string, Read> read_;
};
}
