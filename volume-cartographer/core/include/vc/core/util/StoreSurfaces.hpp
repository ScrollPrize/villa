#pragma once

#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Store-mode output options of tools with hand-written argument parsing:
// --store=PATH, --name=NAME, --tag=TAG (repeatable), --volume-uuid=ID.
struct StoreOutputArgs {
    std::filesystem::path store;
    std::string name;
    std::vector<std::string> tags;
    std::string volume;
    // Consumes `arg` when it is one of the options above.
    bool parse(const std::string& arg);
    // meta.coordinates for outputs: bound to `volume` at its voxel scale.
    std::optional<nlohmann::json> coordinates() const;
};

// The coordinates of a tool's output in `volume`'s voxels (scale 1); none when
// no volume is named.
std::optional<nlohmann::json> volumeCoordinates(const std::string& volume);

// Surface objects for command-line tools: `--store PATH` with `--surface UUID`
// inputs and new surface outputs that record their provenance. A hosted
// replica syncs in the background and finish() waits until the tool's writes
// reached the hub. Several tools may use the same replica at once.
class StoreSurfaces {
public:
    StoreSurfaces(const std::filesystem::path& store, const std::string& tool);
    ~StoreSurfaces();
    StoreSurfaces(const StoreSurfaces&) = delete;
    StoreSurfaces& operator=(const StoreSurfaces&) = delete;

    // The surface's current geometry; channels load on demand from that revision.
    // With `volume`, in that volume's voxels through the store's transformed view
    // (bound to it, or registered to it); otherwise in stored units. Only a surface
    // loaded in stored units, or whose mapping to the volume is the identity, can
    // be written back.
    std::unique_ptr<QuadSurface> load(const std::string& uuid, bool applyValidity = true,
                                      const std::optional<std::string>& volume = std::nullopt);
    // A new surface object holding `surface`. Its meta keeps the surface's meta
    // (e.g. area) except tifxyz file keys and status tags (`tags` are the record's); meta.provenance records the tool, its
    // parameters and the source objects; meta.coordinates is taken from the first
    // source unless given.
    std::string create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                       const std::vector<std::string>& sources = {}, const std::vector<std::string>& tags = {},
                       const std::optional<nlohmann::json>& coordinates = std::nullopt);
    // Publish the edited vertices of a surface this tool loaded, checked against
    // the revision it loaded; relationships are kept as recorded. The grid must
    // keep its size. Returns the new geometry revision.
    std::string update(const std::string& uuid, QuadSurface& surface);
    // Replace the named channels ("mask" is validity) of a surface this tool
    // loaded with the surface's, the same way. Returns the new geometry revision.
    std::string updateChannels(const std::string& uuid, QuadSurface& surface, const std::vector<std::string>& names);
    // Wait until this tool's writes reached the hub (no-op for local stores).
    void finish(std::chrono::seconds timeout = std::chrono::seconds(120));

    const std::shared_ptr<vc::objects::ObjectStoreSession>& session() const { return session_; }
    const vc::objects::AttachmentIdentity& attachment() const { return attachment_; }

private:
    std::string tool_;
    std::shared_ptr<vc::objects::ObjectStoreSession> session_;
    vc::objects::AttachmentIdentity attachment_;
    std::optional<std::chrono::system_clock::time_point> lastWrite_;
    // Object and geometry revisions of each loaded surface, the pins of updates.
    // Loads may run on several threads.
    std::mutex loadedMutex_;
    struct Loaded {
        std::string objectRevision, geometryRevision;
        bool storedUnits = true;
    };
    std::map<std::string, Loaded> loaded_;
    // The pins of a surface this tool loaded in stored units; throws otherwise.
    Loaded writable(const std::string& uuid);
};
