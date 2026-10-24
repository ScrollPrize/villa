#pragma once

#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
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
    std::unique_ptr<QuadSurface> load(const std::string& uuid, bool applyValidity = true);
    // A new surface object holding `surface`. Its meta keeps the surface's meta
    // (e.g. area) except tifxyz file keys; meta.provenance records the tool, its
    // parameters and the source objects; meta.coordinates is taken from the first
    // source unless given.
    std::string create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                       const std::vector<std::string>& sources = {}, const std::vector<std::string>& tags = {},
                       const std::optional<nlohmann::json>& coordinates = std::nullopt);
    // Shared non-geometry files of an object (OBJ exports, models).
    std::filesystem::path artifactDirectory(const std::string& uuid);
    // Wait until this tool's writes reached the hub (no-op for local stores).
    void finish(std::chrono::seconds timeout = std::chrono::seconds(120));

    const std::shared_ptr<vc::objects::ObjectStoreSession>& session() const { return session_; }
    const vc::objects::AttachmentIdentity& attachment() const { return attachment_; }

private:
    std::string tool_;
    std::shared_ptr<vc::objects::ObjectStoreSession> session_;
    vc::objects::AttachmentIdentity attachment_;
    std::optional<std::chrono::system_clock::time_point> lastWrite_;
};
