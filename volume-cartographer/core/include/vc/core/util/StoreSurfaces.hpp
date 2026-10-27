#pragma once

#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>
#include <vc_object_store/objects/ToolObjects.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Store-mode output options of tools with hand-written argument parsing:
// --store=PATH, --name=NAME, --tag=TAG (repeatable), --volume-uuid=ID.
struct StoreOutputArgs {
    std::filesystem::path store;
    std::string name;
    std::vector<std::string> tags;
    std::optional<std::string> volume;
    // Consumes `arg` when it is one of the options above.
    bool parse(const std::string& arg);
};

// Surface objects for command-line tools as QuadSurfaces: `--store PATH` with
// `--surface UUID` inputs and new surface outputs that record their
// provenance. The store's ToolObjects keeps what was read, decides frames and
// pins, and waits for the hub (objects().finish()).
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
    // parameters and the source objects. It is bound to `volume`'s voxels when
    // given, else to the frame the sources were loaded in: the volume they were
    // read through, else the first source's binding. Sources read in different
    // frames throw.
    std::string create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                       const std::vector<std::string>& sources = {}, const std::vector<std::string>& tags = {},
                       const std::optional<std::string>& volume = std::nullopt);
    // Publish the edited vertices of a surface this tool loaded, checked against
    // the revision it loaded; relationships are kept as recorded. The grid must
    // keep its size. Returns the new geometry revision.
    std::string update(const std::string& uuid, QuadSurface& surface);

    vc::objects::ToolObjects& objects() { return objects_; }

private:
    vc::objects::ToolObjects objects_;
};
