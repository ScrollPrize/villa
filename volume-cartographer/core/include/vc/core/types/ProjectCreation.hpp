#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <vc_object_store/objects/ObjectStore.hpp>

namespace vc::project {

// The first store of a new project: one that exists, or one to create.
struct StoreChoice {
    enum class Mode { Existing, New };
    Mode mode = Mode::New;
    std::filesystem::path path;
    // Recorded on the project's reference to an existing store. A new store is
    // created writable and recorded writable.
    bool readOnly = false;
};

struct CreatedProject {
    std::filesystem::path projectPath;
    std::filesystem::path storePath;
    // The volume entry as the project records it, when one was given.
    std::optional<std::string> recordedVolume;
};

// Write a .vcproj referencing one store and, optionally, one volume location
// (local zarr or remote URL) with `tags`. The project is staged beside `path`
// and published atomically; `overwrite` replaces only an existing project file,
// never a store. An existing store is opened read-only to validate it; a new one
// is created with `storeOptions` (its author) and removed again when the project
// cannot be written. Never prompts and never opens the project.
CreatedProject createProject(const std::filesystem::path& path, const std::string& name, const StoreChoice& store,
                             const vc::objects::OpenOptions& storeOptions,
                             const std::optional<std::string>& volume = std::nullopt,
                             std::vector<std::string> tags = {}, bool overwrite = false);

}  // namespace vc::project
