#include "vc/core/types/ProjectCreation.hpp"

#include "vc/core/types/ProjectDocument.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <system_error>

namespace vc::project {

CreatedProject createProject(const std::filesystem::path& path, const std::string& name, const StoreChoice& store,
                             const vc::objects::OpenOptions& storeOptions, const std::optional<std::string>& volume,
                             std::vector<std::string> tags, bool overwrite)
{
    namespace fs = std::filesystem;
    const auto projectPath = fs::absolute(path).lexically_normal();
    const auto storePath = fs::absolute(store.path).lexically_normal();
    auto extension = projectPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    if (extension != ".vcproj") throw std::invalid_argument("A project file must end in .vcproj: " + projectPath.string());
    std::error_code sameError;
    if (storePath == projectPath || fs::equivalent(storePath, projectPath, sameError))
        throw std::invalid_argument("The project and object store must be separate files");
    if (fs::exists(projectPath) && !overwrite) throw std::runtime_error(projectPath.string() + " already exists");

    const auto staging = projectPath.parent_path() /
        ("." + projectPath.stem().string() + "-" + vc::objects::util::newUuid() + ".vcproj");
    bool createdStore = false;
    CreatedProject result{projectPath, storePath, std::nullopt};
    try {
        fs::create_directories(projectPath.parent_path());
        if (store.mode == StoreChoice::Mode::New) {
            auto options = storeOptions;
            options.readOnly = false;
            auto created = vc::objects::ObjectStore::create(storePath, options);
            createdStore = true;
        } else {
            auto options = storeOptions;
            options.readOnly = true;
            auto existing = vc::objects::ObjectStore::open(storePath, options);
        }
        const bool readOnly = store.mode == StoreChoice::Mode::Existing && store.readOnly;
        auto document = ProjectDocument::create(staging, name, storePath, readOnly);
        auto data = document.data();
        if (volume) {
            LoadOptions loadOptions;
            loadOptions.deferResolution = true;
            auto package = VolumePkg::open(staging, loadOptions);
            package->addVolumeEntry(*volume, std::move(tags));
            result.recordedVolume = package->volumeEntries().front().location;
            data = package->document().data();
        }
        vcr::zarr::atomicWrite(projectPath, data.dump(2) + "\n", vcr::zarr::Durability::Fsync);
        fs::remove(staging);
    } catch (...) {
        std::error_code ignored;
        fs::remove(staging, ignored);
        if (createdStore) fs::remove(storePath, ignored);
        throw;
    }
    return result;
}

}  // namespace vc::project
