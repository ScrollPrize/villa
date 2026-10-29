#pragma once
// Projects for tests: a .vcproj file and a separate object store,
// in a directory of this process that is removed when it exits.

#include "vc/core/types/VolumePkg.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/Project.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

namespace vc::test {

struct TestProjectStore {
    std::filesystem::path path;
    std::filesystem::path storePath;
    std::shared_ptr<vc::objects::ObjectStoreSession> session;
    vc::objects::AttachmentIdentity attachment;
    std::string author = "vc-test";

    // The project opened (again) from its document, as reopening it would.
    [[nodiscard]] std::shared_ptr<VolumePkg> open(const vc::project::LoadOptions& opts = {}) const
    {
        return VolumePkg::open(path, opts);
    }
};

inline std::filesystem::path testProjectDirectory()
{
    struct Directory {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("vc_test_projects_" + vc::objects::util::newUuid());
        Directory() { std::filesystem::create_directories(path); }
        ~Directory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    };
    static const Directory directory;
    return directory.path;
}

// A new project and store at `path` (by default fresh files in the test directory).
inline TestProjectStore createTestProject(std::filesystem::path path = {}, const std::string& name = "test")
{
    if (path.empty()) path = testProjectDirectory() / (vc::objects::util::newUuid() + ".vcobj");
    TestProjectStore store;
    store.storePath = path;
    store.storePath.replace_extension(".vcobj");
    path.replace_extension(".vcproj");
    store.path = path;
    vc::objects::OpenOptions options;
    options.author = store.author;
    { auto database = vc::objects::ObjectStore::create(store.storePath, options); }
    vc::project::ProjectDocument::create(path, name, store.storePath);
    store.session = std::make_shared<vc::objects::ObjectStoreSession>();
    store.attachment = store.session->attach(store.storePath, options);
    return store;
}

// A new, empty project.
inline std::shared_ptr<VolumePkg> emptyProject(const vc::project::LoadOptions& opts = {})
{
    return createTestProject().open(opts);
}

} // namespace vc::test
