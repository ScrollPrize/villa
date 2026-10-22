#include "geometry/GeometrySnapshot.hpp"
#include "geometry/SourceMetadata.hpp"
#include "vc_object_store/objects/GeometryExport.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/GeometryOutput.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <climits>
#include <type_traits>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#ifdef __linux__
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/fs.h>
#endif

namespace vc::objects {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Snapshot = detail::GeometrySnapshot;
struct Stage {
    fs::path root;
    ~Stage() { std::error_code ignored; fs::remove_all(root, ignored); }
};
void publish(const fs::path& source, const fs::path& destination, bool overwrite, bool directory)
{
    const auto status = fs::symlink_status(destination);
    const bool exists = fs::exists(status);
    if (exists && !overwrite) throw std::runtime_error("destination already exists (use overwrite): " + destination.string());
    if (exists && (fs::is_symlink(status) || (directory ? !fs::is_directory(status) : !fs::is_regular_file(status))))
        throw std::runtime_error("destination must have the same file/directory type and cannot be a symlink");
#ifdef __linux__
    // NOREPLACE closes the check/rename race; EXCHANGE replaces nonempty dirs
    // without a missing-path window. The old destination is cleaned by Stage.
    const unsigned flags = exists ? RENAME_EXCHANGE : RENAME_NOREPLACE;
    if (syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD, destination.c_str(), flags) != 0)
        throw std::system_error(errno, std::generic_category(), "atomic geometry export publication");
#else
    if (directory) throw std::runtime_error("atomic directory export requires Linux renameat2 support");
    if (overwrite) fs::rename(source, destination);
    else fs::create_hard_link(source, destination);
#endif
}
} // namespace

std::string exportGeometry(ObjectStore& db, const std::string& objectUuid, const fs::path& destination,
    const std::optional<std::string>& revisionUuid, bool overwrite)
{
    if (destination.empty()) throw std::invalid_argument("export destination is empty");
    const auto dest = fs::absolute(destination).lexically_normal();
    const auto resolved = fs::weakly_canonical(dest);
    const auto database = fs::weakly_canonical(db.path());
    // Never replace the open database, a hard link to it, its sidecars, or an
    // ancestor directory containing it, even with explicit overwrite.
    for (auto p = database; !p.empty(); p = p.parent_path()) {
        if (resolved == p) throw std::runtime_error("export destination contains the open object database");
        if (p == p.root_path()) break;
    }
    if (resolved == database.string() + "-wal" || resolved == database.string() + "-shm" || resolved == database.string() + "-journal" ||
        (fs::exists(dest) && fs::equivalent(dest, database)))
        throw std::runtime_error("cannot export over the open object database or its sidecars");
    if (!fs::is_directory(dest.parent_path())) throw std::runtime_error("export parent directory must exist");
    Snapshot snapshot;
    Json attributes;
    {
        auto read = db.read();
        const auto object = db.objectByUuid(read, objectUuid);
        if (!object) throw std::runtime_error("object not found");
        const auto revision = revisionUuid ? db.geometryRevision(read, *revisionUuid) : db.currentGeometry(read, objectUuid);
        if (!revision) throw std::runtime_error("geometry revision not found");
        if (revision->objectUuid != object->uuid) throw std::runtime_error("geometry revision belongs to another object");
        snapshot.revision = *revision;
        attributes = detail::nativeMetadata(db, read, *object, *revision);
        if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz")
            throw std::runtime_error("unsupported geometry format for native export");
        const std::set<std::string> names = revision->kind == "fiber" ? std::set<std::string>{"control_points", "line_points", "control_point_ids"} :
            revision->kind == "point_collection" ? std::set<std::string>{"points", "point_ids"} : std::set<std::string>{"points"};
        for (const auto& array : revision->arrays) {
            if (!names.contains(array.name)) throw std::runtime_error("native export cannot represent auxiliary array: " + array.name);
            snapshot.arrays[array.name] = db.readGeometryArray(read, revision->uuid, array.name);
        }
    }
    const auto& kind = snapshot.revision.kind;
    const bool directory = kind == "patch" || kind == "track";
    const auto stagingPath = dest.parent_path() / (".vc-export-" + util::newUuid());
    if (!fs::create_directory(stagingPath)) throw std::runtime_error("cannot create export staging directory");
    Stage stage{stagingPath};
    const auto output = stage.root / "payload";
    if (directory) fs::create_directory(output);
    if (kind == "fiber") formats::fiber(snapshot, attributes, output);
    else if (kind == "point_collection") formats::collection(snapshot, attributes, output);
    else if (kind == "patch") formats::patch(snapshot, attributes, output);
    else if (kind == "track") formats::track(snapshot, attributes, output);
    else throw std::runtime_error("unsupported geometry kind for export");
    publish(output, dest, overwrite, directory);
    return snapshot.revision.uuid;
}
} // namespace vc::objects
