#include "vc_object_store/objects/StoreReader.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/SourceMetadata.hpp"
#include "vc_object_store/objects/GeometryExport.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/GeometryOutput.hpp"
#include <vc_reader/publication.hpp>
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
    if (fs::is_directory(source) != directory) throw std::runtime_error("geometry export staging type mismatch");
    vcr::files::publishPath(source, destination, overwrite);
}
} // namespace

ObjectVersion exportGeometry(ObjectStore& db, const std::string& objectUuid, const fs::path& destination,
    const std::optional<ObjectVersion>& version, bool overwrite, bool validOnly)
{
    const auto read = db.read();
    return exportGeometry(db, read, objectUuid, destination, version, overwrite, validOnly);
}

ObjectVersion exportGeometry(const ObjectStore& db, const ReadTxn& read, const std::string& objectUuid,
    const fs::path& destination, const std::optional<ObjectVersion>& version, bool overwrite, bool validOnly)
{
    return exportGeometry(LocalStoreReader(db, read), objectUuid, destination, version, overwrite, validOnly);
}

ObjectVersion exportGeometry(const StoreReader& reader, const std::string& objectUuid,
    const fs::path& destination, const std::optional<ObjectVersion>& version, bool overwrite, bool validOnly)
{
    if (destination.empty()) throw std::invalid_argument("export destination is empty");
    const auto dest = fs::absolute(destination).lexically_normal();
    reader.validateExportDestination(dest);
    if (!fs::is_directory(dest.parent_path())) throw std::runtime_error("export parent directory must exist");
    Snapshot snapshot;
    ObjectVersion exported;
    Json attributes;
    std::string fiberJson;
    {
        if (version && util::canonicalUuid(version->objectUuid) != util::canonicalUuid(objectUuid))
            throw std::invalid_argument("export version names another object");
        auto object = reader.selectedObject(objectUuid, {}, {}, version);
        std::vector<ObjectVersion> selection;
        if (object && (object->record.kind == "fiber" || object->record.kind == "point_collection")) {
            if (version) {
                const auto graph = reader.graphReadViewAround(reader.versionsAt(*version));
                for (const auto& selected : *graph.selectedObjects) selection.push_back(selected.reference());
            }
            else {
                ObjectSelection query; query.includeRetired = true;
                for (const auto& selected : reader.selectObjects(query)) selection.push_back(selected.reference());
                const auto root = std::find_if(selection.begin(), selection.end(), [&](const auto& ref) { return ref.objectUuid == object->objectUuid; });
                object = root == selection.end() ? std::nullopt : reader.selectedObject(objectUuid, {}, {}, *root);
            }
        }
        if (!object || object->objectUuid != util::canonicalUuid(objectUuid) || object->deleted || object->record.retired)
            throw std::runtime_error("active object version not found for this object");
        exported = object->reference();
        if (!object->geometryRevision) throw std::runtime_error("object revision has no geometry");
        const auto revision = reader.geometryRevision(*object->geometryRevision);
        if (!revision) throw std::runtime_error("geometry revision not found");
        snapshot.revision = *revision;
        if (revision->kind == "fiber") {
            std::vector<ObjectVersion> fibers;
            for (const auto& selected : reader.objectsAtVersions(selection))
                if (selected && selected->record.kind == "fiber") fibers.push_back(selected->reference());
            fiberJson = fiberNetworkJson(reader, fibers, {object->objectUuid},
                                         std::numeric_limits<uint64_t>::max()).at(object->objectUuid);
        } else attributes = detail::nativeMetadata(reader, object->record, *revision, selection);
        if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz")
            throw std::runtime_error("unsupported geometry format for native export");
        const std::set<std::string> names = revision->kind == "fiber" ? std::set<std::string>{"control_points", "line_points", "control_point_ids"} :
            revision->kind == "point_collection" ? std::set<std::string>{"points", "point_ids"} : std::set<std::string>{"points"};
        if (revision->kind != "fiber") for (const auto& array : revision->arrays) {
            // tifxyz carries surface channels as sidecar TIFFs.
            if (!names.contains(array.name) && revision->kind != "surface")
                throw std::runtime_error("native export cannot represent auxiliary array: " + array.name);
            snapshot.arrays[array.name] = reader.readGeometryArray(revision->uuid, array.name);
        }
    }
    const auto& kind = snapshot.revision.kind;
    const bool directory = kind == "surface" || kind == "track";
    const auto stagingPath = dest.parent_path() / (".vc-export-" + util::newUuid());
    if (!fs::create_directory(stagingPath)) throw std::runtime_error("cannot create export staging directory");
    Stage stage{stagingPath};
    const auto output = stage.root / "payload";
    if (directory) fs::create_directory(output);
    if (kind == "fiber") {
        std::ofstream file(output);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << fiberJson << '\n';
        file.close();
    }
    else if (kind == "point_collection") formats::collection(snapshot, attributes, output);
    else if (kind == "surface") formats::surface(snapshot, attributes, output, validOnly);
    else if (kind == "track") formats::track(snapshot, attributes, output);
    else if (kind == "umbilicus") formats::umbilicus(snapshot, attributes, output);
    else throw std::runtime_error("unsupported geometry kind for export");
    publish(output, dest, overwrite, directory);
    return exported;
}
} // namespace vc::objects
