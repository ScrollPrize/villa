#include "geometry/GeometrySnapshot.hpp"
#include "vc/core/objects/GeometryExport.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/Uuid.hpp"
#include "formats/TrackInput.hpp"
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
void writeBytes(const fs::path& path, const void* data, size_t size)
{
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path, std::ios::binary);
    if (size) output.write(static_cast<const char*>(data), size);
    output.close();
}
void writeJson(const fs::path& path, const Json& data)
{
    const auto text = data.dump(2) + '\n';
    writeBytes(path, text.data(), text.size());
}
template<class T> void writeArray(const fs::path& path, std::vector<T> values)
{
    auto* bytes = reinterpret_cast<uint8_t*>(values.data());
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < values.size() * sizeof(T); i += sizeof(T))
            std::reverse(bytes + i, bytes + i + sizeof(T));
    writeBytes(path, bytes, values.size() * sizeof(T));
}
Json pointAttributes(Json& attributes, const std::string& key, size_t count)
{
    if (!attributes.contains(key)) return Json();
    auto result = attributes.at(key);
    if (!result.is_array() || result.size() != count)
        throw std::runtime_error(key + " must have one entry per point; update annotations after resizing geometry");
    for (const auto& item : result)
        if (!item.is_null() && !item.is_object()) throw std::runtime_error("invalid " + key);
    attributes.erase(key);
    return result;
}
void fiber(const Snapshot& s, Json attributes, const fs::path& path)
{
    attributes["uuid"] = s.revision.objectUuid;
    attributes["type"] = "vc3d_fiber";
    for (const auto* name : {"control_points", "line_points"}) {
        const auto values = s.values<double>(name);
        const auto extras = pointAttributes(attributes, std::string(name) + "_attributes", values.size() / 3);
        auto points = Json::array();
        for (size_t i = 0; i < values.size() / 3; ++i) {
            Json xyz = {values[3*i], values[3*i+1], values[3*i+2]};
            if ((!extras.is_null() && extras[i].is_object()) ||
                (std::string_view(name) == "control_points" && attributes.value("version", 1) >= 3)) {
                auto entry = !extras.is_null() && extras[i].is_object() ? extras[i] : Json::object(); entry["position"] = std::move(xyz); points.push_back(std::move(entry));
            } else points.push_back(std::move(xyz));
        }
        attributes[name] = std::move(points);
    }
    writeJson(path, attributes);
}
void collection(const Snapshot& s, Json attributes, const fs::path& path)
{
    const auto xyz = s.values<double>("points");
    const auto ids = s.values<int64_t>("point_ids");
    const auto extras = pointAttributes(attributes, "point_attributes", ids.size());
    const auto collectionId = attributes.value("source_collection_id", std::string("1"));
    attributes.erase("source_collection_id");
    attributes["uuid"] = s.revision.objectUuid;
    auto points = Json::object();
    std::set<int64_t> seen;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || !seen.insert(ids[i]).second) throw std::runtime_error("point IDs must be nonnegative and unique");
        auto entry = extras.is_null() || extras[i].is_null() ? Json::object() : extras[i];
        entry["p"] = {xyz[3*i], xyz[3*i+1], xyz[3*i+2]};
        points[std::to_string(ids[i])] = std::move(entry);
    }
    attributes["points"] = std::move(points);
    writeJson(path, {{"vc_pointcollections_json_version", "1"}, {"collections", {{collectionId, attributes}}}});
}
void patch(const Snapshot& s, Json attributes, const fs::path& path)
{
    const auto descriptor = std::find_if(s.revision.arrays.begin(), s.revision.arrays.end(), [](const auto& a) { return a.name == "points"; });
    const auto& shape = descriptor->shape;
    if (shape[0] <= 0 || shape[1] <= 0 || shape[0] > INT_MAX || shape[1] > INT_MAX)
        throw std::runtime_error("native patch export requires nonempty dimensions within int32 range");
    const auto xyz = s.values<float>("points");
    std::array<float, 3> low{-1,-1,-1}, high = low;
    bool any = false;
    for (size_t i = 0; i < xyz.size(); i += 3) {
        if (xyz[i+2] <= 0) {
            if (xyz[i] != -1 || xyz[i+1] != -1 || xyz[i+2] != -1)
                throw std::runtime_error("patch points with z <= 0 must use the (-1,-1,-1) invalid sentinel");
            continue;
        }
        for (int a = 0; a < 3; ++a) {
            low[a] = any ? std::min(low[a], xyz[i+a]) : xyz[i+a];
            high[a] = any ? std::max(high[a], xyz[i+a]) : xyz[i+a];
        }
        any = true;
    }
    for (int axis = 0; axis < 3; ++axis) {
        cv::Mat_<float> plane(static_cast<int>(shape[0]), static_cast<int>(shape[1]));
        for (size_t i = 0; i < plane.total(); ++i) plane.ptr<float>()[i] = xyz[3*i+axis];
        const auto file = path / (std::string(1, "xyz"[axis]) + ".tif");
        writeTiff(file, plane);
        if (!isReadableTiff(file)) throw std::runtime_error("failed to write complete TIFF: " + file.string());
    }
    attributes["uuid"] = s.revision.objectUuid;
    attributes["type"] = "seg"; attributes["format"] = "tifxyz";
    attributes["tiff_dimensions"] = {shape[1], shape[0]};
    attributes["bbox"] = {low, high};
    writeJson(path / "meta.json", attributes);
}
void track(const Snapshot& s, const Json& attributes, const fs::path& path)
{
    auto xyz = s.values<int32_t>("points");
    const auto count = xyz.size() / 3;
    const auto id = detail::trackSourceId(attributes.at("external_key").get<std::string>());
    const auto family = attributes.value("family", std::string());
    if (!family.empty() && family != "horizontal" && family != "vertical") throw std::runtime_error("invalid track family");
    double length = 0, chord = 0;
    int32_t zmin = 0, zmax = 0;
    if (count) zmin = zmax = xyz[2];
    for (size_t i = 0; i < count; ++i) {
        zmin = std::min(zmin, xyz[3*i+2]); zmax = std::max(zmax, xyz[3*i+2]);
        if (i) {
            double squared = 0;
            for (int a = 0; a < 3; ++a) { const double d = double(xyz[3*i+a]) - xyz[3*(i-1)+a]; squared += d*d; }
            length += std::sqrt(squared);
        }
    }
    if (count) for (int a = 0; a < 3; ++a) { const double d = double(xyz[3*(count-1)+a]) - xyz[a]; chord += d*d; }
    chord = std::sqrt(chord);
    for (size_t i = 0; i < count; ++i) std::swap(xyz[3*i], xyz[3*i+2]);
    writeArray(path / "coordinates.i32", std::move(xyz));
    writeArray<int64_t>(path / "offsets.i64", {0, static_cast<int64_t>(count)});
    writeArray<uint64_t>(path / "source_ids.u64", {id});
    writeArray<int8_t>(path / "family_codes.i8", {static_cast<int8_t>(family == "horizontal" ? 0 : family == "vertical" ? 1 : -1)});
    writeArray<int32_t>(path / "z_bounds.i32", {zmin, zmax});
    writeArray<double>(path / "arclengths.f64", {length});
    writeArray<double>(path / "tortuosities.f64", {chord > 0 ? length / chord : std::numeric_limits<double>::infinity()});
    std::vector<uint8_t> header(64);
    std::memcpy(header.data(), "VCTRK01", 7);
    auto field = [&](size_t offset, uint64_t value, int size) { for (int i = 0; i < size; ++i) header[offset+i] = static_cast<uint8_t>(value >> (8*i)); };
    field(8, 1, 4); field(12, 64, 4); field(16, 1, 8); field(24, count, 8);
    writeArray(path / "header.bin", std::move(header));
    writeJson(path / "metadata.json", {{"version", 1}, {"track_count", 1}, {"point_count", count},
        {"graph_object_uuid", s.revision.objectUuid}, {"graph_revision_uuid", s.revision.uuid}});
}
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
        if (resolved == p) throw std::runtime_error("export destination contains the open graph database");
        if (p == p.root_path()) break;
    }
    if (resolved == database.string() + "-wal" || resolved == database.string() + "-shm" || resolved == database.string() + "-journal" ||
        (fs::exists(dest) && fs::equivalent(dest, database)))
        throw std::runtime_error("cannot export over the open graph database or its sidecars");
    if (!fs::is_directory(dest.parent_path())) throw std::runtime_error("export parent directory must exist");
    Snapshot snapshot;
    {
        auto read = db.read();
        const auto object = db.objectByUuid(read, objectUuid);
        if (!object) throw std::runtime_error("object not found");
        const auto revision = revisionUuid ? db.geometryRevision(read, *revisionUuid) : db.currentGeometry(read, objectUuid);
        if (!revision) throw std::runtime_error("geometry revision not found");
        if (revision->objectUuid != object->uuid) throw std::runtime_error("geometry revision belongs to another object");
        snapshot.revision = *revision;
        if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz")
            throw std::runtime_error("unsupported geometry format for native export");
        const std::set<std::string> names = revision->kind == "fiber" ? std::set<std::string>{"control_points", "line_points"} :
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
    const auto attributes = Json::parse(snapshot.revision.attributes);
    if (kind == "fiber") fiber(snapshot, attributes, output);
    else if (kind == "point_collection") collection(snapshot, attributes, output);
    else if (kind == "patch") patch(snapshot, attributes, output);
    else if (kind == "track") track(snapshot, attributes, output);
    else throw std::runtime_error("unsupported geometry kind for export");
    publish(output, dest, overwrite, directory);
    return snapshot.revision.uuid;
}
} // namespace vc::objects
