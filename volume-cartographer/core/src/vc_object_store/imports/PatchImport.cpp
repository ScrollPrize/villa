#include "vc/core/objects/import/PatchImport.hpp"
#include "imports/ImportInternal.hpp"
#include "internal/JsonFields.hpp"
#include "internal/RecordSql.hpp"

namespace vc::objects {
ImportReport importPatches(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& segmentsDir,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "scanning patch directory");
    std::vector<std::filesystem::path> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(segmentsDir))
        if (entry.is_directory()) dirs.push_back(entry.path());
    std::sort(dirs.begin(), dirs.end());
    size_t index = 0;
    for (const auto& dir : dirs) {
        util::reportProgress(progress, "registering patches", index++, dirs.size(), dir.filename().string());
        const auto metaPath = dir / "meta.json";
        if (!std::filesystem::exists(metaPath)) continue;
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(readFile(metaPath));
        } catch (const std::exception& e) {
            report.problems.push_back(dir.filename().string() + ": unparseable meta.json: " + e.what());
            continue;
        }
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(dir.filename().string() + ": meta.json has no valid uuid, skipped");
            continue;
        }
        if (db.objectByUuid(txn, uuid)) continue;  // already registered
        ObjectRecord o;
        o.uuid = uuid;
        o.kind = "patch";
        o.name = dir.filename().string();
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, dir);
        o.dataGeneration = 1;
        PatchDetail p;
        try {
            p.gridWidth = root.value("grid_width", int64_t(0));
            p.gridHeight = root.value("grid_height", int64_t(0));
            p.scale = scaleFromJson(root.value("scale", nlohmann::json(0.0)));
            if (root.contains("bbox")) {
                const auto& box = root.at("bbox");
                if (box.is_array() && box.size() == 2 && box.at(0).is_array() && box.at(1).is_array() &&
                    box.at(0).size() == 3 && box.at(1).size() == 3) {
                    p.bbox = std::array<double, 6>{box[0][0].get<double>(), box[0][1].get<double>(), box[0][2].get<double>(),
                        box[1][0].get<double>(), box[1][1].get<double>(), box[1][2].get<double>()};
                } else {
                    p.bbox = bboxFromJson(box);
                    if (!p.bbox) throw std::runtime_error("invalid patch bbox");
                }
            }
        } catch (const std::exception& e) {
            report.problems.push_back(dir.filename().string() + ": invalid patch metadata: " + e.what());
            continue;
        }
        o.patch = p;
        db.upsertObject(txn, o);
        ++report.objectsImported;
    }
    util::reportProgress(progress, "registering patches", dirs.size(), dirs.size());
    return report;
}

} // namespace vc::objects
