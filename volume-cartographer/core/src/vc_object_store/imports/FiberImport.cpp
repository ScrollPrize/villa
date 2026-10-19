#include "vc/core/objects/import/FiberImport.hpp"
#include "imports/ImportInternal.hpp"
#include "internal/EdgeSql.hpp"
#include "formats/FiberInput.hpp"

namespace vc::objects {
namespace {

std::string fnvHashHex(const std::string& text)
{
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : text) h = (h ^ c) * 1099511628211ULL;
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

struct ParsedBranch {
    int localCp = -1;
    std::string targetFile;
    int targetCp = -1;
    bool pending = false;
    bool adjacent = false;
    Anchor local, target;
};

Anchor branchAnchor(const nlohmann::json& j, const char* indexKey, const char* posKey, const char* dirKey)
{
    Anchor a;
    if (j.contains(indexKey) && j.at(indexKey).is_number_integer()) a.index = j.at(indexKey).get<int64_t>();
    for (int i = 0; i < 2; ++i) {
        const char* key = i == 0 ? posKey : dirKey;
        if (j.contains(key) && j.at(key).is_array() && j.at(key).size() == 3) {
            std::array<double, 3> v{};
            for (int k = 0; k < 3; ++k) v[size_t(k)] = j.at(key).at(size_t(k)).get<double>();
            if (i == 0) a.position = v; else a.direction = v;
        }
    }
    return a;
}

}  // namespace

ImportReport importFibers(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& fiberDir,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "scanning fiber directory");
    struct Parsed {
        std::filesystem::path path;
        std::string fileName;
        ObjectRecord object;
        std::vector<ParsedBranch> branches;
        std::string checksum;
    };
    std::vector<Parsed> parsed;
    std::unordered_map<std::string, size_t> byFileName;

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(fiberDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    size_t fileIndex = 0;
    for (const auto& file : files) {
        util::reportProgress(progress, "reading fibers", fileIndex++, files.size(), file.filename().string());
        nlohmann::json root;
        std::string text;
        try {
            text = readFile(file);
            root = nlohmann::json::parse(text);
        } catch (const std::exception& e) {
            report.problems.push_back(file.filename().string() + ": unreadable/unparseable JSON: " + e.what());
            continue;
        }
        if (!root.is_object() || !root.contains("uuid") ||
            !(root.contains("control_points") || root.contains("line_points")))
            continue;  // not a vc3d_fiber file; other JSON in the dir is not our business
        const std::string uuid = root.value("uuid", "");
        if (!util::isUuid(uuid)) {
            report.problems.push_back(file.filename().string() + ": missing/invalid uuid, skipped");
            continue;
        }
        if (db.currentGeometry(txn, uuid)) {
            report.problems.push_back(file.filename().string() +
                ": geometry is database-owned; use ingestGeometrySource to publish a revision (legacy branch import skipped)");
            continue;
        }
        Parsed p;
        p.path = file;
        p.fileName = file.filename().string();
        p.checksum = fnvHashHex(text);
        ObjectRecord& o = p.object;
        o.uuid = formatUuid(parseUuid(uuid));
        o.kind = "fiber";
        o.name = file.stem().string();
        o.formatVersion = root.value("version", int64_t(1));
        o.storageKind = "file";
        o.storageRef = relativeToDb(db, file);
        o.dataGeneration = 1;
        o.dataChecksum = p.checksum;
        if (root.contains("tags") && root.at("tags").is_array())
            for (const auto& t : root.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        FiberDetail f;
        try {
            auto points = objects::detail::fiberPoints(root, "line_points");
            if (points.empty()) points = objects::detail::fiberPoints(root, "control_points");
            f.length = objects::detail::fiberPolylineLength(points);
        } catch (const std::exception& error) {
            report.problems.push_back(p.fileName + ": invalid fiber geometry: " + error.what());
            continue;
        }
        if (root.contains("control_points") && root.at("control_points").is_array())
            f.controlPointCount = int64_t(root.at("control_points").size());
        if (root.contains("line_points") && root.at("line_points").is_array())
            f.linePointCount = int64_t(root.at("line_points").size());
        f.width = root.value("width", 0.0);
        f.widthGapFraction = root.value("width_gap_fraction", 0.2);
        f.optimizationMode = root.value("optimization_mode", "");
        if (root.contains("hv_classification") && root.at("hv_classification").is_object()) {
            const auto& hv = root.at("hv_classification");
            const std::string manual = hv.value("manual_tag", "");
            const std::string automatic = hv.value("automatic_tag", "");
            f.manualFamily = !manual.empty();
            f.family = f.manualFamily ? manual : automatic;
        }
        o.fiber = f;
        for (const auto* arrayKey : {"branches", "adjacent_branches"}) {
            if (!root.contains(arrayKey) || !root.at(arrayKey).is_array()) continue;
            for (const auto& bj : root.at(arrayKey)) {
                if (!bj.is_object()) continue;
                ParsedBranch b;
                b.adjacent = std::string(arrayKey) == "adjacent_branches";
                b.localCp = bj.value("control_point_index", -1);
                b.targetCp = bj.value("branch_control_point_index", -1);
                b.targetFile = std::filesystem::path(bj.value("branch_file", "")).filename().string();
                b.pending = bj.value("pending", false);
                b.local = branchAnchor(bj, "control_point_index", "control_point_position", "control_point_direction");
                b.target = branchAnchor(bj, "branch_control_point_index", "branch_control_point_position", "branch_control_point_direction");
                p.branches.push_back(std::move(b));
            }
        }
        byFileName[p.fileName] = parsed.size();
        parsed.push_back(std::move(p));
    }

    util::reportProgress(progress, "reading fibers", files.size(), files.size());

    // Objects first (edges need their endpoints to exist). Unchanged
    // files (same uuid + content hash) are skipped: re-import is a no-op.
    size_t objectIndex = 0;
    for (const auto& p : parsed) {
        util::reportProgress(progress, "registering fibers", objectIndex++, parsed.size());
        auto refreshed = p.object;
        if (auto existing = db.objectByUuid(txn, p.object.uuid)) {
            if (existing->dataChecksum && *existing->dataChecksum == p.checksum) {
                db.setFiberLength(txn, p.object.uuid, p.object.fiber->length);
                continue;
            }
            // Files seed annotations once. Subsequent imports own only the
            // source descriptor and measurements, not graph names/tags/meta.
            refreshed.name = existing->name;
            refreshed.tags = existing->tags;
            refreshed.meta = existing->meta;
            refreshed.dataGeneration = existing->dataGeneration + 1;
            for (const auto& edge : db.edgesOf(txn, existing->uuid))
                stampContentChange(ObjectStoreImplAccess::conn(txn), db.author(), db.hostname(), "edges",
                    *edgeIdByUuid(ObjectStoreImplAccess::conn(txn), parseUuid(edge.uuid)), parseUuid(edge.uuid), "geometry_invalidated");
        }
        db.upsertObject(txn, refreshed);
        ++report.objectsImported;
    }

    util::reportProgress(progress, "registering fibers", parsed.size(), parsed.size());

    auto hasReciprocal = [&](const Parsed& owner, const ParsedBranch& b) {
        auto it = byFileName.find(b.targetFile);
        if (it == byFileName.end()) return false;
        for (const auto& other : parsed[it->second].branches)
            if (other.targetFile == owner.fileName && other.localCp == b.targetCp && other.targetCp == b.localCp &&
                other.adjacent == b.adjacent)
                return true;
        return false;
    };

    std::set<std::string> seenPairs, retainedBranches, importedObjects;
    for (const auto& p : parsed) importedObjects.insert(p.object.uuid);
    size_t branchIndex = 0;
    for (const auto& p : parsed) {
        util::reportProgress(progress, "importing fiber branches", branchIndex++, parsed.size());
        for (const auto& b : p.branches) {
            auto it = byFileName.find(b.targetFile);
            if (it == byFileName.end()) {
                report.problems.push_back(p.fileName + ": branch target not in imported set: " + b.targetFile);
                continue;
            }
            const Parsed& target = parsed[it->second];
            std::string keyA = p.fileName + "#" + std::to_string(b.localCp);
            std::string keyB = b.targetFile + "#" + std::to_string(b.targetCp);
            const std::string pairKey = keyA < keyB ? keyA + "|" + keyB : keyB + "|" + keyA;
            const bool reciprocal = hasReciprocal(p, b);
            if (!reciprocal)
                report.problems.push_back(p.fileName + ": one-sided branch to " + b.targetFile +
                                          " (edge imported from this side only)");
            if (!seenPairs.insert(pairKey).second)
                continue;
            std::optional<EdgeRecord> existing;
            for (const auto& edge : db.edgesOf(txn, p.object.uuid)) {
                const bool forward = edge.aUuid == p.object.uuid;
                if ((forward ? edge.bUuid : edge.aUuid) == target.object.uuid && edge.kind == "fiber_branch" &&
                    (forward ? edge.aAnchor : edge.bAnchor).index == b.local.index &&
                    (forward ? edge.bAnchor : edge.aAnchor).index == b.target.index) {
                    existing = edge;
                    break;
                }
            }
            if (existing) {
                retainedBranches.insert(existing->uuid);
                auto meta = nlohmann::json::parse(existing->meta);
                // Unmarked legacy/manual edges have ambiguous ownership. Do
                // not silently claim or delete them based on external files.
                if (meta.value("source_import", "") != "fiber-json") continue;
                auto refreshed = *existing;
                refreshed.aUuid = p.object.uuid; refreshed.bUuid = target.object.uuid;
                refreshed.aAnchor = b.local; refreshed.bAnchor = b.target;
                meta["adjacent"] = b.adjacent;
                refreshed.meta = meta.dump();
                db.updateEdge(txn, refreshed);
                continue;
            }
            EdgeRecord edge;
            edge.kind = "fiber_branch";
            edge.aUuid = p.object.uuid; edge.bUuid = target.object.uuid;
            edge.aAnchor = b.local; edge.bAnchor = b.target;
            edge.meta = nlohmann::json{{"adjacent", b.adjacent}, {"source_import", "fiber-json"}}.dump();
            edge.approvalState = b.pending ? "pending" : "approved";
            retainedBranches.insert(addEdgeImpl(ObjectStoreImplAccess::conn(txn), db.author(), db.hostname(), edge, true));
            ++report.edgesImported;
        }
    }
    // Only reconcile this importer's edges when both files were parsed.
    // Missing, unreadable, or omitted files cannot establish branch removal.
    for (const auto& edge : db.edgesByKind(txn, "fiber_branch")) {
        if (!importedObjects.contains(edge.aUuid) || !importedObjects.contains(edge.bUuid) || retainedBranches.contains(edge.uuid)) continue;
        const auto meta = nlohmann::json::parse(edge.meta);
        if (meta.value("source_import", "") == "fiber-json") db.removeEdge(txn, edge.uuid);
    }
    util::reportProgress(progress, "importing fiber branches", parsed.size(), parsed.size());
    return report;
}

} // namespace vc::objects
