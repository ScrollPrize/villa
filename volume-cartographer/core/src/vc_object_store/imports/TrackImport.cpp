#include "vc/core/objects/import/TrackImport.hpp"
#include "imports/ImportInternal.hpp"
#include "internal/JsonFields.hpp"
#include "internal/RecordSql.hpp"

namespace vc::objects {
ImportReport importTrackList(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& trackListJson,
    const util::ProgressCallback& progress)
{
    ImportReport report;
    util::reportProgress(progress, "reading track list");
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(readFile(trackListJson));
    } catch (const std::exception& e) {
        throw std::runtime_error("object store: cannot parse track list " + trackListJson.string() + ": " + e.what());
    }
    if (!root.is_array())
        throw std::runtime_error("object store: track list must be a JSON array: " + trackListJson.string());

    // Index already-registered tracks by identity so re-registration
    // reuses the minted UUID instead of colliding with the UNIQUE index.
    std::map<std::pair<std::string, std::string>, ObjectRecord> registered;
    for (auto& existing : db.objectsByKind(txn, "track"))
        if (existing.track)
            registered[{existing.track->datasetRef, existing.track->externalKey}] = existing;

    for (size_t i = 0; i < root.size(); ++i) {
        util::reportProgress(progress, "registering tracks", i, root.size());
        const auto& entry = root.at(i);
        const std::string label = "entry " + std::to_string(i);
        if (!entry.is_object()) {
            report.problems.push_back(label + ": not an object, skipped");
            continue;
        }
        const std::string datasetRef = entry.value("dataset_ref", "");
        const std::string externalKey = entry.value("external_key", "");
        if (datasetRef.empty() || externalKey.empty()) {
            report.problems.push_back(label + ": missing dataset_ref/external_key, skipped");
            continue;
        }
        ObjectRecord o;
        o.kind = "track";
        o.name = entry.value("name", externalKey);
        o.storageKind = "dataset";
        o.storageRef = datasetRef;
        o.dataGeneration = 1;
        if (entry.contains("tags") && entry.at("tags").is_array())
            for (const auto& t : entry.at("tags"))
                if (t.is_string()) o.tags.push_back(t.get<std::string>());
        if (entry.contains("meta")) o.meta = entry.at("meta").dump();
        TrackDetail t;
        t.datasetRef = datasetRef;
        t.externalKey = externalKey;
        t.family = entry.value("family", "");
        t.pointCount = entry.value("point_count", int64_t(0));
        t.length = entry.value("length", 0.0);
        if (entry.contains("bbox")) t.bbox = bboxFromJson(entry.at("bbox"));
        o.track = t;

        auto it = registered.find({datasetRef, externalKey});
        if (it != registered.end()) {
            const ObjectRecord& cur = it->second;
            o.name = cur.name; o.tags = cur.tags; o.meta = cur.meta;
            if (db.currentGeometry(txn, cur.uuid)) {
                report.problems.push_back(label + ": geometry is database-owned; use ingestGeometrySource to publish a revision");
                continue;
            }
            const bool unchanged = cur.track &&
                                   cur.track->family == t.family && cur.track->pointCount == t.pointCount &&
                                   cur.track->length == t.length && cur.track->bbox == t.bbox;
            if (unchanged)
                continue;  // idempotent re-registration
            o.uuid = cur.uuid;
            db.upsertObject(txn, o);
            ++report.objectsImported;
        } else {
            const std::string uuid = db.upsertObject(txn, o);
            o.uuid = uuid;
            registered[{datasetRef, externalKey}] = o;
            ++report.objectsImported;
        }
    }
    util::reportProgress(progress, "registering tracks", root.size(), root.size());
    return report;
}

} // namespace vc::objects
