#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/TerminalProgress.hpp"
#include <iostream>

int main(int argc, char** argv)
try {
    std::filesystem::path path;
    std::string author;
    std::optional<std::filesystem::path> root;
    std::vector<std::string> uuids;
    std::vector<std::pair<std::string, std::filesystem::path>> sources;
    std::optional<std::string> volumeUuid, scaleFactor, uuidNamespace;
    bool registered = false, showProgress = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--db") path = next();
        else if (arg == "--author") author = next();
        else if (arg == "--dataset-root") root = next();
        else if (arg == "--registered") registered = true;
        else if (arg == "--uuid") { uuids.push_back(next()); registered = true; }
        else if (arg == "--patch") sources.emplace_back("patch", next());
        else if (arg == "--fiber") sources.emplace_back("fiber", next());
        else if (arg == "--point-collections") sources.emplace_back("point_collection", next());
        else if (arg == "--tracks") sources.emplace_back("track", next());
        else if (arg == "--volume-uuid") volumeUuid = next();
        else if (arg == "--scale-factor") scaleFactor = next();
        else if (arg == "--derive-uuids") uuidNamespace = next();
        else if (arg == "--no-progress") showProgress = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_object_ingest --db objects.vcobj --author NAME [options]\n"
                "  --registered              load initial geometry for registered placeholders\n"
                "  --uuid UUID               load selected placeholders (flag repeatable)\n"
                "  --dataset-root DIR        resolve registered source paths relative to DIR\n"
                "  --patch DIR               import a new tifxyz patch\n"
                "  --fiber PATH              import new fibers from JSON or a directory\n"
                "  --point-collections FILE  import new collections from a JSON\n"
                "  --tracks PATH             import new packed tracks (or DBM with packed mirror)\n"
                "  --volume-uuid ID          coordinate frame: catalog volume ID (metadata.json id)\n"
                "  --scale-factor S          coordinate frame: stored units per volume voxel\n"
                "  --derive-uuids NAMESPACE  identities for sources without UUIDs (see below)\n"
                "  --no-progress             suppress progress display\n"
                "Fiber/collection sources and registered batches commit atomically; tracks commit individually. Failures are reported and exit status is nonzero.\n"
                "Source flags are repeatable. Existing object identities are errors, even when unchanged.\n"
                "--volume-uuid and --scale-factor go together and apply to every object ingested by this run:\n"
                "they fill in a frame the source does not declare, and a conflicting declaration is an error.\n"
                "--derive-uuids NAMESPACE (a UUID, one per dataset) admits patches, fibers and PCL collections\n"
                "that lack canonical UUIDs, as VC3D-era files do. Each gets uuid5(NAMESPACE, KIND:KEY): a patch\n"
                "keyed by its meta.json uuid or directory name, a fiber by its file stem (version-3 branches\n"
                "resolve the same way), a PCL collection by DOCUMENT_STEM:ID. Rerunning yields the same UUIDs,\n"
                "so a repeated import fails as an existing identity. Canonical UUIDs are kept; tracks are unaffected.\n"
                "Edit imported geometry, annotations and relationships through the object API.\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (path.empty() || author.empty() || (!registered && sources.empty()))
        throw std::runtime_error("specify --db, --author and --registered/--uuid or source flags; see --help");
    if (uuidNamespace && !vc::objects::util::isUuid(*uuidNamespace))
        throw std::runtime_error("--derive-uuids requires a UUID namespace");
    if (volumeUuid.has_value() != scaleFactor.has_value())
        throw std::runtime_error("--volume-uuid and --scale-factor must be given together");
    nlohmann::json frame = nlohmann::json::object();
    if (volumeUuid) {
        std::size_t parsed = 0;
        double scale = 0;
        try { scale = std::stod(*scaleFactor, &parsed); } catch (const std::exception&) {}
        if (parsed == 0 || parsed != scaleFactor->size())
            throw std::runtime_error("--scale-factor must be a number");
        frame = vc::objects::geometryCoordinates({{"volume_uuid", *volumeUuid}, {"scale_factor", scale}});
    }
    const auto coordinateContext = frame.dump();
    vc::objects::OpenOptions options; options.author = author;
    auto db = std::filesystem::exists(path) ? vc::objects::ObjectStore::open(path, options) : vc::objects::ObjectStore::create(path, options);
    bool failed = false;
    auto print = [&](const vc::objects::GeometryIngestReport& report) {
        std::cout << report.ingested << " ingested, " << report.problems.size() << " problems\n";
        for (const auto& problem : report.problems) std::cerr << problem << '\n';
        failed |= !report.problems.empty();
    };
    for (const auto& [kind, source] : sources) {
        vc::objects::util::TerminalProgress progress("Ingest " + kind, showProgress);
        const auto report = vc::objects::ingestGeometrySource(db, kind, source, progress.callback(), coordinateContext, uuidNamespace);
        progress.finish(); print(report);
    }
    if (registered) {
        vc::objects::util::TerminalProgress progress("Ingest registered geometry", showProgress);
        const auto report = vc::objects::ingestRegisteredGeometry(db, root, uuids, progress.callback(), coordinateContext);
        progress.finish(); print(report);
    }
    db.checkpoint();
    return failed ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_ingest: " << error.what() << '\n';
    return 1;
}
