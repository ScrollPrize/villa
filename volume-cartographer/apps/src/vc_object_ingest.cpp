#include "vc/core/objects/GeometryIngest.hpp"
#include "vc/core/util/TerminalProgress.hpp"
#include <iostream>

int main(int argc, char** argv)
try {
    std::filesystem::path path;
    std::string author;
    std::optional<std::filesystem::path> root;
    std::vector<std::string> uuids;
    std::vector<std::pair<std::string, std::filesystem::path>> sources;
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
        else if (arg == "--no-progress") showProgress = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_object_ingest --db objects.vcobj --author NAME [options]\n"
                "  --registered              ingest all registered linkable objects\n"
                "  --uuid UUID               ingest selected registered objects (repeatable)\n"
                "  --dataset-root DIR        resolve registered source paths relative to DIR\n"
                "  --patch DIR               ingest/register a tifxyz patch\n"
                "  --fiber FILE              ingest/register a vc3d_fiber JSON\n"
                "  --point-collections FILE  ingest/register every collection in a JSON\n"
                "  --tracks PATH             ingest/register packed tracks (or DBM with packed mirror)\n"
                "  --no-progress             suppress progress display\n"
                "Each object commits atomically. Failures are reported and exit status is nonzero.\n"
                "Source flags are repeatable. Existing identities, tags and metadata are retained;\n"
                "new geometry revisions reset object approval; graphs adopt revisions explicitly.\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (path.empty() || author.empty() || (!registered && sources.empty()))
        throw std::runtime_error("specify --db, --author and --registered/--uuid or source flags; see --help");
    vc::objects::OpenOptions options; options.author = author;
    auto db = std::filesystem::exists(path) ? vc::objects::ObjectStore::open(path, options) : vc::objects::ObjectStore::create(path, options);
    bool failed = false;
    auto print = [&](const vc::objects::GeometryIngestReport& report) {
        std::cout << report.ingested << " ingested, " << report.unchanged << " unchanged, " << report.problems.size() << " problems\n";
        for (const auto& problem : report.problems) std::cerr << problem << '\n';
        failed |= !report.problems.empty();
    };
    for (const auto& [kind, source] : sources) {
        vc::util::TerminalProgress progress("Ingest " + kind, showProgress);
        const auto report = vc::objects::ingestGeometrySource(db, kind, source, progress.callback());
        progress.finish(); print(report);
    }
    if (registered) {
        vc::util::TerminalProgress progress("Ingest registered geometry", showProgress);
        const auto report = vc::objects::ingestRegisteredGeometry(db, root, uuids, progress.callback());
        progress.finish(); print(report);
    }
    db.checkpoint();
    return failed ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_ingest: " << error.what() << '\n';
    return 1;
}
