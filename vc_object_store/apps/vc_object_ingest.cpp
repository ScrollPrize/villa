#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/TerminalProgress.hpp"
#include <iostream>

int main(int argc, char** argv)
try {
    std::filesystem::path path;
    std::string author;
    std::vector<std::pair<std::string, std::filesystem::path>> sources;
    struct Channel { std::string uuid, name; std::filesystem::path image; };
    std::vector<Channel> channels;
    std::vector<std::string> tags;
    bool showProgress = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--db") path = next();
        else if (arg == "--author") author = next();
        else if (arg == "--surface") sources.emplace_back("surface", next());
        else if (arg == "--fiber") sources.emplace_back("fiber", next());
        else if (arg == "--point-collections") sources.emplace_back("point_collection", next());
        else if (arg == "--umbilicus") sources.emplace_back("umbilicus", next());
        else if (arg == "--tracks") sources.emplace_back("track", next());
        else if (arg == "--surface-channel") {
            Channel channel; channel.uuid = next(); channel.name = next(); channel.image = next();
            channels.push_back(std::move(channel));
        }
        else if (arg == "--tag") tags.push_back(next());
        else if (arg == "--no-progress") showProgress = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_object_ingest --db objects.vcobj --author NAME [options]\n"
                "  --surface DIR               import a new tifxyz surface, or a folder of them\n"
                "  --fiber PATH              import new fibers from JSON or a directory\n"
                "  --point-collections FILE  import new collections from a JSON\n"
                "  --umbilicus FILE          import a new umbilicus from a vc_umbilicus JSON\n"
                "  --tracks PATH             import new packed tracks (or DBM with packed mirror)\n"
                "  --surface-channel UUID NAME TIFF\n"
                "                            add or replace a channel of a stored surface\n"
                "  --tag NAME                tag every imported object (repeatable)\n"
                "  --no-progress             suppress progress display\n"
                "Fiber/collection sources commit atomically; tracks commit individually. Failures are reported and exit status is nonzero.\n"
                "Source flags are repeatable. Existing object identities are errors, even when unchanged.\n"
                "Edit imported geometry, annotations and relationships through the object API.\n";
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (path.empty() || author.empty() || (sources.empty() && channels.empty()))
        throw std::runtime_error("specify --db, --author and source or channel flags; see --help");
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
        const auto report = vc::objects::ingestGeometrySource(db, kind, source, progress.callback(), "{}", tags);
        progress.finish(); print(report);
    }
    for (const auto& channel : channels) {
        try {
            auto txn = db.write();
            const auto revision = vc::objects::importSurfaceChannel(db, txn, channel.uuid, channel.name, channel.image);
            txn.commit();
            std::cout << channel.uuid << ": channel " << channel.name << " in geometry revision " << revision << '\n';
        } catch (const std::exception& error) {
            std::cerr << channel.uuid << ": " << error.what() << '\n';
            failed = true;
        }
    }
    db.checkpoint();
    return failed ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_ingest: " << error.what() << '\n';
    return 1;
}
