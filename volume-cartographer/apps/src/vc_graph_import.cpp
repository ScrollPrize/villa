// Import existing fiber/segment metadata into a vc_graph database and
// export the metadata graph as JSON. See
// docs/plans/graph_db_initial_implementation.md (Phase 4).

#include "vc/core/graph/GraphDb.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace vc::graph;

namespace {

void usage()
{
    std::cerr
        << "usage: vc_graph_import --db <graph.vcgr> --author <name> [options]\n"
        << "  --fibers <dir>       import vc3d_fiber JSON files (objects + branch edges)\n"
        << "  --segments <dir>     register segment dirs with meta.json as patch objects\n"
        << "  --tracks <file>      register tracks from a JSON interchange list\n"
        << "                       ([{dataset_ref, external_key, family, point_count, ...}])\n"
        << "  --import-json <path> import a vc_graph_export JSON document\n"
        << "  --export-json <path> write a vc_graph_export JSON document (- for stdout)\n"
        << "Creates the database if it does not exist. Imports run in one write\n"
        << "transaction each and print a report (problems go to stderr).\n";
}

void printReport(const char* what, const ImportReport& report)
{
    std::cout << what << ": " << report.objectsImported << " objects, " << report.edgesImported
              << " edges, " << report.problems.size() << " problems\n";
    for (const auto& problem : report.problems)
        std::cerr << "  ! " << problem << "\n";
}

}  // namespace

int main(int argc, char** argv)
try {
    fs::path dbPath, fiberDir, segmentsDir, tracksPath, importJsonPath, exportJsonPath;
    std::string author;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (arg == "--db") dbPath = next("--db");
        else if (arg == "--author") author = next("--author");
        else if (arg == "--fibers") fiberDir = next("--fibers");
        else if (arg == "--segments") segmentsDir = next("--segments");
        else if (arg == "--tracks") tracksPath = next("--tracks");
        else if (arg == "--import-json") importJsonPath = next("--import-json");
        else if (arg == "--export-json") exportJsonPath = next("--export-json");
        else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (dbPath.empty() || author.empty()) {
        usage();
        return 2;
    }

    OpenOptions options;
    options.author = author;
    std::error_code ec;
    const bool exists = fs::exists(dbPath) && fs::file_size(dbPath, ec) > 0;
    GraphDb db = exists ? GraphDb::open(dbPath, options) : GraphDb::create(dbPath, options);
    if (!exists)
        std::cout << "created " << dbPath.string() << "\n";

    if (!fiberDir.empty()) {
        auto w = db.write();
        ImportReport report = importFibers(db, w, fiberDir);
        w.commit();
        printReport("fibers", report);
    }
    if (!segmentsDir.empty()) {
        auto w = db.write();
        ImportReport report = importPatches(db, w, segmentsDir);
        w.commit();
        printReport("segments", report);
    }
    if (!tracksPath.empty()) {
        auto w = db.write();
        ImportReport report = importTrackList(db, w, tracksPath);
        w.commit();
        printReport("tracks", report);
    }
    if (!importJsonPath.empty()) {
        std::ifstream in(importJsonPath);
        if (!in) throw std::runtime_error("cannot read " + importJsonPath.string());
        std::ostringstream ss;
        ss << in.rdbuf();
        auto w = db.write();
        db.importJson(w, ss.str());
        w.commit();
        std::cout << "imported JSON from " << importJsonPath.string() << "\n";
    }
    if (!exportJsonPath.empty()) {
        std::string json;
        {
            auto r = db.read();
            json = db.exportJson(r);
        }
        db.checkpoint();
        if (exportJsonPath == "-") {
            std::cout << json << "\n";
        } else {
            std::ofstream(exportJsonPath) << json << "\n";
            std::cout << "exported to " << exportJsonPath.string() << "\n";
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "vc_graph_import: " << e.what() << "\n";
    return 1;
}
