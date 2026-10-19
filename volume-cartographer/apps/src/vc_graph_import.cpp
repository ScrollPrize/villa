#include "vc/core/objects/import/TrackImport.hpp"
#include "vc/core/objects/import/PatchImport.hpp"
#include "vc/core/objects/import/FiberImport.hpp"
// Import existing fiber/segment metadata into an object database and
// export the metadata graph as JSON. See
// docs/plans/graph_db_initial_implementation.md (Phase 4).

#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/TerminalProgress.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace vc::objects;

namespace {

void usage()
{
    std::cerr
        << "usage: vc_graph_import --db <graph.vcobj> --author <name> [options]\n"
        << "  --no-progress        disable terminal progress reporting (stderr)\n"
        << "  --fibers <dir>       import vc3d_fiber JSON files (objects + branch edges)\n"
        << "  --segments <dir>     register segment dirs with meta.json as patch objects\n"
        << "  --link-pcl-patches <json>       link point collections to --segments patches (repeatable; no CT)\n"
        << "  --pcl-tolerance <vx>           maximum point-to-patch distance (default 2.5)\n"
        << "  --point-scale <factor>         point XYZ -> working XYZ (default 1)\n"
        << "  --pcl-window-points <n>        centered support window (default 1)\n"
        << "  --pcl-window-min-points <n>    required hits in that window (default 1)\n"
        << "  --pcl-absolute / --pcl-relative override abs_winding.json filename rule\n"
        << "  --tracks <file>      register tracks from a JSON interchange list\n"
        << "                       ([{dataset_ref, external_key, family, point_count, ...}])\n"
        << "  --link-track-tracks <dbm>       register tracks and compute crossings directly into SQLite\n"
        << "  --track-source-id <id>         optional subset of source IDs (repeatable)\n"
        << "  --track-dataset <ref>           restrict to this exact registered dataset_ref\n"
        << "  --track-uuid <uuid>             restrict to these tracks (repeatable)\n"
        << "  --track-workers <n>             crossing workers (default 1)\n"
        << "  --track-min-angle <degrees>     minimum acute tangent angle (default 30)\n"
        << "  --track-tangent-radius <vx>     tangent radius in dataset voxels (default 12)\n"
        << "  --track-cluster-radius <n>      event index clustering radius (default 4)\n"
        << "  --link-patch-overlaps <dir>\n"
        << "                       run the patch-overlap detector over the tifxyz\n"
        << "                       segment dirs in <dir>; creates pending edges\n"
        << "  --overlap-workers <n>            search workers (default 0: auto, up to 8)\n"
        << "  --overlap-tolerance <vx>          (default 2)\n"
        << "  --overlap-uv-tolerance <vx>       (default 3)\n"
        << "  --overlap-sample-spacing <vx>     (default 4)\n"
        << "  --overlap-min-inliers <n>         (default 32)\n"
        << "  --overlap-min-major-spread <vx>   (default 20)\n"
        << "  --overlap-min-minor-spread <vx>   (default 5, the 'thickness' gate)\n"
        << "  --overlap-max-rms <vx>            (default 2)\n"
        << "  --overlap-agreement <0..1>        (default 0.6)\n"
        << "  --overlap-min-fraction <0..1>     (default 0 = off)\n"
        << "  --overlap-edge-kind <name>        (default same-surface)\n"
        << "  --link-track-patches <dbm>      link tracks to --segments patches (same strip rules)\n"
        << "  --track-scale <factor>          track XYZ -> working XYZ (default 1)\n"
        << "  --strip-rows <n>                shared strip rows each side (default 1)\n"
        << "  --anchor-contact <vx>           shared minimum anchor contact (default 75)\n"
        << "  --link-fiber-patches <dir>       link fibers in <dir> to --segments patches\n"
        << "                       strips are cached automatically in <dir>/.strip-cache/\n"
        << "  --rebuild-fiber-strips          refresh strip caches (e.g. after CT voxel edits)\n"
        << "  --ct <path>                     uint8 CT volume for linking (default normal source)\n"
        << "  --normals-manifest <path>        use Lasagna normals in the same linker instead of CT\n"
        << "  --ct-level <level>              CT pyramid level (default 0)\n"
        << "  --ct-to-base-scale <factor>     selected CT voxel size in base units (default 1)\n"
        << "  --fiber-scale <factor>           fiber XYZ -> working XYZ (default 1)\n"
        << "  --patch-scale <factor>           patch XYZ -> working XYZ (default 1)\n"
        << "  --working-to-base-scale <factor> working voxel size in common base units (default 1)\n"
        << "  --fiber-strip-rows <n>           rows each side (default 1, at 20 vx)\n"
        << "  --fiber-anchor-contact <vx>      minimum anchor contact (default 75)\n"
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
    fs::path dbPath, fiberDir, segmentsDir, tracksPath, overlapDir, importJsonPath, exportJsonPath;
    fs::path linkFiberDir, linkTrackPath, ctVolume;
    std::vector<fs::path> pclFiles;
    vc::graph_links::PointCollectionPatchParams pclParams;
    vc::graph_links::TrackPatchParams trackPatchParams;
    vc::graph_links::FiberPatchParams fiberPatchParams;
    bool linkTracks = false;
    bool showProgress = true;
    vc::graph_links::TrackTrackParams trackParams;
    std::string author;
    vc::graph_links::PatchOverlapParams overlapParams;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (arg == "--no-progress") showProgress = false;
        else if (arg == "--db") dbPath = next("--db");
        else if (arg == "--author") author = next("--author");
        else if (arg == "--fibers") fiberDir = next("--fibers");
        else if (arg == "--segments") segmentsDir = next("--segments");
        else if (arg == "--link-pcl-patches") pclFiles.emplace_back(next(arg.c_str()));
        else if (arg == "--pcl-tolerance") pclParams.tolerance = std::stod(next(arg.c_str()));
        else if (arg == "--point-scale") pclParams.pointScale = std::stod(next(arg.c_str()));
        else if (arg == "--pcl-window-points") pclParams.windowPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-window-min-points") pclParams.windowMinPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-absolute") pclParams.absoluteWinding = true;
        else if (arg == "--pcl-relative") pclParams.absoluteWinding = false;
        else if (arg == "--tracks") tracksPath = next("--tracks");
        else if (arg == "--link-track-tracks") {
            linkTracks = true;
            // A bare flag retains the already-registered-track API.
            if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0)
                trackParams.tracksPath = next(arg.c_str());
        }
        else if (arg == "--track-source-id") {
            const auto text = next(arg.c_str());
            size_t used = 0;
            const auto id = std::stoull(text, &used);
            if (text.empty() || text[0] == '-' || used != text.size())
                throw std::runtime_error("invalid track source ID: " + text);
            trackParams.sourceIds.push_back(id);
        }
        else if (arg == "--track-dataset") trackParams.datasetRef = next(arg.c_str());
        else if (arg == "--track-uuid") trackParams.trackUuids.push_back(next(arg.c_str()));
        else if (arg == "--track-workers") trackParams.workers = std::stoi(next(arg.c_str()));
        else if (arg == "--track-min-angle") trackParams.minAngleDegrees = std::stod(next(arg.c_str()));
        else if (arg == "--track-tangent-radius") trackParams.tangentRadiusVoxels = std::stod(next(arg.c_str()));
        else if (arg == "--track-cluster-radius") trackParams.clusterIndexRadius = std::stoi(next(arg.c_str()));
        else if (arg == "--link-patch-overlaps") overlapDir = next("--link-patch-overlaps");
        else if (arg == "--overlap-workers") overlapParams.workers = std::stoi(next(arg.c_str()));
        else if (arg == "--overlap-tolerance") overlapParams.tolerance = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-uv-tolerance") overlapParams.uvInlierTolerance = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-sample-spacing") overlapParams.sampleSpacing = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-min-inliers") overlapParams.minInliers = std::stoull(next(arg.c_str()));
        else if (arg == "--overlap-min-major-spread") overlapParams.minMajorSpread = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-min-minor-spread") overlapParams.minMinorSpread = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-max-rms") overlapParams.maxRefitRms = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-agreement") overlapParams.minAgreementRatio = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-min-fraction") overlapParams.minOverlapFraction = std::stod(next(arg.c_str()));
        else if (arg == "--overlap-edge-kind") overlapParams.edgeKind = next(arg.c_str());
        else if (arg == "--link-track-patches") linkTrackPath = next(arg.c_str());
        else if (arg == "--track-scale") trackPatchParams.trackScale = std::stod(next(arg.c_str()));
        else if (arg == "--link-fiber-patches") linkFiberDir = next(arg.c_str());
        else if (arg == "--rebuild-fiber-strips") fiberPatchParams.rebuildStrips = true;
        else if (arg == "--ct") ctVolume = next(arg.c_str());
        else if (arg == "--normals-manifest") fiberPatchParams.normalsManifest = next(arg.c_str());
        else if (arg == "--ct-level") fiberPatchParams.ctLevel = std::stoi(next(arg.c_str()));
        else if (arg == "--ct-to-base-scale") fiberPatchParams.ctToBaseScale = std::stod(next(arg.c_str()));
        else if (arg == "--fiber-scale") fiberPatchParams.fiberScale = std::stod(next(arg.c_str()));
        else if (arg == "--patch-scale") fiberPatchParams.patchScale = std::stod(next(arg.c_str()));
        else if (arg == "--working-to-base-scale") fiberPatchParams.workingToBaseScale = std::stod(next(arg.c_str()));
        else if ((arg == "--fiber-strip-rows" || arg == "--strip-rows")) fiberPatchParams.stripRowsEachSide = std::stoi(next(arg.c_str()));
        else if ((arg == "--fiber-anchor-contact" || arg == "--anchor-contact")) fiberPatchParams.anchorContact = std::stod(next(arg.c_str()));
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

    if ((!linkFiberDir.empty() || !linkTrackPath.empty()) && (segmentsDir.empty() || (ctVolume.empty() && fiberPatchParams.normalsManifest.empty())))
        throw std::runtime_error("patch linking requires --segments and either --ct or --normals-manifest");
    if (!pclFiles.empty() && segmentsDir.empty())
        throw std::runtime_error("PCL linking requires --segments");
    pclParams.patchScale = fiberPatchParams.patchScale;

    static_cast<vc::graph_links::LinePatchParams&>(trackPatchParams) = fiberPatchParams;
    trackPatchParams.sourceIds = trackParams.sourceIds;
    trackPatchParams.trackUuids = trackParams.trackUuids;

    OpenOptions options;
    options.author = author;
    std::error_code ec;
    const bool exists = fs::exists(dbPath) && fs::file_size(dbPath, ec) > 0;
    ObjectStore db = [&] {
        vc::util::TerminalProgress progress("Open graph database", showProgress);
        auto graph = exists ? ObjectStore::open(dbPath, options) : ObjectStore::create(dbPath, options);
        progress.finish();
        return graph;
    }();
    if (!exists)
        std::cout << "created " << dbPath.string() << "\n";

    if (!fiberDir.empty()) {
        vc::util::TerminalProgress progress("Import fibers", showProgress);
        auto w = db.write();
        ImportReport report = importFibers(db, w, fiberDir, progress.callback());
        vc::util::reportProgress(progress.callback(), "committing transaction");
        w.commit();
        progress.finish();
        printReport("fibers", report);
    }
    if (!segmentsDir.empty() && linkFiberDir.empty() && linkTrackPath.empty() && pclFiles.empty()) {
        vc::util::TerminalProgress progress("Import patches", showProgress);
        auto w = db.write();
        ImportReport report = importPatches(db, w, segmentsDir, progress.callback());
        vc::util::reportProgress(progress.callback(), "committing transaction");
        w.commit();
        progress.finish();
        printReport("segments", report);
    }
    if (!tracksPath.empty()) {
        vc::util::TerminalProgress progress("Import tracks", showProgress);
        auto w = db.write();
        ImportReport report = importTrackList(db, w, tracksPath, progress.callback());
        vc::util::reportProgress(progress.callback(), "committing transaction");
        w.commit();
        progress.finish();
        printReport("tracks", report);
    }
    bool trackLinkFailed = false;
    if (linkTracks) {
        vc::util::TerminalProgress progress("Track crossings", showProgress);
        trackParams.progress = progress.callback();
        const auto report = vc::graph_links::linkTracksToTracks(db, trackParams);
        progress.finish();
        std::cout << "track-tracks: " << report.objectsConsidered << " tracks, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
        trackLinkFailed = !report.problems.empty();
    }
    if (!overlapDir.empty()) {
        vc::util::TerminalProgress progress("Patch overlaps", showProgress);
        overlapParams.progress = progress.callback();
        const auto report = vc::graph_links::linkPatchOverlaps(db, overlapDir, overlapParams);
        progress.finish();
        std::cout << "patch-overlaps: " << report.objectsConsidered << " patches, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    if (!linkTrackPath.empty()) {
        vc::util::TerminalProgress progress("Track to patch", showProgress);
        trackPatchParams.progress = progress.callback();
        const auto report = vc::graph_links::linkTracksToPatches(
            db, linkTrackPath, segmentsDir, ctVolume, trackPatchParams);
        progress.finish();
        std::cout << "track-patches: " << report.objectsConsidered << " objects, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
        trackLinkFailed |= !report.problems.empty();
    }
    if (!linkFiberDir.empty()) {
        vc::util::TerminalProgress progress("Fiber to patch", showProgress);
        fiberPatchParams.progress = progress.callback();
        const auto report = vc::graph_links::linkFibersToPatches(
            db, linkFiberDir, segmentsDir, ctVolume, fiberPatchParams);
        progress.finish();
        std::cout << "fiber-patches: " << report.objectsConsidered << " objects, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    for (const auto& pclFile : pclFiles) {
        vc::util::TerminalProgress progress("Point collections to patches", showProgress);
        pclParams.progress = progress.callback();
        const auto report = vc::graph_links::linkPointCollectionsToPatches(db, pclFile, segmentsDir, pclParams);
        progress.finish();
        std::cout << "pcl-patches: " << report.objectsConsidered << " objects, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems) std::cerr << "  ! " << problem << "\n";
    }
    if (!importJsonPath.empty()) {
        vc::util::TerminalProgress progress("Import JSON", showProgress);
        vc::util::reportProgress(progress.callback(), "reading JSON");
        std::ifstream in(importJsonPath);
        if (!in) throw std::runtime_error("cannot read " + importJsonPath.string());
        std::ostringstream ss;
        ss << in.rdbuf();
        vc::util::reportProgress(progress.callback(), "importing objects and edges");
        auto w = db.write();
        db.importJson(w, ss.str());
        vc::util::reportProgress(progress.callback(), "committing transaction");
        w.commit();
        progress.finish();
        std::cout << "imported JSON from " << importJsonPath.string() << "\n";
    }
    if (!exportJsonPath.empty()) {
        vc::util::TerminalProgress progress("Export JSON", showProgress);
        vc::util::reportProgress(progress.callback(), "serializing graph");
        std::string json;
        {
            auto r = db.read();
            json = db.exportJson(r);
        }
        vc::util::reportProgress(progress.callback(), "checkpointing database");
        db.checkpoint();
        vc::util::reportProgress(progress.callback(), "writing JSON");
        if (exportJsonPath == "-") {
            std::cout << json << "\n";
            progress.finish();
        } else {
            std::ofstream(exportJsonPath) << json << "\n";
            progress.finish();
            std::cout << "exported to " << exportJsonPath.string() << "\n";
        }
    }
    return trackLinkFailed ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "vc_graph_import: " << e.what() << "\n";
    return 1;
}
