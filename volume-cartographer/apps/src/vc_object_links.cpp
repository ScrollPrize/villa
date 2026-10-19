#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/object_links/ObjectLinks.hpp"
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
        << "usage: vc_object_links --db <objects.vcobj> --author <name> [options]\n"
        << "  --no-progress        disable terminal progress reporting (stderr)\n"
        << "  --link-pcl-patches             link stored point collections to stored patches\n"
        << "  --pcl-tolerance <vx>           maximum point distance (default 2.5)\n"
        << "  --point-scale <factor>         point XYZ -> working XYZ\n"
        << "  --pcl-window-points <n>        support window (default 1)\n"
        << "  --pcl-window-min-points <n>    required support (default 1)\n"
        << "  --pcl-absolute / --pcl-relative override stored winding role\n"
        << "  --link-track-tracks            discover crossings between stored tracks\n"
        << "  --track-dataset <ref>          restrict to stored dataset identity\n"
        << "  --track-uuid <uuid>            select track (repeatable)\n"
        << "  --patch-uuid <uuid>            select patch (repeatable)\n"
        << "  --fiber-uuid <uuid>            select fiber (repeatable)\n"
        << "  --collection-uuid <uuid>       select collection (repeatable)\n"
        << "  --track-workers <n>            crossing workers (default 1)\n"
        << "  --track-min-angle <deg>        crossing angle gate (default 30)\n"
        << "  --track-tangent-radius <vx>    tangent radius (default 12)\n"
        << "  --track-cluster-radius <n>     cluster index radius (default 4)\n"
        << "  --link-patch-overlaps          discover overlaps between stored patches\n"
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
        << "  --link-track-patches            link stored tracks to stored patches\n"
        << "  --track-scale <factor>          track XYZ -> working XYZ (default 1)\n"
        << "  --strip-rows <n>                shared strip rows each side (default 1)\n"
        << "  --anchor-contact <vx>           shared minimum anchor contact (default 75)\n"
        << "  --link-fiber-patches            link stored fibers to stored patches\n"
        << "  --strip-cache <dir>             optional disposable fiber-strip cache\n"
        << "  --rebuild-fiber-strips          refresh strip caches (e.g. after CT voxel edits)\n"
        << "  --ct <path>                     uint8 CT volume for linking (default normal source)\n"
        << "  --normals-manifest <path>        use Lasagna normals in the same linker instead of CT\n"
        << "  --ct-level <level>              CT pyramid level (default 0)\n"
        << "  --ct-to-base-scale <factor>     selected CT voxel size in base units (default 1)\n"
        << "  --fiber-scale <factor>           fiber XYZ -> working XYZ (default 1)\n"
        << "  --patch-scale <factor>           patch XYZ -> working XYZ (default 1)\n"
        << "  --working-to-base-scale <factor> working voxel size in common base units (default 1)\n"
        << "  --import-json <path> import a vc_object_metadata JSON document\n"
        << "  --export-json <path> write a vc_object_metadata JSON document (- for stdout)\n"
        << "Ingest geometry with vc_object_ingest before discovery.\n";
}

}  // namespace

int main(int argc, char** argv)
try {
    fs::path dbPath, importJsonPath, exportJsonPath;
    fs::path ctVolume;
    bool linkFibers = false, linkTrackPatches = false, linkPcl = false, linkOverlaps = false;
    vc::object_links::PointCollectionPatchParams pclParams;
    vc::object_links::TrackPatchParams trackPatchParams;
    vc::object_links::FiberPatchParams fiberPatchParams;
    bool linkTracks = false;
    bool showProgress = true;
    vc::object_links::TrackTrackParams trackParams;
    std::string author;
    vc::object_links::PatchOverlapParams overlapParams;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (arg == "--no-progress") showProgress = false;
        else if (arg == "--db") dbPath = next("--db");
        else if (arg == "--author") author = next("--author");
        else if (arg == "--link-pcl-patches") linkPcl = true;
        else if (arg == "--pcl-tolerance") pclParams.tolerance = std::stod(next(arg.c_str()));
        else if (arg == "--point-scale") pclParams.pointScale = std::stod(next(arg.c_str()));
        else if (arg == "--pcl-window-points") pclParams.windowPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-window-min-points") pclParams.windowMinPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-absolute") pclParams.absoluteWinding = true;
        else if (arg == "--pcl-relative") pclParams.absoluteWinding = false;
        else if (arg == "--link-track-tracks") linkTracks = true;
        else if (arg == "--patch-uuid") {
            const auto uuid = next(arg.c_str());
            overlapParams.patchUuids.push_back(uuid);
            fiberPatchParams.patchUuids.push_back(uuid);
            pclParams.patchUuids.push_back(uuid);
        }
        else if (arg == "--fiber-uuid") fiberPatchParams.fiberUuids.push_back(next(arg.c_str()));
        else if (arg == "--collection-uuid") pclParams.collectionUuids.push_back(next(arg.c_str()));
        else if (arg == "--strip-cache") fiberPatchParams.cacheDirectory = next(arg.c_str());
        else if (arg == "--track-dataset") trackParams.datasetRef = next(arg.c_str());
        else if (arg == "--track-uuid") trackParams.trackUuids.push_back(next(arg.c_str()));
        else if (arg == "--track-workers") trackParams.workers = std::stoi(next(arg.c_str()));
        else if (arg == "--track-min-angle") trackParams.minAngleDegrees = std::stod(next(arg.c_str()));
        else if (arg == "--track-tangent-radius") trackParams.tangentRadiusVoxels = std::stod(next(arg.c_str()));
        else if (arg == "--track-cluster-radius") trackParams.clusterIndexRadius = std::stoi(next(arg.c_str()));
        else if (arg == "--link-patch-overlaps") linkOverlaps = true;
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
        else if (arg == "--link-track-patches") linkTrackPatches = true;
        else if (arg == "--track-scale") trackPatchParams.trackScale = std::stod(next(arg.c_str()));
        else if (arg == "--link-fiber-patches") linkFibers = true;
        else if (arg == "--rebuild-fiber-strips") fiberPatchParams.rebuildStrips = true;
        else if (arg == "--ct") ctVolume = next(arg.c_str());
        else if (arg == "--normals-manifest") fiberPatchParams.normalsManifest = next(arg.c_str());
        else if (arg == "--ct-level") fiberPatchParams.ctLevel = std::stoi(next(arg.c_str()));
        else if (arg == "--ct-to-base-scale") fiberPatchParams.ctToBaseScale = std::stod(next(arg.c_str()));
        else if (arg == "--fiber-scale") fiberPatchParams.fiberScale = std::stod(next(arg.c_str()));
        else if (arg == "--patch-scale") fiberPatchParams.patchScale = std::stod(next(arg.c_str()));
        else if (arg == "--working-to-base-scale") fiberPatchParams.workingToBaseScale = std::stod(next(arg.c_str()));
        else if (arg == "--strip-rows") fiberPatchParams.stripRowsEachSide = std::stoi(next(arg.c_str()));
        else if (arg == "--anchor-contact") fiberPatchParams.anchorContact = std::stod(next(arg.c_str()));
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

    if ((linkFibers || linkTrackPatches) && ctVolume.empty() && fiberPatchParams.normalsManifest.empty())
        throw std::runtime_error("patch linking requires --ct or --normals-manifest");
    pclParams.patchScale = fiberPatchParams.patchScale;

    static_cast<vc::object_links::LinePatchParams&>(trackPatchParams) = fiberPatchParams;
    trackPatchParams.trackUuids = trackParams.trackUuids;

    OpenOptions options;
    options.author = author;
    std::error_code ec;
    const bool exists = fs::exists(dbPath) && fs::file_size(dbPath, ec) > 0;
    ObjectStore db = [&] {
        vc::util::TerminalProgress progress("Open object database", showProgress);
        auto store = exists ? ObjectStore::open(dbPath, options) : ObjectStore::create(dbPath, options);
        progress.finish();
        return store;
    }();
    if (!exists)
        std::cout << "created " << dbPath.string() << "\n";

    bool linkFailed = false;
    if (linkTracks) {
        vc::util::TerminalProgress progress("Track crossings", showProgress);
        trackParams.progress = progress.callback();
        const auto report = vc::object_links::linkTracksToTracks(db, trackParams);
        progress.finish();
        linkFailed |= !report.problems.empty();
        std::cout << "track-tracks: " << report.objectsConsidered << " tracks, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    if (linkOverlaps) {
        vc::util::TerminalProgress progress("Patch overlaps", showProgress);
        overlapParams.progress = progress.callback();
        const auto report = vc::object_links::linkPatchOverlaps(db, overlapParams);
        progress.finish();
        linkFailed |= !report.problems.empty();
        std::cout << "patch-overlaps: " << report.objectsConsidered << " patches, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    if (linkTrackPatches) {
        vc::util::TerminalProgress progress("Track to patch", showProgress);
        trackPatchParams.progress = progress.callback();
        const auto report = vc::object_links::linkTracksToPatches(
            db, ctVolume, trackPatchParams);
        progress.finish();
        linkFailed |= !report.problems.empty();
        std::cout << "track-patches: " << report.objectsConsidered << " objects, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    if (linkFibers) {
        vc::util::TerminalProgress progress("Fiber to patch", showProgress);
        fiberPatchParams.progress = progress.callback();
        const auto report = vc::object_links::linkFibersToPatches(
            db, ctVolume, fiberPatchParams);
        progress.finish();
        linkFailed |= !report.problems.empty();
        std::cout << "fiber-patches: " << report.objectsConsidered << " objects, "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems)
            std::cerr << "  ! " << problem << "\n";
    }
    if (linkPcl) {
        vc::util::TerminalProgress progress("Point collections to patches", showProgress);
        pclParams.progress = progress.callback();
        const auto report = vc::object_links::linkPointCollectionsToPatches(db, pclParams);
        progress.finish();
        linkFailed |= !report.problems.empty();
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
        vc::util::reportProgress(progress.callback(), "serializing object metadata");
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
    return linkFailed ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "vc_object_links: " << e.what() << "\n";
    return 1;
}
