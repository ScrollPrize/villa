#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/TerminalProgress.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
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
        << "  --link-pcl-surfaces             link stored point collections to stored surfaces\n"
        << "  --pcl-tolerance <vx>           maximum point distance (default 2.5)\n"
        << "  --point-scale <factor>         point XYZ -> working XYZ\n"
        << "  --pcl-window-points <n>        support window (default 1)\n"
        << "  --pcl-window-min-points <n>    required support (default 1)\n"
        << "  --pcl-absolute / --pcl-relative override stored winding role\n"
        << "  --link-track-tracks            discover crossings between stored tracks\n"
        << "  --track-dataset <ref>          restrict to stored dataset identity\n"
        << "  --track-uuid <uuid>            select track (repeatable)\n"
        << "  --surface-uuid <uuid>            select surface (repeatable)\n"
        << "  --fiber-uuid <uuid>            select fiber (repeatable)\n"
        << "  --collection-uuid <uuid>       select collection (repeatable)\n"
        << "  --track-workers <n>            crossing workers (default 1)\n"
        << "  --track-min-angle <deg>        crossing angle gate (default 30)\n"
        << "  --track-tangent-radius <vx>    tangent radius (default 12)\n"
        << "  --track-cluster-radius <n>     cluster index radius (default 4)\n"
        << "  --link-surface-overlaps          discover overlaps between stored surfaces\n"
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
        << "  --link-track-surfaces            link stored tracks to stored surfaces\n"
        << "  --track-scale <factor>          track XYZ -> working XYZ (default 1)\n"
        << "  --strip-rows <n>                shared strip rows each side (default 1)\n"
        << "  --anchor-contact <vx>           shared minimum anchor contact (default 75)\n"
        << "  --link-fiber-surfaces            link stored fibers to stored surfaces\n"
        << "  --strip-cache <dir>             optional disposable fiber-strip cache\n"
        << "  --rebuild-fiber-strips          refresh strip caches (e.g. after CT voxel edits)\n"
        << "  --ct <path>                     uint8 CT volume for linking (default normal source)\n"
        << "  --ct-level <level>              CT pyramid level (default 0)\n"
        << "  --ct-to-base-scale <factor>     selected CT voxel size in base units (default 1)\n"
        << "  --fiber-scale <factor>           fiber XYZ -> working XYZ (default 1)\n"
        << "  --surface-scale <factor>           surface XYZ -> working XYZ (default 1)\n"
        << "  --working-to-base-scale <factor> working voxel size in common base units (default 1)\n"
        << "Ingest geometry with vc_object_ingest before discovery.\n";
}

}  // namespace

int main(int argc, char** argv)
try {
    fs::path dbPath;
    fs::path ctVolume;
    bool linkFibers = false, linkTrackSurfaces = false, linkPcl = false, linkOverlaps = false;
    vc::object_links::PointCollectionSurfaceParams pclParams;
    vc::object_links::TrackSurfaceParams trackSurfaceParams;
    vc::object_links::FiberSurfaceParams fiberSurfaceParams;
    bool linkTracks = false;
    bool showProgress = true;
    vc::object_links::TrackTrackParams trackParams;
    std::string author;
    vc::object_links::SurfaceOverlapParams overlapParams;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[i];
        };
        if (arg == "--no-progress") showProgress = false;
        else if (arg == "--db") dbPath = next("--db");
        else if (arg == "--author") author = next("--author");
        else if (arg == "--link-pcl-surfaces") linkPcl = true;
        else if (arg == "--pcl-tolerance") pclParams.tolerance = std::stod(next(arg.c_str()));
        else if (arg == "--point-scale") pclParams.pointScale = std::stod(next(arg.c_str()));
        else if (arg == "--pcl-window-points") pclParams.windowPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-window-min-points") pclParams.windowMinPoints = std::stoi(next(arg.c_str()));
        else if (arg == "--pcl-absolute") pclParams.absoluteWinding = true;
        else if (arg == "--pcl-relative") pclParams.absoluteWinding = false;
        else if (arg == "--link-track-tracks") linkTracks = true;
        else if (arg == "--surface-uuid") {
            const auto uuid = next(arg.c_str());
            overlapParams.surfaceUuids.push_back(uuid);
            fiberSurfaceParams.surfaceUuids.push_back(uuid);
            pclParams.surfaceUuids.push_back(uuid);
        }
        else if (arg == "--fiber-uuid") fiberSurfaceParams.fiberUuids.push_back(next(arg.c_str()));
        else if (arg == "--collection-uuid") pclParams.collectionUuids.push_back(next(arg.c_str()));
        else if (arg == "--strip-cache") fiberSurfaceParams.cacheDirectory = next(arg.c_str());
        else if (arg == "--track-dataset") trackParams.datasetRef = next(arg.c_str());
        else if (arg == "--track-uuid") trackParams.trackUuids.push_back(next(arg.c_str()));
        else if (arg == "--track-workers") trackParams.workers = std::stoi(next(arg.c_str()));
        else if (arg == "--track-min-angle") trackParams.minAngleDegrees = std::stod(next(arg.c_str()));
        else if (arg == "--track-tangent-radius") trackParams.tangentRadiusVoxels = std::stod(next(arg.c_str()));
        else if (arg == "--track-cluster-radius") trackParams.clusterIndexRadius = std::stoi(next(arg.c_str()));
        else if (arg == "--link-surface-overlaps") linkOverlaps = true;
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
        else if (arg == "--link-track-surfaces") linkTrackSurfaces = true;
        else if (arg == "--track-scale") trackSurfaceParams.trackScale = std::stod(next(arg.c_str()));
        else if (arg == "--link-fiber-surfaces") linkFibers = true;
        else if (arg == "--rebuild-fiber-strips") fiberSurfaceParams.rebuildStrips = true;
        else if (arg == "--ct") ctVolume = next(arg.c_str());
        else if (arg == "--ct-level") fiberSurfaceParams.ctLevel = std::stoi(next(arg.c_str()));
        else if (arg == "--ct-to-base-scale") fiberSurfaceParams.ctToBaseScale = std::stod(next(arg.c_str()));
        else if (arg == "--fiber-scale") fiberSurfaceParams.fiberScale = std::stod(next(arg.c_str()));
        else if (arg == "--surface-scale") fiberSurfaceParams.surfaceScale = std::stod(next(arg.c_str()));
        else if (arg == "--working-to-base-scale") fiberSurfaceParams.workingToBaseScale = std::stod(next(arg.c_str()));
        else if (arg == "--strip-rows") fiberSurfaceParams.stripRowsEachSide = std::stoi(next(arg.c_str()));
        else if (arg == "--anchor-contact") fiberSurfaceParams.anchorContact = std::stod(next(arg.c_str()));
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

    if ((linkFibers || linkTrackSurfaces) && ctVolume.empty())
        throw std::runtime_error("surface linking requires --ct");
    pclParams.surfaceScale = fiberSurfaceParams.surfaceScale;

    static_cast<vc::object_links::LineSurfaceParams&>(trackSurfaceParams) = fiberSurfaceParams;
    trackSurfaceParams.trackUuids = trackParams.trackUuids;

    OpenOptions options;
    options.author = author;
    std::error_code ec;
    const bool exists = fs::exists(dbPath) && fs::file_size(dbPath, ec) > 0;
    ObjectStore db = [&] {
        vc::objects::util::TerminalProgress progress("Open object database", showProgress);
        auto store = exists ? ObjectStore::open(dbPath, options) : ObjectStore::create(dbPath, options);
        progress.finish();
        return store;
    }();
    if (!exists)
        std::cout << "created " << dbPath.string() << "\n";

    bool linkFailed = false;
    auto run = [&](const char* title, const char* label, const char* noun,
                   auto& params, auto detector, const auto&... inputs) {
        vc::objects::util::TerminalProgress progress(title, showProgress);
        params.progress = progress.callback();
        const auto report = detector(db, inputs..., params, nullptr);
        progress.finish();
        linkFailed |= !report.problems.empty();
        std::cout << label << ": " << report.objectsConsidered << " " << noun << ", "
                  << report.candidatesFound << " candidates, " << report.edgesCreated << " edges created, "
                  << report.edgesUpdated << " updated, " << report.edgesSkippedExisting << " skipped, " << report.staleRemoved << " stale removed\n";
        for (const auto& problem : report.problems) std::cerr << "  ! " << problem << "\n";
    };
    using namespace vc::object_links;
    if (linkTracks) run("Track crossings", "track-tracks", "tracks", trackParams, linkTracksToTracks);
    if (linkOverlaps) run("Surface overlaps", "surface-overlaps", "surfaces", overlapParams, linkSurfaceOverlaps);
    if (linkTrackSurfaces) run("Track to surface", "track-surfaces", "objects", trackSurfaceParams, linkTracksToSurfaces, ctVolume);
    if (linkFibers) run("Fiber to surface", "fiber-surfaces", "objects", fiberSurfaceParams, linkFibersToSurfaces, ctVolume);
    if (linkPcl) run("Point collections to surfaces", "pcl-surfaces", "objects", pclParams, linkPointCollectionsToSurfaces);
    return linkFailed ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "vc_object_links: " << e.what() << "\n";
    return 1;
}
