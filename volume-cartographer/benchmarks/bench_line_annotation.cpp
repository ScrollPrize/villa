// The line annotation window on real fibers: for each fiber, the views VC3D's
// Line Annotation builds (vc::lasagna::buildLineViewSurfaces: the bottom strip,
// the side strip, and the current/side cut planes) shown in four annotation
// viewers, then stepped along the fiber the way scrolling the line position
// does: the cuts follow LineAnnotationDialog::updatePlaneSurface /
// updateSidePlaneSurface and the strip cameras recentre on the position.
// Fibers are VC3D fiber JSON files (line_points in level-0 voxels).

#include "bench_common.hpp"

#include "CState.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

using namespace vcbench;

namespace {

struct Fiber {
    std::string name;
    std::vector<cv::Vec3d> points;
};

// The `count` fibers with the most line points (ties by file name): the longest
// walks along a fiber, and a deterministic choice.
std::vector<Fiber> loadFibers(const std::filesystem::path& folder, int count, double coordinateScale)
{
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(folder))
        if (entry.path().extension() == ".json")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    std::vector<Fiber> fibers;
    for (const auto& file : files) {
        std::ifstream in(file);
        const auto doc = nlohmann::json::parse(in, nullptr, false);
        if (doc.is_discarded() || !doc.contains("line_points") || !doc["line_points"].is_array())
            continue;
        Fiber fiber{file.filename().string(), {}};
        for (const auto& p : doc["line_points"])
            if (p.is_array() && p.size() == 3)
                fiber.points.emplace_back(p[0].get<double>() * coordinateScale, p[1].get<double>() * coordinateScale,
                                          p[2].get<double>() * coordinateScale);
        if (fiber.points.size() >= 2)
            fibers.push_back(std::move(fiber));
    }
    std::stable_sort(fibers.begin(), fibers.end(),
                     [](const Fiber& a, const Fiber& b) { return a.points.size() > b.points.size(); });
    if (int(fibers.size()) > count)
        fibers.resize(std::size_t(count));
    return fibers;
}

cv::Vec3f normalized(const cv::Vec3f& v)
{
    const float n = float(cv::norm(v));
    return n > 1e-6f ? v / n : cv::Vec3f(NAN, NAN, NAN);
}

bool finite(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

struct Views {
    vc::lasagna::LineViewSurfaces surfaces;
    vc::lasagna::LineModel line;
    std::shared_ptr<PlaneSurface> cut = std::make_shared<PlaneSurface>();
    std::shared_ptr<PlaneSurface> sideCut = std::make_shared<PlaneSurface>();
};

// The cut planes and strip cameras at fractional line position `position`.
void moveTo(Views& views, double position, std::vector<Viewer*>& viewers)
{
    const auto& pts = views.line.points;
    const auto last = pts.size() - 1;
    const auto i0 = std::min(last, std::size_t(std::floor(position)));
    const auto i1 = std::min(last, i0 + 1);
    const double t = position - double(i0);
    const cv::Vec3f origin(pts[i0].position * (1 - t) + pts[i1].position * t);
    const auto a = std::min(i0, last - 1);
    const cv::Vec3f tangent = normalized(cv::Vec3f(pts[a + 1].position - pts[a].position));
    // The cut's up: the line's display frame normal, else its transported up vector.
    cv::Vec3f up(pts[i0].sampledNormal.normal);
    if (!finite(normalized(up)) && i0 < views.surfaces.lineUpVectors.size())
        up = views.surfaces.lineUpVectors[i0];
    if (finite(origin) && finite(tangent) && finite(normalized(up))) {
        views.cut->setFromNormalAndUp(origin, tangent, up);
        const auto sideNormal = normalized(up.cross(tangent));
        if (finite(sideNormal))
            views.sideCut->setFromNormalAndUp(origin, sideNormal, tangent);
    }
    const double column = views.surfaces.stripPositionMap.originalPositionToStripGridColumn(position);
    for (auto* strip : {views.surfaces.lineSurface.get(), views.surfaces.lineSideSlice.get()}) {
        if (!strip || !std::isfinite(column))
            continue;
        const auto uv = strip->gridToSurface({column, double(strip->rawPointsPtr()->rows) / 2});
        for (auto* viewer : viewers)
            if (viewer->get().currentSurface() == strip)
                viewer->get().centerOnSurfacePoint(cv::Vec2f(float(uv[0]), float(uv[1])));
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        Args args(argc, argv);
        Session session(argc, argv, args);
        const auto fibersDir = args.str("fibers");
        if (fibersDir.empty())
            throw std::invalid_argument("--fibers <folder of fiber JSON> is required");
        const int fiberCount = int(args.num("fiber-count", 4));
        const double fiberScale = args.num("fiber-scale", 1);
        const int steps = int(args.num("steps", 24));
        const int passes = int(args.num("passes", 2));
        const int stripW = session.viewSize(int(args.num("strip-width", 1600)));
        const int stripH = session.viewSize(int(args.num("strip-height", 320)));
        const int cutSize = session.viewSize(int(args.num("cut-size", 640)));
        const double timeoutMs = args.num("timeout-ms", 120000);
        const auto json = args.str("json");
        const auto dump = args.str("dump");
        args.rejectUnused();

        const auto fibers = loadFibers(fibersDir, fiberCount, fiberScale);
        if (fibers.empty())
            throw std::runtime_error("no fiber with line_points in " + fibersDir);
        Report report("line_annotation", session.pipelineName(), {});

        const std::string stripName = "line strip", sideName = "line side strip", cutName = "line cut",
                          sideCutName = "line side cut";
        std::vector<std::unique_ptr<Viewer>> owned;
        owned.push_back(std::make_unique<Viewer>(session, stripName, true, stripW, stripH));
        owned.push_back(std::make_unique<Viewer>(session, sideName, true, stripW, stripH));
        owned.push_back(std::make_unique<Viewer>(session, cutName, true, cutSize, cutSize));
        owned.push_back(std::make_unique<Viewer>(session, sideCutName, true, cutSize, cutSize));
        std::vector<Viewer*> viewers;
        for (auto& v : owned)
            viewers.push_back(v.get());
        session.showVolume();
        waitSettled(session, viewers, Clock::now(), timeoutMs);

        std::string names;
        for (int pass = 0; pass < passes; ++pass) {
            const std::string tag = pass == 0 ? "first " : "repeat ";
            for (const auto& fiber : fibers) {
                if (pass == 0)
                    names += (names.empty() ? "" : ",") + fiber.name + "(" + std::to_string(fiber.points.size()) + ")";
                // Open: build the views, register them, show the middle of the fiber.
                auto start = Clock::now();
                Views views;
                views.line = vc::lasagna::lineModelForInspection(fiber.points);
                vc::lasagna::LineViewConfig config;
                config.buildLineZSlices = false;
                views.surfaces = vc::lasagna::buildLineViewSurfaces(views.line, config);
                report.add(tag + "build line views", "ms", msSince(start));
                session.state().setSurface(stripName, views.surfaces.lineSurface);
                session.state().setSurface(sideName, views.surfaces.lineSideSlice);
                const double middle = double(fiber.points.size() - 1) / 2;
                moveTo(views, middle, viewers);
                session.state().setSurface(cutName, views.cut);
                session.state().setSurface(sideCutName, views.sideCut);
                record(report, tag + "open fiber (4 viewers)", waitSettled(session, viewers, start, timeoutMs));
                if (!dump.empty() && pass == 0) {
                    std::filesystem::create_directories(dump);
                    for (auto* v : viewers)
                        v->save(std::filesystem::path(dump) /
                                (session.pipelineName() + "-" + fiber.name + "-" + v->surfaceName() + ".png"));
                }

                // Scroll along the whole fiber.
                for (int s = 0; s < steps; ++s) {
                    const double position = double(fiber.points.size() - 1) * s / std::max(1, steps - 1);
                    start = Clock::now();
                    moveTo(views, position, viewers);
                    session.state().setSurface(cutName, views.cut, false, true);
                    session.state().setSurface(sideCutName, views.sideCut, false, true);
                    record(report, tag + "step along fiber (4 viewers)",
                           waitSettled(session, viewers, start, timeoutMs));
                }
            }
        }

        session.note("fibers", names);
        session.noteCaches();
        report.setConfig(session.config());
        report.finish(json);
        owned.clear();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_line_annotation: %s\n", e.what());
        return 1;
    }
}
